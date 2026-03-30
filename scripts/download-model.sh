#!/usr/bin/env bash
# =============================================================================
# Chimere Model Downloader
#
# Downloads the appropriate GGUF model based on .env configuration
# (written by detect-gpu.py) and places it in the Docker volume.
#
# Usage:
#   ./scripts/download-model.sh              # auto from .env
#   ./scripts/download-model.sh --quant iq3s # override quant level
#   ./scripts/download-model.sh --help
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

# Defaults
DOCKER_VOLUME_NAME="chimere_chimere-models"
LOCAL_MODEL_DIR=""
FORCE=false
QUANT_OVERRIDE=""

# ---------------------------------------------------------------------------
# HuggingFace model registry
# ---------------------------------------------------------------------------
declare -A HF_REPOS=(
    ["q4km"]="Kevletesteur/Qwen3.5-35B-A3B-Q4_K_M-GGUF"
    ["iq3s"]="Kevletesteur/Qwen3.5-35B-A3B-RAMP-v2-15G"
    ["iq2"]="Kevletesteur/Qwen3.5-35B-A3B-IQ2_M-GGUF"
    ["iq2xs"]="Kevletesteur/Qwen3.5-35B-A3B-IQ2_XS-GGUF"
)

declare -A MODEL_FILES=(
    ["q4km"]="Qwen3.5-35B-A3B-Q4_K_M.gguf"
    ["iq3s"]="Qwen3.5-35B-A3B-RAMP-v2-15g.gguf"
    ["iq2"]="Qwen3.5-35B-A3B-IQ2_M.gguf"
    ["iq2xs"]="Qwen3.5-35B-A3B-IQ2_XS.gguf"
)

# Expected file sizes in bytes (approximate, for verification)
declare -A EXPECTED_SIZES=(
    ["q4km"]=19500000000    # ~19.5 GB
    ["iq3s"]=15200000000    # ~15.2 GB
    ["iq2"]=11500000000     # ~11.5 GB
    ["iq2xs"]=9800000000    # ~9.8 GB
)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage: download-model.sh [OPTIONS]

Downloads the appropriate GGUF model for your GPU configuration.
Reads .env (from detect-gpu.py) or accepts manual overrides.

Options:
  --quant LEVEL    Override quant level: q4km, iq3s, iq2, iq2xs
  --output DIR     Download to a local directory instead of Docker volume
  --force          Re-download even if the file already exists
  --list           List available quant levels and exit
  --help           Show this help message

Environment (.env):
  QUANT_TAG        Quant level (set by detect-gpu.py)
  MODEL_FILENAME   Expected filename
  MODEL_URL        HuggingFace repo URL

Examples:
  ./scripts/download-model.sh                   # auto from .env
  ./scripts/download-model.sh --quant q4km      # force Q4_K_M
  ./scripts/download-model.sh --output ./models  # download locally
USAGE
}

list_quants() {
    echo "Available quantisation levels:"
    echo ""
    printf "  %-8s %-45s %s\n" "TAG" "MODEL FILE" "APPROX SIZE"
    printf "  %-8s %-45s %s\n" "---" "----------" "-----------"
    printf "  %-8s %-45s %s\n" "q4km"  "${MODEL_FILES[q4km]}"  "~19.5 GB (24 GB+ VRAM)"
    printf "  %-8s %-45s %s\n" "iq3s"  "${MODEL_FILES[iq3s]}"  "~15.2 GB (16 GB VRAM)"
    printf "  %-8s %-45s %s\n" "iq2"   "${MODEL_FILES[iq2]}"   "~11.5 GB (12 GB VRAM)"
    printf "  %-8s %-45s %s\n" "iq2xs" "${MODEL_FILES[iq2xs]}" "~9.8 GB  (8 GB VRAM)"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

info() {
    echo "  [*] $*"
}

warn() {
    echo "  [!] $*" >&2
}

# Load .env file
load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        info "Loading ${ENV_FILE}"
        # Source only safe key=value lines (no export, no commands)
        while IFS='=' read -r key value; do
            key="$(echo "$key" | xargs)"
            [[ -z "$key" || "$key" == \#* ]] && continue
            value="$(echo "$value" | xargs)"
            export "$key=$value" 2>/dev/null || true
        done < "$ENV_FILE"
    else
        warn ".env not found at ${ENV_FILE}"
        warn "Run scripts/detect-gpu.py first, or use --quant to override."
    fi
}

# Auto-detect VRAM if no .env
auto_detect_vram() {
    if command -v nvidia-smi &>/dev/null; then
        local vram_mb
        vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
        if [[ -n "$vram_mb" ]]; then
            local vram_gb=$(( ${vram_mb%.*} / 1024 ))
            info "Auto-detected VRAM: ${vram_gb} GB"
            if (( vram_gb >= 24 )); then
                echo "q4km"
            elif (( vram_gb >= 16 )); then
                echo "iq3s"
            elif (( vram_gb >= 12 )); then
                echo "iq2"
            else
                echo "iq2xs"
            fi
            return 0
        fi
    fi
    return 1
}

# Resolve the Docker volume mount point
get_volume_path() {
    local vol_name="$1"
    # Create the volume if it does not exist
    if ! docker volume inspect "$vol_name" &>/dev/null; then
        info "Creating Docker volume: ${vol_name}"
        docker volume create "$vol_name" >/dev/null
    fi
    docker volume inspect "$vol_name" --format '{{ .Mountpoint }}'
}

# Download using huggingface-cli (preferred) or wget fallback
download_model() {
    local repo="$1"
    local filename="$2"
    local dest_dir="$3"
    local dest_path="${dest_dir}/${filename}"

    # Check if file already exists
    if [[ -f "$dest_path" ]] && [[ "$FORCE" != "true" ]]; then
        local existing_size
        existing_size="$(stat -c%s "$dest_path" 2>/dev/null || echo 0)"
        if (( existing_size > 1000000000 )); then
            info "Model already exists: ${dest_path} ($(numfmt --to=iec "$existing_size"))"
            info "Use --force to re-download."
            return 0
        fi
    fi

    info "Downloading: ${repo} -> ${filename}"
    info "Destination: ${dest_path}"
    echo ""

    if command -v huggingface-cli &>/dev/null; then
        info "Using huggingface-cli"
        huggingface-cli download "$repo" "$filename" \
            --local-dir "$dest_dir" \
            --local-dir-use-symlinks False
    elif command -v wget &>/dev/null; then
        info "Using wget (install huggingface-cli for better experience)"
        local url="https://huggingface.co/${repo}/resolve/main/${filename}"
        wget -c --show-progress -O "$dest_path" "$url"
    elif command -v curl &>/dev/null; then
        info "Using curl (install huggingface-cli for better experience)"
        local url="https://huggingface.co/${repo}/resolve/main/${filename}"
        curl -L -C - --progress-bar -o "$dest_path" "$url"
    else
        die "No download tool found. Install huggingface-cli, wget, or curl."
    fi
}

# Verify downloaded file
verify_model() {
    local filepath="$1"
    local quant="$2"

    if [[ ! -f "$filepath" ]]; then
        die "Download failed: file not found at ${filepath}"
    fi

    local actual_size
    actual_size="$(stat -c%s "$filepath")"
    local expected="${EXPECTED_SIZES[$quant]:-0}"

    info "File size: $(numfmt --to=iec "$actual_size")"

    if (( expected > 0 )); then
        # Allow 10% tolerance
        local lower=$(( expected * 90 / 100 ))
        local upper=$(( expected * 110 / 100 ))
        if (( actual_size < lower || actual_size > upper )); then
            warn "Size mismatch! Expected ~$(numfmt --to=iec "$expected"), got $(numfmt --to=iec "$actual_size")"
            warn "The file may be corrupted or incomplete. Re-run with --force."
            return 1
        fi
    fi

    info "Verification passed."
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo "=== Chimere Model Downloader ==="
    echo ""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --quant)
                QUANT_OVERRIDE="$2"
                shift 2
                ;;
            --output)
                LOCAL_MODEL_DIR="$2"
                shift 2
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --list)
                list_quants
                exit 0
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1 (try --help)"
                ;;
        esac
    done

    # Load configuration
    load_env

    # Determine quant level
    local quant=""
    if [[ -n "$QUANT_OVERRIDE" ]]; then
        quant="$QUANT_OVERRIDE"
    elif [[ -n "${QUANT_TAG:-}" ]]; then
        quant="$QUANT_TAG"
    else
        info "No QUANT_TAG in .env, auto-detecting from GPU..."
        quant="$(auto_detect_vram)" || die "Cannot detect GPU. Use --quant to specify."
    fi

    # Validate quant level
    if [[ -z "${HF_REPOS[$quant]+x}" ]]; then
        die "Unknown quant level: '${quant}'. Use --list to see options."
    fi

    local repo="${HF_REPOS[$quant]}"
    local filename="${MODEL_FILES[$quant]}"

    info "Quant level:  ${quant}"
    info "Repository:   ${repo}"
    info "Model file:   ${filename}"
    echo ""

    # Determine destination directory
    local dest_dir=""
    if [[ -n "$LOCAL_MODEL_DIR" ]]; then
        dest_dir="$LOCAL_MODEL_DIR"
        mkdir -p "$dest_dir"
        info "Download target: local directory ${dest_dir}"
    else
        # Use Docker volume
        if ! command -v docker &>/dev/null; then
            die "Docker not found. Use --output DIR to download locally instead."
        fi
        dest_dir="$(get_volume_path "$DOCKER_VOLUME_NAME")"
        info "Download target: Docker volume ${DOCKER_VOLUME_NAME} (${dest_dir})"
    fi
    echo ""

    # Download
    download_model "$repo" "$filename" "$dest_dir"
    echo ""

    # Verify
    local dest_path="${dest_dir}/${filename}"
    verify_model "$dest_path" "$quant"

    echo ""
    echo "=== Download complete ==="
    echo ""
    echo "  Model: ${dest_path}"
    echo ""
    if [[ -z "$LOCAL_MODEL_DIR" ]]; then
        echo "  The model is in Docker volume '${DOCKER_VOLUME_NAME}'."
        echo "  Set MODEL_FILENAME=${filename} in .env (should already be set)."
        echo ""
        echo "  Next step: cd docker && docker compose up -d"
    else
        echo "  To use with Docker, copy or bind-mount this file into the container."
    fi
}

main "$@"
