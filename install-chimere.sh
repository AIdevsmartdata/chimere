#!/usr/bin/env bash
# =============================================================================
# install-chimere.sh
#
# One-command bootstrap for Chimere on Linux (Ubuntu / Debian / Arch / Fedora).
# Clones the ik_llama.cpp backend fork, builds it with CUDA sm_120 (Blackwell)
# or sm_89 (Ada), builds the Rust server, and prints next-step instructions.
#
# This script is intentionally plain bash and does not assume Docker. For a
# Docker-based install, see docker/README.md.
#
# Usage:
#   ./install-chimere.sh                 # auto-detect SM, build release, no model pull
#   ./install-chimere.sh --sm 89         # force sm_89 (RTX 4090, etc.)
#   ./install-chimere.sh --with-model    # also pull the Chimere v3 RAMP GGUF
#   ./install-chimere.sh --skip-backend  # skip ik_llama.cpp build (already built)
#   ./install-chimere.sh --help
#
# Requirements (checked on start): git, curl, cmake, rustc, cargo, nvcc.
# This script does NOT install them for you — it tells you which are missing
# and leaves the package-manager choice to you.
# =============================================================================

set -euo pipefail

# --- paths and defaults -----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
IK_LLAMA_DIR="${IK_LLAMA_DIR:-$HOME/ik_llama.cpp}"
BUILD_DIR_NAME=""                  # set after SM detection
SM_ARCH=""                         # auto unless --sm
CUDA_PREFIX="${CUDA_PREFIX:-/usr/local/cuda-12.8}"
WITH_MODEL=0
SKIP_BACKEND=0
JOBS="${JOBS:-$(nproc)}"

# --- helpers ----------------------------------------------------------------
c_red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
c_green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
c_yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
c_cyan()   { printf '\033[0;36m%s\033[0m\n' "$*"; }

info() { c_cyan  "  [*] $*"; }
ok()   { c_green "  [+] $*"; }
warn() { c_yellow "  [!] $*" >&2; }
die()  { c_red   "  [x] $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
install-chimere.sh — Chimere local-first install helper

OPTIONS
  --sm N              CUDA SM arch (120 for RTX 50xx, 89 for RTX 40xx, 86 for RTX 30xx)
  --with-model        Also pull the Chimere v3 RAMP GGUF (~15.2 GB)
  --skip-backend      Do not rebuild ik_llama.cpp (assumes already built)
  --ik-dir PATH       Path where ik_llama.cpp lives (default: $HOME/ik_llama.cpp)
  --cuda PATH         CUDA toolkit prefix  (default: /usr/local/cuda-12.8)
  --jobs N            Parallel build jobs  (default: nproc)
  -h, --help          Show this help

ENVIRONMENT OVERRIDES
  IK_LLAMA_DIR, CUDA_PREFIX, JOBS

EXAMPLES
  ./install-chimere.sh                       # full auto on an RTX 5060 Ti
  ./install-chimere.sh --sm 89 --with-model  # RTX 4090, grab the weights too
  JOBS=4 ./install-chimere.sh                # slow laptop build

This script never runs `sudo`. Missing system packages are reported, not auto-installed.
USAGE
}

detect_sm() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found — NVIDIA driver required. Install NVIDIA driver first."
  fi
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  [[ -z "$name" ]] && die "Could not read GPU name from nvidia-smi."
  info "Detected GPU: $name"
  case "$name" in
    *"RTX 50"*|*"RTX 5090"*|*"B100"*|*"B200"*)      echo 120 ;;
    *"RTX 40"*|*"L40"*|*"L4"*|*"RTX 6000 Ada"*)     echo  89 ;;
    *"H100"*|*"H200"*|*"H20"*)                       echo  90 ;;
    *"RTX 30"*|*"RTX A"*)                            echo  86 ;;
    *"A100"*|*"A800"*)                               echo  80 ;;
    *"T4"*|*"RTX 20"*|*"Titan V"*)                   echo  75 ;;
    *)
      warn "Unknown GPU family ($name). Falling back to sm_89 (Ada)."
      echo 89
      ;;
  esac
}

check_tool() {
  local tool="$1" hint="$2"
  if ! command -v "$tool" >/dev/null 2>&1; then
    warn "Missing: $tool  — install with: $hint"
    return 1
  fi
  return 0
}

preflight() {
  info "Pre-flight checks"
  local missing=0
  check_tool git      "apt install git"       || missing=1
  check_tool curl     "apt install curl"      || missing=1
  check_tool cmake    "apt install cmake"     || missing=1
  check_tool rustc    "https://rustup.rs"     || missing=1
  check_tool cargo    "https://rustup.rs"     || missing=1
  if [[ ! -x "$CUDA_PREFIX/bin/nvcc" ]]; then
    warn "nvcc not found at $CUDA_PREFIX/bin/nvcc — install CUDA toolkit 12.x"
    missing=1
  fi
  if (( missing )); then
    die "Missing prerequisites. Install them and re-run."
  fi
  ok "Pre-flight OK"
}

build_backend() {
  if (( SKIP_BACKEND )); then
    ok "Skipping backend build (--skip-backend)"
    return 0
  fi

  BUILD_DIR_NAME="build_sm${SM_ARCH}"
  local build_dir="$IK_LLAMA_DIR/$BUILD_DIR_NAME"

  if [[ ! -d "$IK_LLAMA_DIR" ]]; then
    info "Cloning ik_llama.cpp fork -> $IK_LLAMA_DIR"
    git clone https://github.com/AIdevsmartdata/ik_llama.cpp.git "$IK_LLAMA_DIR"
  else
    info "ik_llama.cpp already present at $IK_LLAMA_DIR — fetching latest"
    git -C "$IK_LLAMA_DIR" fetch --quiet origin || true
  fi

  # Prefer the multi-arch backport branch if it exists.
  if git -C "$IK_LLAMA_DIR" rev-parse --verify mamba2-nemotron-h-backport >/dev/null 2>&1; then
    info "Checking out mamba2-nemotron-h-backport"
    git -C "$IK_LLAMA_DIR" checkout mamba2-nemotron-h-backport --quiet
  fi

  info "Configuring CMake (sm_${SM_ARCH}, CUDA at $CUDA_PREFIX)"
  CUDA_TOOLKIT_ROOT_DIR="$CUDA_PREFIX" \
    cmake -S "$IK_LLAMA_DIR" -B "$build_dir" \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${SM_ARCH}" \
      -DGGML_NATIVE=OFF >/dev/null

  info "Building ik_llama.cpp  (this takes 5-15 min, -j$JOBS)"
  cmake --build "$build_dir" -j"$JOBS"
  ok "Backend built at $build_dir"
}

build_server() {
  info "Building chimere-server (cargo release)"
  local ld_path="$IK_LLAMA_DIR/$BUILD_DIR_NAME/ggml/src:$IK_LLAMA_DIR/$BUILD_DIR_NAME/src:$CUDA_PREFIX/lib64"
  (
    cd "$REPO_ROOT/chimere-server"
    # Expose build-time paths for build.rs + ffi/build.rs
    # (they resolve IKLLAMACPP_DIR + IK_LLAMA_BUILD_SUBDIR with sensible
    #  fallbacks, but we pin them here for deterministic builds).
    IKLLAMACPP_DIR="$IK_LLAMA_DIR" \
    IK_LLAMA_BUILD_SUBDIR="$BUILD_DIR_NAME" \
    LD_LIBRARY_PATH="$ld_path" \
      cargo build --release --features server --bin chimere-server
  )
  ok "Server binary at $REPO_ROOT/chimere-server/target/release/chimere-server"
}

pull_model() {
  if (( ! WITH_MODEL )); then return 0; fi
  local model_dir="$HOME/.openclaw/models/Qwen3.5-35B-A3B-Chimere-v3-GGUF"
  mkdir -p "$model_dir"
  if command -v huggingface-cli >/dev/null 2>&1; then
    info "Pulling Chimere v3 RAMP GGUF (~15.2 GB)"
    huggingface-cli download Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF \
      chimere-v3-ramp.gguf --local-dir "$model_dir" --local-dir-use-symlinks False
    ok "Model at $model_dir/chimere-v3-ramp.gguf"
  else
    warn "huggingface-cli not found — install with: pip install --user 'huggingface_hub[cli]'"
    warn "Or download manually: https://huggingface.co/Kevletesteur/Qwen3.5-35B-A3B-Chimere-v3-GGUF"
  fi
}

print_next_steps() {
  local ld_path="$IK_LLAMA_DIR/$BUILD_DIR_NAME/ggml/src:$IK_LLAMA_DIR/$BUILD_DIR_NAME/src:$CUDA_PREFIX/lib64"
  local bin="$REPO_ROOT/chimere-server/target/release/chimere-server"

  cat <<EOF

$(c_green "=== Install complete ===")

Binary:   $bin
Backend:  $IK_LLAMA_DIR/$BUILD_DIR_NAME
SM arch:  sm_${SM_ARCH}

To start the server:

  export LD_LIBRARY_PATH="$ld_path"
  export CHIMERE_MODEL=/path/to/your.gguf
  export CHIMERE_TOKENIZER=/path/to/tokenizer.json
  export CHIMERE_LLAMA_BACKEND=1
  export CHIMERE_PORT=8081
  "$bin"

Smoke test:

  curl -s http://127.0.0.1:8081/health
  curl -s http://127.0.0.1:8081/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

Next:
  - Install the ODO orchestrator:   https://github.com/AIdevsmartdata/chimere-odo
  - Install the Studio desktop UI:  https://github.com/AIdevsmartdata/chimere-studio

EOF
}

# --- argument parse ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sm)            SM_ARCH="$2"; shift 2 ;;
    --with-model)    WITH_MODEL=1; shift ;;
    --skip-backend)  SKIP_BACKEND=1; shift ;;
    --ik-dir)        IK_LLAMA_DIR="$2"; shift 2 ;;
    --cuda)          CUDA_PREFIX="$2"; shift 2 ;;
    --jobs)          JOBS="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "Unknown option: $1 (try --help)" ;;
  esac
done

# --- main -------------------------------------------------------------------
echo "=== Chimere install ==="
preflight
[[ -z "$SM_ARCH" ]] && SM_ARCH="$(detect_sm)"
BUILD_DIR_NAME="build_sm${SM_ARCH}"
info "Target SM: sm_${SM_ARCH}"
build_backend
build_server
pull_model
print_next_steps
