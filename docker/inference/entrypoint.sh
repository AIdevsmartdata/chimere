#!/bin/sh
set -eu

# Chimere Inference Entrypoint
# Constructs and executes llama-server from environment variables.

CHIMERE_MODEL="${CHIMERE_MODEL:-/models/model.gguf}"
CHIMERE_PORT="${CHIMERE_PORT:-8081}"
CHIMERE_CTX="${CHIMERE_CTX:-32768}"
CHIMERE_NGL="${CHIMERE_NGL:-99}"
CHIMERE_NCMOE="${CHIMERE_NCMOE:-4}"
CHIMERE_FLASH_ATTN="${CHIMERE_FLASH_ATTN:-on}"
CHIMERE_KV_K="${CHIMERE_KV_K:-q8_0}"
CHIMERE_KV_V="${CHIMERE_KV_V:-q4_0}"
CHIMERE_NP="${CHIMERE_NP:-1}"

# Validate model file exists
if [ ! -f "${CHIMERE_MODEL}" ]; then
    echo "ERROR: Model file not found: ${CHIMERE_MODEL}" >&2
    echo "Mount your GGUF model to /models/ or set CHIMERE_MODEL." >&2
    exit 1
fi

echo "--- Chimere Inference ---"
echo "Model:       ${CHIMERE_MODEL}"
echo "Port:        ${CHIMERE_PORT}"
echo "Context:     ${CHIMERE_CTX}"
echo "GPU layers:  ${CHIMERE_NGL}"
echo "CPU MoE:     ${CHIMERE_NCMOE}"
echo "Flash Attn:  ${CHIMERE_FLASH_ATTN}"
echo "KV cache:    K=${CHIMERE_KV_K} V=${CHIMERE_KV_V}"
echo "Parallel:    ${CHIMERE_NP}"
echo "-------------------------"

# Build args as a proper set to avoid word-splitting issues
set -- \
    /usr/local/bin/llama-server \
    --model "${CHIMERE_MODEL}" \
    --port "${CHIMERE_PORT}" \
    --ctx-size "${CHIMERE_CTX}" \
    --n-gpu-layers "${CHIMERE_NGL}" \
    --n-cpu-moe "${CHIMERE_NCMOE}" \
    --flash-attn "${CHIMERE_FLASH_ATTN}" \
    --cache-type-k "${CHIMERE_KV_K}" \
    --cache-type-v "${CHIMERE_KV_V}" \
    --parallel "${CHIMERE_NP}" \
    --host 0.0.0.0 \
    --metrics \
    --jinja \
    --alias chimere-default \
    "$@"

exec "$@"
