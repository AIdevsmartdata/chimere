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

# Build the command
CMD="/usr/local/bin/llama-server"
CMD="${CMD} --model ${CHIMERE_MODEL}"
CMD="${CMD} --port ${CHIMERE_PORT}"
CMD="${CMD} --ctx-size ${CHIMERE_CTX}"
CMD="${CMD} --n-gpu-layers ${CHIMERE_NGL}"
CMD="${CMD} --n-cpu-moe ${CHIMERE_NCMOE}"
CMD="${CMD} --flash-attn ${CHIMERE_FLASH_ATTN}"
CMD="${CMD} --cache-type-k ${CHIMERE_KV_K}"
CMD="${CMD} --cache-type-v ${CHIMERE_KV_V}"
CMD="${CMD} --parallel ${CHIMERE_NP}"
CMD="${CMD} --host 0.0.0.0"
CMD="${CMD} --metrics"
CMD="${CMD} --jinja"

# Append any extra arguments passed to the container
if [ $# -gt 0 ]; then
    CMD="${CMD} $*"
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

exec ${CMD}
