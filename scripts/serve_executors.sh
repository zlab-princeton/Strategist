#!/bin/bash
# Serve the paper's three executor tiers as OpenAI-compatible vLLM endpoints
# on one GPU node (paper Sec. 4: Qwen3-0.6B fast, 1.7B balanced,
# 4B-Thinking complex). Cluster-agnostic: run directly on a GPU node or
# submit the absolute path through Hive. Never submit via sbatch.
#
#   bash /absolute/path/strategist/serve_tiers.sh
#
# Env overrides:
#   FAST_MODEL / BALANCED_MODEL / COMPLEX_MODEL   (HF model ids)
#   FAST_GPU=0 BALANCED_GPU=1 COMPLEX_GPU=2       (CUDA device per tier)
#   FAST_PORT=8001 BALANCED_PORT=8002 COMPLEX_PORT=8003
#   MAX_MODEL_LEN=32768                            (matches STRATEGIST_MODEL_CTX)
#
# Writes the per-model endpoint registries that search.py's
# endpoint-refresh reads (strategist/logs/vllm_endpoints_<model_key>.json)
# and prints the --endpoints flag to paste into a run.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/strategist/logs"
mkdir -p "${LOG_DIR}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

FAST_MODEL="${FAST_MODEL:-Qwen/Qwen3-0.6B}"
BALANCED_MODEL="${BALANCED_MODEL:-Qwen/Qwen3-1.7B}"
COMPLEX_MODEL="${COMPLEX_MODEL:-Qwen/Qwen3-4B}"
FAST_GPU="${FAST_GPU:-0}"
BALANCED_GPU="${BALANCED_GPU:-1}"
COMPLEX_GPU="${COMPLEX_GPU:-2}"
FAST_PORT="${FAST_PORT:-8001}"
BALANCED_PORT="${BALANCED_PORT:-8002}"
COMPLEX_PORT="${COMPLEX_PORT:-8003}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HOST="$(hostname -s)"

model_key () {  # Qwen/Qwen3-0.6B -> qwen_qwen3_0_6b (matches search.py)
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's#[/.-]#_#g'
}

serve () {
    local model="$1" gpu="$2" port="$3" tier="$4"
    local log="${LOG_DIR}/vllm_${tier}.log"
    echo "[serve] ${tier}: ${model} on GPU ${gpu}, port ${port} (log: ${log})"
    CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m vllm.entrypoints.openai.api_server \
        --model "${model}" \
        --host 0.0.0.0 \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization 0.90 \
        --disable-log-requests \
        > "${log}" 2>&1 &
    echo "http://${HOST}:${port}/v1" > "${LOG_DIR}/endpoint_${tier}.txt"
    # Endpoint registry search.py's refresh_llms_dict reads on failure.
    printf '["http://%s:%s/v1"]\n' "${HOST}" "${port}" \
        > "${LOG_DIR}/vllm_endpoints_$(model_key "${model}").json"
}

serve "${FAST_MODEL}"     "${FAST_GPU}"     "${FAST_PORT}"     fast
serve "${BALANCED_MODEL}" "${BALANCED_GPU}" "${BALANCED_PORT}" balanced
serve "${COMPLEX_MODEL}"  "${COMPLEX_GPU}"  "${COMPLEX_PORT}"  complex

ENDPOINTS="fast=http://${HOST}:${FAST_PORT}/v1,balanced=http://${HOST}:${BALANCED_PORT}/v1,complex=http://${HOST}:${COMPLEX_PORT}/v1"
echo "${ENDPOINTS}" > "${LOG_DIR}/endpoints_flag.txt"
echo
echo "[serve] waiting for all three tiers to answer /models ..."
for tier in fast balanced complex; do
    url="$(cat "${LOG_DIR}/endpoint_${tier}.txt")"
    for _ in $(seq 1 120); do
        if curl -s --max-time 3 "${url}/models" | grep -q model; then
            echo "[serve] ${tier} UP at ${url}"
            break
        fi
        sleep 5
    done
done
echo
echo "--endpoints ${ENDPOINTS}"
echo "[serve] servers run in the background; logs in ${LOG_DIR}/vllm_*.log"
wait
