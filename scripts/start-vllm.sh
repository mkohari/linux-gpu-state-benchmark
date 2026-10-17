#!/usr/bin/env bash
# Foreground, loopback only. Ctrl-C stops this server; no other process is touched.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
model_path="${1:-models/qwen2.5-1.5b-instruct}"
if [[ ! -f "$model_path/config.json" || ! -x .venv-vllm/bin/vllm ]]; then
  echo 'First run gpu-state-bench download-model and bash scripts/setup-vllm.sh; or supply a local model directory as argument.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${BENCH_GPU:-0}"
export HF_HUB_OFFLINE=1
exec .venv-vllm/bin/vllm serve "$model_path" \
  --served-model-name benchmark-public-model \
  --host 127.0.0.1 --port "${BENCH_VLLM_PORT:-8000}" \
  --dtype float16 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.7 --enforce-eager \
  --no-enable-prefix-caching --generation-config vllm
