#!/usr/bin/env bash
# Optional serving environment; never replace the reference engine's dependencies.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python3 -c 'import sys; assert sys.platform == "linux" and (3,10) <= sys.version_info[:2] < (3,13), "Optional vLLM recipe requires Linux and Python 3.10–3.12"'
python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install vllm==0.11.0
.venv-vllm/bin/python -m pip check
echo 'Optional environment ready. Start in a separate terminal: bash scripts/start-vllm.sh'
