#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Moiz Kohari
# SPDX-License-Identifier: MIT
# Public dependencies only; no sudo, model download or GPU job is started here.
set -euo pipefail
mode="${1:---gpu}"
if [[ "$mode" != --gpu && "$mode" != --cpu ]]; then
  echo 'Usage: bash scripts/setup.sh [--gpu|--cpu]' >&2
  exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,14), "Use Python 3.10–3.13 (3.12 recommended)"'
host_platform="$(python3 -c 'import sys; print(sys.platform)')"
if [[ "$mode" == --gpu && "$host_platform" != linux ]]; then
  echo 'GPU measurements require Linux. Use --cpu for a development demonstration.' >&2
  exit 2
fi
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
torch_index='https://download.pytorch.org/whl/cu128'
if [[ "$mode" == --cpu ]]; then
  torch_index='https://download.pytorch.org/whl/cpu'
fi
if [[ "$host_platform" == darwin && "$mode" == --cpu ]]; then
  .venv/bin/python -m pip install --upgrade --force-reinstall torch==2.9.0
else
  .venv/bin/python -m pip install --upgrade --force-reinstall torch==2.9.0 --index-url "$torch_index"
fi
.venv/bin/python -m pip install -e '.[run]'
.venv/bin/python -m pip check
echo 'Setup finished. Run: source .venv/bin/activate'
if [[ "$mode" == --cpu ]]; then
  echo 'Then run: gpu-state-bench demo'
else
  echo 'Then run: gpu-state-bench doctor'
  echo 'Then run: gpu-state-bench download-model'
  echo 'Then run: gpu-state-bench start'
fi
