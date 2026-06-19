#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT/venv}"
USE_EXISTING_ENV="${USE_EXISTING_ENV:-1}"
INSTALL_TORCH_IF_MISSING="${INSTALL_TORCH_IF_MISSING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"

if [[ "$USE_EXISTING_ENV" != "1" ]]; then
  if [[ ! -d "$VENV_PATH" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_PATH"
  fi
  source "$VENV_PATH/bin/activate"
else
  echo "Using current Python environment: $(command -v "$PYTHON_BIN")"
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if ! "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"; then
  if [[ "$INSTALL_TORCH_IF_MISSING" == "1" ]]; then
    "$PYTHON_BIN" -m pip install torch
  else
    echo "Torch is missing. On Lightning AI, switch to a PyTorch image or rerun with INSTALL_TORCH_IF_MISSING=1."
    exit 1
  fi
fi

"$PYTHON_BIN" -m pip install -r requirements-lightning.txt

mkdir -p data/raw outputs/logs outputs/tables outputs/evalplus outputs/livecodebench external

"$PYTHON_BIN" scripts/download_data.py --dataset humaneval
"$PYTHON_BIN" scripts/download_data.py --dataset mbpp
"$PYTHON_BIN" scripts/build_heavy_rebuttal_runs.py
"$PYTHON_BIN" scripts/audit_heavy_rebuttal_outputs.py --phase t4_priority

echo "Lightning setup complete."
echo "Run smoke test:"
echo "  SMOKE=1 PHASE=t4_priority bash run_heavy_rebuttal_suite.sh"
