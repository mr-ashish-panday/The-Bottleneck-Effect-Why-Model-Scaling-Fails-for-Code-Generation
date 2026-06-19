#!/usr/bin/env bash
set -euo pipefail

if python3 -m venv .venv 2>/tmp/trump_occurrence_venv.err; then
  source .venv/bin/activate
else
  echo "venv creation unavailable; using current Python environment"
  cat /tmp/trump_occurrence_venv.err
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
