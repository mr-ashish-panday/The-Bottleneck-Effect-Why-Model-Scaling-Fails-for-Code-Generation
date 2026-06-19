# Trump Occurrence GPU Upgrade

This is a separate V2 upgrade path. It does not mutate the completed `trump-occurrence-sprint/` client package.

Goal: give the LLM/LoRA direction a fairer GPU-backed attempt while preserving benchmark integrity.

## Current Status

- V1 client package is complete and frozen.
- V2 protocol is defined in `GPU_UPGRADE_PROTOCOL.md`.
- SSH access to the GPU box must be working before remote training can start.

## Recommended Remote Layout

```text
~/trump-occurrence-work/
  trump-occurrence-sprint/
  trump-occurrence-gpu-upgrade/
```

## Remote Setup

```bash
cd ~/trump-occurrence-work/trump-occurrence-gpu-upgrade
bash scripts/setup_remote.sh
source .venv/bin/activate
python -m src.inspect_gpu
```

## First GPU Smoke Run

Use a small model and a small row cap first. This only proves the loop.

```bash
python -m src.train_lora_scorer \
  --config config_v2.yaml \
  --row-cap-train 2000 \
  --row-cap-val 800 \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir runs/smoke_qwen15
```

## Full V2 Training Run

Run this only after the smoke run succeeds.

```bash
python -m src.train_lora_scorer \
  --config config_v2.yaml \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/qwen7b_lora_seed20260615
```

## Integrity Rules

- Do not tune on the completed V1 locked test.
- Do not overwrite V1 predictions or metrics.
- Treat validation results as model-selection evidence only.
- A client-grade V2 result requires a fresh sealed holdout or must be marked exploratory.

