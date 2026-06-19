# GPU Run Status - 2026-06-15

## Current Verdict

The GPU upgrade path is technically working, but the run is blocked by Lightning account balance.

Do not change the locked 48h client handoff based on these GPU experiments yet. The original package remains the valid client deliverable. This V2 track is an upgrade candidate only.

## What Worked

- Uploaded and unpacked `trump-occurrence-gpu-transfer-2026-06-15.zip` on Lightning Studio `01kqkzqkvv9snw0jghw8ma4dn6`.
- Confirmed H200 availability before shutdown:
  - GPU: `NVIDIA H200`
  - Memory: `143771 MiB`
- Installed the missing training stack in the default Lightning conda environment:
  - `transformers`
  - `peft`
  - `bitsandbytes`
  - `pyarrow`
- Patched the V2 runner so large runs can use `eval_strategy: "no"` and avoid repeated full-validation passes during training.
- Added `config_h200_validation.yaml` for a serious H200 validation run.

## Completed Smoke Run

Remote output pulled locally to:

`remote_artifacts_20260615/`

Smoke run:

```bash
python -m src.train_lora_scorer \
  --config config_v2.yaml \
  --row-cap-train 1000 \
  --row-cap-val 400 \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir runs/smoke_qwen05b
```

Result:

- Train rows: `1000`
- Validation rows: `400`
- Train time: `118.0s`
- Full-prefix Brier: `0.0282964745`
- Timing baseline Brier: `0.0290544078`
- Brier improvement over timing: `0.0007579333`
- Blanked-prefix Brier: `0.0817825325`
- Content delta, full minus blanked: `-0.0534860579`

Interpretation: the path works end to end, and the tiny smoke run shows content signal, but it is not strong enough to claim a client-facing upgrade.

## Interrupted 7B Run

Started command:

```bash
python -m src.train_lora_scorer \
  --config config_h200_validation.yaml \
  --row-cap-train 20000 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/qwen7b_lora_val20k_seed20260615
```

Observed before shutdown:

- Qwen 7B loaded successfully as a sequence classifier.
- LoRA trainable params: `40,377,344`
- Total params: `7,111,003,648`
- Training reached about `262/1250` steps, roughly `21%`.
- Latest parsed progress from pulled log: `262/1250`, elapsed `11:23`, remaining estimate `46:19`, about `2.81s/step`.

The job did not finish. No final 7B validation metric exists.

## Blocker

Lightning stopped the running instance with:

`USER_STOP_WORKLOAD_REASON_OUT_OF_FUNDS`

Restart attempts failed for `L40S`, `H100`, and `H200` with:

`insufficient balance to start the cloud space, top up and try again`

The original SSH target is still inaccessible from this local account:

`s_01kv4xj262ce96tjdepmyc708w@ssh.lightning.ai: Permission denied (publickey)`

## Resume Command After Top-Up

After topping up Lightning balance, restart the visible Studio `01kqkzqkvv9snw0jghw8ma4dn6` and run:

```bash
cd trump-occurrence-work/trump-occurrence-gpu-upgrade

python -m src.train_lora_scorer \
  --config config_h200_validation.yaml \
  --row-cap-train 20000 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/qwen7b_lora_val20k_seed20260615_rerun
```

Keep `test_path: null` unless a new locked evaluation split is explicitly created. The original test has already been spent by V1.

## Promotion Gate

Only promote V2 to the client if it clears all of these:

- Full-prefix validation Brier beats timing baseline by a meaningful margin.
- Blanked-prefix validation is materially worse than full-prefix validation.
- Calibration is not obviously broken.
- Metrics are saved with run metadata and logs.
- No original locked-test claims are rewritten without a new proper evaluation gate.
