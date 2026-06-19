#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROBLEMS="${NUM_PROBLEMS:-164}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"

cd "$ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

OUT_ROOT="data/results_heavy_rebuttal/jss_decoding_robustness_20s"
SUMMARY_ROOT="outputs/tables/jss_decoding_robustness_20s"
PROMPT_BASELINE_ROOT="data/results_heavy_rebuttal/jss_prompt_robustness_20s"
mkdir -p "$OUT_ROOT" "$SUMMARY_ROOT" outputs/logs

configs=(
  "configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical.yaml"
  "configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical.yaml"
)

baseline_for() {
  local label="$1"
  case "$label" in
    decoding_robustness__gpt2_medium__humaneval__standard__canonical)
      echo "prompt_robustness__gpt2_medium__humaneval__standard__canonical"
      ;;
    decoding_robustness__codegen_mono_350m__humaneval__standard__canonical)
      echo "prompt_robustness__codegen_mono_350m__humaneval__standard__canonical"
      ;;
    decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical)
      echo "prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical"
      ;;
    *)
      echo ""
      ;;
  esac
}

write_summary() {
  local label="$1"
  local out_dir="$2"
  local summary_file="$SUMMARY_ROOT/${label}_summary.json"
  "$PYTHON_BIN" - "$out_dir/evaluation_results.json" "$summary_file" "$label" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
label = sys.argv[3]

data = json.loads(input_file.read_text())
category_counts = Counter()
total_samples = 0
success_samples = 0
problems_with_success = 0

for problem in data:
    problem_success = False
    for sample in problem.get("samples", []):
        total_samples += 1
        category = sample.get("category", "missing_category")
        category_counts[category] += 1
        if category == "success" or sample.get("execution_result", {}).get("success"):
            success_samples += 1
            problem_success = True
    if problem_success:
        problems_with_success += 1

summary = {
    "label": label,
    "tasks": len(data),
    "samples_per_task_min": min((len(p.get("samples", [])) for p in data), default=0),
    "samples_per_task_max": max((len(p.get("samples", [])) for p in data), default=0),
    "total_samples": total_samples,
    "success_samples": success_samples,
    "success_sample_rate": success_samples / total_samples if total_samples else 0.0,
    "problems_with_success": problems_with_success,
    "problem_success_rate": problems_with_success / len(data) if data else 0.0,
    "category_counts": dict(category_counts),
}

output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

for config in "${configs[@]}"; do
  label="$(basename "$config" .yaml)"
  out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"

  echo "[$(date '+%F %T')] JSS decoding robustness job: $label"

  baseline_label="$(baseline_for "$label")"
  baseline_dir="$PROMPT_BASELINE_ROOT/$baseline_label"
  if [[ -n "$baseline_label" && -s "$baseline_dir/generated_samples.json" && -s "$baseline_dir/evaluation_results.json" ]]; then
    echo "[$(date '+%F %T')] Reusing standard canonical baseline: $baseline_label"
    cp "$baseline_dir/generated_samples.json" "$out_dir/generated_samples.json"
    cp "$baseline_dir/evaluation_results.json" "$out_dir/evaluation_results.json"
  else
    "$PYTHON_BIN" scripts/generate_samples_safe.py \
      --config "$config" \
      --resume \
      --num_problems "$NUM_PROBLEMS" \
      --num_samples "$NUM_SAMPLES" \
      --output_dir "$out_dir"

    "$PYTHON_BIN" scripts/run_evaluation.py \
      --config "$config" \
      --input_file "$out_dir/generated_samples.json" \
      --output_file "$out_dir/evaluation_results.json"
  fi

  write_summary "$label" "$out_dir"
done

"$PYTHON_BIN" - "$SUMMARY_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = []
for path in sorted(root.glob("*_summary.json")):
    summaries.append(json.loads(path.read_text()))

aggregate = {
    "run": "jss_decoding_robustness_20s",
    "jobs": len(summaries),
    "summaries": summaries,
}
out = root / "aggregate_summary.json"
out.write_text(json.dumps(aggregate, indent=2) + "\n")
print(f"Wrote {out}")
PY

echo "[$(date '+%F %T')] JSS decoding robustness 20-sample subset complete"
