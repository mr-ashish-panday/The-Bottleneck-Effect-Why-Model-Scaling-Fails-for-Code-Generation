#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
LCB_DIR="${LCB_DIR:-$ROOT/external/LiveCodeBench}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v2}"
NUM_PROBLEMS="${NUM_PROBLEMS:-511}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"

cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables outputs/livecodebench external

if [ ! -f "data/raw/livecodebench_${RELEASE_VERSION}.jsonl" ]; then
  echo "[$(date '+%F %T')] Downloading LiveCodeBench ${RELEASE_VERSION}"
  python scripts/download_data.py --dataset livecodebench --version_tag "$RELEASE_VERSION"
fi

if [ ! -d "$LCB_DIR/.git" ]; then
  echo "[$(date '+%F %T')] Cloning LiveCodeBench repo"
  git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
fi

if [ ! -x "$LCB_DIR/.venv/bin/python" ]; then
  echo "[$(date '+%F %T')] Creating LiveCodeBench environment"
  if command -v uv >/dev/null 2>&1; then
    (
      cd "$LCB_DIR"
      uv venv --python 3.11
      source .venv/bin/activate
      uv pip install -e .
    )
  else
    python3 -m venv "$LCB_DIR/.venv"
    "$LCB_DIR/.venv/bin/python" -m pip install --upgrade pip
    "$LCB_DIR/.venv/bin/pip" install -e "$LCB_DIR"
  fi
fi

run_case() {
  local label="$1"
  local config="$2"
  local output_json="$ROOT/outputs/livecodebench/${label}_custom_outputs.json"
  local log_file="$ROOT/outputs/logs/livecodebench_${label}.log"
  local summary_file="$ROOT/outputs/tables/livecodebench_${label}_summary.json"
  local eval_all_file=""

  echo "[$(date '+%F %T')] Generating LiveCodeBench samples for ${label}"
  python scripts/generate_samples_safe.py \
    --config "$config" \
    --resume \
    --num_problems "$NUM_PROBLEMS" \
    --num_samples "$NUM_SAMPLES"

  python scripts/export_livecodebench_custom_outputs.py \
    --config "$config" \
    --output_file "$output_json"

  echo "[$(date '+%F %T')] Evaluating LiveCodeBench outputs for ${label}"
  (
    cd "$LCB_DIR"
    source .venv/bin/activate
    python -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$output_json" \
      --release_version "$RELEASE_VERSION"
  ) | tee "$log_file"

  eval_all_file="$(find "$LCB_DIR" -type f -name '*eval_all*.json' | sort | tail -n 1 || true)"
  if [ -n "$eval_all_file" ]; then
    (
      cd "$LCB_DIR"
      source .venv/bin/activate
      python -m lcb_runner.evaluation.compute_scores --eval_all_file "$eval_all_file"
    ) | tee -a "$log_file" || true
  fi

  python scripts/summarize_livecodebench_scores.py \
    --search_root "$LCB_DIR" \
    --log_file "$log_file" \
    --output_file "$summary_file"
}

echo "[$(date '+%F %T')] Starting LiveCodeBench ${RELEASE_VERSION} overnight run"

run_case "gpt2" "config_livecodebench_gpt2.yaml"
run_case "gpt2_medium" "config_livecodebench_gpt2_medium.yaml"
run_case "codegen" "config_livecodebench_codegen.yaml"

echo "[$(date '+%F %T')] LiveCodeBench overnight run completed"
