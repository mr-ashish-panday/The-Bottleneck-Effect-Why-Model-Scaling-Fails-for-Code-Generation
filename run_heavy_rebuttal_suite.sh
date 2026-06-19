#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
PHASE="${PHASE:-core_scaling}"
FORCE="${FORCE:-0}"
RUN_PROMPT_PPL="${RUN_PROMPT_PPL:-0}"
SMOKE="${SMOKE:-0}"
CONFIRM_PAID_RUN="${CONFIRM_PAID_RUN:-0}"
SMOKE_NUM_PROBLEMS="${SMOKE_NUM_PROBLEMS:-2}"
SMOKE_NUM_SAMPLES="${SMOKE_NUM_SAMPLES:-2}"
VENV_PATH="${VENV_PATH:-$ROOT/venv}"
LCB_DIR="${LCB_DIR:-$ROOT/external/LiveCodeBench}"
RELEASE_VERSION="${RELEASE_VERSION:-release_v2}"

cd "$ROOT"
if [[ "$SMOKE" != "1" && "$CONFIRM_PAID_RUN" != "1" ]]; then
  echo "Refusing full paid run. Set CONFIRM_PAID_RUN=1 after explicit approval, or use SMOKE=1."
  exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "[$(date '+%F %T')] Using active virtualenv: $VIRTUAL_ENV"
elif [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
  echo "[$(date '+%F %T')] Activated virtualenv: $VENV_PATH"
else
  echo "[$(date '+%F %T')] No venv found at $VENV_PATH; using current Python: $(command -v python)"
fi
mkdir -p outputs/logs outputs/tables outputs/evalplus outputs/livecodebench external

num_problems_for() {
  if [[ "$SMOKE" == "1" ]]; then
    echo "$SMOKE_NUM_PROBLEMS"
  else
    echo "$1"
  fi
}

num_samples_for() {
  if [[ "$SMOKE" == "1" ]]; then
    echo "$SMOKE_NUM_SAMPLES"
  else
    echo "$1"
  fi
}

ensure_evalplus() {
  if ! python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('evalplus') else 1)"; then
    python -m pip install "evalplus==0.3.1"
  fi
}

should_run_phase() {
  local job_phase="$1"
  [[ "$PHASE" == "all" || "$PHASE" == "$job_phase" ]]
}

run_generation() {
  local config="$1"
  local num_problems="$2"
  local num_samples="$3"
  if [[ "$SMOKE" == "1" ]]; then
    python scripts/generate_samples_safe.py --config "$config" --resume --num_problems "$num_problems" --num_samples "$num_samples" --output_dir "$(smoke_results_dir "$config")"
    return 0
  fi
  python scripts/generate_samples_safe.py --config "$config" --resume --num_problems "$num_problems" --num_samples "$num_samples"
}

run_local_eval() {
  local config="$1"
  if [[ "$SMOKE" == "1" ]]; then
    local smoke_dir
    smoke_dir="$(smoke_results_dir "$config")"
    python scripts/run_evaluation.py --config "$config" --input_file "$smoke_dir/generated_samples.json" --output_file "$smoke_dir/evaluation_results.json"
    return 0
  fi
  python scripts/run_evaluation.py --config "$config"
}

smoke_results_dir() {
  local config="$1"
  local stem
  stem="$(basename "$config" .yaml)"
  echo "data/results_heavy_rebuttal_smoke/${PHASE}/${stem}"
}

run_extraction_sweep() {
  local config="$1"
  local output_file="$2"
  if [[ "$SMOKE" == "1" ]]; then
    echo "[$(date '+%F %T')] Smoke mode: skip extraction sweep"
    return 0
  fi
  if [[ "$FORCE" != "1" && -s "$output_file" ]]; then
    echo "[$(date '+%F %T')] Skip extraction sweep; exists: $output_file"
    return 0
  fi
  python scripts/evaluate_extraction_sweep.py --config "$config" --output_file "$output_file"
}

run_prompt_ppl() {
  local config="$1"
  local output_file="$2"
  if [[ "$RUN_PROMPT_PPL" != "1" ]]; then
    return 0
  fi
  if [[ "$FORCE" != "1" && -s "$output_file" ]]; then
    echo "[$(date '+%F %T')] Skip prompt PPL; exists: $output_file"
    return 0
  fi
  python scripts/compute_prompt_perplexity.py --config "$config" --output_file "$output_file"
}

run_evalplus_rescore() {
  local label="$1"
  local dataset="$2"
  local config="$3"
  local max_samples="$4"
  if [[ "$SMOKE" == "1" ]]; then
    echo "[$(date '+%F %T')] Smoke mode: skip EvalPlus scoring"
    return 0
  fi
  ensure_evalplus
  local case_dir="outputs/evalplus/${label}_${dataset}"
  local samples_file="${case_dir}/samples.jsonl"
  local log_file="outputs/logs/evalplus_${label}_${dataset}.log"
  local summary_file="outputs/tables/evalplus_${label}_${dataset}_summary.json"
  if [[ "$FORCE" != "1" && -s "$summary_file" ]]; then
    echo "[$(date '+%F %T')] Skip EvalPlus; exists: $summary_file"
    return 0
  fi
  mkdir -p "$case_dir"
  python scripts/export_evalplus_samples.py --config "$config" --max_samples_per_task "$max_samples" --output_file "$samples_file"
  (cd "$case_dir" && evalplus.evaluate "$dataset" --samples samples.jsonl) | tee "$log_file"
  python scripts/analyze_evalplus_results.py --search_root "$case_dir" --log_file "$log_file" --output_file "$summary_file"
}

run_clean_mbppplus() {
  local label="$1"
  local config="$2"
  local num_problems="$3"
  local num_samples="$4"
  num_problems="$(num_problems_for "$num_problems")"
  num_samples="$(num_samples_for "$num_samples")"
  ensure_evalplus
  local case_dir="outputs/evalplus/${label}_mbppplus"
  local samples_file="${case_dir}/samples.jsonl"
  local log_file="outputs/logs/evalplus_${label}_mbppplus.log"
  local summary_file="outputs/tables/evalplus_${label}_mbppplus_summary.json"
  if [[ "$FORCE" != "1" && -s "$summary_file" ]]; then
    echo "[$(date '+%F %T')] Skip MBPP+; exists: $summary_file"
    return 0
  fi
  python scripts/generate_mbppplus_evalplus.py --config "$config" --resume --num_problems "$num_problems" --num_samples "$num_samples"
  if [[ "$SMOKE" == "1" ]]; then
    echo "[$(date '+%F %T')] Smoke mode: generated MBPP+ samples only; skip EvalPlus scoring"
    return 0
  fi
  mkdir -p "$case_dir"
  python scripts/export_evalplus_samples.py --config "$config" --output_file "$samples_file"
  (cd "$case_dir" && evalplus.evaluate mbpp --samples samples.jsonl) | tee "$log_file"
  python scripts/analyze_evalplus_results.py --search_root "$case_dir" --log_file "$log_file" --output_file "$summary_file"
}

ensure_livecodebench() {
  if [[ ! -f "data/raw/livecodebench_${RELEASE_VERSION}.jsonl" ]]; then
    python scripts/download_data.py --dataset livecodebench --version_tag "$RELEASE_VERSION"
  fi
  if [[ ! -d "$LCB_DIR/.git" ]]; then
    git clone https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_DIR"
  fi
}

run_livecodebench_case() {
  local label="$1"
  local config="$2"
  local num_problems="$3"
  local num_samples="$4"
  num_problems="$(num_problems_for "$num_problems")"
  num_samples="$(num_samples_for "$num_samples")"
  local output_json="$ROOT/outputs/livecodebench/${label}_custom_outputs.json"
  local log_file="$ROOT/outputs/logs/livecodebench_${label}.log"
  local summary_file="$ROOT/outputs/tables/livecodebench_${label}_summary.json"
  if [[ "$FORCE" != "1" && -s "$summary_file" ]]; then
    echo "[$(date '+%F %T')] Skip LiveCodeBench; exists: $summary_file"
    return 0
  fi
  ensure_livecodebench
  run_generation "$config" "$num_problems" "$num_samples"
  if [[ "$SMOKE" == "1" ]]; then
    echo "[$(date '+%F %T')] Smoke mode: generated LiveCodeBench samples only; skip external scoring"
    return 0
  fi
  python scripts/export_livecodebench_custom_outputs.py --config "$config" --output_file "$output_json"
  (cd "$LCB_DIR" && PYTHONPATH="$LCB_DIR:${PYTHONPATH:-}" python -m lcb_runner.runner.custom_evaluator --custom_output_file "$output_json" --release_version "$RELEASE_VERSION") | tee "$log_file"
  local eval_all_file
  eval_all_file="$(find "$LCB_DIR" -type f -name '*eval_all*.json' | sort | tail -n 1 || true)"
  if [[ -n "$eval_all_file" ]]; then
    (cd "$LCB_DIR" && PYTHONPATH="$LCB_DIR:${PYTHONPATH:-}" python -m lcb_runner.evaluation.compute_scores --eval_all_file "$eval_all_file") | tee -a "$log_file" || true
  fi
  python scripts/summarize_livecodebench_scores.py --search_root "$LCB_DIR" --log_file "$log_file" --output_file "$summary_file"
}

echo "[$(date '+%F %T')] Starting heavy rebuttal suite phase=$PHASE"

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__gpt2_small__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__gpt2_small__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__gpt2_small__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__gpt2_small__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_small__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__gpt2_small__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_small__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__gpt2_small__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__gpt2_small__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__gpt2_small__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__gpt2_small__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_small__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__gpt2_small__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_small__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__gpt2_medium__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__gpt2_medium__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__gpt2_medium__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__gpt2_medium__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_medium__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__gpt2_medium__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_medium__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__gpt2_medium__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__gpt2_medium__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__gpt2_medium__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__gpt2_medium__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_medium__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__gpt2_medium__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_medium__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__gpt2_large__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__gpt2_large__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__gpt2_large__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__gpt2_large__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_large__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__gpt2_large__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_large__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__gpt2_large__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__gpt2_large__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__gpt2_large__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__gpt2_large__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_large__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__gpt2_large__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__gpt2_large__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__pythia_70m__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__pythia_70m__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__pythia_70m__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__pythia_70m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_70m__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__pythia_70m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_70m__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__pythia_70m__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__pythia_70m__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__pythia_70m__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__pythia_70m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_70m__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__pythia_70m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_70m__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__pythia_160m__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__pythia_160m__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__pythia_160m__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__pythia_160m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_160m__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__pythia_160m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_160m__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__pythia_160m__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__pythia_160m__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__pythia_160m__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__pythia_160m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_160m__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__pythia_160m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_160m__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__pythia_410m__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__pythia_410m__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__pythia_410m__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__pythia_410m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_410m__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__pythia_410m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_410m__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__pythia_410m__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__pythia_410m__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__pythia_410m__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__pythia_410m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_410m__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__pythia_410m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__pythia_410m__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_nl_350m__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__codegen_nl_350m__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_nl_350m__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__codegen_nl_350m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_nl_350m__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_multi_350m__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__codegen_multi_350m__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_multi_350m__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__codegen_multi_350m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_multi_350m__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_mono_350m__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_priority'; then
  echo "[$(date '+%F %T')] Job t4_priority__codegen_mono_350m__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_mono_350m__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_priority__codegen_mono_350m__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_priority__codegen_mono_350m__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_stretch'; then
  echo "[$(date '+%F %T')] Job t4_stretch__gpt2_xl__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__gpt2_xl__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__gpt2_xl__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_stretch'; then
  echo "[$(date '+%F %T')] Job t4_stretch__gpt2_xl__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__gpt2_xl__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_stretch__gpt2_xl__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__gpt2_xl__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_stretch'; then
  echo "[$(date '+%F %T')] Job t4_stretch__pythia_1b__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_stretch__pythia_1b__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_stretch__pythia_1b__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_stretch__pythia_1b__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__pythia_1b__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_stretch__pythia_1b__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__pythia_1b__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_stretch'; then
  echo "[$(date '+%F %T')] Job t4_stretch__pythia_1b__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_stretch__pythia_1b__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_stretch__pythia_1b__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_stretch__pythia_1b__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__pythia_1b__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_stretch__pythia_1b__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__pythia_1b__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_stretch'; then
  echo "[$(date '+%F %T')] Job t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical"
  run_generation 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__qwen25_coder_05b__humaneval_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 't4_stretch'; then
  echo "[$(date '+%F %T')] Job t4_stretch__qwen25_coder_05b__mbpp_t4__standard"
  run_generation 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__mbpp_t4__standard.yaml' $(num_problems_for 257) $(num_samples_for 10)
  run_local_eval 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__mbpp_t4__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__qwen25_coder_05b__mbpp_t4__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/t4_stretch__qwen25_coder_05b__mbpp_t4__standard.yaml' 'outputs/tables/heavy_rebuttal/t4_stretch__qwen25_coder_05b__mbpp_t4__standard__canonical_prompt_perplexity.json'
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_small__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_small__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__gpt2_small__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_small__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_small__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__gpt2_small__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_small__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__gpt2_small__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__gpt2_small__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_medium__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_medium__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__gpt2_medium__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_medium__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_medium__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__gpt2_medium__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_medium__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__gpt2_medium__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__gpt2_medium__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_large__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_large__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__gpt2_large__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_large__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_large__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__gpt2_large__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_large__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__gpt2_large__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__gpt2_large__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_xl__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_xl__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__gpt2_xl__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_xl__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__gpt2_xl__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__gpt2_xl__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__gpt2_xl__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__gpt2_xl__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__gpt2_xl__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_70m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_70m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__pythia_70m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_70m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_70m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__pythia_70m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_70m__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__pythia_70m__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__pythia_70m__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_160m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_160m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__pythia_160m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_160m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_160m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__pythia_160m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_160m__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__pythia_160m__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__pythia_160m__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_410m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_410m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__pythia_410m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_410m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_410m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__pythia_410m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_410m__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__pythia_410m__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__pythia_410m__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_1b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_1b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__pythia_1b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_1b__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__pythia_1b__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__pythia_1b__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__pythia_1b__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__pythia_1b__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__pythia_1b__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_nl_350m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__codegen_nl_350m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_nl_350m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__codegen_nl_350m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_nl_350m__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__codegen_nl_350m__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__codegen_nl_350m__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_multi_350m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__codegen_multi_350m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_multi_350m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__codegen_multi_350m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_multi_350m__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__codegen_multi_350m__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__codegen_multi_350m__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_mono_350m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__codegen_mono_350m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_mono_350m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'core_scaling__codegen_mono_350m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'core_scaling'; then
  echo "[$(date '+%F %T')] Job core_scaling__codegen_mono_350m__mbppplus__standard"
  run_clean_mbppplus 'core_scaling__codegen_mono_350m__mbppplus__standard__canonical' 'configs/heavy_rebuttal/core_scaling__codegen_mono_350m__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'modern_code_validation'; then
  echo "[$(date '+%F %T')] Job modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'modern_code_validation'; then
  echo "[$(date '+%F %T')] Job modern_code_validation__qwen25_coder_05b__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'modern_code_validation__qwen25_coder_05b__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'modern_code_validation'; then
  echo "[$(date '+%F %T')] Job modern_code_validation__qwen25_coder_05b__mbppplus__standard"
  run_clean_mbppplus 'modern_code_validation__qwen25_coder_05b__mbppplus__standard__canonical' 'configs/heavy_rebuttal/modern_code_validation__qwen25_coder_05b__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'modern_code_validation'; then
  echo "[$(date '+%F %T')] Job modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'modern_code_validation'; then
  echo "[$(date '+%F %T')] Job modern_code_validation__deepseek_coder_13b__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'modern_code_validation__deepseek_coder_13b__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'modern_code_validation'; then
  echo "[$(date '+%F %T')] Job modern_code_validation__deepseek_coder_13b__mbppplus__standard"
  run_clean_mbppplus 'modern_code_validation__deepseek_coder_13b__mbppplus__standard__canonical' 'configs/heavy_rebuttal/modern_code_validation__deepseek_coder_13b__mbppplus__standard.yaml' 378 20
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_medium__humaneval__low_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_medium__humaneval__low_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__low_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_medium__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_medium__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_medium__humaneval__high_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_medium__humaneval__high_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__humaneval__high_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_medium__mbpp__low_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_medium__mbpp__low_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__low_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_medium__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_medium__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_medium__mbpp__high_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_medium__mbpp__high_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__gpt2_medium__mbpp__high_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_xl__humaneval__low_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_xl__humaneval__low_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__low_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_xl__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_xl__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_xl__humaneval__high_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_xl__humaneval__high_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__humaneval__high_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_xl__mbpp__low_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_xl__mbpp__low_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__low_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_xl__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_xl__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__gpt2_xl__mbpp__high_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__gpt2_xl__mbpp__high_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__gpt2_xl__mbpp__high_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__pythia_1b__humaneval__low_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__pythia_1b__humaneval__low_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__low_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__pythia_1b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__pythia_1b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__pythia_1b__humaneval__high_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__pythia_1b__humaneval__high_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__humaneval__high_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__pythia_1b__mbpp__low_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__pythia_1b__mbpp__low_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__low_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__pythia_1b__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__pythia_1b__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__pythia_1b__mbpp__high_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__pythia_1b__mbpp__high_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__pythia_1b__mbpp__high_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__low_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__codegen_mono_350m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__codegen_mono_350m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__humaneval__high_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__codegen_mono_350m__mbpp__low_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__codegen_mono_350m__mbpp__low_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__low_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__codegen_mono_350m__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__codegen_mono_350m__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__codegen_mono_350m__mbpp__high_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__codegen_mono_350m__mbpp__high_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__codegen_mono_350m__mbpp__high_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__low_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical' 'humaneval' 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__humaneval__high_temp__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__qwen25_coder_05b__mbpp__low_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__qwen25_coder_05b__mbpp__low_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__low_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__qwen25_coder_05b__mbpp__standard"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__qwen25_coder_05b__mbpp__standard__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__standard.yaml' $(num_samples_for 20)
fi

if should_run_phase 'decoding_robustness'; then
  echo "[$(date '+%F %T')] Job decoding_robustness__qwen25_coder_05b__mbpp__high_temp"
  run_generation 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp.yaml' $(num_problems_for 257) $(num_samples_for 20)
  run_local_eval 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp.yaml' 'outputs/tables/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'decoding_robustness__qwen25_coder_05b__mbpp__high_temp__canonical' 'mbpp' 'configs/heavy_rebuttal/decoding_robustness__qwen25_coder_05b__mbpp__high_temp.yaml' $(num_samples_for 20)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__gpt2_medium__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__gpt2_medium__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__gpt2_medium__humaneval__standard__signature_only"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__gpt2_medium__humaneval__standard__signature_only' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__signature_only.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__gpt2_medium__humaneval__standard__comment_plus_signature.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__gpt2_xl__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__gpt2_xl__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__gpt2_xl__humaneval__standard__signature_only"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__gpt2_xl__humaneval__standard__signature_only' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__signature_only.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__gpt2_xl__humaneval__standard__comment_plus_signature.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__pythia_1b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__pythia_1b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__pythia_1b__humaneval__standard__signature_only"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__pythia_1b__humaneval__standard__signature_only' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__signature_only.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__pythia_1b__humaneval__standard__comment_plus_signature.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__codegen_mono_350m__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__codegen_mono_350m__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__signature_only.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__codegen_mono_350m__humaneval__standard__comment_plus_signature.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__canonical.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__signature_only.yaml' $(num_samples_for 100)
fi

if should_run_phase 'prompt_robustness'; then
  echo "[$(date '+%F %T')] Job prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature"
  run_generation 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature.yaml' $(num_problems_for 164) $(num_samples_for 100)
  run_local_eval 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature.yaml'
  run_extraction_sweep 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature_extraction_sweep.json'
  run_prompt_ppl 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature.yaml' 'outputs/tables/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature_prompt_perplexity.json'
  run_evalplus_rescore 'prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature' 'humaneval' 'configs/heavy_rebuttal/prompt_robustness__qwen25_coder_05b__humaneval__standard__comment_plus_signature.yaml' $(num_samples_for 100)
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__gpt2_small__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__gpt2_small__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__gpt2_small__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__gpt2_medium__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__gpt2_medium__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__gpt2_medium__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__gpt2_large__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__gpt2_large__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__gpt2_large__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__gpt2_xl__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__gpt2_xl__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__gpt2_xl__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__pythia_410m__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__pythia_410m__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__pythia_410m__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__pythia_1b__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__pythia_1b__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__pythia_1b__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__codegen_mono_350m__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__codegen_mono_350m__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__codegen_mono_350m__livecodebench__standard.yaml' 511 10
fi

if should_run_phase 'livecodebench_stress'; then
  echo "[$(date '+%F %T')] Job livecodebench_stress__qwen25_coder_05b__livecodebench__standard"
  run_livecodebench_case 'livecodebench_stress__qwen25_coder_05b__livecodebench__standard__canonical' 'configs/heavy_rebuttal/livecodebench_stress__qwen25_coder_05b__livecodebench__standard.yaml' 511 10
fi

echo "[$(date '+%F %T')] Heavy rebuttal suite completed"
