#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

source venv/bin/activate
mkdir -p outputs/logs outputs/tables

echo "[$(date '+%F %T')] Starting remaining CodeGen ladder queue"

echo "[$(date '+%F %T')] Step 1/3: sanitized MBPP ladder"
bash run_codegen_ladder_mbpp.sh

echo "[$(date '+%F %T')] Step 2/3: prompt perplexity diagnostics"
bash run_codegen_ladder_prompt_ppl.sh

echo "[$(date '+%F %T')] Step 3/3: mechanism discovery rerun"
bash run_codegen_ladder_mechanism.sh

echo "[$(date '+%F %T')] Remaining CodeGen ladder queue finished"
