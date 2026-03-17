#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

mkdir -p outputs/logs

echo "[$(date '+%F %T')] Waiting for clean MBPP+ queue to finish"
while pgrep -f "run_mbppplus_clean_validation.sh" >/dev/null || \
      pgrep -f "generate_mbppplus_evalplus.py" >/dev/null || \
      pgrep -f "evalplus.evaluate mbpp --samples samples.jsonl" >/dev/null; do
  sleep 300
done

echo "[$(date '+%F %T')] MBPP+ queue finished; launching CodeGen ladder"
source venv/bin/activate
nohup bash run_codegen_pretraining_ladder.sh > outputs/logs/codegen_ladder_nohup.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] CodeGen ladder launch submitted"
