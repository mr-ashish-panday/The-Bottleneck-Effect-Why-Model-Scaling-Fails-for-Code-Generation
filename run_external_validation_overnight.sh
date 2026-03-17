#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ashish/paper11_code_execution_failures"
cd "$ROOT"

mkdir -p outputs/logs

echo "[$(date '+%F %T')] Starting external-validation overnight queue" | tee outputs/logs/external_validation_overnight.log

./run_evalplus_rescoring.sh 2>&1 | tee -a outputs/logs/external_validation_overnight.log
./run_livecodebench_overnight.sh 2>&1 | tee -a outputs/logs/external_validation_overnight.log

echo "[$(date '+%F %T')] External-validation overnight queue completed" | tee -a outputs/logs/external_validation_overnight.log
