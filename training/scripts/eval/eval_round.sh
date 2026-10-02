#!/usr/bin/env bash
# L2 eval of one SFT round (DEV_SPEC I6/I7): serve each GGUF with the CPU llama-server in WSL,
# run main + holdout through the online extractor path, write eval/reports/<round>_<size>_*.
# Run from the repo root in Git Bash on the Windows host:
#   bash training/scripts/eval/eval_round.sh r002            # both sizes
#   bash training/scripts/eval/eval_round.sh r002 0.6b       # one size
set -euo pipefail
round=$1
sizes=${2:-"0.6b 1.7b"}
repo_wsl=${RIFT_REPO_WSL:-/mnt/c/Users/1/rift-companion-agent}
server=${LLAMA_SERVER_BIN:-'~/rift-train/llama.cpp/build/bin/llama-server'}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-'C:\Users\1\.uv\python'}
export LLAMA_SERVER_URL=http://localhost:8080/v1
export LOCAL_SLOT_TIMEOUT_S=60
export PYTHONIOENCODING=utf-8

stop_server() { wsl -e bash -lc 'pkill -x llama-server || true'; }
trap stop_server EXIT

for size in $sizes; do
  gguf="training/outputs/gguf/rift-slot-${size}-${round}-f16.gguf"
  stop_server
  # Thinking off, exactly like training (template qwen3 + enable_thinking=false).
  wsl -e bash -lc "cd $repo_wsl && nohup $server -m $gguf --host 0.0.0.0 --port 8080 -c 4096 -np 1 -t 10 \
    --jinja --chat-template-kwargs '{\"enable_thinking\":false}' > ~/rift-train/logs/llama_server_${size}_${round}.log 2>&1 &"
  until curl -s localhost:8080/health | grep -q ok; do sleep 2; done
  tag="${round}_${size/./}"
  for ds in main holdout; do
    python -m uv run --env-file .env python eval/runners/run_slot_eval.py \
      --extractor local --dataset "$ds" --concurrency 1 --out "eval/reports/${tag}_slot_${ds}" \
      | grep -E "^pass" | sed "s/^/$tag $ds: /"
  done
done
