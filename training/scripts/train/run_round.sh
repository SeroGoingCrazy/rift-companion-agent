#!/usr/bin/env bash
# Train, merge and convert one SFT round for both sizes (run inside the WSL training env).
#   bash training/scripts/train/run_round.sh r002
set -euo pipefail
round=$1
venv=${RIFT_TRAIN_VENV:-$HOME/rift-train/.venv}
llama=${LLAMA_CPP_DIR:-$HOME/rift-train/llama.cpp}
logs=${RIFT_TRAIN_LOGS:-$HOME/rift-train/logs}
cfg=training/configs/training/llamafactory
mkdir -p "$logs" training/outputs/gguf
for size in 0.6b 1.7b; do
  echo "[$(date +%T)] train $size $round"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$venv/bin/llamafactory-cli" train "$cfg/sft_${size}_${round}.yaml" > "$logs/sft_${size}_${round}.log" 2>&1
  echo "[$(date +%T)] merge $size"
  "$venv/bin/llamafactory-cli" export "$cfg/merge_${size}_${round}.yaml" > "$logs/merge_${size}_${round}.log" 2>&1
  echo "[$(date +%T)] gguf $size"
  "$venv/bin/python" "$llama/convert_hf_to_gguf.py" "training/outputs/merged/sft_${size}_${round}" \
    --outtype f16 --outfile "training/outputs/gguf/rift-slot-${size}-${round}-f16.gguf" > "$logs/gguf_${size}_${round}.log" 2>&1
done
echo "[$(date +%T)] done"
