#!/usr/bin/env bash
# DPO ablation (DEV_SPEC I9): train, merge and convert each group on top of SFT r003 0.6B.
# Run inside the WSL training env from the repo root:
#   bash training/scripts/train/run_dpo.sh                       # all four groups
#   bash training/scripts/train/run_dpo.sh rule_b01 onpolicy_b03  # some of them
# GGUFs land as training/outputs/gguf/rift-slot-0.6b-dpo_<group>-f16.gguf, so
#   bash training/scripts/eval/eval_round.sh dpo_<group> 0.6b
# evaluates one group.
set -euo pipefail
groups=${*:-"rule_b01 rule_b03 onpolicy_b01 onpolicy_b03"}
venv=${RIFT_TRAIN_VENV:-$HOME/rift-train/.venv}
llama=${LLAMA_CPP_DIR:-$HOME/rift-train/llama.cpp}
logs=${RIFT_TRAIN_LOGS:-$HOME/rift-train/logs}
cfg=training/configs/training/llamafactory
mkdir -p "$logs" training/outputs/gguf
for g in $groups; do
  echo "[$(date +%T)] train dpo $g"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$venv/bin/llamafactory-cli" train "$cfg/dpo_0.6b_${g}.yaml" > "$logs/dpo_0.6b_${g}.log" 2>&1
  echo "[$(date +%T)] merge $g"
  "$venv/bin/llamafactory-cli" export "$cfg/merge_dpo_0.6b_${g}.yaml" > "$logs/merge_dpo_0.6b_${g}.log" 2>&1
  echo "[$(date +%T)] gguf $g"
  "$venv/bin/python" "$llama/convert_hf_to_gguf.py" "training/outputs/merged/dpo_0.6b_${g}" \
    --outtype f16 --outfile "training/outputs/gguf/rift-slot-0.6b-dpo_${g}-f16.gguf" \
    > "$logs/gguf_dpo_0.6b_${g}.log" 2>&1
done
echo "[$(date +%T)] done"
