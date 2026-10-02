"""On-policy sampling for DPO (DEV_SPEC I8 ②), on the GPU of the training env (WSL).

Samples ``k`` answers per SFT train prompt from a merged SFT model with the same chat template
as training and serving (Qwen3, ``enable_thinking=False``). Scoring and pairing happen in the
uv workspace (``build_dpo.py --mode onpolicy``), which owns the L2 scorer::

    python training/scripts/train/sample_onpolicy.py \
        --model training/outputs/merged/sft_0.6b_r003 \
        --data training/data/processed/sft/v0.4/train.jsonl \
        --out training/data/raw/onpolicy_r003_0.6b.jsonl --k 4 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--batch", type=int, default=8, help="prompts per generate(); 16 x k=4 spills 16GB VRAM"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    with open(args.data, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    start = time.time()
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for i in range(0, len(rows), args.batch):
            batch = rows[i : i + args.batch]
            prompts = [
                tokenizer.apply_chat_template(
                    r["messages"][:2],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for r in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.k,
                    pad_token_id=tokenizer.pad_token_id,
                )
            texts = tokenizer.batch_decode(
                gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for j, row in enumerate(batch):
                samples = [t.strip() for t in texts[j * args.k : (j + 1) * args.k]]
                out.write(
                    json.dumps({"id": row["id"], "samples": samples}, ensure_ascii=False) + "\n"
                )
            done = i + len(batch)
            print(f"{done}/{len(rows)} prompts, {time.time() - start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
