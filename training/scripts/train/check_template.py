"""Check that LLaMA-Factory renders SFT samples exactly like the official chat template.

Runs in the training environment (WSL, LLaMA-Factory installed), not in the uv workspace.
For each sample, the prompt LLaMA-Factory trains on must equal the official Qwen3 template
with ``enable_thinking=False`` (what llama-server renders online), and prompt + answer must
equal the full rendered conversation. Also prints token lengths to size ``cutoff_len``::

    python training/scripts/train/check_template.py \
        --data training/data/processed/sft/v0.2/train.jsonl --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
from transformers import AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--template", default="qwen3")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = get_template_and_fix_tokenizer(
        tokenizer, DataArguments(template=args.template, enable_thinking=False)
    )
    with open(args.data, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if args.limit:
        rows = rows[: args.limit]

    mismatches, lengths, answer_lengths = 0, [], []
    for row in rows:
        system, user, assistant = row["messages"]
        prompt_ids, answer_ids = template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": user["content"]},
                {"role": "assistant", "content": assistant["content"]},
            ],
            system["content"],
        )
        official_prompt = tokenizer.apply_chat_template(
            [system, user], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        official_full = tokenizer.apply_chat_template(
            [system, user, assistant], tokenize=False, enable_thinking=False
        )
        trained_prompt = tokenizer.decode(prompt_ids)
        trained_full = tokenizer.decode(prompt_ids + answer_ids)
        if trained_prompt != official_prompt or trained_full.rstrip() != official_full.rstrip():
            mismatches += 1
            if mismatches <= 2:
                print(f"--- mismatch {row['id']}")
                print("trained prompt tail :", repr(trained_prompt[-80:]))
                print("official prompt tail:", repr(official_prompt[-80:]))
                print("trained full tail   :", repr(trained_full[-120:]))
                print("official full tail  :", repr(official_full[-120:]))
        lengths.append(len(prompt_ids) + len(answer_ids))
        answer_lengths.append(len(answer_ids))

    print(f"samples {len(rows)}, template mismatches {mismatches}")
    print(
        f"tokens per sample: median {statistics.median(lengths)}, max {max(lengths)}; "
        f"answer: median {statistics.median(answer_lengths)}, max {max(answer_lengths)}"
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
