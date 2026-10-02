"""Quantize the final F16 GGUF into every configured variant (DEV_SPEC I10).

    uv run python training/scripts/quantize/quantize.py
    uv run python training/scripts/quantize/quantize.py --variants q4_k_m --force

Runs llama.cpp's ``llama-quantize`` inside WSL (the same CPU build that serves the model), skips
variants whose GGUF already exists unless ``--force``, and writes ``manifest.json`` (file, type,
size, sha256 per variant) next to the benchmark report. Variants marked ``imatrix: true`` first
need the importance matrix: calibration text rendered from the SFT train split, then
``llama-imatrix`` over it (both skipped when the files exist).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rift_training.data.specs import REPO_ROOT
from rift_training.quantization import (
    SOURCE_VARIANT,
    QuantConfig,
    calibration_text,
    imatrix_command,
    load_config,
    quantize_command,
)

DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "quantization" / "rift_slot_0.6b.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "training" / "experiments" / "quant" / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def wsl(command: str) -> None:
    subprocess.run(["wsl", "-e", "bash", "-lc", command], check=True)


def ensure_imatrix(cfg: QuantConfig, *, force: bool) -> None:
    assert cfg.imatrix is not None
    im = cfg.imatrix
    calibration = REPO_ROOT / im.calibration
    if force or not calibration.exists():
        with (REPO_ROOT / im.dataset).open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        text = calibration_text(rows, im.samples, im.seed)
        calibration.write_text(text, encoding="utf-8", newline="\n")
        print(f"calibration: {min(im.samples, len(rows))} train samples -> {im.calibration}")
    if force or not (REPO_ROOT / im.file).exists():
        print(f"imatrix -> {im.file}")
        wsl(f"cd {cfg.llama_cpp['repo_wsl']} && {imatrix_command(cfg)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize the final GGUF with llama-quantize.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*", help="default: every configured variant")
    parser.add_argument("--force", action="store_true", help="re-quantize existing files")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    source = REPO_ROOT / cfg.source
    if not source.exists():
        print(f"source GGUF missing: {cfg.source}", file=sys.stderr)
        return 1
    todo = [
        v
        for v in args.variants or list(cfg.variants)
        if v != SOURCE_VARIANT and (args.force or not (REPO_ROOT / cfg.gguf(v)).exists())
    ]
    if cfg.imatrix_variants & set(todo):
        ensure_imatrix(cfg, force=args.force)
    for variant in todo:
        print(f"quantize {variant} ({cfg.variants[variant]}) -> {cfg.gguf(variant)}")
        wsl(f"cd {cfg.llama_cpp['repo_wsl']} && {quantize_command(cfg, variant)}")

    entries = {}
    for variant, qtype in cfg.variants.items():
        path = REPO_ROOT / cfg.gguf(variant)
        if not path.exists():
            continue
        entries[variant] = {
            "file": path.name,
            "type": qtype,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        print(f"{variant:>7}: {entries[variant]['size_bytes'] / 2**20:7.0f} MB  {path.name}")
    manifest = {
        "model": cfg.model,
        "source": cfg.source,
        "imatrix": asdict(cfg.imatrix) if cfg.imatrix else None,
        "imatrix_variants": sorted(cfg.imatrix_variants),
        "variants": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"manifest: {args.manifest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
