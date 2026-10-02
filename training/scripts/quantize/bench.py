"""CPU benchmark of every quantized variant in one run (DEV_SPEC I10).

    uv run --env-file .env python training/scripts/quantize/bench.py
    uv run --env-file .env python training/scripts/quantize/bench.py --variants f16 q4_k_m \
        --limit 10

Run on the Windows host from the repo root after ``quantize.py``. For each variant, one after
another on the same machine and llama.cpp build:

1. start the CPU llama-server in WSL (same flags as the I6-I9 evals) and time the load;
2. warm the shared prefix: the first request carries the static system prompt into the KV
   cache, and its timings are kept as the cold-start number;
3. speed: stream every L2 prompt (main + holdout, byte-identical to production, temperature 0,
   max_tokens 160, ``json_object``) and record client TTFT / E2E plus the server's prompt and
   decode timings; a few more requests with ``cache_prompt: false`` show what the warm prefix saves;
4. quality: L2 main + holdout through the production ``local`` extractor, reports in
   ``eval/reports/quant_<variant>_06b_slot_<dataset>.*``;
5. peak RSS of the server, then stop it.

Writes ``training/experiments/quant/bench.{json,md}`` and exits 2 if the candidate
(Q4_K_M) fails more than ``gate.max_extra_failures`` samples beyond the baseline (F16).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

from rift_agent.extractors.prompt import build_messages
from rift_training.contract import context_from_sample
from rift_training.data.specs import REPO_ROOT
from rift_training.evaluation.dataset import load_samples
from rift_training.quantization import (
    DONE,
    QuantConfig,
    RequestTiming,
    VariantResult,
    apply_server_timings,
    check_gate,
    chunk_content,
    keyed,
    load_config,
    parse_sse_line,
    render_markdown,
    server_command,
    speed_summary,
)

DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "quantization" / "rift_slot_0.6b.yaml"
DEFAULT_OUT = REPO_ROOT / "training" / "experiments" / "quant" / "bench"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"


def load_runner() -> ModuleType:
    path = REPO_ROOT / "eval" / "runners" / "run_slot_eval.py"
    spec = importlib.util.spec_from_file_location("run_slot_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_slot_eval"] = module
    spec.loader.exec_module(module)
    return module


# --- WSL llama-server -----------------------------------------------------------------------


def wsl(command: str, *, check: bool = True) -> str:
    done = subprocess.run(
        ["wsl", "-e", "bash", "-lc", command],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return done.stdout.strip()


def stop_server() -> None:
    wsl("pkill -x llama-server; sleep 1; true", check=False)


def start_server(cfg: QuantConfig, variant: str) -> float:
    """Start one variant and wait for ``/health`` inside the same wsl call; seconds to healthy.

    A job backgrounded as the last command of ``wsl bash -lc`` dies with the session before it
    starts, hence the health loop in the same call (see eval_round.sh).
    """
    stop_server()
    port = cfg.server["port"]
    log = f"{cfg.llama_cpp['logs']}/llama_server_quant_{variant}.log"
    t0 = time.perf_counter()
    wsl(
        f"cd {cfg.llama_cpp['repo_wsl']} && nohup {server_command(cfg, variant)} > {log} 2>&1 &\n"
        f"for _ in $(seq 600); do curl -s localhost:{port}/health | grep -q ok && exit 0; "
        "sleep 0.1; done; echo 'llama-server did not become healthy' >&2; exit 1"
    )
    return time.perf_counter() - t0


def rss_peak_mb() -> float:
    out = wsl("grep VmHWM /proc/$(pgrep -x llama-server)/status", check=False)
    try:
        return int(out.split()[1]) / 1024
    except (IndexError, ValueError):
        return 0.0


def machine_info(cfg: QuantConfig) -> dict[str, str]:
    cpu = wsl("lscpu | sed -n 's/^Model name: *//p'", check=False)
    root = cfg.llama_cpp["server"].rsplit("/build/", 1)[0]
    commit = wsl(f"git -C {root} rev-parse --short HEAD", check=False)
    return {"cpu": cpu, "llama_cpp": commit}


# --- requests -------------------------------------------------------------------------------


def stream_request(
    client: httpx.Client,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    cache_prompt: bool = True,
) -> RequestTiming:
    """One streamed chat completion with the production payload."""
    payload: dict[str, Any] = {
        "model": "local",
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "cache_prompt": cache_prompt,
    }
    parts: list[str] = []
    ttft: float | None = None
    server_timings: dict[str, Any] = {}
    t0 = time.perf_counter()
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            chunk = parse_sse_line(line)
            if chunk is DONE:
                break
            if not isinstance(chunk, dict):
                continue
            content = chunk_content(chunk)
            if content:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(content)
            if isinstance(chunk.get("timings"), dict):
                server_timings = chunk["timings"]
    e2e = (time.perf_counter() - t0) * 1000
    timing = RequestTiming(
        ttft_ms=round(ttft if ttft is not None else e2e, 1), e2e_ms=round(e2e, 1)
    )
    timing.text = "".join(parts)
    apply_server_timings(timing, server_timings)
    return timing


# --- main -----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark quantized GGUFs on the CPU.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*", help="default: every configured variant")
    parser.add_argument("--limit", type=int, default=None, help="first N samples per dataset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report stem (.json + .md)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    variants = args.variants or list(cfg.variants)
    missing = [v for v in variants if not (REPO_ROOT / cfg.gguf(v)).exists()]
    if missing:
        print(f"GGUF missing for {missing}; run quantize.py first", file=sys.stderr)
        return 1

    # 127.0.0.1, not localhost: on Windows localhost tries ::1 first, which the WSL port
    # forward does not answer, and Python waits ~2 s per request before falling back to IPv4.
    base_url = f"http://127.0.0.1:{cfg.server['port']}"
    os.environ["LLAMA_SERVER_URL"] = f"{base_url}/v1"
    os.environ.setdefault("LOCAL_SLOT_TIMEOUT_S", str(cfg.bench.get("timeout_s", 60)))

    from rift_common.settings import load_settings
    from rift_training.evaluation.preferences import make_style_matcher

    runner = load_runner()
    settings = load_settings(runner.DEFAULT_SETTINGS)
    local = settings.llm["local_slot"]
    temperature = local.temperature if local.temperature is not None else 0.0
    max_tokens = local.max_tokens or 160
    style = make_style_matcher(settings.embedding)
    extractor = runner.make_eval_extractor("local", settings)

    datasets: list[str] = list(cfg.bench.get("datasets", ["main", "holdout"]))
    samples = {d: load_samples(runner.resolve_dataset(d))[: args.limit] for d in datasets}
    prompts = [
        (d, s.id, build_messages(context_from_sample(s))) for d in datasets for s in samples[d]
    ]
    nocache_n = int(cfg.bench.get("nocache_samples", 0))
    started = datetime.now()
    meta: dict[str, Any] = {
        "model": cfg.model,
        "source": cfg.source,
        "started_at": f"{started:%Y-%m-%d %H:%M:%S}",
        "datasets": datasets,
        "limit": args.limit,
        "threads": cfg.server["threads"],
        "ctx": cfg.server["ctx"],
        "parallel": cfg.server["parallel"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "nocache_samples": nocache_n,
        **machine_info(cfg),
    }

    results: dict[str, VariantResult] = {}
    client = httpx.Client(base_url=base_url, timeout=cfg.bench.get("timeout_s", 60))
    try:
        for variant in variants:
            gguf = cfg.gguf(variant)
            r = VariantResult(variant, cfg.variants[variant], gguf)
            r.size_bytes = (REPO_ROOT / gguf).stat().st_size
            print(f"\n== {variant} ({r.qtype}, {r.size_bytes / 2**20:.0f} MB)")
            r.load_s = round(start_server(cfg, variant), 2)

            # Warm the shared prefix; the first request is the cold-start number.
            cold = stream_request(
                client, prompts[0][2], temperature=temperature, max_tokens=max_tokens
            )
            r.cold = {k: v for k, v in asdict(cold).items() if k != "text"}

            streamed: dict[str, str] = {}
            timings: list[RequestTiming] = []
            for d, sid, messages in prompts:
                t = stream_request(client, messages, temperature=temperature, max_tokens=max_tokens)
                timings.append(t)
                streamed[f"{d}/{sid}"] = t.text
            r.speed = speed_summary(timings)
            r.nocache = speed_summary(
                [
                    stream_request(
                        client,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        cache_prompt=False,
                    )
                    for _, _, messages in prompts[:nocache_n]
                ]
            )
            s = r.speed
            print(
                f"load {r.load_s:.1f}s; cold TTFT {cold.ttft_ms:.0f} ms; "
                f"TTFT P50 {s['ttft_p50']:.0f} ms; "
                f"E2E P50 {s['e2e_p50']:.0f} ms; decode {s['decode_tps_p50']:.1f} tok/s; "
                f"no-cache TTFT P50 {r.nocache['ttft_p50']:.0f} ms"
            )

            for d in datasets:
                out = REPORTS_DIR / f"quant_{variant}_06b_slot_{d}"
                ns = argparse.Namespace(
                    extractor="local",
                    dataset=d,
                    settings=runner.DEFAULT_SETTINGS,
                    limit=args.limit,
                    concurrency=1,
                    threshold=runner.PASS_THRESHOLD,
                    style_threshold=style.threshold,
                    exact_style=False,
                    examples=15,
                    out=out,
                )
                report = runner.run(ns, extractor=extractor, style=style)
                summ = report["summary"]
                r.quality[d] = {
                    "passed": summ["passed"],
                    "n": summ["n"],
                    "protocol_pass_rate": summ["protocol_pass_rate"],
                    "report": str(out.relative_to(REPO_ROOT)).replace("\\", "/") + ".md",
                }
                r.passed.update(keyed(d, ((i["id"], i["passed"]) for i in report["items"])))
                r.stream_mismatch += [
                    f"{d}/{i['id']}"
                    for i in report["items"]
                    if streamed.get(f"{d}/{i['id']}") != i["raw"]
                ]
                print(f"{variant} {d}: {summ['passed']}/{summ['n']}")
            r.rss_peak_mb = round(rss_peak_mb(), 1)
            results[variant] = r
    finally:
        client.close()
        stop_server()

    gate_cfg = cfg.gate
    if gate_cfg["baseline"] in results and gate_cfg["candidate"] in results:
        gate = check_gate(results, gate_cfg)
    else:
        gate = {
            **gate_cfg,
            "ok": True,
            "extra_failures": 0,
            "baseline_failed": 0,
            "candidate_failed": 0,
            "lost": [],
            "gained": [],
            "skipped": True,
        }
    ordered = [results[v] for v in variants]
    bench = {"meta": meta, "gate": gate, "results": [asdict(r) for r in ordered]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(bench, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    args.out.with_suffix(".md").write_text(
        render_markdown(ordered, gate, meta), encoding="utf-8", newline="\n"
    )
    print(
        f"\ngate {gate['candidate']} vs {gate['baseline']}: +{gate['extra_failures']} failures "
        f"(max {gate['max_extra_failures']}) -> {'PASS' if gate['ok'] else 'FAIL'}"
    )
    print(f"report: {args.out.with_suffix('.md')}")
    return 0 if gate["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
