"""Quantization and CPU benchmark helpers (DEV_SPEC I10).

``training/scripts/quantize/quantize.py`` turns the final F16 GGUF into the configured
variants with llama.cpp's ``llama-quantize``. ``training/scripts/quantize/bench.py`` then
serves every variant in turn from the same CPU llama-server build in one run and measures
speed and L2 quality. This module holds the parts that do not touch a server: config, the
server command, SSE timing parsing, the statistics, the quality gate and the Markdown report.
"""

from __future__ import annotations

import json
import math
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: The variant that is the source GGUF itself (never re-quantized).
SOURCE_VARIANT = "f16"


@dataclass(frozen=True)
class QuantConfig:
    model: str
    source: str
    out_dir: str
    variants: dict[str, str]
    llama_cpp: dict[str, str]
    server: dict[str, Any]
    bench: dict[str, Any]
    gate: dict[str, Any]
    imatrix: str | None = None

    def gguf(self, variant: str) -> str:
        """Repo-relative path of one variant's GGUF (the source for ``f16``)."""
        if variant not in self.variants:
            raise KeyError(f"unknown variant {variant!r}; known: {', '.join(self.variants)}")
        if variant == SOURCE_VARIANT:
            return self.source
        return f"{self.out_dir}/{self.model}-{variant}.gguf"


def load_config(path: Path) -> QuantConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = {str(k): str(v) for k, v in raw["variants"].items()}
    if SOURCE_VARIANT not in variants:
        raise ValueError(f"{path}: variants must include {SOURCE_VARIANT!r} (the source GGUF)")
    gate = raw["gate"]
    for key in ("baseline", "candidate"):
        if gate[key] not in variants:
            raise ValueError(f"{path}: gate.{key} {gate[key]!r} is not a variant")
    return QuantConfig(
        model=raw["model"],
        source=raw["source"],
        out_dir=raw["out_dir"],
        variants=variants,
        llama_cpp=dict(raw["llama_cpp"]),
        server=dict(raw["server"]),
        bench=dict(raw.get("bench") or {}),
        gate=dict(gate),
        imatrix=raw.get("imatrix"),
    )


def quantize_command(cfg: QuantConfig, variant: str) -> str:
    """The ``llama-quantize`` call (run from the repo root inside WSL)."""
    parts = [cfg.llama_cpp["quantize"]]
    if cfg.imatrix:
        parts += ["--imatrix", shlex.quote(cfg.imatrix)]
    parts += [shlex.quote(cfg.source), shlex.quote(cfg.gguf(variant)), cfg.variants[variant]]
    return " ".join(parts)


def server_command(cfg: QuantConfig, variant: str) -> str:
    """The ``llama-server`` call for one variant, same flags as the L2 evals of I6-I9."""
    s = cfg.server
    kwargs = json.dumps(s.get("chat_template_kwargs") or {}, separators=(",", ":"))
    return " ".join(
        [
            cfg.llama_cpp["server"],
            "-m",
            shlex.quote(cfg.gguf(variant)),
            "--host 0.0.0.0",
            f"--port {s['port']}",
            f"-c {s['ctx']}",
            f"-np {s['parallel']}",
            f"-t {s['threads']}",
            "--jinja",
            "--chat-template-kwargs",
            shlex.quote(kwargs),
        ]
    )


# --- one streamed request ------------------------------------------------------------------


@dataclass
class RequestTiming:
    """Client-side TTFT / E2E plus llama-server's own ``timings`` for one request."""

    ttft_ms: float
    e2e_ms: float
    text: str = ""
    prompt_n: int = 0  # prompt tokens actually evaluated (excludes the cached prefix)
    prompt_ms: float = 0.0
    cache_n: int = 0  # prompt tokens reused from the KV cache
    predicted_n: int = 0
    predicted_ms: float = 0.0

    @property
    def decode_tps(self) -> float:
        return self.predicted_n / self.predicted_ms * 1000 if self.predicted_ms > 0 else 0.0


DONE = object()


def parse_sse_line(line: str) -> dict[str, Any] | object | None:
    """One SSE line: the chunk dict, ``DONE`` at the end, ``None`` for anything else."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return DONE
    try:
        chunk = json.loads(data)
    except ValueError:
        return None
    return chunk if isinstance(chunk, dict) else None


def chunk_content(chunk: Mapping[str, Any]) -> str:
    try:
        content = chunk["choices"][0].get("delta", {}).get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return content if isinstance(content, str) else ""


def apply_server_timings(timing: RequestTiming, timings: Mapping[str, Any]) -> None:
    timing.prompt_n = int(timings.get("prompt_n", 0))
    timing.prompt_ms = float(timings.get("prompt_ms", 0.0))
    timing.cache_n = int(timings.get("cache_n", 0))
    timing.predicted_n = int(timings.get("predicted_n", 0))
    timing.predicted_ms = float(timings.get("predicted_ms", 0.0))


# --- statistics ----------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def speed_summary(timings: Sequence[RequestTiming]) -> dict[str, float]:
    ttft = [t.ttft_ms for t in timings]
    e2e = [t.e2e_ms for t in timings]
    tps = [t.decode_tps for t in timings if t.predicted_n > 1]
    return {
        "n": len(timings),
        "ttft_p50": round(percentile(ttft, 50), 1),
        "ttft_p95": round(percentile(ttft, 95), 1),
        "e2e_p50": round(percentile(e2e, 50), 1),
        "e2e_p95": round(percentile(e2e, 95), 1),
        "decode_tps_p50": round(percentile(tps, 50), 1),
        "prompt_eval_mean": round(_mean([t.prompt_n for t in timings]), 1),
        "cached_mean": round(_mean([t.cache_n for t in timings]), 1),
        "output_tokens_mean": round(_mean([t.predicted_n for t in timings]), 1),
    }


# --- quality -------------------------------------------------------------------------------


@dataclass
class VariantResult:
    variant: str
    qtype: str
    gguf: str
    size_bytes: int = 0
    load_s: float = 0.0
    rss_peak_mb: float = 0.0
    cold: dict[str, Any] = field(default_factory=dict)
    speed: dict[str, float] = field(default_factory=dict)
    nocache: dict[str, float] = field(default_factory=dict)
    #: dataset -> {"passed", "n", "protocol_pass_rate", "report"}
    quality: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: "<dataset>/<sample id>" -> passed
    passed: dict[str, bool] = field(default_factory=dict)
    #: samples whose streamed speed-pass text differs from the extractor's output
    stream_mismatch: list[str] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(self.passed.values())

    @property
    def n_failed(self) -> int:
        return len(self.passed) - self.n_passed


def flips(base: VariantResult, other: VariantResult) -> dict[str, list[str]]:
    """Samples that pass in one variant but not the other."""
    shared = sorted(base.passed.keys() & other.passed.keys())
    return {
        "lost": [k for k in shared if base.passed[k] and not other.passed[k]],
        "gained": [k for k in shared if not base.passed[k] and other.passed[k]],
    }


def check_gate(results: Mapping[str, VariantResult], gate: Mapping[str, Any]) -> dict[str, Any]:
    """Candidate may fail at most ``max_extra_failures`` more samples than the baseline."""
    base, cand = results[gate["baseline"]], results[gate["candidate"]]
    if base.passed.keys() != cand.passed.keys():
        raise ValueError("baseline and candidate were scored on different samples")
    extra = cand.n_failed - base.n_failed
    return {
        "baseline": base.variant,
        "candidate": cand.variant,
        "baseline_failed": base.n_failed,
        "candidate_failed": cand.n_failed,
        "extra_failures": extra,
        "max_extra_failures": gate["max_extra_failures"],
        "ok": extra <= gate["max_extra_failures"],
        **flips(base, cand),
    }


# --- Markdown ------------------------------------------------------------------------------


def _mb(n: float) -> str:
    return f"{n / 1024 / 1024:.0f}"


def render_markdown(
    results: Sequence[VariantResult], gate: Mapping[str, Any], meta: Mapping[str, Any]
) -> str:
    datasets: list[str] = list(meta.get("datasets", []))
    base = next((r for r in results if r.variant == gate["baseline"]), None)
    lines = [
        f"# 量化基准：{meta['model']}",
        "",
        f"- 时间：{meta['started_at']}；机器：{meta.get('cpu', '-')}，"
        f"llama.cpp `{meta.get('llama_cpp', '-')}`",
        f"- 源模型：`{meta['source']}`",
        f"- llama-server：`-t {meta['threads']} -c {meta['ctx']} -np {meta['parallel']}`，"
        "temperature 0，max_tokens 160，`response_format: json_object`，thinking 关闭",
        "- 所有变体在同一次运行中依次加载、预热共享前缀后测量；速度用 L2 全部样本的线上 prompt，"
        "质量走线上 `local` 抽取器路径（与 I6–I9 相同）。",
        "",
        "## 质量（L2）",
        "",
        "| 变体 | 大小 MB | "
        + " | ".join(f"{d}" for d in datasets)
        + " | 合计 | 协议 | 相对基准 |",
        "|---|---:|" + "---:|" * len(datasets) + "---:|---:|---|",
    ]
    for r in results:
        cells = [
            f"{r.quality[d]['passed']}/{r.quality[d]['n']}"
            f"（{r.quality[d]['passed'] / r.quality[d]['n'] * 100:.1f}%）"
            for d in datasets
        ]
        proto = min((r.quality[d]["protocol_pass_rate"] for d in datasets), default=0.0)
        if base is None or r is base:
            delta = "基准" if r is base else "-"
        else:
            f = flips(base, r)
            delta = f"−{len(f['lost'])} / +{len(f['gained'])}"
        lines.append(
            f"| `{r.variant}` | {_mb(r.size_bytes)} | {' | '.join(cells)} | "
            f"{r.n_passed}/{len(r.passed)} | {proto * 100:.1f}% | {delta} |"
        )

    ok = "**通过**" if gate["ok"] else "**未通过**"
    lines += [
        "",
        f"门槛：`{gate['candidate']}` 比 `{gate['baseline']}` 多错 {gate['extra_failures']} 条"
        f"（{gate['candidate_failed']} vs {gate['baseline_failed']}，"
        f"上限 {gate['max_extra_failures']}）→ {ok}。",
    ]
    if gate["lost"] or gate["gained"]:
        lines.append(
            f"变差：{', '.join(gate['lost']) or '无'}；变好：{', '.join(gate['gained']) or '无'}。"
        )

    lines += [
        "",
        "## 速度（CPU）",
        "",
        "| 变体 | 加载 s | 峰值 RSS MB | 冷启动首请求 TTFT ms | TTFT P50 / P95 ms | "
        "无缓存 TTFT P50 ms | E2E P50 / P95 ms | decode tok/s | "
        "平均 prompt 计算 / 复用 tok | 平均输出 tok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        s, nc = r.speed, r.nocache
        lines.append(
            f"| `{r.variant}` | {r.load_s:.1f} | {r.rss_peak_mb:.0f} | "
            f"{r.cold.get('ttft_ms', 0):.0f} | {s['ttft_p50']:.0f} / {s['ttft_p95']:.0f} | "
            f"{nc.get('ttft_p50', 0):.0f} | {s['e2e_p50']:.0f} / {s['e2e_p95']:.0f} | "
            f"{s['decode_tps_p50']:.1f} | {s['prompt_eval_mean']:.0f} / {s['cached_mean']:.0f} | "
            f"{s['output_tokens_mean']:.0f} |"
        )
    mismatched = [(r.variant, len(r.stream_mismatch)) for r in results if r.stream_mismatch]
    lines += [
        "",
        "冷启动首请求即预热请求（共享 system prompt 尚未进入 KV 缓存）；无缓存 TTFT 为 "
        f"`cache_prompt: false` 的 {meta.get('nocache_samples', 0)} 条请求。",
        "速度阶段的流式输出与质量阶段抽取器输出逐字比较："
        + (
            "，".join(f"`{v}` {n} 条不同" for v, n in mismatched)
            if mismatched
            else "全部一致（temperature 0 下可复现）。"
        ),
    ]
    reports: list[str] = []
    for r in results:
        reports += [r.quality[d]["report"] for d in datasets if r.quality[d].get("report")]
    if reports:
        lines += ["", "L2 逐条报告：" + "、".join(f"`{p}`" for p in reports) + "。"]
    return "\n".join(lines).rstrip() + "\n"


def keyed(dataset: str, ids_passed: Iterable[tuple[str, bool]]) -> dict[str, bool]:
    return {f"{dataset}/{i}": ok for i, ok in ids_passed}
