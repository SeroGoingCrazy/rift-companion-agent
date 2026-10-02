"""I10: quantization config, server command, SSE timings, statistics, gate and report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rift_training.quantization import (
    DONE,
    RequestTiming,
    VariantResult,
    apply_server_timings,
    calibration_text,
    check_gate,
    chunk_content,
    flips,
    imatrix_command,
    keyed,
    load_config,
    parse_sse_line,
    percentile,
    quantize_command,
    render_chatml,
    render_markdown,
    server_command,
    speed_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "training" / "configs" / "quantization" / "rift_slot_0.6b.yaml"


def test_repo_config_loads_with_f16_source_and_gate() -> None:
    cfg = load_config(CONFIG)
    assert cfg.gguf("f16") == cfg.source
    assert cfg.gguf("q4_k_m") == "training/outputs/gguf/rift-slot-0.6b-q4_k_m.gguf"
    assert cfg.variants["q8_0"] == "Q8_0"
    assert cfg.variants["q4_k_m_imat"] == "Q4_K_M"
    assert cfg.imatrix_variants == {"q4_k_m_imat"}
    assert cfg.imatrix is not None and cfg.imatrix.dataset.endswith("sft/v0.4/train.jsonl")
    assert cfg.gate == {"baseline": "f16", "candidate": "q4_k_m_imat", "max_extra_failures": 2}
    with pytest.raises(KeyError):
        cfg.gguf("q2_k")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "q.yaml"
    path.write_text(CONFIG.read_text(encoding="utf-8").replace(*text.split("|")), encoding="utf-8")
    return path


def test_config_requires_the_f16_source_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="f16"):
        load_config(_write(tmp_path, "  f16: F16\n|"))


def test_config_rejects_a_gate_on_an_unknown_variant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate"):
        load_config(_write(tmp_path, "candidate: q4_k_m|candidate: q3_k_s"))


def test_quantize_command_passes_the_imatrix_only_to_imatrix_variants() -> None:
    cfg = load_config(CONFIG)
    plain = quantize_command(cfg, "q4_k_m")
    assert plain.endswith(f"{cfg.source} training/outputs/gguf/rift-slot-0.6b-q4_k_m.gguf Q4_K_M")
    assert "--imatrix" not in plain
    imat = quantize_command(cfg, "q4_k_m_imat")
    assert "--imatrix training/outputs/gguf/rift-slot-0.6b-imatrix.gguf " in imat
    assert imat.endswith("rift-slot-0.6b-q4_k_m_imat.gguf Q4_K_M")


def test_imatrix_variants_need_an_imatrix_and_never_apply_to_f16(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="none is configured"):
        load_config(_write(tmp_path, "\nimatrix:\n|\nunused_imatrix:\n"))
    with pytest.raises(ValueError, match="never quantized"):
        load_config(_write(tmp_path, "  f16: F16|  f16: {type: F16, imatrix: true}"))


def test_imatrix_command_reads_the_calibration_text_with_special_tokens() -> None:
    cmd = imatrix_command(load_config(CONFIG))
    assert "-m training/outputs/gguf/rift-slot-0.6b-dpo_rule_b03-f16.gguf" in cmd
    assert "-f training/outputs/gguf/rift-slot-0.6b-calibration.txt" in cmd
    assert "-o training/outputs/gguf/rift-slot-0.6b-imatrix.gguf" in cmd
    assert "-c 2048" in cmd and cmd.endswith("--parse-special")


def test_calibration_renders_chat_samples_reproducibly() -> None:
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "{}"},
    ]
    assert render_chatml(msgs) == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n{}<|im_end|>\n"
    )
    rows = [
        {"id": f"r{i}", "messages": [*msgs[:1], {"role": "user", "content": str(i)}]}
        for i in range(10)
    ]
    text = calibration_text(rows, 3, seed=1)
    assert text.count("<|im_start|>user") == 3
    assert text == calibration_text(list(reversed(rows)), 3, seed=1)  # order-independent
    assert calibration_text(rows, 50, seed=1).count("<|im_start|>user") == 10


def test_server_command_turns_thinking_off_like_training() -> None:
    cmd = server_command(load_config(CONFIG), "q8_0")
    assert "-m training/outputs/gguf/rift-slot-0.6b-q8_0.gguf" in cmd
    assert "-t 6" in cmd and "-np 1" in cmd and "--jinja" in cmd
    assert """--chat-template-kwargs '{"enable_thinking":false}'""" in cmd


def test_sse_lines_and_server_timings() -> None:
    assert parse_sse_line("data: [DONE]") is DONE
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("data: {broken") is None
    chunk = parse_sse_line('data: {"choices":[{"delta":{"content":"{\\"a\\""}}]}')
    assert isinstance(chunk, dict) and chunk_content(chunk) == '{"a"'
    assert chunk_content({"choices": [{"delta": {}}]}) == ""
    assert chunk_content({"choices": []}) == ""

    t = RequestTiming(ttft_ms=80.0, e2e_ms=700.0)
    apply_server_timings(
        t,
        {
            "prompt_n": 23,
            "prompt_ms": 60.5,
            "cache_n": 736,
            "predicted_n": 50,
            "predicted_ms": 625.0,
        },
    )
    assert (t.prompt_n, t.cache_n, t.predicted_n) == (23, 736, 50)
    assert t.decode_tps == pytest.approx(80.0)
    assert RequestTiming(ttft_ms=1, e2e_ms=1).decode_tps == 0.0


def test_percentile_and_speed_summary() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([3, 1, 2, 4], 50) == 2
    assert percentile([3, 1, 2, 4], 95) == 4
    timings = [
        RequestTiming(
            ttft_ms=float(i),
            e2e_ms=10.0 * i,
            prompt_n=20,
            cache_n=700,
            predicted_n=40,
            predicted_ms=500.0,
        )
        for i in range(1, 11)
    ]
    timings.append(RequestTiming(ttft_ms=5, e2e_ms=5, predicted_n=1, predicted_ms=1.0))
    s = speed_summary(timings)
    assert s["n"] == 11
    assert s["ttft_p50"] == 5 and s["e2e_p95"] == 100
    assert s["decode_tps_p50"] == 80.0  # one-token outputs are left out of the decode rate
    assert s["cached_mean"] == pytest.approx(636.4)


def _variant(name: str, passed: dict[str, bool]) -> VariantResult:
    r = VariantResult(name, name.upper(), f"{name}.gguf", size_bytes=400 * 2**20)
    r.passed = dict(passed)
    main = {k: v for k, v in passed.items() if k.startswith("main/")}
    r.quality = {
        "main": {
            "passed": sum(main.values()),
            "n": len(main),
            "protocol_pass_rate": 1.0,
            "report": f"eval/reports/quant_{name}_06b_slot_main.md",
        },
    }
    r.speed = speed_summary(
        [RequestTiming(ttft_ms=80, e2e_ms=700, predicted_n=50, predicted_ms=600)]
    )
    r.nocache = speed_summary([RequestTiming(ttft_ms=800, e2e_ms=1400)])
    return r


GATE = {"baseline": "f16", "candidate": "q4_k_m", "max_extra_failures": 2}


def test_gate_counts_net_extra_failures_and_lists_flips() -> None:
    base = _variant("f16", keyed("main", [("a", True), ("b", True), ("c", True), ("d", False)]))
    cand = _variant("q4_k_m", keyed("main", [("a", False), ("b", False), ("c", True), ("d", True)]))
    assert flips(base, cand) == {"lost": ["main/a", "main/b"], "gained": ["main/d"]}
    gate = check_gate({"f16": base, "q4_k_m": cand}, GATE)
    assert gate["extra_failures"] == 1 and gate["ok"]
    assert (gate["baseline_failed"], gate["candidate_failed"]) == (1, 2)


def test_gate_fails_beyond_the_limit_and_on_mismatched_samples() -> None:
    base = _variant("f16", keyed("main", [(c, True) for c in "abcd"]))
    worse = _variant(
        "q4_k_m", keyed("main", [("a", False), ("b", False), ("c", False), ("d", True)])
    )
    assert not check_gate({"f16": base, "q4_k_m": worse}, GATE)["ok"]
    other = _variant("q4_k_m", keyed("main", [("x", True)]))
    with pytest.raises(ValueError, match="different samples"):
        check_gate({"f16": base, "q4_k_m": other}, GATE)


def test_markdown_report_has_quality_gate_and_speed() -> None:
    base = _variant("f16", keyed("main", [("a", True), ("b", True)]))
    cand = _variant("q4_k_m", keyed("main", [("a", True), ("b", False)]))
    cand.stream_mismatch = ["main/b"]
    gate = check_gate({"f16": base, "q4_k_m": cand}, GATE)
    meta = {
        "model": "rift-slot-0.6b",
        "source": "src.gguf",
        "started_at": "2026-10-02 13:00:00",
        "datasets": ["main"],
        "threads": 6,
        "ctx": 4096,
        "parallel": 1,
        "nocache_samples": 1,
    }
    md = render_markdown([base, cand], gate, meta)
    assert "| `f16` | 400 | 2/2（100.0%） | 2/2 | 100.0% | 基准 |" in md
    assert "| `q4_k_m` | 400 | 1/2（50.0%） | 1/2 | 100.0% | −1 / +0 |" in md
    assert "多错 1 条" in md and "**通过**" in md and "变差：main/b" in md
    assert "`q4_k_m` 1 条不同" in md
    assert "eval/reports/quant_q4_k_m_06b_slot_main.md" in md
    json.dumps(gate)  # the gate goes into bench.json as is
