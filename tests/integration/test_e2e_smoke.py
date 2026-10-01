"""H4: L4 scenarios and runner — the CI smoke replays one scenario with no model at all
(mock intents + scripted extractions, in-process booking-mcp, keyword knowledge stub)."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = REPO_ROOT / "eval" / "scenarios"
SMOKE = "interject_refund_then_continue"


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_e2e", REPO_ROOT / "eval" / "runners" / "run_e2e.py")


def smoke_data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((SCENARIOS / f"{SMOKE}.yaml").read_text("utf-8"))
    return data


def mock_runtime(domain: Any) -> Any:
    return runner.Runtime("mock", domain, runner.make_knowledge("stub"))


def run_one(data: dict[str, Any], domain: Any) -> Any:
    return asyncio.run(runner.run_scenario(runner.parse_scenario(data), mock_runtime(domain)))


def booked(intent: str = "booking", **delta: Any) -> dict[str, Any]:
    return {"turn_intent": intent, "delta": delta, "confirmation": "none"}


# --- scenario files ------------------------------------------------------------------------


def test_scenario_files_parse() -> None:
    scenarios = runner.load_scenarios(SCENARIOS)
    assert len(scenarios) >= 20
    assert len({s.id for s in scenarios}) == len(scenarios)
    by_id = {s.id: s for s in scenarios}
    assert by_id[SMOKE].has_mock
    assert by_id["double_session_same_slot"].nicknames() == ["tester_01", "tester_02"]
    assert any(t.restart for t in by_id["restart_resume"].turns)
    for s in scenarios:
        assert s.expect, f"{s.id} asserts nothing about the outcome"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"extra": 1}, "unknown keys"),
        ({"turns": [{"text": "hi"}]}, "needs 'say'"),
        ({"turns": [{"say": "hi", "expect": 1}]}, "unknown keys"),
        ({"turns": [{"say": "hi", "mock": {"foo": 1}}]}, "mock needs"),
        ({"expect": {"orders": 1}}, "unknown expect keys"),
    ],
)
def test_malformed_scenarios(patch: dict[str, Any], message: str) -> None:
    data = {**smoke_data(), **patch}
    with pytest.raises(runner.ScenarioError, match=message):
        runner.parse_scenario(data)


def test_missing_required_key() -> None:
    data = smoke_data()
    del data["now"]
    with pytest.raises(runner.ScenarioError, match="missing 'now'"):
        runner.parse_scenario(data)


# --- CI smoke ------------------------------------------------------------------------------


def test_smoke_scenario_passes_with_mock(domain: Any) -> None:
    result = run_one(smoke_data(), domain)
    assert result.passed, result.failures
    assert [t.reply_type for t in result.turns] == [
        "ask_missing",
        "consult",
        "candidates",
        "confirm_booking",
        "booked",
    ]
    assert "query_knowledge_hub" in result.turns[1].tools
    assert result.turns[0].remote_llm_calls == 1  # the classifier
    (order,) = result.bookings["tester_01"]
    assert order["total"] == "218.40" and order["status"] == "pending_payment"


def test_wrong_expectations_are_reported(domain: Any) -> None:
    data = copy.deepcopy(smoke_data())
    data["turns"][2]["expect_reply_type"] = "no_candidates"
    data["turns"][2]["expect_slots"] = {"rank_requirement": "master"}
    data["turns"][2]["expect_relaxations"] = True
    data["turns"][3]["expect_span"] = {"tool": "create_booking"}
    data["expect"]["booking"]["total"] = "999.00"
    data["expect"]["bookings"] = {"tester_01": 2}
    data["expect"]["state_phase"] = "BOOKING"
    data["expect"]["no_booking"] = True
    result = run_one(data, domain)
    assert not result.passed
    text = "\n".join(result.failures)
    for fragment in (
        "turn 3 (那就两小时吧): reply_type",
        "slot rank_requirement: expected 'master', got 'diamond'",
        "relaxations: expected some",
        "turn 4 (选第一个): span: expected tool:create_booking",
        "booking.total: expected '999.00', got '218.40'",
        "bookings[tester_01]: expected 2, got 1",
        "state_phase: expected BOOKING, got IDLE",
        "no_booking: 1 booking(s) were created",
    ):
        assert fragment in text


def test_two_sessions_conflict_and_restart_with_mock(domain: Any) -> None:
    ask = "帮我约个明晚八点的单双排，挑战者，两小时"
    want = booked(game_mode="ranked_solo_duo", start_time_expr="明晚八点",
                  duration_hours=2, rank_requirement="challenger")  # fmt: skip
    pick = booked(companion_name="星河滚烫")
    yes = {"turn_intent": "booking", "delta": {}, "confirmation": "yes"}
    data = {
        "id": "mock_conflict",
        "now": "2026-10-01 14:00",
        "user": {"nickname": "tester_01"},
        "turns": [
            {"say": ask, "mock": {"intent": "booking", "extraction": want}},
            {
                "say": ask,
                "session": "b",
                "user": "tester_02",
                "mock": {"intent": "booking", "extraction": want},
            },
            {"say": "第一个", "mock": {"extraction": pick}},
            {"say": "第一个", "session": "b", "mock": {"extraction": pick}},
            {
                "say": "确认",
                "restart": True,
                "expect_reply_type": "booked",
                "mock": {"extraction": yes},
            },
            {
                "say": "确认",
                "session": "b",
                "expect_reply_type": ["candidates", "no_candidates"],
                "mock": {"extraction": yes},
            },
        ],
        "expect": {"bookings": {"tester_01": 1, "tester_02": 0}},
    }
    result = run_one(data, domain)
    assert result.passed, result.failures
    assert "create_booking" in result.turns[-1].tools


def test_setup_bookings_and_refund_ratio(domain: Any) -> None:
    data = {
        "id": "mock_cancel",
        "now": "2026-10-01 14:00",
        "setup": {
            "bookings": [
                {
                    "game_mode": "aram",
                    "start_time": "2026-10-01 20:00",
                    "duration_hours": 1,
                    "paid": True,
                }
            ]
        },
        "turns": [
            {
                "say": "帮我取消订单",
                "expect_reply_type": "cancel_confirm",
                "mock": {"intent": "manage"},
            },
            {"say": "确认", "expect_reply_type": "cancelled", "mock": {"intent": "manage"}},
        ],
        "expect": {
            "setup_bookings": [{"status": "cancelled", "refund_ratio": 0.5}],
            "no_booking": True,
        },
    }
    result = run_one(data, domain)
    assert result.passed, result.failures

    data["expect"]["setup_bookings"] = [{"status": "cancelled", "refund_ratio": 1.0}, {}]
    failures = run_one(data, domain).failures
    assert "setup_bookings[0].refund_ratio: expected 1.0, got 0.50" in failures
    assert "setup_bookings[1]: no such setup booking" in failures


def test_unbookable_setup_is_an_error(domain: Any) -> None:
    data = {
        "id": "mock_bad_setup",
        "now": "2026-10-01 14:00",
        "setup": {
            "bookings": [
                {"game_mode": "aram", "start_time": "2026-10-01 03:00", "duration_hours": 1}
            ]
        },
        "turns": [{"say": "看看我的订单", "mock": {"intent": "manage"}}],
        "expect": {"no_booking": True},
    }
    result = run_one(data, domain)
    assert not result.passed and result.error is not None


# --- runner helpers ------------------------------------------------------------------------


def test_compare_helpers() -> None:
    assert runner._same("2026-10-02 20:00", "2026-10-02T20:00:00")
    assert not runner._same("2026-10-02 20:00", "2026-10-02T21:00:00")
    assert runner._same(2, "2.0") and runner._same(218.4, "218.40")
    assert runner._same(["jungle"], ("jungle",)) and runner._same(None, None)
    booking = {"total": "100.00", "refund_amount": "50.00", "status": "cancelled", "hours": "1.0"}
    assert runner.check_booking({"refund_ratio": 0.5, "duration_hours": 1}, booking, "b") == []
    assert runner.check_booking({"status": "confirmed"}, booking, "b") == [
        "b.status: expected 'confirmed', got 'cancelled'"
    ]


def test_knowledge_stub_falls_back_without_kb(tmp_path: Path) -> None:
    sections = runner.load_kb_sections(tmp_path / "missing")
    assert sections["platform_rules"] and sections["modes_and_ranks"]


def test_summary_and_markdown() -> None:
    record = runner.TurnRecord(
        "main", "hi", "ok", "consult", "IDLE", 120.0, 2, ["query_knowledge_hub"]
    )
    slow = runner.TurnRecord("main", "确认", "x", "booked", "IDLE", 900.0, 1, ["create_booking"])
    results = [
        runner.ScenarioResult("a", True, [], [record, slow]),
        runner.ScenarioResult("b", False, ["booking: expected a booking"], [record]),
    ]
    summary = runner.summarize(results)
    assert summary["completion_rate"] == 0.5 and summary["avg_turns"] == 1.5
    assert summary["latency_ms"]["p50"] == 120.0 and summary["latency_ms"]["p95"] == 900.0
    assert summary["remote_llm_calls_per_session"] == 2.5 and summary["failed"] == ["b"]
    report = {
        "meta": {
            "config": "llm",
            "model": "deepseek-chat",
            "knowledge": "rag",
            "started_at": "2026-10-01 14:00:00",
        },
        "summary": summary,
        "scenarios": [runner.asdict(r) for r in results],
    }
    md = runner.render_markdown(report)
    assert "**50.0%**（1/2）" in md and "| `b` | ❌ | 1 | 2 | 120 |" in md
    assert "- booking: expected a booking" in md


def test_run_mock_skips_unscripted_scenarios(
    tmp_path: Path, domain: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    (scenarios / "smoke.yaml").write_text(
        (SCENARIOS / f"{SMOKE}.yaml").read_text("utf-8"), encoding="utf-8"
    )
    (scenarios / "real.yaml").write_text(
        (SCENARIOS / "one_shot_booking.yaml").read_text("utf-8"), encoding="utf-8"
    )
    args = argparse.Namespace(
        config="mock",
        knowledge="stub",
        scenarios=scenarios,
        only=None,
        out=tmp_path / "reports" / "e2e_mock",
    )
    report = runner.run(args, runtime=mock_runtime(domain))
    assert "skipping 1 scenario(s)" in capsys.readouterr().out
    assert report["summary"]["scenarios"] == 1 and report["summary"]["passed"] == 1
    assert (tmp_path / "reports" / "e2e_mock.md").read_text("utf-8").startswith("# L4")
