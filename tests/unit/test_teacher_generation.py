"""I3: Teacher requests, the OpenAI backend, answer validation and the resumable generator."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from rift_agent.prompts import load_prompt
from rift_domain.config import DomainConfig
from rift_domain.enums import Confirmation, GameMode, TurnIntent
from rift_training.contract import render_conversation
from rift_training.data.generate import (
    ACCEPTED,
    REJECTED,
    RUN,
    TASKS,
    call_teacher,
    generate,
    stratified,
    summarize_dir,
)
from rift_training.data.prompting import (
    PERSONAS,
    answer_schema,
    build_request,
    persona,
    system_prompt,
    task_prompt,
)
from rift_training.data.specs import GenerationTask, load_spec, read_tasks, sample_tasks
from rift_training.data.validate import assemble, validate_answer
from rift_training.evaluation.dataset import CATEGORIES, load_samples
from rift_training.inference.base import TeacherError, TeacherRequest
from rift_training.inference.factory import load_teacher_config, make_teacher
from rift_training.inference.mock import MockTeacher, fill_schema
from rift_training.inference.teacher_openai import OpenAITeacher, TeacherConfig, describe

SCRIPT = Path(__file__).resolve().parents[2] / "training" / "scripts" / "data" / "generate.py"
NOW = datetime(2026, 10, 1, 14, 0)


def make_task(**kw: Any) -> GenerationTask:
    base: dict[str, Any] = {
        "id": "t-0001",
        "spec_version": "test",
        "category": "multi_turn",
        "variant": "modify_field",
        "hint": "用户修改之前说过的信息",
        "style": "colloquial",
        "style_hint": "口语化",
        "now": NOW,
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "state_fields": ("game_mode", "start_time_expr", "duration_hours"),
        "num_candidates": 0,
        "pending_confirmation": False,
        "history_turns": 1,
        "turn_intent": TurnIntent.BOOKING,
        "confirmation": Confirmation.NONE,
        "delta": {"start_time_expr": "value", "duration_hours": "value", "style_preference": "any"},
    }
    return GenerationTask.model_validate({**base, **kw})


def good_answer(**kw: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "candidates": [],
        "state": {
            "game_mode": "ranked_solo_duo",
            "start_time": "2026-10-02 20:00",
            "duration_hours": 2,
        },
        "history": [
            {"role": "user", "content": "明晚八点双排，打两小时"},
            {"role": "assistant", "content": "好的，想要什么段位的陪玩？"},
        ],
        "user_input": "改成晚一小时吧，打三个小时，风格无所谓",
        "values": {"start_time_expr": "晚一小时", "duration_hours": 3},
    }
    answer.update(kw)
    return answer


def check(task: GenerationTask, answer: Any, domain: DomainConfig) -> list[str]:
    text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    return validate_answer(task, text, domain).problems


# --- request and schema --------------------------------------------------------------------


def _strict(schema: dict[str, Any]) -> None:
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(schema["properties"])
        for sub in schema["properties"].values():
            _strict(sub)
    if schema.get("type") == "array":
        _strict(schema["items"])


@pytest.mark.unit
def test_answer_schema_is_strict_and_sized_by_the_task(domain: DomainConfig) -> None:
    task = make_task(num_candidates=3, history_turns=2)
    schema = answer_schema(task, domain)
    _strict(schema)
    props = schema["properties"]
    assert list(props) == ["candidates", "state", "history", "user_input", "values"]
    assert props["candidates"]["minItems"] == props["candidates"]["maxItems"] == 3
    assert props["history"]["minItems"] == props["history"]["maxItems"] == 4
    assert list(props["state"]["properties"]) == ["game_mode", "start_time", "duration_hours"]
    assert "pattern" in props["state"]["properties"]["start_time"]
    # only "value" keys are asked for; "any" / null come from the task
    assert list(props["values"]["properties"]) == ["start_time_expr", "duration_hours"]


@pytest.mark.unit
def test_schema_pins_the_conversation_mode(domain: DomainConfig) -> None:
    task = make_task(delta={"game_mode": "value"}, state_fields=("game_mode",))
    props = answer_schema(task, domain)["properties"]
    assert props["values"]["properties"]["game_mode"]["enum"] == ["ranked_solo_duo"]
    state_modes = props["state"]["properties"]["game_mode"]["enum"]
    assert "ranked_solo_duo" not in state_modes and len(state_modes) == 5

    unchanged = answer_schema(make_task(), domain)["properties"]["state"]["properties"]
    assert unchanged["game_mode"]["enum"] == ["ranked_solo_duo"]
    rank = answer_schema(make_task(delta={"rank_requirement": "value"}), domain)
    assert len(rank["properties"]["values"]["properties"]["rank_requirement"]["enum"]) == 10


@pytest.mark.unit
def test_prompts_carry_the_extractor_rules_and_the_task(domain: DomainConfig) -> None:
    system = system_prompt(domain)
    assert load_prompt("slot_extract.txt") in system
    assert "{{" not in system and "时长：1–8 小时，0.5 小时为步长" in system
    assert "aram_mayhem=海克斯大乱斗" in system and "jungle=打野" in system
    user = task_prompt(make_task(), domain)
    assert "2026-10-01 14:00 星期四" in user
    assert "start_time_expr：给出具体值" in user
    assert "style_preference：明确表示不限" in user
    assert persona(make_task()) in user
    assert persona(make_task()) == persona(make_task())
    assert len({persona(make_task(id=f"t-{i}")) for i in range(60)}) == len(PERSONAS)
    empty = task_prompt(make_task(delta={}, turn_intent=TurnIntent.CONSULT), domain)
    assert "delta 为空" in empty


@pytest.mark.unit
def test_retry_request_shows_the_previous_answer_and_problems(domain: DomainConfig) -> None:
    from rift_training.data.prompting import retry_request

    request = retry_request(build_request(make_task(), domain), '{"x": 1}', ["a is wrong"])
    messages = request.messages()
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == '{"x": 1}'
    assert "a is wrong" in messages[3]["content"]


# --- validation ----------------------------------------------------------------------------


@pytest.mark.unit
def test_good_answer_becomes_an_l2_sample_with_task_tri_state(domain: DomainConfig) -> None:
    task = make_task()
    outcome = validate_answer(task, json.dumps(good_answer(), ensure_ascii=False), domain)
    assert outcome.ok, outcome.problems
    assert outcome.sample is not None
    assert outcome.sample.expected["delta"] == {
        "start_time_expr": "晚一小时",
        "duration_hours": 3,
        "style_preference": "any",
    }
    assert outcome.sample.state().start_time == datetime(2026, 10, 2, 20, 0)
    withdrawn = assemble(make_task(delta={"duration_hours": "null"}), good_answer(values={}))
    assert withdrawn["expected"]["delta"] == {"duration_hours": None}


BAD_ANSWERS = [
    ("not json", "invalid JSON"),
    ([1, 2], "not a JSON object"),
    ({"user_input": "x"}, "answer keys"),
    (good_answer(values={"start_time_expr": "晚一小时"}), "values keys"),
    (good_answer(history=[]), "history must have 2"),
    (good_answer(candidates=["阿狸酱"]), "candidates must have 0"),
    (good_answer(user_input="  "), "user_input is empty"),
    (
        good_answer(
            history=[{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
        ),
        "history[0] should be a user message",
    ),
    (good_answer(values={"start_time_expr": "晚两小时", "duration_hours": 3}), "not copied"),
    (
        good_answer(
            user_input="下辈子再说，打三个小时",
            values={"start_time_expr": "下辈子", "duration_hours": 3},
        ),
        "does not parse",
    ),
    (
        good_answer(values={"start_time_expr": "晚一小时", "duration_hours": 9}),
        "outside the allowed",
    ),
    (good_answer(values={"start_time_expr": "晚一小时", "duration_hours": 2.2}), "multiple of 0.5"),
    (good_answer(values={"start_time_expr": "晚一小时", "duration_hours": 2}), "current value"),
    (
        good_answer(
            state={"game_mode": "aram", "start_time": "2026-10-02 20:00", "duration_hours": 2}
        ),
        "not the task's ranked_solo_duo",
    ),
    (
        good_answer(
            state={
                "game_mode": "ranked_solo_duo",
                "start_time": "2026-09-30 20:00",
                "duration_hours": 2,
            }
        ),
        "not after now",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("answer", "message"), BAD_ANSWERS)
def test_bad_answers_are_rejected(domain: DomainConfig, answer: Any, message: str) -> None:
    problems = check(make_task(), answer, domain)
    assert any(message in p for p in problems), problems


@pytest.mark.unit
def test_names_must_come_from_the_candidates(domain: DomainConfig) -> None:
    task = make_task(
        category="candidate_ref",
        variant="ordinal",
        num_candidates=2,
        state_fields=("game_mode",),
        delta={"companion_name": "value"},
    )
    answer = good_answer(
        candidates=["阿狸酱", "夜雨声烦"],
        state={"game_mode": "ranked_solo_duo"},
        user_input="就第二个吧",
        values={"companion_name": "夜雨声烦"},
    )
    assert check(task, answer, domain) == []
    answer["values"] = {"companion_name": "第二个"}
    assert any("not a candidate" in p for p in check(task, answer, domain))
    answer["candidates"] = ["阿狸酱", "阿狸酱"]
    assert any("unique" in p for p in check(task, answer, domain))


@pytest.mark.unit
def test_vague_times_are_kept(domain: DomainConfig) -> None:
    task = make_task(state_fields=("game_mode",), delta={"start_time_expr": "value"})
    answer = good_answer(
        state={"game_mode": "ranked_solo_duo"},
        user_input="晚上吧",
        values={"start_time_expr": "晚上"},
    )
    assert check(task, answer, domain) == []


# --- OpenAI backend ------------------------------------------------------------------------

OPENAI = TeacherConfig(
    provider="openai",
    model="gpt-test",
    reasoning_effort="low",
    max_completion_tokens=500,
    api_key_env="TEST_TEACHER_KEY",
)


def _completion(content: str = "{}", refusal: str | None = None, **choice: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"content": content}
    if refusal is not None:
        message["refusal"] = refusal
    return {
        "model": "gpt-test-2026",
        "choices": [{"message": message, "finish_reason": "stop", **choice}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _teacher(handler: Any, monkeypatch: pytest.MonkeyPatch) -> OpenAITeacher:
    monkeypatch.setenv("TEST_TEACHER_KEY", "sk-test")
    return OpenAITeacher(OPENAI, transport=httpx.MockTransport(handler))


REQUEST = TeacherRequest(system="s", user="u", schema={"type": "object"}, schema_name="x")


@pytest.mark.unit
async def test_openai_payload_uses_strict_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion('{"a": 1}'))

    teacher = _teacher(handler, monkeypatch)
    response = await teacher.generate(REQUEST)
    await teacher.aclose()
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer sk-test"
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "x", "strict": True, "schema": {"type": "object"}},
    }
    assert body["max_completion_tokens"] == 500 and body["reasoning_effort"] == "low"
    assert "temperature" not in body and "max_tokens" not in body
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert response.text == '{"a": 1}' and response.model == "gpt-test-2026"
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert describe(OPENAI) == {
        "provider": "openai",
        "model": "gpt-test",
        "reasoning_effort": "low",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "retryable", "message"),
    [
        (httpx.Response(429, text="slow down"), True, "HTTP 429"),
        (httpx.Response(503, text="busy"), True, "HTTP 503"),
        (httpx.Response(400, text="bad schema"), False, "HTTP 400"),
        (httpx.Response(200, json=_completion(refusal="no")), False, "refused"),
        (httpx.Response(200, json=_completion(finish_reason="length")), False, "truncated"),
        (httpx.Response(200, text="<html>"), True, "malformed"),
    ],
)
async def test_openai_errors(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response, retryable: bool, message: str
) -> None:
    teacher = _teacher(lambda _: response, monkeypatch)
    with pytest.raises(TeacherError, match=message) as info:
        await teacher.generate(REQUEST)
    assert info.value.retryable is retryable


@pytest.mark.unit
def test_openai_key_only_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_TEACHER_KEY", raising=False)
    with pytest.raises(TeacherError, match="TEST_TEACHER_KEY is not set"):
        OpenAITeacher(OPENAI)


@pytest.mark.unit
def test_teacher_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEACHER_MODEL", raising=False)
    config = load_teacher_config("openai")
    assert config.provider == "openai" and config.api_key_env == "OPENAI_API_KEY"
    monkeypatch.setenv("TEACHER_MODEL", "from-env")
    assert load_teacher_config("openai").model == "from-env"
    assert load_teacher_config("openai", model="from-cli").model == "from-cli"
    assert isinstance(make_teacher(load_teacher_config("mock")), MockTeacher)


@pytest.mark.unit
async def test_call_teacher_backs_off_then_gives_up() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky(_: TeacherRequest) -> dict[str, Any]:
        calls.append(1)
        raise TeacherError("busy", retryable=True)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.raises(TeacherError):
        await call_teacher(MockTeacher(flaky), REQUEST, max_retries=3, sleep=sleep)
    assert len(calls) == 4 and sleeps == [2.0, 4.0, 8.0]

    def fatal(_: TeacherRequest) -> dict[str, Any]:
        raise TeacherError("bad", retryable=False)

    sleeps.clear()
    with pytest.raises(TeacherError):
        await call_teacher(MockTeacher(fatal), REQUEST, max_retries=3, sleep=sleep)
    assert sleeps == []


# --- generator -----------------------------------------------------------------------------


async def _no_sleep(_: float) -> None:
    return None


@pytest.mark.unit
async def test_generate_retries_with_feedback_and_writes_files(
    domain: DomainConfig, tmp_path: Path
) -> None:
    task = make_task()
    answers = iter([good_answer(user_input="改成晚两小时"), good_answer()])
    teacher = MockTeacher(lambda _: next(answers))
    stats = await generate([task], teacher, domain, tmp_path, sleep=_no_sleep)
    assert stats.accepted == 1 and stats.accepted_first_try == 0 and stats.teacher_calls == 2
    retry = teacher.requests[1].messages()
    assert retry[-2]["role"] == "assistant" and "没有通过校验" in retry[-1]["content"]

    samples = load_samples(tmp_path / ACCEPTED)
    assert [s.id for s in samples] == ["t-0001"]
    assert render_conversation(samples[0])[-1]["role"] == "assistant"
    assert read_tasks(tmp_path / TASKS) == [task]
    manifest = json.loads((tmp_path / RUN).read_text("utf-8"))
    assert manifest["status"]["pass_rate"] == 1.0
    assert manifest["runs"][0]["teacher"] == {"provider": "mock", "model": "mock-teacher"}


@pytest.mark.unit
async def test_generate_records_rejections_and_resumes(
    domain: DomainConfig, tmp_path: Path
) -> None:
    good, bad, broken = make_task(id="t-good"), make_task(id="t-bad"), make_task(id="t-api")

    def responder(request: TeacherRequest) -> dict[str, Any] | str:
        if "t-api" in request.user:
            raise TeacherError("HTTP 400", retryable=False)
        return good_answer() if "t-good" in request.user else "nope"

    teacher = MockTeacher(responder)
    stats = await generate([good, bad, broken], teacher, domain, tmp_path, sleep=_no_sleep)
    assert (stats.accepted, stats.rejected_validation, stats.rejected_api) == (1, 1, 1)
    rejected = [json.loads(line) for line in (tmp_path / REJECTED).read_text("utf-8").splitlines()]
    assert {(r["task_id"], r["kind"]) for r in rejected} == {
        ("t-bad", "validation"),
        ("t-api", "api"),
    }
    assert next(r for r in rejected if r["task_id"] == "t-bad")["raw"] == "nope"
    assert summarize_dir(tmp_path)["pass_rate"] == 0.5

    calls = len(teacher.requests)
    again = await generate([good, bad, broken], teacher, domain, tmp_path, sleep=_no_sleep)
    assert again.skipped == 3 and len(teacher.requests) == calls

    fixed = MockTeacher(lambda _: good_answer())
    final = await generate(
        [good, bad, broken], fixed, domain, tmp_path, retry_rejected=True, sleep=_no_sleep
    )
    assert final.skipped == 1 and final.accepted == 2
    status = summarize_dir(tmp_path)
    assert status["accepted"] == 3 and status["rejected_validation"] == 0
    assert len(json.loads((tmp_path / RUN).read_text("utf-8"))["runs"]) == 3


@pytest.mark.unit
async def test_mock_teacher_runs_the_whole_spec(domain: DomainConfig, tmp_path: Path) -> None:
    tasks = sample_tasks(load_spec("v0.1"), domain=domain)
    stats = await generate(tasks, MockTeacher(fill_schema), domain, tmp_path, sleep=_no_sleep)
    assert stats.accepted + stats.rejected_validation == 800
    assert stats.accepted >= 700  # canned values only fail where a slot must change
    assert len(load_samples(tmp_path / ACCEPTED)) == stats.accepted


@pytest.mark.unit
def test_stratified_trial_covers_every_category(domain: DomainConfig) -> None:
    tasks = sample_tasks(load_spec("v0.1"), domain=domain)
    trial = stratified(tasks, 20)
    assert len(trial) == 20
    assert {t.category for t in trial} == set(CATEGORIES)
    assert stratified(tasks, 20) == trial
    assert len(stratified(tasks, 5000)) == 800


@pytest.mark.unit
def test_script_dry_run_with_mock(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module_spec = importlib.util.spec_from_file_location("generate_script", SCRIPT)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    out = tmp_path / "run"
    assert module.main(["--teacher", "mock", "--limit", "20", "--out", str(out)]) == 0
    assert (out / ACCEPTED).exists()
    assert '"pass_rate"' in capsys.readouterr().out
