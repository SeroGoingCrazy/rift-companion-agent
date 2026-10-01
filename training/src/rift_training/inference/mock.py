"""Offline Teacher: answers any request by filling its JSON Schema with canned values.

It knows nothing about the task beyond the schema, so it exercises the whole generation
pipeline (schema -> answer -> validation -> files) without a network or a key. Tests can pass
their own ``responder`` to script specific answers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rift_training.inference.base import Teacher, TeacherRequest, TeacherResponse

Responder = Callable[[TeacherRequest], dict[str, Any] | str]

NAMES = ("阿狸酱", "夜雨声烦", "小鹿乱撞", "北极星")
STRINGS = {
    "start_time_expr": "明晚八点",
    "style_preference": "温柔会聊天",
}


def _value(name: str, schema: dict[str, Any], candidates: list[str]) -> Any:
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "array":
        return [_value(name, schema["items"], candidates)]
    if kind == "boolean":
        return True
    if kind == "number":
        return {"budget_per_hour": 80}.get(name, 2)
    if name == "start_time":
        return "2099-01-01 20:00"
    if name == "companion_name":
        return candidates[0] if candidates else NAMES[0]
    return STRINGS.get(name, "随便")


def fill_schema(request: TeacherRequest) -> dict[str, Any]:
    """A schema-valid answer whose user input mentions every value it reports."""
    props = request.schema["properties"]
    n_history = props["history"]["minItems"]
    n_candidates = props["candidates"]["minItems"]
    candidates = list(NAMES[:n_candidates])
    state = {
        name: _value(name, spec, candidates) for name, spec in props["state"]["properties"].items()
    }
    values = {
        name: _value(name, spec, candidates) for name, spec in props["values"]["properties"].items()
    }
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"第{i + 1}句"}
        for i in range(n_history)
    ]
    mentioned = [str(v) for v in values.values() if isinstance(v, str)]
    return {
        "candidates": candidates,
        "state": state,
        "history": history,
        "user_input": "，".join(["好的", *mentioned]),
        "values": values,
    }


class MockTeacher(Teacher):
    name = "mock"

    def __init__(self, responder: Responder = fill_schema, *, model: str = "mock-teacher") -> None:
        self.responder = responder
        self._model = model
        self.requests: list[TeacherRequest] = []

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, request: TeacherRequest) -> TeacherResponse:
        self.requests.append(request)
        answer = self.responder(request)
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return TeacherResponse(text=text, model=self._model, usage={"total_tokens": 0})
