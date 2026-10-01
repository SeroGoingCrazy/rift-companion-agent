"""Teacher requests: the instructions and the per-task answer schema (DEV_SPEC I3).

Structured Outputs in strict mode cannot say "this key may be absent", so the Teacher never
writes the delta itself. The task (I2) already fixes which keys appear and whether each is a
value, ``"any"`` or ``null``; the answer schema asks only for:

- ``candidates`` / ``state`` / ``history``: the conversation so far (exact sizes and keys);
- ``user_input``: this turn's message;
- ``values``: the concrete value of every delta key the task marks as "value".

``validate.assemble`` then builds the expected extraction in code, so tri-state labels are
right by construction and the validator only has to check business consistency.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from rift_agent.prompts import load_prompt
from rift_domain.config import DomainConfig
from rift_training.contract import PROMPT_FILE
from rift_training.data.specs import GenerationTask
from rift_training.inference.base import TeacherRequest

TEACHER_SYSTEM = (
    Path(__file__).resolve().parents[3] / "configs" / "inference" / "teacher_system.txt"
)
TIME_PATTERN = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
WEEKDAYS = "一二三四五六日"
KIND_TEXT = {
    "value": "给出具体值（填进 values）",
    "any": '明确表示不限 / 无所谓 / 都行（标签为 "any"，不填 values）',
    "null": "撤回之前说的、暂时不定（标签为 null，不填 values）",
}
#: Who is typing; picked per task from its id so reruns stay identical.
PERSONAS = (
    "大学生，晚上宿舍开黑",
    "上班族，下班后想放松",
    "刚入坑的新手，很多术语不熟",
    "打了好几年的老玩家，说话很简洁",
    "想冲分的段位党，比较在意效率",
    "女生玩家，想找人一起玩得开心",
    "话比较多、爱开玩笑的玩家",
    "性子急、句子很短的玩家",
    "客气礼貌、喜欢用敬语的玩家",
    "第一次用陪玩平台，有点犹豫",
)


def _enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": values}


def field_schemas(domain: DomainConfig) -> dict[str, dict[str, Any]]:
    """JSON Schema of each slot value, enums taken from ``domain.yaml``."""
    roles = [r.value for r in domain.roles]
    return {
        "game_mode": _enum([m.value for m in domain.game_modes]),
        "start_time_expr": {"type": "string"},
        "duration_hours": {"type": "number"},
        "rank_requirement": _enum([r.key.value for r in domain.ranks]),
        "role_preference": {"type": "array", "items": _enum(roles), "minItems": 1, "maxItems": 5},
        "service_type": _enum([s.value for s in domain.service_types]),
        "companion_gender": _enum([g.value for g in domain.genders]),
        "voice_required": {"type": "boolean"},
        "budget_per_hour": {"type": "number"},
        "style_preference": {"type": "string"},
        "companion_name": {"type": "string"},
    }


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def state_key(name: str) -> str:
    """The state stores the resolved time, the delta the raw expression."""
    return "start_time" if name == "start_time_expr" else name


def answer_schema(task: GenerationTask, domain: DomainConfig) -> dict[str, Any]:
    fields = field_schemas(domain)
    state_mode = fields["game_mode"]
    if task.game_mode is not None:
        # The conversation's mode is fixed by the task; a modified mode starts from another one.
        mode = task.game_mode.value
        modified = "game_mode" in task.state_fields and task.delta.get("game_mode") == "value"
        others = [m for m in fields["game_mode"]["enum"] if m != mode]
        fields["game_mode"] = _enum([mode])
        state_mode = _enum(others) if modified else _enum([mode])
    state = {state_key(name): fields[name] for name in task.state_fields}
    if "start_time" in state:
        state["start_time"] = {"type": "string", "pattern": TIME_PATTERN}
    if "game_mode" in state:
        state["game_mode"] = state_mode
    values = {name: fields[name] for name, kind in task.delta.items() if kind == "value"}
    n_history = 2 * task.history_turns
    message = _object({"role": _enum(["user", "assistant"]), "content": {"type": "string"}})
    return _object(
        {
            "candidates": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": task.num_candidates,
                "maxItems": task.num_candidates,
            },
            "state": _object(state),
            "history": {
                "type": "array",
                "items": message,
                "minItems": n_history,
                "maxItems": n_history,
            },
            "user_input": {"type": "string"},
            "values": _object(values),
        }
    )


def _aliases(domain: DomainConfig) -> str:
    def line(title: str, entries: list[tuple[str, str, tuple[str, ...]]]) -> str:
        items = "；".join(
            f"{key}={label}（{'/'.join(aliases)}）" for key, label, aliases in entries
        )
        return f"- {title}：{items}"

    return "\n".join(
        [
            line("模式", [(k.value, v.label, v.aliases) for k, v in domain.game_modes.items()]),
            line("段位（低→高）", [(r.key.value, r.label, r.aliases) for r in domain.ranks]),
            line("位置", [(k.value, v.label, v.aliases) for k, v in domain.roles.items()]),
            line(
                "服务类型", [(k.value, v.label, v.aliases) for k, v in domain.service_types.items()]
            ),
            line("性别", [(k.value, v.label, v.aliases) for k, v in domain.genders.items()]),
        ]
    )


def system_prompt(domain: DomainConfig) -> str:
    """``teacher_system.txt`` with the domain values and the online extractor rules filled in."""
    hours = domain.duration
    slots = {
        "aliases": _aliases(domain),
        "min_hours": str(hours.min_hours),
        "max_hours": str(hours.max_hours),
        "step_hours": str(hours.step_hours),
        "extractor_rules": load_prompt(PROMPT_FILE),
    }
    text = TEACHER_SYSTEM.read_text("utf-8").strip()
    for key, value in slots.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _when(task: GenerationTask) -> str:
    now = task.now
    return f"{now:%Y-%m-%d %H:%M} 星期{WEEKDAYS[now.weekday()]}"


def persona(task: GenerationTask) -> str:
    digest = hashlib.sha256(task.id.encode()).digest()
    return PERSONAS[digest[0] % len(PERSONAS)]


def task_prompt(task: GenerationTask, domain: DomainConfig) -> str:
    lines = [
        f"任务 {task.id}（场景：{task.category} / {task.variant}）",
        f"- 场景说明：{task.hint}",
        f"- 当前时间：{_when(task)}",
        f"- 说法风格：{task.style}——{task.style_hint}",
        f"- 用户人设：{persona(task)}",
    ]
    if task.game_mode is not None:
        label = domain.game_modes[task.game_mode].label
        lines.append(f"- 本次预约的模式：{task.game_mode.value}（{label}）")
    if task.state_fields:
        keys = "、".join(state_key(n) for n in task.state_fields)
        lines.append(f"- 之前已收集的信息（state）：{keys}")
    else:
        lines.append("- 之前没有收集到任何预约信息（state 为空）")
    lines.append(f"- 之前的对话：{task.history_turns} 轮（history 共 {2 * task.history_turns} 条）")
    if task.num_candidates:
        lines.append(f"- 助手已展示 {task.num_candidates} 位候选陪玩师")
    if task.pending_confirmation:
        lines.append("- 助手刚刚请用户确认下单（pending_confirmation = true）")
    lines.append(
        f"- 这一轮的 turn_intent：{task.turn_intent.value}；confirmation：{task.confirmation.value}"
    )
    if task.delta:
        lines.append("- 用户这一轮涉及的字段（其他字段都不要提）：")
        lines += [f"  - {name}：{KIND_TEXT[kind]}" for name, kind in task.delta.items()]
    else:
        lines.append("- 用户这一轮不涉及任何预约字段（delta 为空），values 为空对象")
    return "\n".join(lines)


def build_request(task: GenerationTask, domain: DomainConfig) -> TeacherRequest:
    return TeacherRequest(
        system=system_prompt(domain),
        user=task_prompt(task, domain),
        schema=answer_schema(task, domain),
        schema_name="slot_sample",
    )


def retry_request(request: TeacherRequest, answer: str, problems: list[str]) -> TeacherRequest:
    """Ask again, showing the previous answer and what the validator rejected."""
    feedback = "上面的答案没有通过校验：\n" + "\n".join(f"- {p}" for p in problems[:8])
    return TeacherRequest(
        system=request.system,
        user=request.user,
        schema=request.schema,
        schema_name=request.schema_name,
        followups=(
            *request.followups,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": feedback + "\n请修正后重新输出完整的 JSON。"},
        ),
    )
