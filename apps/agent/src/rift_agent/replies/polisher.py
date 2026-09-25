"""Optional LLM polish of template replies (``reply.polish: true``).

The template text stays the source of truth: the polished version is used only if the
LLM answered and every number of the template (prices, times, order ids) survived.
Anything else falls back to the template, so polishing can never change a fact.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from rift_agent.prompts import load_prompt
from rift_common.llm import BaseLLM, LLMError, Message

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    return _NUMBER.findall(text)


def keeps_facts(template: str, polished: str) -> bool:
    """Every number of the template appears in the polished text (as often as before)."""
    have = numbers(polished)
    return all(have.count(n) >= numbers(template).count(n) for n in set(numbers(template)))


class ReplyPolisher:
    def __init__(self, llm: BaseLLM, *, max_tokens: int = 400) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    async def polish(self, template_reply: str, facts: dict[str, Any]) -> tuple[str, bool]:
        """(text, polished?) — the template when polishing failed or changed a fact."""
        messages: list[Message] = [
            {"role": "system", "content": load_prompt("reply_polish.txt")},
            {
                "role": "user",
                "content": json.dumps(
                    {"template_reply": template_reply, "facts": facts},
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]
        try:
            result = await asyncio.to_thread(
                self.llm.chat, messages, temperature=0.3, max_tokens=self.max_tokens
            )
        except LLMError:
            logger.warning("reply polish failed; using the template", exc_info=True)
            return template_reply, False
        text = result.text.strip()
        if not text or not keeps_facts(template_reply, text):
            logger.info("polished reply dropped facts; using the template")
            return template_reply, False
        return text, True
