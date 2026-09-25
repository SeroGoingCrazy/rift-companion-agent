"""Slot extractors: LLM (remote), local (llama-server, stage J) and the router."""

from rift_agent.extractors.base import ExtractionContext, ExtractionResult, SlotExtractor
from rift_agent.extractors.llm import LLMSlotExtractor
from rift_agent.extractors.routed import RoutedSlotExtractor, compare_extractions

__all__ = [
    "ExtractionContext",
    "ExtractionResult",
    "LLMSlotExtractor",
    "RoutedSlotExtractor",
    "SlotExtractor",
    "compare_extractions",
]
