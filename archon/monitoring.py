"""Proactive monitoring helpers built on top of the memory store."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel

from .memory import recall_from_memory


def run_daily_monitor(*, significance_llm: BaseLanguageModel, event_description: str) -> dict[str, Any]:
    """Simulate a proactive monitoring cycle and return the structured result."""

    user_interests = recall_from_memory("AI risk")
    prompt = (
        "You are a significance auditor. Decide if the event warrants notifying the user.\n"
        f"User Interests: {user_interests}\nEvent: {event_description}"
    )
    result = significance_llm.invoke(prompt)
    payload = result.dict() if hasattr(result, "dict") else result
    payload.setdefault("user_interests", user_interests)
    payload.setdefault("event", event_description)
    return payload


__all__ = ["run_daily_monitor"]
