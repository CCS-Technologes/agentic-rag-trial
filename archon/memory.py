"""Lightweight in-memory store for cognitive insights."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MemoryStore:
    """Dictionary-backed memory with simple keyword retrieval."""

    store: Dict[str, str] = field(default_factory=dict)

    def save(self, key: str, insight: str) -> str:
        self.store[key] = insight
        return f"Insight '{key}' saved."

    def recall(self, query: str) -> List[str]:
        tokens = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        results = []
        for key, value in self.store.items():
            key_tokens = set(key.lower().split("_"))
            if tokens & key_tokens:
                results.append(value)
        return results


# Convenience module-level instance mirroring the notebook API -----------------
_MEMORY = MemoryStore()


def save_to_memory(key: str, insight: str) -> str:
    return _MEMORY.save(key, insight)


def recall_from_memory(query: str) -> List[str]:
    return _MEMORY.recall(query)


__all__ = ["MemoryStore", "save_to_memory", "recall_from_memory"]
