"""LLM-powered enrichment helpers for parsed document chunks."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field, ValidationError
from unstructured.documents.elements import Element


class ChunkMetadata(BaseModel):
    """Structured metadata captured for each enriched chunk."""

    summary: str = Field(description="A concise 1-2 sentence summary of the chunk.")
    keywords: list[str] = Field(
        default_factory=list,
        description="Key topics or entities mentioned in the chunk.",
    )
    hypothetical_questions: list[str] = Field(
        default_factory=list,
        description="Questions the chunk could help answer.",
    )
    table_summary: str | None = Field(
        default=None,
        description="If the chunk is a table, a natural language summary of its insights.",
    )


def generate_enrichment_prompt(chunk_text: str, *, is_table: bool) -> str:
    """Create a deterministic prompt for the enrichment LLM."""

    table_instruction = """
    This chunk is a TABLE. Describe the main data points and trends in natural language.
    """ if is_table else ""

    return f"""
You are an expert financial analyst. Analyse the following document chunk and produce structured metadata.
{table_instruction}
Chunk Content:
---
{chunk_text}
---
"""


def _get_chunk_content(chunk: Element) -> tuple[str, bool]:
    metadata = chunk.metadata.to_dict()
    if "text_as_html" in metadata:
        return metadata["text_as_html"], True
    return chunk.text, False


def enrich_chunk(
    chunk: Element,
    *,
    llm: BaseLanguageModel,
    max_characters: int = 3000,
) -> Dict[str, Any] | None:
    """Enrich a chunk with metadata using the supplied language model.

    Args:
        chunk: Unstructured document chunk.
        llm: Language model with ``invoke`` returning a ``ChunkMetadata`` instance
            or an object exposing ``dict``.
        max_characters: Maximum number of characters sent to the model to avoid
            overly long prompts.
    """

    content, is_table = _get_chunk_content(chunk)
    truncated_content = content[:max_characters]
    prompt = generate_enrichment_prompt(truncated_content, is_table=is_table)

    try:
        metadata = llm.invoke(prompt)
    except Exception:  # pragma: no cover - network/LLM specific
        # Re-raise with additional context for easier debugging while still
        # signalling failure to the caller.
        raise RuntimeError("Failed to enrich chunk via language model")

    try:
        if isinstance(metadata, ChunkMetadata):
            return metadata.model_dump()
        if hasattr(metadata, "dict"):
            data = metadata.dict()
        else:
            data = json.loads(metadata)
        return ChunkMetadata.model_validate(data).model_dump()
    except (ValidationError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Language model returned invalid chunk metadata") from exc


__all__ = ["ChunkMetadata", "enrich_chunk", "generate_enrichment_prompt"]
