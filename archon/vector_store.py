"""Helpers for preparing enriched chunks for vector stores."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - optional dependency during static analysis
    from qdrant_client.http.models import PointStruct
except Exception:  # pragma: no cover - fallback when qdrant isn't installed
    PointStruct = None  # type: ignore[misc]


def create_embedding_text(
    chunk: Mapping[str, object],
    *,
    content_limit: int = 1000,
) -> str:
    """Create a deterministic text representation for embedding.

    The previous implementation embedded arbitrary string interpolation that was
    hard to reuse.  This helper normalises whitespace and allows callers to
    control how much of the original content is embedded.
    """

    summary = str(chunk.get("summary", "")).strip()
    keywords = ", ".join(str(keyword).strip() for keyword in chunk.get("keywords", []) or [])
    content = str(chunk.get("content", ""))[:content_limit]

    sections = []
    if summary:
        sections.append(f"Summary: {summary}")
    if keywords:
        sections.append(f"Keywords: {keywords}")
    if content:
        sections.append(f"Content: {content}")

    return "\n".join(sections)


def prepare_qdrant_points(
    chunks: Sequence[Mapping[str, object]],
    embeddings: Iterable[Sequence[float]],
) -> list[PointStruct]:
    """Combine payloads and vectors into ``PointStruct`` instances.

    This keeps the original metadata intact while guaranteeing a 1:1 mapping
    between embeddings and payloads.  The helper raises a ``ValueError`` if the
    lengths do not match, catching subtle pipeline bugs early.
    """

    if PointStruct is None:  # pragma: no cover - helpful error when optional dependency missing
        raise RuntimeError("qdrant-client must be installed to build PointStruct payloads")

    chunk_list = list(chunks)
    embedding_list = list(embeddings)

    if len(chunk_list) != len(embedding_list):
        raise ValueError("Number of embeddings does not match number of chunks")

    points: list[PointStruct] = []
    for idx, (chunk, embedding) in enumerate(zip(chunk_list, embedding_list)):
        points.append(
            PointStruct(
                id=idx,
                payload=dict(chunk),
                vector=list(embedding),
            )
        )

    return points


__all__ = ["create_embedding_text", "prepare_qdrant_points"]
