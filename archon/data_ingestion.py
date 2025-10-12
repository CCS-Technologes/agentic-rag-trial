"""Utilities for loading SEC HTML filings and chunking them for downstream tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element, element_from_dict
from unstructured.partition.html import partition_html

LOGGER = logging.getLogger(__name__)


def parse_html_file(file_path: str | Path) -> List[Dict]:
    """Parse an SEC filing saved as HTML into Unstructured element dictionaries.

    The original notebook performed the parsing inline and ignored the specific
    exception details.  This helper centralises error handling so callers can
    decide what to do with failures and instrumentation can report more
    detailed diagnostics.

    Args:
        file_path: Path to the ``full-submission.txt`` file produced by
            ``sec-edgar-downloader``.

    Returns:
        A list of dictionaries representing Unstructured elements.  The
        function returns an empty list when parsing fails so downstream code can
        simply skip the file.
    """

    path = Path(file_path)
    if not path.exists():
        LOGGER.warning("Skipping missing HTML file: %s", path)
        return []

    try:
        elements = partition_html(
            filename=str(path),
            infer_table_structure=True,
            strategy="fast",
        )
    except Exception:  # pragma: no cover - defensive: library exceptions vary
        LOGGER.exception("Failed to parse SEC filing: %s", path)
        return []

    return [element.to_dict() for element in elements]


def _ensure_elements(elements: Sequence[Element | Dict]) -> List[Element]:
    """Normalise a sequence of Element objects or dictionaries."""

    normalised: List[Element] = []
    for element in elements:
        if isinstance(element, Element):
            normalised.append(element)
        else:
            normalised.append(element_from_dict(element))
    return normalised


def chunk_document(
    elements: Sequence[Element | Dict],
    *,
    max_characters: int = 2048,
    combine_text_under_n_chars: int = 256,
    new_after_n_chars: int = 1800,
) -> List[Element]:
    """Chunk a parsed document using the ``chunk_by_title`` heuristic.

    The defaults mirror the notebook but can now be overridden for different
    document types.  Accepting either ``Element`` instances or dictionaries
    makes the function convenient for callers that cache parsed output on disk.
    """

    if not elements:
        return []

    normalised = _ensure_elements(elements)
    return chunk_by_title(
        normalised,
        max_characters=max_characters,
        combine_text_under_n_chars=combine_text_under_n_chars,
        new_after_n_chars=new_after_n_chars,
    )


__all__ = ["parse_html_file", "chunk_document"]
