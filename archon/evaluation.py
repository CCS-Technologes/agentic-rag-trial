"""Evaluation helpers for retrieval quality and LLM-as-a-judge scoring."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field


def evaluate_retrieval(retrieved_docs: Sequence[Mapping[str, str]], golden_docs: Iterable[str]) -> Dict[str, float]:
    """Compute precision and recall against a golden set."""

    retrieved_contents = [doc.get("content", "") for doc in retrieved_docs]
    golden_set = set(golden_docs)
    tp = len(set(retrieved_contents) & golden_set)
    precision = tp / len(retrieved_contents) if retrieved_contents else 0.0
    recall = tp / len(golden_set) if golden_set else 0.0
    return {"precision": precision, "recall": recall}


class AdvancedEvaluationResult(BaseModel):
    """Structured response from the advanced judge."""

    faithfulness_score: int = Field(description="Score from 1-5 for faithfulness.")
    relevance_score: int = Field(description="Score from 1-5 for answer relevance.")
    plan_soundness_score: int = Field(description="Score from 1-5 for the plan's logic and efficiency.")
    analytical_depth_score: int = Field(description="Score from 1-5 for analytical depth and causal insight.")
    reasoning: str = Field(description="Detailed reasoning for the scores.")


def _build_judge_prompt(request: str, plan: Sequence[str], context: Sequence[Mapping[str, object]], answer: str) -> str:
    plan_text = "\n".join(map(str, plan))
    context_text = "\n".join(str(step) for step in context)
    return (
        "You are an impartial AI evaluator. Assess the agent's behaviour using the rubric.\n"
        f"Request:\n{request}\nPlan:\n{plan_text}\nContext:\n{context_text}\nAnswer:\n{answer}\n"
        "Provide structured scores (1-5) and detailed reasoning."
    )


def evaluate_with_advanced_judge(
    *,
    request: str,
    plan: Sequence[str],
    context: Sequence[Mapping[str, object]],
    answer: str,
    judge_llm: BaseLanguageModel,
) -> AdvancedEvaluationResult:
    prompt = _build_judge_prompt(request, plan, context, answer)
    result = judge_llm.invoke(prompt)
    if isinstance(result, AdvancedEvaluationResult):
        return result
    if hasattr(result, "dict"):
        return AdvancedEvaluationResult.model_validate(result.dict())
    return AdvancedEvaluationResult.model_validate(result)


__all__ = ["evaluate_retrieval", "AdvancedEvaluationResult", "evaluate_with_advanced_judge"]
