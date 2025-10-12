"""Vision analyst utilities."""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field


class ChartAnalysis(BaseModel):
    chart_type: str = Field(description="Type of chart being analysed.")
    key_insight: str = Field(description="Primary takeaway from the chart.")
    data_points: list[str] = Field(description="Key supporting data points.")


def vision_analyst_tool(*, description: str, vision_llm: BaseLanguageModel) -> ChartAnalysis:
    prompt = (
        "You are a financial analyst with vision capabilities. Analyse the chart described.\n"
        f"Description: {description}"
    )
    result = vision_llm.invoke(prompt)
    if isinstance(result, ChartAnalysis):
        return result
    if hasattr(result, "dict"):
        return ChartAnalysis.model_validate(result.dict())
    return ChartAnalysis.model_validate(result)


__all__ = ["ChartAnalysis", "vision_analyst_tool"]
