"""Agent graph nodes with improved safety and dependency injection."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, TypedDict

from langchain_core.language_models import BaseLanguageModel

from .tools import parse_tool_invocation


class AgentState(TypedDict, total=False):
    """State container shared across Archon's graph nodes."""

    original_request: str
    clarification_question: Optional[str]
    plan: List[str]
    intermediate_steps: List[Dict[str, Any]]
    verification_history: List[Dict[str, Any]]
    final_response: str


@dataclass
class AgentNodes:
    """Bundle of callable nodes that mirrors the notebook's behaviour."""

    ambiguity_llm: BaseLanguageModel
    planner_llm: BaseLanguageModel
    auditor_llm: BaseLanguageModel
    synthesizer_llm: BaseLanguageModel
    tools: Mapping[str, Any]
    planner_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.planner_prompt is None:
            self.planner_prompt = self._build_planner_prompt(self.tools.values())

    @staticmethod
    def _build_planner_prompt(tools: Iterable[Any]) -> str:
        tool_descriptions = []
        for tool in tools:
            name = getattr(tool, "name", tool.__class__.__name__)
            description = getattr(tool, "description", "No description provided.")
            tool_descriptions.append(f"- {name}: {description.strip()}")
        tool_block = "\n".join(tool_descriptions)
        return (
            "You are a master financial analyst agent. Create a step-by-step plan using the available tools.\n"
            f"Available Tools:\n{tool_block}\n"
            "Return a JSON list of tool invocations ending with 'FINISH'."
        )

    def _call_llm(self, llm: BaseLanguageModel, prompt: str) -> Any:
        response = llm.invoke(prompt)
        return getattr(response, "content", response)

    def ambiguity_check_node(self, state: AgentState) -> Dict[str, Any]:
        request = state["original_request"]
        prompt = (
            "You identify ambiguity in financial analysis requests.\n"
            "If the request is clear respond with 'OK'. Otherwise ask a single clarifying question.\n"
            f"User Request: {request}\nResponse:"
        )
        response = self._call_llm(self.ambiguity_llm, prompt)
        if str(response).strip() == "OK":
            return {"clarification_question": None}
        return {"clarification_question": str(response).strip()}

    def planner_node(self, state: AgentState) -> Dict[str, Any]:
        prompt = f"{self.planner_prompt}\nUser Request: {state['original_request']}\nPlan:"
        plan_text = self._call_llm(self.planner_llm, prompt)
        try:
            plan = ast.literal_eval(str(plan_text))
        except Exception as exc:
            raise ValueError(f"Planner produced invalid plan: {plan_text}") from exc
        if not isinstance(plan, list):
            raise ValueError("Planner output must be a list of steps")
        return {"plan": plan}

    def tool_executor_node(self, state: AgentState) -> Dict[str, Any]:
        if not state.get("plan"):
            return {"plan": [], "intermediate_steps": state.get("intermediate_steps", [])}

        next_step = state["plan"][0]
        tool_name, args, kwargs = parse_tool_invocation(next_step)
        tool = self.tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown tool: {tool_name}")

        if hasattr(tool, "run"):
            result = tool.run(*args, **kwargs)
        elif hasattr(tool, "invoke"):
            result = tool.invoke(*args, **kwargs)
        else:
            result = tool(*args, **kwargs)

        steps = state.get("intermediate_steps", []) + [
            {"tool_name": tool_name, "tool_input": args or kwargs, "tool_output": result}
        ]
        return {"plan": state["plan"][1:], "intermediate_steps": steps}

    def verification_node(self, state: AgentState) -> Dict[str, Any]:
        last_step = state["intermediate_steps"][-1]
        prompt = (
            "You are a meticulous fact-checker. Evaluate the relevance and coherence of the tool output.\n"
            f"User Request: {state['original_request']}\n"
            f"Tool: {last_step['tool_name']}\n"
            f"Output: {json.dumps(last_step['tool_output'], ensure_ascii=False)}"
        )
        audit = self.auditor_llm.invoke(prompt)
        record = audit.dict() if hasattr(audit, "dict") else audit
        history = state.get("verification_history", []) + [record]
        return {"verification_history": history}

    def router_node(self, state: AgentState) -> str:
        if state.get("clarification_question"):
            return "__end__"
        plan = state.get("plan", [])
        if not plan:
            return "planner"
        history = state.get("verification_history", [])
        if history and history[-1].get("confidence_score", 5) < 3:
            return "planner"
        if plan[0] == "FINISH":
            return "synthesize"
        return "execute_tool"

    def synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        context = "\n\n".join(
            f"## Tool: {step['tool_name']}\nInput: {step['tool_input']}\nOutput: {json.dumps(step['tool_output'], ensure_ascii=False)}"
            for step in state.get("intermediate_steps", [])
        )
        prompt = (
            "You are an expert financial strategist. Synthesise a coherent answer, highlighting causal links when possible.\n"
            f"User Request:\n{state['original_request']}\n---\nContext:\n{context}\n---\nFinal Answer:"
        )
        response = self._call_llm(self.synthesizer_llm, prompt)
        return {"final_response": str(response).strip()}


__all__ = ["AgentState", "AgentNodes"]
