"""Tool abstractions used by the Archon agent."""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, Iterable, List, Protocol

LOGGER = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol describing the embedding interface required by the tools."""

    def embed(self, texts: Iterable[str], batch_size: int | None = None) -> Iterable[List[float]]:  # pragma: no cover - protocol
        ...


class VectorClient(Protocol):
    """Subset of the Qdrant client used by the librarian tool."""

    def search(self, *, collection_name: str, query_vector: Iterable[float], limit: int, with_payload: bool) -> List[Any]:
        ...  # pragma: no cover - protocol


class CrossEncoderModel(Protocol):
    """Subset of the cross-encoder API used for reranking."""

    def predict(self, pairs: Iterable[List[str]]) -> Iterable[float]:  # pragma: no cover - protocol
        ...


class QueryOptimizer(Protocol):
    """Protocol for the optional query optimiser LLM."""

    def invoke(self, prompt: str) -> Any:  # pragma: no cover - protocol
        ...


class LibrarianRAGTool:
    """High-level wrapper around the retrieval stack used in the notebook."""

    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        vector_client: VectorClient,
        collection_name: str,
        reranker: CrossEncoderModel,
        query_optimizer: QueryOptimizer | None = None,
        initial_limit: int = 20,
        top_k: int = 5,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_client = vector_client
        self.collection_name = collection_name
        self.reranker = reranker
        self.query_optimizer = query_optimizer
        self.initial_limit = initial_limit
        self.top_k = top_k

    def optimise_query(self, query: str) -> str:
        if not self.query_optimizer:
            return query

        try:
            response = self.query_optimizer.invoke(
                "You are a query optimisation expert. Rewrite the following query for financial document retrieval.\n"
                f"Query: {query}\nOptimised Query:"
            )
        except Exception:  # pragma: no cover - optimisation is optional
            LOGGER.exception("Query optimisation failed; using original query")
            return query

        optimised = getattr(response, "content", None) or str(response)
        return optimised.strip() or query

    def _rerank(self, query: str, results: List[Any]) -> List[Any]:
        rerank_pairs = [[query, result.payload["content"]] for result in results]
        scores = list(self.reranker.predict(rerank_pairs))
        for result, score in zip(results, scores):
            result.score = float(score)
        return sorted(results, key=lambda item: getattr(item, "score", 0), reverse=True)

    def run(self, query: str) -> List[Dict[str, Any]]:
        optimised_query = self.optimise_query(query)
        embedding = list(self.embedding_model.embed([optimised_query]))[0]
        candidates = self.vector_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=self.initial_limit,
            with_payload=True,
        )
        reranked = self._rerank(optimised_query, candidates)
        return [
            {
                "source": result.payload.get("source"),
                "content": result.payload.get("content"),
                "summary": result.payload.get("summary"),
                "rerank_score": float(getattr(result, "score", 0.0)),
            }
            for result in reranked[: self.top_k]
        ]


class AnalystSQLTool:
    """Wrapper around the LangChain SQL agent used in the notebook."""

    def __init__(self, executor: Any) -> None:
        self.executor = executor

    def run(self, query: str) -> str:
        response = self.executor.invoke({"input": query})
        return str(response.get("output", ""))


class AnalystTrendTool:
    """Performs deterministic trend calculations over a financial DataFrame."""

    def __init__(self, dataframe) -> None:  # using pandas.DataFrame without importing pandas eagerly
        self.dataframe = dataframe

    def run(self, metric: str = "revenue_usd_billions") -> str:
        df = self.dataframe.copy()
        df["period"] = df["year"].astype(str) + "-" + df["quarter"]
        df.set_index("period", inplace=True)
        df["QoQ_Growth"] = df[metric].pct_change()
        df["YoY_Growth"] = df[metric].pct_change(4)

        latest_period = df.index[-1]
        start_period = df.index[0]
        latest_val = df.loc[latest_period, metric]
        start_val = df.loc[start_period, metric]
        latest_qoq = df.loc[latest_period, "QoQ_Growth"]
        latest_yoy = df.loc[latest_period, "YoY_Growth"]

        return (
            f"Analysis of {metric} from {start_period} to {latest_period}:\n"
            f"- The series starts at ${start_val:.1f}B and ends at ${latest_val:.1f}B.\n"
            f"- Latest quarter QoQ growth: {latest_qoq:.1%}.\n"
            f"- Latest quarter YoY growth: {latest_yoy:.1%}."
        )


def parse_tool_invocation(invocation: str) -> tuple[str, list[Any], Dict[str, Any]]:
    """Parse a ``planner`` step into a callable name and arguments.

    The original notebook used ``eval`` which executes arbitrary code.  This
    helper relies on ``ast`` so untrusted planner output cannot execute
    side-effects while still supporting positional and keyword arguments.
    """

    try:
        call = ast.parse(invocation, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive against malformed plans
        raise ValueError(f"Invalid tool invocation: {invocation}") from exc

    if not isinstance(call.body, ast.Call) or not isinstance(call.body.func, ast.Name):
        raise ValueError(f"Unsupported tool invocation: {invocation}")

    args = [ast.literal_eval(arg) for arg in call.body.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.body.keywords}
    return call.body.func.id, args, kwargs


__all__ = [
    "LibrarianRAGTool",
    "AnalystSQLTool",
    "AnalystTrendTool",
    "parse_tool_invocation",
]
