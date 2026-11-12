from __future__ import annotations

import logging
import math
import typing as t

from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.executor import Executor
from ragas.messages import AIMessage, HumanMessage, Message, ToolCall, ToolMessage
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from llama_index.core.base.response.schema import Response as LlamaIndexResponse
    from llama_index.core.workflow import Event

    from ragas.callbacks import Callbacks
    from ragas.cost import TokenUsageParser
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.base import SimpleBaseMetric


logger = logging.getLogger(__name__)


def evaluate(
    query_engine,
    dataset: EvaluationDataset,
    metrics: t.List[SimpleBaseMetric],
    llm: t.Optional[InstructorBaseRagasLLM] = None,
    embeddings: t.Optional[BaseRagasEmbedding] = None,
    callbacks: t.Optional[Callbacks] = None,
    in_ci: bool = False,
    run_config: t.Optional[RunConfig] = None,
    batch_size: t.Optional[int] = None,
    token_usage_parser: t.Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: t.Optional[t.Dict[str, str]] = None,
    show_progress: bool = True,
) -> EvaluationResult:
    """Run a LlamaIndex query engine over ``dataset`` and score the answers.

    ``metrics`` are ``ragas.metrics.collections`` metrics; ``llm`` and
    ``embeddings`` are ragas models (``llm_factory()``, ``embedding_factory()``)
    used for metrics that take them per call and for the default metrics.
    LlamaIndex LLMs are no longer wrapped (#215): reach the same provider with
    ``llm_factory()``.
    """
    column_map = column_map or {}

    # validate and transform dataset
    if dataset is None or not isinstance(dataset, EvaluationDataset):
        raise ValueError("Please provide a dataset that is of type EvaluationDataset")

    exec = Executor(
        desc="Running Query Engine",
        keep_progress_bar=True,
        show_progress=show_progress,
        raise_exceptions=raise_exceptions,
        run_config=run_config,
        batch_size=batch_size,
    )

    # check if multi-turn
    if dataset.is_multi_turn():
        raise NotImplementedError(
            "Multi-turn evaluation is not implemented yet. Please do raise an issue on GitHub if you need this feature and we will prioritize it"
        )
    samples = t.cast(t.List[SingleTurnSample], dataset.samples)

    # get query and make jobs
    queries = [sample.user_input for sample in samples]
    for i, q in enumerate(queries):
        exec.submit(query_engine.aquery, q, name=f"query-{i}")

    # get responses and retrieved contexts
    responses: t.List[t.Optional[str]] = []
    retrieved_contexts: t.List[t.Optional[t.List[str]]] = []
    results = exec.results()
    for i, r in enumerate(results):
        # Handle failed jobs which are recorded as NaN in the executor
        if isinstance(r, float) and math.isnan(r):
            responses.append(None)
            retrieved_contexts.append(None)
            logger.warning(f"Query engine failed for query {i}: '{queries[i]}'")
            continue

        # Cast to LlamaIndex Response type for proper type checking
        response: LlamaIndexResponse = t.cast("LlamaIndexResponse", r)
        responses.append(response.response if response.response is not None else "")
        retrieved_contexts.append([n.get_text() for n in response.source_nodes])

    # append the extra information to the dataset
    for i, sample in enumerate(samples):
        sample.response = responses[i]
        sample.retrieved_contexts = retrieved_contexts[i]

    results = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=raise_exceptions,
        _callbacks=callbacks,
        show_progress=show_progress,
        run_config=run_config or RunConfig(),
        token_usage_parser=token_usage_parser,
        return_executor=False,
    )

    # Type assertion since return_executor=False guarantees EvaluationResult
    return t.cast(EvaluationResult, results)


def convert_to_ragas_messages(events: t.List[Event]) -> t.List[Message]:
    """
    Convert a sequence of LlamIndex agent events into Ragas message objects.

    This function processes a list of `Event` objects (e.g., `AgentInput`, `AgentOutput`,
    and `ToolCallResult`) and converts them into a list of `Message` objects (`HumanMessage`,
    `AIMessage`, and `ToolMessage`) that can be used for evaluation with the Ragas framework.

    Parameters
    ----------
    events : List[Event]
        A list of agent events that represent a conversation trace. These can include
        user inputs (`AgentInput`), model outputs (`AgentOutput`), and tool responses
        (`ToolCallResult`).

    Returns
    -------
    List[Message]
        A list of Ragas `Message` objects corresponding to the structured conversation.
        Tool calls are de-duplicated using their tool ID to avoid repeated entries.
    """
    try:
        from llama_index.core.agent.workflow import (
            AgentInput,
            AgentOutput,
            ToolCallResult,
        )
        from llama_index.core.base.llms.types import MessageRole, TextBlock
    except ImportError:
        raise ImportError(
            "Please install the llama_index package to use this function."
        )
    ragas_messages = []
    tool_call_ids = set()

    for event in events:
        if isinstance(event, AgentInput):
            last_chat_message = event.input[-1]

            content = ""
            if last_chat_message.blocks:
                content = "\n".join(
                    str(block.text)
                    for block in last_chat_message.blocks
                    if isinstance(block, TextBlock)
                )

            if last_chat_message.role == MessageRole.USER:
                if ragas_messages and isinstance(ragas_messages[-1], ToolMessage):
                    continue
                ragas_messages.append(HumanMessage(content=content))

        elif isinstance(event, AgentOutput):
            content = "\n".join(
                str(block.text)
                for block in event.response.blocks
                if isinstance(block, TextBlock)
            )
            ragas_tool_calls = None

            if hasattr(event, "tool_calls"):
                raw_tool_calls = event.tool_calls
                ragas_tool_calls = []
                for tc in raw_tool_calls:
                    if tc.tool_id not in tool_call_ids:
                        tool_call_ids.add(tc.tool_id)
                        ragas_tool_calls.append(
                            ToolCall(
                                name=tc.tool_name,
                                args=tc.tool_kwargs,
                            )
                        )
            ragas_messages.append(
                AIMessage(
                    content=content,
                    tool_calls=ragas_tool_calls if ragas_tool_calls else None,
                )
            )
        elif isinstance(event, ToolCallResult):
            if event.return_direct:
                ragas_messages.append(AIMessage(content=event.tool_output.content))
            else:
                ragas_messages.append(ToolMessage(content=event.tool_output.content))

    return ragas_messages
