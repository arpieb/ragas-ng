from __future__ import annotations

import logging
import random
import typing as t
import warnings
from dataclasses import dataclass, field

from ragas.callbacks import new_group
from ragas.cost import TokenUsageParser
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample, _json_safe
from ragas.executor import Executor
from ragas.run_config import RunConfig, run_scope
from ragas.testset._modern import optional, require_embedding, require_llm
from ragas.testset.document import Document, DocumentLike
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.graph_queries import get_parent_nodes
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.transforms import (
    Transforms,
    apply_transforms,
    default_transforms,
    default_transforms_for_prechunked,
)

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.testset.synthesizers import QueryDistribution
    from ragas.testset.synthesizers.base import BaseScenario


RAGAS_TESTSET_GENERATION_GROUP_NAME = "ragas testset generation"
logger = logging.getLogger(__name__)

DocumentAdapter = t.Callable[[t.Any], DocumentLike]
"""Converts one of your documents to a ``DocumentLike``; see ``generate_with_docs``."""


@dataclass
class TestsetGenerator:
    """
    Generates an evaluation dataset based on given scenarios and parameters.

    Attributes
    ----------
    llm : InstructorBaseRagasLLM
        The language model to use for the generation process: the output of
        ``llm_factory()``.
    embedding_model : BaseRagasEmbedding
        The embedding model used to build the knowledge graph, such as the
        output of ``embedding_factory()``.
    knowledge_graph : KnowledgeGraph, default empty
        The knowledge graph to use for the generation process.
    llm_context : Optional[str], default None
        Additional context to provide to the LLM when generating responses.
        This context will be used to guide how the LLM generates queries and answers.
    """

    llm: InstructorBaseRagasLLM
    embedding_model: BaseRagasEmbedding
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    persona_list: t.Optional[t.List[Persona]] = None
    llm_context: t.Optional[str] = None

    def __post_init__(self):
        # None is let through: the generate_with_* methods say which one is
        # missing, and generate() on a prebuilt graph needs no embeddings.
        optional(require_llm, self.llm, "TestsetGenerator")
        optional(require_embedding, self.embedding_model, "TestsetGenerator")

    def generate_with_docs(
        self,
        documents: t.Sequence[t.Any],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[InstructorBaseRagasLLM] = None,
        transforms_embedding_model: t.Optional[BaseRagasEmbedding] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
        document_adapter: t.Optional[DocumentAdapter] = None,
    ) -> t.Union[Testset, Executor]:
        """
        Generates an evaluation dataset from whole documents.

        ``transforms`` split, enrich and link the documents into a knowledge
        graph, then queries are generated from it.

        Parameters
        ----------
        documents : Sequence
            The source documents: objects with ``page_content`` and
            ``metadata`` attributes (``ragas.testset.document.Document``, a
            LangChain ``Document``), strings, or objects that
            ``document_adapter`` converts. Documents with no text are skipped.
        testset_size : int
            The number of test samples to generate
        transforms : Optional[Transforms], optional
            Custom transforms to apply to the documents, by default None
        transforms_llm : Optional[InstructorBaseRagasLLM], optional
            LLM to use for transforms if different from instance LLM, by default None
        transforms_embedding_model : Optional[BaseRagasEmbedding], optional
            Embedding model to use for transforms if different from instance model, by default None
        query_distribution : Optional[QueryDistribution], optional
            Distribution of query types to generate, by default None
        run_config : Optional[RunConfig], optional
            Configuration for the generation run, by default None
        callbacks : Optional[Callbacks], optional
            Callbacks to use during generation, by default None
        token_usage_parser : Optional[TokenUsageParser], optional
            Parse the LLMResult object and return a TokenUsage object. This is used to
            calculate the cost of the generation process.
        with_debugging_logs : bool, optional
            Whether to include debug logs, by default False
        raise_exceptions : bool, optional
            Whether to raise exceptions during generation, by default True
        return_executor : bool, optional
            If True, returns the Executor instance instead of running generation.
            The returned executor can be used to cancel execution by calling executor.cancel().
            To get results, call executor.results(). Default is False.
        document_adapter : Optional[Callable[[Any], DocumentLike]], optional
            Converts each document that is not a string to an object with
            ``page_content`` and ``metadata``, for documents whose type uses
            other names. For LlamaIndex documents:
            ``lambda d: Document(page_content=d.text, metadata=d.metadata)``.
            By default the documents are used as they are.

        Returns
        -------
        Testset or Executor
            If return_executor is False, returns the generated evaluation dataset.
            If return_executor is True, returns the Executor instance for cancellable execution.

        Raises
        ------
        ValueError
            If no LLM or embedding model is provided either during initialization or as arguments
        TypeError
            If an LLM is not an ``llm_factory()`` LLM, or an embedding model
            is not a ``BaseRagasEmbedding``
        """

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                """An llm client was not provided.
                       Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter.
                       Alternatively you can provide your own transforms through the `transforms` parameter."""
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                """An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."""
            )
        transforms_llm = optional(require_llm, transforms_llm, "transforms_llm")
        transforms_embedding_model = optional(
            require_embedding, transforms_embedding_model, "transforms_embedding_model"
        )

        docs = _to_documents(documents, document_adapter)
        if not transforms:
            transforms = default_transforms(
                documents=docs,
                llm=transforms_llm or self.llm,
                embedding_model=transforms_embedding_model or self.embedding_model,
            )

        kg = KnowledgeGraph(nodes=_to_nodes(docs, NodeType.DOCUMENT))

        # transforms and generation share one run scope (#90)
        return self._transform_and_generate(
            kg,
            transforms,
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            token_usage_parser=token_usage_parser,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
            return_executor=return_executor,
        )

    def generate_with_langchain_docs(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """Deprecated alias for :meth:`generate_with_docs`.

        The method no longer has anything to do with LangChain -- it accepts any
        object exposing ``page_content`` and ``metadata``, which LangChain
        ``Document`` objects still satisfy.
        """
        warnings.warn(
            "generate_with_langchain_docs() is deprecated and will be removed in a "
            "future version. Use generate_with_docs(), which accepts any object "
            "with `page_content` and `metadata` attributes -- including LangChain "
            "Document objects.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.generate_with_docs(*args, **kwargs)

    def generate_with_chunks(
        self,
        chunks: t.Sequence[t.Any],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[InstructorBaseRagasLLM] = None,
        transforms_embedding_model: t.Optional[BaseRagasEmbedding] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
        document_adapter: t.Optional[DocumentAdapter] = None,
    ) -> t.Union[Testset, Executor]:
        """
        Generates an evaluation dataset based on provided pre-chunked documents.

        This method allows users to skip the internal chunking process by providing
        documents that are already chunked. The input documents are treated as
        `NodeType.CHUNK` directly.

        Parameters
        ----------
        chunks : Sequence
            The chunks: strings, objects with ``page_content`` and
            ``metadata`` attributes, or objects that ``document_adapter``
            converts. Chunks with no text are skipped.
        testset_size : int
            The number of test samples to generate
        transforms : Optional[Transforms], optional
            Custom transforms to apply to the chunks, by default None
        transforms_llm : Optional[InstructorBaseRagasLLM], optional
            LLM to use for transforms if different from instance LLM, by default None
        transforms_embedding_model : Optional[BaseRagasEmbedding], optional
            Embedding model to use for transforms if different from instance model, by default None
        query_distribution : Optional[QueryDistribution], optional
            Distribution of query types to generate, by default None
        run_config : Optional[RunConfig], optional
            Configuration for the generation run, by default None
        callbacks : Optional[Callbacks], optional
            Callbacks to use during generation, by default None
        token_usage_parser : Optional[TokenUsageParser], optional
            Parse the LLMResult object and return a TokenUsage object.
        with_debugging_logs : bool, optional
            Whether to include debug logs, by default False
        raise_exceptions : bool, optional
            Whether to raise exceptions during generation, by default True
        return_executor : bool, optional
            If True, returns the Executor instance instead of running generation.
        document_adapter : Optional[Callable[[Any], DocumentLike]], optional
            Converts each chunk that is not a string, as in
            :meth:`generate_with_docs`.

        Returns
        -------
        Testset or Executor
            If return_executor is False, returns the generated evaluation dataset.
            If return_executor is True, returns the Executor instance.
        """

        # force the user to provide an llm and embedding client
        if not self.llm and not transforms_llm:
            raise ValueError(
                """An llm client was not provided.
                       Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter.
                       Alternatively you can provide your own transforms through the `transforms` parameter."""
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                """An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."""
            )
        transforms_llm = optional(require_llm, transforms_llm, "transforms_llm")
        transforms_embedding_model = optional(
            require_embedding, transforms_embedding_model, "transforms_embedding_model"
        )

        if transforms is None:
            transforms = default_transforms_for_prechunked(
                llm=transforms_llm or self.llm,
                embedding_model=transforms_embedding_model or self.embedding_model,
            )

        docs = _to_documents(chunks, document_adapter)
        kg = KnowledgeGraph(nodes=_to_nodes(docs, NodeType.CHUNK))

        # transforms and generation share one run scope (#90)
        return self._transform_and_generate(
            kg,
            transforms,
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            token_usage_parser=token_usage_parser,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
            return_executor=return_executor,
        )

    def generate(
        self,
        testset_size: int,
        query_distribution: t.Optional[QueryDistribution] = None,
        num_personas: int = 3,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
    ) -> t.Union[Testset, Executor]:
        """
        Generate an evaluation dataset based on given scenarios and parameters.

        Parameters
        ----------
        testset_size : int
            The number of samples to generate.
        query_distribution : Optional[QueryDistribution], optional
            A list of tuples containing scenario simulators and their probabilities.
            If None, default simulators will be used.
        num_personas : int, default 3
            The number of personas to generate or use from the persona_list.
        run_config : Optional[RunConfig], optional
            Configuration for running the generation process.
        batch_size: int, optional
            How large should batches be.  If set to None (default), no batching is done.
        callbacks : Optional[Callbacks], optional
            Langchain style callbacks to use for the generation process. You can use
            this to log the generation process or add other metadata.
        token_usage_parser : Optional[TokenUsageParser], optional
            Parse the LLMResult object and return a TokenUsage object. This is used to
            calculate the cost of the generation process.
        with_debugging_logs : bool, default False
            If True, enable debug logging for various components.
        raise_exceptions : bool, default True
            If True, raise exceptions during the generation process.
        return_executor : bool, default False
            If True, returns the Executor instance instead of running generation.
            The returned executor can be used to cancel execution by calling executor.cancel().
            To get results, call executor.results().

        Returns
        -------
        Testset or Executor
            If return_executor is False, returns a dataset containing the generated TestsetSamples.
            If return_executor is True, returns the Executor instance for cancellable execution.

        Notes
        -----
        This function performs the following steps:
        1. Set up scenarios and debug logging if required.
        2. Generate scenarios using an Executor.
        3. Calculate split values for different scenario types.
        4. Generate samples for each scenario.
        5. Compile the results into an EvaluationDataset.
        """
        # Token accounting: run-scoped, like the RunConfig, rather than set on
        # self.llm, which may be shared -- and was never detached (#76).
        cost_cb = _usage_collector(token_usage_parser)

        with run_scope(run_config=run_config, usage_collector=cost_cb):
            return self._generate(
                testset_size=testset_size,
                query_distribution=query_distribution,
                num_personas=num_personas,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=callbacks,
                cost_cb=cost_cb,
                with_debugging_logs=with_debugging_logs,
                raise_exceptions=raise_exceptions,
                return_executor=return_executor,
            )

    def _transform_and_generate(
        self,
        kg: KnowledgeGraph,
        transforms: Transforms,
        *,
        testset_size: int,
        query_distribution: t.Optional[QueryDistribution],
        run_config: t.Optional[RunConfig],
        callbacks: t.Optional[Callbacks],
        token_usage_parser: t.Optional[TokenUsageParser],
        with_debugging_logs: bool,
        raise_exceptions: bool,
        return_executor: bool,
    ) -> t.Union[Testset, Executor]:
        """Build the knowledge graph, then generate, in one run scope.

        Building the graph is most of the LLM cost, so the token collector
        and the RunConfig (timeouts, retries) cover the transforms as well
        as generation (#90). The graph is stored only once every transform
        has succeeded.
        """
        cost_cb = _usage_collector(token_usage_parser)
        with run_scope(run_config=run_config, usage_collector=cost_cb):
            apply_transforms(kg, transforms, run_config=run_config or RunConfig())
            self.knowledge_graph = kg
            return self._generate(
                testset_size=testset_size,
                query_distribution=query_distribution,
                num_personas=3,
                run_config=run_config,
                batch_size=None,
                callbacks=callbacks,
                cost_cb=cost_cb,
                with_debugging_logs=with_debugging_logs,
                raise_exceptions=raise_exceptions,
                return_executor=return_executor,
            )

    def _generate(
        self,
        testset_size: int,
        query_distribution: t.Optional[QueryDistribution],
        num_personas: int,
        run_config: t.Optional[RunConfig],
        batch_size: t.Optional[int],
        callbacks: t.Optional[Callbacks],
        cost_cb: t.Optional[t.Any],
        with_debugging_logs: bool,
        raise_exceptions: bool,
        return_executor: bool,
    ) -> t.Union[Testset, Executor]:
        query_distribution = query_distribution or default_query_distribution(
            self.llm, self.knowledge_graph, self.llm_context
        )
        callbacks = callbacks or []

        # dict to store any callbacks we define
        ragas_callbacks = {}
        # `callbacks` may be a CallbackGroup or a plain list of handlers
        for cb in ragas_callbacks.values():
            if hasattr(callbacks, "add_handler"):
                callbacks.add_handler(cb)  # type: ignore[union-attr]
            else:
                callbacks.append(cb)  # type: ignore[union-attr]

        # new group for Testset Generation
        testset_generation_rm, testset_generation_grp = new_group(
            name=RAGAS_TESTSET_GENERATION_GROUP_NAME,
            inputs={"testset_size": testset_size},
            callbacks=callbacks,
        )

        if with_debugging_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.testset.synthesizers", logging.DEBUG)
            patch_logger("ragas.testset.graph", logging.DEBUG)
            patch_logger("ragas.testset.transforms", logging.DEBUG)

        # Personas generated on the first call are kept on the generator, so
        # later calls reuse them: consistent runs, no repeat LLM calls. Shuffle
        # a copy, never the stored list (it may be the caller's own list).
        if self.persona_list is None:
            self.persona_list = generate_personas_from_kg(
                llm=self.llm,
                kg=self.knowledge_graph,
                num_personas=num_personas,
                callbacks=callbacks,
            )
        personas = random.sample(self.persona_list, len(self.persona_list))
        personas = personas[:num_personas]

        # testset_size is a ceiling: the shares sum to exactly testset_size,
        # and a synthesizer with a share of 0 is not run at all (#81).
        splits, _ = calculate_split_values(
            [prob for _, prob in query_distribution], testset_size
        )
        planned = [
            (synthesizer, split)
            for (synthesizer, _), split in zip(query_distribution, splits)
            if split > 0
        ]

        # new group for Generation of Scenarios
        scenario_generation_rm, scenario_generation_grp = new_group(
            name="Scenario Generation",
            inputs={"splits": splits},
            callbacks=testset_generation_grp,
        )

        # generate scenarios
        exec = Executor(
            desc="Generating Scenarios",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
        )
        for synthesizer, split in planned:
            exec.submit(
                synthesizer.generate_scenarios,
                n=split,
                knowledge_graph=self.knowledge_graph,
                persona_list=personas,
                callbacks=scenario_generation_grp,
            )

        try:
            scenario_results: t.List[t.Any] = exec.results()
        except Exception as e:
            scenario_generation_rm.on_chain_error(e)
            raise e

        # With raise_exceptions=False a failed job yields NaN instead of a
        # list; skip that synthesizer rather than crash on it (#82).
        failed_synthesizers: t.List[str] = []
        scenario_sample_list: t.List[t.List[BaseScenario]] = []
        for (synthesizer, split), scenarios in zip(planned, scenario_results):
            if isinstance(scenarios, list):
                scenario_sample_list.append(scenarios[:split])
            else:
                failed_synthesizers.append(synthesizer.name)
                scenario_sample_list.append([])
        scenario_generation_rm.on_chain_end(
            outputs={"scenario_sample_list": scenario_sample_list}
        )

        # new group for Generation of Samples
        sample_generation_rm, sample_generation_grp = new_group(
            name="Sample Generation",
            inputs={"scenario_sample_list": scenario_sample_list},
            callbacks=testset_generation_grp,
        )
        exec = Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
            batch_size=batch_size,
        )
        additional_testset_info: t.List[t.Dict] = []
        submitted_scenarios: t.List[BaseScenario] = []
        for (synthesizer, _), scenarios in zip(planned, scenario_sample_list):
            for scenario in scenarios:
                submitted_scenarios.append(scenario)
                exec.submit(
                    synthesizer.generate_sample,
                    scenario=scenario,
                    callbacks=sample_generation_grp,
                )
                # fill out the additional info for the TestsetSample
                additional_testset_info.append(
                    {
                        "synthesizer_name": synthesizer.name,
                    }
                )

        # Return executor for cancellable execution if requested
        if return_executor:
            return exec

        try:
            eval_samples = exec.results()
        except Exception as e:
            sample_generation_rm.on_chain_error(e)
            raise e
        else:
            sample_generation_rm.on_chain_end(outputs={"eval_samples": eval_samples})

        # build the testset; a failed sample is NaN under raise_exceptions=False
        # and is left out rather than failing TestsetSample validation (#82)
        testsets = []
        failed_samples = 0
        for sample, scenario, additional_info in zip(
            eval_samples, submitted_scenarios, additional_testset_info
        ):
            if not isinstance(sample, (SingleTurnSample, MultiTurnSample)):
                failed_samples += 1
                continue
            if isinstance(sample, SingleTurnSample):
                _attach_source_metadata(sample, scenario, self.knowledge_graph)
            testsets.append(TestsetSample(eval_sample=sample, **additional_info))

        _warn_on_shortfall(
            testset_size, len(testsets), failed_synthesizers, failed_samples
        )
        testset = Testset(samples=testsets, cost_cb=cost_cb, personas=personas)
        testset_generation_rm.on_chain_end({"testset": testset})

        return testset


def _to_documents(
    items: t.Iterable[t.Any], adapter: t.Optional[DocumentAdapter]
) -> t.List[DocumentLike]:
    """``items`` as documents, leaving out those with no text.

    A string becomes a ``Document``; anything else goes through ``adapter``
    when one is given, else is used as it is.
    """
    docs: t.List[DocumentLike] = []
    for item in items:
        if isinstance(item, str):
            doc: DocumentLike = Document(page_content=item)
        else:
            doc = adapter(item) if adapter is not None else item
        text = doc.page_content
        if text is not None and text.strip():
            docs.append(doc)
    return docs


def _to_nodes(docs: t.Sequence[DocumentLike], node_type: NodeType) -> t.List[Node]:
    return [
        Node(
            type=node_type,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        for doc in docs
    ]


def _source_metadata(node: Node, kg: KnowledgeGraph) -> t.Dict[str, t.Any]:
    """The document metadata for ``node``: its own, else its document's."""
    metadata = node.get_property("document_metadata")
    if metadata is None:
        for parent in get_parent_nodes(node, kg, level=3):
            metadata = parent.get_property("document_metadata")
            if metadata is not None:
                break
    return dict(metadata or {})


def _attach_source_metadata(
    sample: SingleTurnSample, scenario: BaseScenario, kg: KnowledgeGraph
) -> None:
    """Record each reference context's source document metadata (#91).

    The built-in synthesizers make one reference context per scenario node,
    in order. A synthesizer that sets the field itself, or whose contexts
    don't line up with its nodes, is left alone.
    """
    if sample.reference_contexts_metadata is not None:
        return
    contexts = sample.reference_contexts or []
    if len(contexts) != len(scenario.nodes):
        logger.debug(
            "Not attaching source metadata: %d reference contexts for %d nodes.",
            len(contexts),
            len(scenario.nodes),
        )
        return
    # assignment skips pydantic validation, so make the values JSON-safe here
    sample.reference_contexts_metadata = [
        _json_safe(_source_metadata(node, kg)) for node in scenario.nodes
    ]


def _usage_collector(
    token_usage_parser: t.Optional[TokenUsageParser],
) -> t.Optional[t.Any]:
    if token_usage_parser is None:
        return None
    from ragas.cost import TokenUsageCollector

    return TokenUsageCollector(token_usage_parser=token_usage_parser)


def _warn_on_shortfall(
    requested: int,
    generated: int,
    failed_synthesizers: t.List[str],
    failed_samples: int,
) -> None:
    """Warn when the testset is smaller than ``testset_size``, and say why."""
    if generated >= requested:
        return
    reasons = []
    if failed_synthesizers:
        reasons.append(
            "scenario generation failed for " + ", ".join(failed_synthesizers)
        )
    if failed_samples:
        reasons.append(f"{failed_samples} sample(s) failed")
    if not reasons:
        reasons.append(
            "the knowledge graph did not yield enough scenarios; add documents "
            "or change the query distribution"
        )
    warnings.warn(
        f"Generated {generated} of the {requested} requested samples: "
        + "; ".join(reasons)
        + ".",
        UserWarning,
        stacklevel=4,
    )
