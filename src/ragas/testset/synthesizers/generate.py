from __future__ import annotations

import logging
import random
import typing as t
import warnings
from dataclasses import dataclass, field

from ragas.callbacks import new_group
from ragas.cost import TokenUsageParser
from ragas.dataset_schema import MultiTurnSample, SingleTurnSample, _json_safe
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LlamaIndexEmbeddingsWrapper,
)
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LlamaIndexLLMWrapper
from ragas.run_config import RunConfig, run_scope
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
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding as LlamaIndexEmbedding,
    )
    from llama_index.core.base.llms.base import BaseLLM as LlamaIndexLLM
    from llama_index.core.schema import Document as LlamaIndexDocument

    from ragas.callbacks import Callbacks
    from ragas.embeddings.base import BaseRagasEmbedding, BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM, InstructorBaseRagasLLM
    from ragas.testset.synthesizers import QueryDistribution
    from ragas.testset.synthesizers.base import BaseScenario


RAGAS_TESTSET_GENERATION_GROUP_NAME = "ragas testset generation"
logger = logging.getLogger(__name__)


@dataclass
class TestsetGenerator:
    """
    Generates an evaluation dataset based on given scenarios and parameters.

    Attributes
    ----------
    llm : BaseRagasLLM or InstructorBaseRagasLLM
        The language model to use for the generation process, such as the
        output of ``llm_factory``.
    embedding_model : BaseRagasEmbeddings or BaseRagasEmbedding
        The embedding model used to build the knowledge graph, such as the
        output of ``embedding_factory``.
    knowledge_graph : KnowledgeGraph, default empty
        The knowledge graph to use for the generation process.
    llm_context : Optional[str], default None
        Additional context to provide to the LLM when generating responses.
        This context will be used to guide how the LLM generates queries and answers.
    """

    llm: t.Union[BaseRagasLLM, InstructorBaseRagasLLM]
    embedding_model: t.Union[BaseRagasEmbeddings, BaseRagasEmbedding]
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    persona_list: t.Optional[t.List[Persona]] = None
    llm_context: t.Optional[str] = None

    @classmethod
    def from_llama_index(
        cls,
        llm: LlamaIndexLLM,
        embedding_model: LlamaIndexEmbedding,
        knowledge_graph: t.Optional[KnowledgeGraph] = None,
        llm_context: t.Optional[str] = None,
    ) -> TestsetGenerator:
        """
        Creates a `TestsetGenerator` from a LlamaIndex LLM and embedding model.
        """
        knowledge_graph = knowledge_graph or KnowledgeGraph()
        return cls(
            LlamaIndexLLMWrapper(llm),
            LlamaIndexEmbeddingsWrapper(embedding_model),
            knowledge_graph,
            llm_context=llm_context,
        )

    def generate_with_docs(
        self,
        documents: t.Sequence[DocumentLike],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[
            t.Union[BaseRagasLLM, InstructorBaseRagasLLM]
        ] = None,
        transforms_embedding_model: t.Optional[
            t.Union[BaseRagasEmbeddings, BaseRagasEmbedding]
        ] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
    ) -> t.Union[Testset, Executor]:
        """
        Generates an evaluation dataset based on given Langchain documents and parameters.

        Parameters
        ----------
        documents : Sequence[DocumentLike]
            A sequence of Langchain documents to use as source material
        testset_size : int
            The number of test samples to generate
        transforms : Optional[Transforms], optional
            Custom transforms to apply to the documents, by default None
        transforms_llm : Optional[BaseRagasLLM or InstructorBaseRagasLLM], optional
            LLM to use for transforms if different from instance LLM, by default None
        transforms_embedding_model : Optional[BaseRagasEmbeddings or BaseRagasEmbedding], optional
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

        Returns
        -------
        Testset or Executor
            If return_executor is False, returns the generated evaluation dataset.
            If return_executor is True, returns the Executor instance for cancellable execution.

        Raises
        ------
        ValueError
            If no LLM or embedding model is provided either during initialization or as arguments
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

        if not transforms:
            transforms = default_transforms(
                documents=list(documents),
                llm=transforms_llm or self.llm,
                embedding_model=transforms_embedding_model or self.embedding_model,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
            nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

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

    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaIndexDocument],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[LlamaIndexLLM] = None,
        transforms_embedding_model: t.Optional[LlamaIndexEmbedding] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ):
        """
        Generates an evaluation dataset based on given scenarios and parameters.
        """

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                "An llm client was not provided. Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter. Alternatively you can provide your own transforms through the `transforms` parameter."
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                "An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."
            )

        if not transforms:
            # use TestsetGenerator's LLM and embedding model if no transforms_llm or transforms_embedding_model is provided
            if transforms_llm is None:
                llm_for_transforms = self.llm
            else:
                llm_for_transforms = LlamaIndexLLMWrapper(transforms_llm)
            if transforms_embedding_model is None:
                embedding_model_for_transforms = self.embedding_model
            else:
                embedding_model_for_transforms = LlamaIndexEmbeddingsWrapper(
                    transforms_embedding_model
                )

            # create the transforms
            transforms = default_transforms(
                documents=[Document(page_content=doc.text) for doc in documents],
                llm=llm_for_transforms,
                embedding_model=embedding_model_for_transforms,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            if doc.text is not None and doc.text.strip() != "":
                node = Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.text,
                        "document_metadata": doc.metadata,
                    },
                )
                nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

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
            return_executor=False,
        )

    def generate_with_chunks(
        self,
        chunks: t.Sequence[t.Union[DocumentLike, str]],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[
            t.Union[BaseRagasLLM, InstructorBaseRagasLLM]
        ] = None,
        transforms_embedding_model: t.Optional[
            t.Union[BaseRagasEmbeddings, BaseRagasEmbedding]
        ] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
    ) -> t.Union[Testset, Executor]:
        """
        Generates an evaluation dataset based on provided pre-chunked documents.

        This method allows users to skip the internal chunking process by providing
        documents that are already chunked. The input documents are treated as
        `NodeType.CHUNK` directly.

        Parameters
        ----------
        chunks : Sequence[Union[DocumentLike, str]]
            A sequence of Langchain documents or strings to use as chunks.
            Strings will be automatically converted to Documents.
        testset_size : int
            The number of test samples to generate
        transforms : Optional[Transforms], optional
            Custom transforms to apply to the chunks, by default None
        transforms_llm : Optional[BaseRagasLLM or InstructorBaseRagasLLM], optional
            LLM to use for transforms if different from instance LLM, by default None
        transforms_embedding_model : Optional[BaseRagasEmbeddings or BaseRagasEmbedding], optional
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

        if transforms is None:
            transforms = default_transforms_for_prechunked(
                llm=transforms_llm or self.llm,
                embedding_model=transforms_embedding_model or self.embedding_model,
            )

        # convert the chunks to Ragas nodes
        nodes = []
        for chunk in chunks:
            if isinstance(chunk, str):
                page_content = chunk
                metadata = {}
            else:
                page_content = chunk.page_content
                metadata = chunk.metadata

            if page_content is not None and page_content.strip() != "":
                node = Node(
                    type=NodeType.CHUNK,
                    properties={
                        "page_content": page_content,
                        "document_metadata": metadata,
                    },
                )
                nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

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
