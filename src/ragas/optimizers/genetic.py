from __future__ import annotations

import copy
import logging
import re
import typing as t
from uuid import UUID

import numpy as np
from pydantic import BaseModel
from tqdm.auto import tqdm

from ragas.async_utils import run_sync
from ragas.callbacks import new_group
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    SampleAnnotation,
    SingleMetricAnnotation,
)
from ragas.executor import Executor
from ragas.losses import Loss
from ragas.optimizers.base import Optimizer
from ragas.optimizers.utils import hamming_distance
from ragas.prompt.metrics.base_prompt import BasePrompt
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.metrics.collections.base import BaseMetric

logger = logging.getLogger(__name__)

RAGAS_OPTIMIZATION_GROUP = "ragas_optimization"
MIN_ANNOTATIONS = 10

example_type = t.TypeVar(
    "example_type", bound=t.Dict[t.Dict[str, t.Any], t.Dict[str, t.Any]]
)

# Where BasePrompt.to_string puts the input, after the instruction, schema
# and examples.
_INPUT_MARKER = "Now perform the same with the following input\n"
_REPEAT_SUFFIX = re.compile(r" \(\d+\)$")


class FormattedExamples(BaseModel):
    examples: t.List[t.Tuple[str, t.Any]]

    @classmethod
    def from_examples(cls, examples: t.List[example_type]) -> "FormattedExamples":
        formated_examples = []
        for example in examples:
            input_, output = example.values()
            input_ = "".join(f"\n{key}:\n\t{val}\n" for key, val in input_.items())
            formated_examples.append((input_, output))

        return cls(examples=formated_examples)


class OutputInstruction(BaseModel):
    instruction: str


class ReverseEngineerPrompt(BasePrompt[FormattedExamples, OutputInstruction]):
    instruction = "Given a set of (input containing (user_input, response, reference, etc), expected output) pairs that were manually annotated, guess and generate the instruction given to the annotator."
    input_model = FormattedExamples
    output_model = OutputInstruction


class ParentPrompts(BaseModel):
    parent_1: str
    parent_2: str


class CrossOverPrompt(BasePrompt[ParentPrompts, OutputInstruction]):
    instruction = (
        "You are a mutator who is familiar with the concept of cross-over in genetic algorithm, namely "
        "combining the genetic information of two parents to generate new offspring. Given two parent "
        "prompts, you will perform a cross-over to generate an offspring prompt that covers the same "
        "semantic meaning as both parents."
    )
    input_model = ParentPrompts
    output_model = OutputInstruction
    examples = [
        (
            ParentPrompts(
                parent_1="Now you are a categorizer, your mission is to ascertain the sentiment of the provided text, either favorable or unfavorable.",
                parent_2="Assign a sentiment label to the given sentence from [’negative’, ’positive’] and return only the label without any other text.",
            ),
            OutputInstruction(
                instruction="Your mission is to ascertain the sentiment of the provided text and assign a sentiment label from [’negative’, ’positive’].",
            ),
        )
    ]


class FeedbackExample(BaseModel):
    input: str
    output: t.Dict[str, t.Any]
    expected_output: t.Dict[str, t.Any]


class FeedbackMutationInput(BaseModel):
    instruction: str
    examples: t.List[FeedbackExample]


class FeedbackMutationOutput(BaseModel):
    feedbacks: t.List[str]


class FeedbackMutationPrompt(BasePrompt[FeedbackMutationInput, FeedbackMutationOutput]):
    instruction = (
        "You're an expert reviewer. Given an instruction and a set of (input  containing (user_input, response, reference, etc), output, expected_output) examples. After analyzing the examples, give maximum 3 concrete feedbacks on how the instruction can be modified so that the model arrives at the expected output."
        "Do not provide the feedback to add examples with the instruction."
    )
    input_model = FeedbackMutationInput
    output_model = FeedbackMutationOutput


class FeedbackMutationPromptInput(BaseModel):
    instruction: str
    feedbacks: t.List[str]


class FeedbackMutationPromptGeneration(
    BasePrompt[FeedbackMutationPromptInput, OutputInstruction]
):
    instruction = "You are a mutator. Given an instruction and a set of feedbacks on how the instruction can be improved generate a new instruction that incorporates the feedback."
    input_model = FeedbackMutationPromptInput
    output_model = OutputInstruction


def _traced_prompt_io(
    metric_trace: t.Mapping[str, t.Any], prompt: BasePrompt
) -> t.Optional[t.Tuple[str, t.Dict[str, t.Any]]]:
    """``(input text, output)`` of ``prompt``'s LLM call in a metric's trace.

    A collections metric renders its prompts and calls the LLM itself, so
    the trace holds one LLM-call run per call, named after the response
    model, with the rendered prompt as input. The call whose response model
    is the prompt's output model and whose text starts with the prompt's
    instruction is taken; failing the instruction match, the first call with
    that response model.
    """
    name = prompt.output_model.__name__
    calls = [
        call
        for key, call in metric_trace.items()
        if isinstance(call, dict) and _REPEAT_SUFFIX.sub("", key) == name
    ]
    if not calls:
        return None

    def text_of(call: t.Mapping[str, t.Any]) -> str:
        data = call.get("input", {})
        text = data.get("prompt", data) if isinstance(data, dict) else data
        return text if isinstance(text, str) else str(text)

    instruction = prompt.instruction.strip()
    call = next(
        (c for c in calls if text_of(c).lstrip().startswith(instruction)), calls[0]
    )
    text = text_of(call)
    if _INPUT_MARKER in text:
        text = text.split(_INPUT_MARKER, 1)[1]
        text = text.removesuffix("Output: ").rstrip()
    output = call.get("output", {})
    if isinstance(output, BaseModel):
        output = output.model_dump(exclude_none=True)
    elif not isinstance(output, dict):
        output = {"output": output}
    return text, output


class GeneticOptimizer(Optimizer):
    """
    A genetic algorithm optimizer that balances exploration and exploitation.

    Tunes the instructions of a collections metric's prompts. Each candidate
    is scored on a copy of the metric through ``evaluate()``, so the metric
    being optimized is never modified. ``config`` (``optimizer_config``)
    takes ``population_size`` (default 3), ``num_demonstrations`` (examples
    per reverse-engineered instruction, default 3) and ``sample_size``
    (samples per feedback mutation, default 12).
    """

    reverse_engineer_prompt = ReverseEngineerPrompt(name="reverse_engineer")
    cross_over_prompt = CrossOverPrompt(name="crossover")
    feedback_generation_prompt = FeedbackMutationPrompt(name="feedback_mutation")
    feedback_mutation_prompt = FeedbackMutationPromptGeneration(
        name="feedback_mutation_generation"
    )

    def optimize(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Optional[t.Mapping[str, t.Any]] = None,
        *,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
    ) -> t.Dict[str, BasePrompt]:
        callbacks = callbacks or []
        metric, _ = self._require()
        config = config or {}

        if len(dataset) < MIN_ANNOTATIONS:
            raise ValueError(
                f"Number of annotations should be greater than {MIN_ANNOTATIONS}. Please annotate {MIN_ANNOTATIONS - len(dataset)} more samples"
            )

        population_size = config.get("population_size", 3)
        num_demonstrations = config.get("num_demonstrations", 3)
        sample_size = config.get("sample_size", 12)

        # new group for optimization
        optimization_generation_rm, optimization_generation_grp = new_group(
            name=RAGAS_OPTIMIZATION_GROUP,
            inputs={"metric": metric.name},
            callbacks=callbacks,
        )

        stages = [
            {"name": "Initializing Population", "steps": population_size - 1},
            {
                "name": "Feedback Mutation",
                "steps": population_size * sample_size + population_size,
            },
            {
                "name": "Cross-over Mutation",
                "steps": population_size * len(dataset) + population_size,
            },
            {"name": "Fitness Evaluation", "steps": population_size * len(dataset)},
        ]
        total_steps = sum([stage["steps"] for stage in stages])
        with tqdm(
            total=total_steps, desc="Overall Progress", dynamic_ncols=True
        ) as parent_pbar:
            parent_pbar.set_description(f"{stages[0]['name']} Step 1/{len(stages)}")
            initial_population = self.initialize_population(
                dataset=dataset,
                population_size=population_size - 1,
                num_demonstrations=num_demonstrations,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=optimization_generation_grp,
                raise_exceptions=raise_exceptions,
                parent_pbar=parent_pbar,
            )

            # get the default prompt used in the metric as seed prompt
            if len(initial_population) > 0:
                seed_prompts = {
                    key: val.instruction
                    for key, val in metric.get_prompts().items()
                    if key in initial_population[0].keys()
                }
                initial_population.append(seed_prompts)

            parent_pbar.set_description(f"{stages[1]['name']} Step 2/{len(stages)}")
            improved_prompts = self.feedback_mutation(
                initial_population,
                dataset,
                sample_size=sample_size,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=optimization_generation_grp,
                raise_exceptions=raise_exceptions,
                parent_pbar=parent_pbar,
            )

            parent_pbar.set_description(f"{stages[2]['name']} Step 3/{len(stages)}")
            improved_prompts = self.cross_over_mutation(
                candidates=improved_prompts,
                dataset=dataset,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=optimization_generation_grp,
                raise_exceptions=raise_exceptions,
                parent_pbar=parent_pbar,
            )

            parent_pbar.set_description(f"{stages[3]['name']} Step 4/{len(stages)}")
            fitness_scores = self.evaluate_fitness(
                candidates=improved_prompts,
                dataset=dataset,
                loss_fn=loss,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=optimization_generation_grp,
                raise_exceptions=raise_exceptions,
                parent_pbar=parent_pbar,
            )
        # evaluate_fitness returns higher-is-better values (Loss.fitness); a
        # NaN (a candidate whose every score failed) never wins.
        ranked = np.nan_to_num(np.asarray(fitness_scores, dtype=float), nan=-np.inf)
        best_candidate = improved_prompts[int(np.argmax(ranked))]

        optimization_generation_rm.on_chain_end(
            outputs={"best_candidate": best_candidate}
        )

        return self._with_instructions(best_candidate)

    def initialize_population(
        self,
        *,
        dataset: SingleMetricAnnotation,
        population_size: int,
        num_demonstrations: int = 3,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        parent_pbar: t.Optional[tqdm] = None,
    ) -> t.List[t.Dict[str, str]]:
        initialize_population_rm, initialize_population_grp = new_group(
            name="Initializing Population",
            inputs={"population_size": population_size},
            callbacks=callbacks,
        )

        exec = Executor(
            desc="Initializing Population",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
            pbar=parent_pbar,
        )

        candidates = []
        dataset = dataset.filter(lambda x: x["is_accepted"])
        batches = dataset.stratified_batches(
            batch_size=num_demonstrations,
            stratify_key="metric_output",
            replace=False,
            drop_last_batch=False,
        )
        for batch in batches[:population_size]:
            exec.submit(
                self._reverse_engineer_instruction,
                batch=batch,
                callbacks=initialize_population_grp,
            )

        try:
            candidates = exec.results()
        except Exception as e:
            initialize_population_rm.on_chain_error(e)
            raise e
        else:
            initialize_population_rm.on_chain_end(
                outputs={"initial_population": candidates}
            )

        return candidates

    async def _reverse_engineer_instruction(
        self, batch: t.List[SampleAnnotation], callbacks: Callbacks = None
    ) -> t.Dict[str, str]:
        prompt_annotations = {key: [] for key in batch[0]["prompts"].keys()}
        candidates = {}
        for sample in batch:
            input_ouputs = sample["prompts"]
            for name, example in input_ouputs.items():
                input_ = {
                    key: val
                    for key, val in example["prompt_input"].items()
                    if val is not None
                }
                output = (
                    example["edited_output"]
                    if example["edited_output"]
                    else example["prompt_output"]
                )
                prompt_annotations[name].append({"input": input_, "output": output})

        for prompt_name, examples in prompt_annotations.items():
            formatted_examples = FormattedExamples.from_examples(examples)
            instruction = await self.reverse_engineer_prompt.generate(
                data=formatted_examples, llm=self._llm, callbacks=callbacks
            )
            candidates[prompt_name] = instruction.instruction

        return candidates

    async def _cross_over_prompts(
        self, parent_1: str, parent_2: str, callbacks: Callbacks = None
    ) -> str:
        parents = ParentPrompts(parent_1=parent_1, parent_2=parent_2)
        offspring = await self.cross_over_prompt.generate(
            data=parents, llm=self._llm, callbacks=callbacks
        )
        return offspring.instruction

    def candidate_metric(self, candidate: t.Dict[str, str]) -> BaseMetric:
        """A copy of the metric with ``candidate``'s instructions.

        The metric under optimization is never changed, so candidates scored
        concurrently can't overwrite each other's instructions.
        """
        scored = copy.copy(self._metric)
        scored.set_prompts(**self._with_instructions(candidate))
        return scored

    def feedback_mutation(
        self,
        candidates: t.List[t.Dict[str, str]],
        dataset: SingleMetricAnnotation,
        sample_size: int,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        parent_pbar: t.Optional[tqdm] = None,
    ) -> t.List[t.Dict[str, str]]:
        feedback_rm, feedback_grp = new_group(
            name="Feedback mutation",
            inputs={"candidates": candidates},
            callbacks=callbacks,
        )
        improved_candidates = []
        dataset = dataset.filter(lambda x: x["is_accepted"])
        sample_size = min(sample_size, len(dataset))
        exec = Executor(
            desc="Feedback Mutation",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
            pbar=parent_pbar,
        )

        for candidate in candidates:
            dataset_sample = dataset.sample(sample_size, stratify_key="metric_output")
            exec.submit(
                self._feedback_mutation,
                candidate=candidate,
                dataset=dataset_sample,
                callbacks=feedback_grp,
                raise_exceptions=raise_exceptions,
                batch_size=batch_size,
                run_config=run_config,
                parent_pbar=parent_pbar,
            )

        try:
            improved_candidates = exec.results()
        except Exception as e:
            feedback_rm.on_chain_error(e)
            raise e
        else:
            feedback_rm.on_chain_end(
                outputs={"improved_candidates": improved_candidates}
            )

        return improved_candidates

    async def _feedback_mutation(
        self,
        candidate: t.Dict[str, str],
        dataset: SingleMetricAnnotation,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        parent_pbar: t.Optional[tqdm] = None,
    ) -> t.Dict[str, str]:
        candidate_rm, candidate_grp = new_group(
            name="Candidate feedback mutation",
            inputs={"candidate": candidate},
            callbacks=callbacks,
        )
        batch, target = self._get_evaluation_dataset(dataset)
        results = await self.aevaluate_candidate(
            candidate=candidate,
            eval_dataset=batch,
            run_config=run_config,
            batch_size=batch_size,
            callbacks=candidate_grp,
            raise_exceptions=raise_exceptions,
            run_id=candidate_rm.run_id,
            parent_pbar=parent_pbar,
        )

        feedback_candidate = await self._get_feedbacks(
            candidate, dataset, results, target, candidate_grp
        )
        improved_candidate = await self._implement_feedbacks(
            candidate, feedback_candidate, candidate_grp
        )

        candidate_rm.on_chain_end(outputs={"improved_candidate": improved_candidate})
        return improved_candidate

    async def _implement_feedbacks(
        self,
        candidate: t.Dict[str, str],
        feedbacks: t.Dict[str, t.List[str]],
        callbacks: Callbacks = None,
    ) -> t.Dict[str, str]:
        improved_candidate = {}
        for key in candidate.keys():
            feedback = feedbacks[key]
            if feedback:
                feedback_input = FeedbackMutationPromptInput(
                    instruction=candidate[key], feedbacks=feedback
                )
                output = await self.feedback_mutation_prompt.generate(
                    data=feedback_input, llm=self._llm, callbacks=callbacks
                )
                improved_candidate[key] = output.instruction
            else:
                improved_candidate[key] = candidate[key]
                logger.warning(
                    f"No feedbacks found for the prompt {key}. Returning the original prompt."
                )

        return improved_candidate

    async def _get_feedbacks(
        self,
        candidate: t.Dict[str, str],
        dataset: SingleMetricAnnotation,
        results: EvaluationResult,
        target: t.List[float],
        callbacks: Callbacks = None,
    ) -> t.Dict[str, t.List[str]]:
        prediction = self._predictions(results)
        indices = [idx for idx in range(len(target)) if target[idx] != prediction[idx]]
        traces = [trace.get(self._metric.name, {}) for trace in results.traces]
        prompts = self._with_instructions(candidate)
        feedback_candidates: t.Dict[str, t.List[str]] = {
            prompt_name: [] for prompt_name in candidate.keys()
        }
        if not indices:
            logger.warning("No samples found for the feedback generation.")
            return feedback_candidates

        for prompt_name in candidate.keys():
            feedback_data = []
            for idx in indices:
                if idx >= len(traces):
                    continue
                io = _traced_prompt_io(traces[idx], prompts[prompt_name])
                annotation = dataset[idx]["prompts"].get(prompt_name)
                if io is None or annotation is None:
                    continue
                feedback_data.append(
                    FeedbackExample(
                        input=io[0],
                        output=io[1],
                        expected_output=annotation["edited_output"]
                        or annotation["prompt_output"],
                    )
                )
            if not feedback_data:
                # e.g. an LLM that doesn't trace its calls
                logger.warning(
                    "No traced calls of prompt %s to give feedback on.", prompt_name
                )
                continue
            prompt_input = FeedbackMutationInput(
                instruction=candidate[prompt_name], examples=feedback_data
            )
            feedbacks = await self.feedback_generation_prompt.generate(
                data=prompt_input, llm=self._llm, callbacks=callbacks
            )
            feedback_candidates[prompt_name] = feedbacks.feedbacks

        return feedback_candidates

    def _get_evaluation_dataset(
        self, dataset: SingleMetricAnnotation
    ) -> t.Tuple[EvaluationDataset, t.List[float]]:
        output_type = getattr(self._metric, "output_type", None)
        if output_type is None:
            raise ValueError("No output type provided for the metric.")

        training_ids = []
        y_true = []
        for idx, sample in enumerate(dataset):
            if sample["is_accepted"]:
                training_ids.append(idx)
                y_true.append(sample.metric_output)
            elif not sample["is_accepted"] and output_type.name == "BINARY":
                training_ids.append(idx)
                y_true.append(int(not sample.metric_output))

        dataset = dataset.select(training_ids)
        eval_dataset = dataset.to_evaluation_dataset()
        return eval_dataset, y_true

    async def aevaluate_candidate(
        self,
        *,
        candidate: t.Dict[str, str],
        eval_dataset: EvaluationDataset,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        run_id: t.Optional[UUID] = None,
        parent_pbar: t.Optional[tqdm] = None,
    ) -> EvaluationResult:
        """Score ``candidate`` on ``eval_dataset`` (a copy of the metric)."""
        # evaluate() without its deprecation warning, which would fire once
        # per candidate.
        from ragas.evaluation import _aevaluate_in_run_scope

        results = await _aevaluate_in_run_scope(
            dataset=eval_dataset,
            # typed for legacy metrics; runs collections metrics natively (#221)
            metrics=[t.cast(t.Any, self.candidate_metric(candidate))],
            llm=None,
            embeddings=None,
            experiment_name=None,
            run_config=run_config,
            token_usage_parser=None,
            raise_exceptions=raise_exceptions,
            column_map=None,
            show_progress=True,
            batch_size=batch_size,
            _run_id=run_id,
            _pbar=parent_pbar,
            _callbacks=callbacks,
            return_executor=False,
        )
        return t.cast(EvaluationResult, results)

    def evaluate_candidate(
        self,
        *,
        candidate: t.Dict[str, str],
        eval_dataset: EvaluationDataset,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        run_id: t.Optional[UUID] = None,
        parent_pbar: t.Optional[tqdm] = None,
    ) -> EvaluationResult:
        """Sync ``aevaluate_candidate``."""
        return run_sync(
            self.aevaluate_candidate(
                candidate=candidate,
                eval_dataset=eval_dataset,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=callbacks,
                raise_exceptions=raise_exceptions,
                run_id=run_id,
                parent_pbar=parent_pbar,
            )
        )

    def _predictions(self, results: EvaluationResult) -> t.List[float]:
        """The metric's score for each sample in ``results``."""
        values = results.to_pandas()[self._metric.name].values
        y_pred = values.tolist() if isinstance(values, np.ndarray) else [values]
        return t.cast(t.List[float], y_pred)

    def _predict(
        self,
        candidate: t.Dict[str, str],
        eval_dataset: EvaluationDataset,
        *,
        run_config: t.Optional[RunConfig],
        batch_size: t.Optional[int],
        callbacks: t.Optional[Callbacks],
        raise_exceptions: bool,
        run_id: t.Optional[UUID],
        parent_pbar: t.Optional[tqdm],
    ) -> t.List[float]:
        """``candidate``'s score for each sample of ``eval_dataset``."""
        results = self.evaluate_candidate(
            candidate=candidate,
            eval_dataset=eval_dataset,
            run_config=run_config,
            batch_size=batch_size,
            callbacks=callbacks,
            raise_exceptions=raise_exceptions,
            run_id=run_id,
            parent_pbar=parent_pbar,
        )
        return self._predictions(results)

    def evaluate_fitness(
        self,
        *,
        candidates: t.List[t.Dict[str, str]],
        dataset: SingleMetricAnnotation,
        loss_fn: Loss,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        parent_pbar: t.Optional[tqdm] = None,
    ) -> t.List[float]:
        fitness_scores = []

        eval_dataset, y_true = self._get_evaluation_dataset(dataset)

        initialize_population_rm, initialize_population_grp = new_group(
            name="Evaluating candidate fitness",
            inputs={"candidates": candidates},
            callbacks=callbacks,
        )
        run_id = initialize_population_rm.run_id
        for candidate in candidates:
            y_pred = self._predict(
                candidate,
                eval_dataset,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=initialize_population_grp,
                raise_exceptions=raise_exceptions,
                run_id=run_id,
                parent_pbar=parent_pbar,
            )
            # Loss takes (predicted, actual). fitness() negates lower-is-better
            # losses, so a larger value is always better.
            fitness_scores.append(loss_fn.fitness(y_pred, y_true))

        initialize_population_rm.on_chain_end(
            outputs={"fitness_scores": fitness_scores}
        )

        return fitness_scores

    async def _cross_over_chain(
        self,
        parent_x: t.Dict[str, str],
        parent_y: t.Dict[str, str],
        callbacks: Callbacks,
    ):
        if parent_x.keys() != parent_y.keys():
            raise ValueError("The parents must have the same prompt names.")

        chain_offsprings = {}
        for key in parent_x.keys():
            offspring = await self._cross_over_prompts(
                parent_x[key], parent_y[key], callbacks
            )
            chain_offsprings[key] = offspring

        return chain_offsprings

    def cross_over_mutation(
        self,
        *,
        candidates: t.List[t.Dict[str, str]],
        dataset: SingleMetricAnnotation,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        raise_exceptions: bool = True,
        parent_pbar: t.Optional[tqdm] = None,
    ):
        eval_dataset, y_true = self._get_evaluation_dataset(dataset)

        cross_over_rm, cross_over_grp = new_group(
            name="Cross-over mutation",
            inputs={"candidates": candidates},
            callbacks=callbacks,
        )
        run_id = cross_over_rm.run_id
        prediction_vectors = []
        for candidate in candidates:
            y_pred = self._predict(
                candidate,
                eval_dataset,
                run_config=run_config,
                batch_size=batch_size,
                callbacks=cross_over_grp,
                raise_exceptions=raise_exceptions,
                run_id=run_id,
                parent_pbar=parent_pbar,
            )
            prediction = [int(pred == true) for pred, true in zip(y_pred, y_true)]
            prediction_vectors.append(prediction)

        prediction_vectors = np.array(prediction_vectors)
        distance_matrix = hamming_distance(prediction_vectors).astype(float)
        if len(candidates) > 1:
            # A candidate must not be paired with itself.
            np.fill_diagonal(distance_matrix, np.inf)

        exec = Executor(
            desc="Mutating candidates",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
            pbar=parent_pbar,
        )

        offspring_candidates = []
        for idx, candidate in enumerate(candidates):
            parent_x = candidates[idx]
            parent_y = candidates[np.argmin(distance_matrix[idx])]
            exec.submit(
                self._cross_over_chain,
                parent_x=parent_x,
                parent_y=parent_y,
                callbacks=cross_over_grp,
            )

        try:
            offspring_candidates = exec.results()
        except Exception as e:
            cross_over_rm.on_chain_error(e)
            raise e
        else:
            cross_over_rm.on_chain_end(
                outputs={"offspring_candidates": offspring_candidates}
            )

        return offspring_candidates
