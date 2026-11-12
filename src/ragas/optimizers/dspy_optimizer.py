from __future__ import annotations

import hashlib
import json
import logging
import sys
import typing as t
from dataclasses import dataclass, field

from ragas.cache import CacheInterface
from ragas.losses import Loss
from ragas.optimizers.annotations import SingleMetricAnnotation
from ragas.optimizers.base import Optimizer
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks
    from ragas.prompt.metrics.base_prompt import BasePrompt
logger = logging.getLogger(__name__)

_SCORE_TYPES = (int, float, bool)


def _score_field(prompt: "BasePrompt", metric_name: str) -> str:
    """The output field DSPy's metric compares: one named after the metric,
    else the prompt's first numeric output field (a verdict or score)."""
    fields = prompt.output_model.model_fields
    if metric_name in fields:
        return metric_name
    for name, info in fields.items():
        if info.annotation in _SCORE_TYPES:
            return name
    return metric_name


@dataclass
class DSPyOptimizer(Optimizer):
    """
    Advanced prompt optimizer using DSPy's MIPROv2.

    MIPROv2 performs sophisticated prompt optimization by combining:
    - Instruction optimization (prompt engineering)
    - Demonstration optimization (few-shot examples)
    - Combined search over both spaces

    Requires: pip install 'ragas-ng[dspy]' (DSPy 3 with optuna)

    Each of the metric's prompts is compiled on its own annotations. The
    result is a copy of the prompt with MIPROv2's instruction and its chosen
    demonstrations as examples. DSPy calls the model ``llm`` calls, through
    its own LiteLLM ``dspy.LM`` (see ``dspy_adapter.dspy_lm_from_llm``), or
    ``lm`` when given.

    Parameters
    ----------
    lm : Any, optional
        A DSPy LM to optimize with, instead of one built from ``llm``. Use it
        for an LLM over a ``litellm.Router`` or Azure client.
    num_candidates : int
        Number of prompt variants to try during optimization.
    max_bootstrapped_demos : int
        Maximum number of auto-generated examples to use.
    max_labeled_demos : int
        Maximum number of human-annotated examples to use.
    init_temperature : float
        Exploration temperature for optimization.
    auto : str, optional
        Automatic configuration level: 'light', 'medium', or 'heavy'.
        Controls the depth of optimization search.
    num_threads : int, optional
        Number of parallel threads for optimization.
    max_errors : int, optional
        Maximum errors tolerated during optimization before stopping. When
        unset, ``optimize()``'s ``raise_exceptions`` decides: DSPy's own
        limit (``dspy.settings.max_errors``, 10 by default) when True, or
        never stop on errors when False.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Enable verbose logging during optimization.
    track_stats : bool
        Track and report optimization statistics.
    log_dir : str, optional
        Directory for saving optimization logs and progress.
    metric_threshold : float, optional
        Minimum acceptable metric value to achieve.
    cache : CacheInterface, optional
        Cache backend for storing optimization results.
    """

    num_candidates: int = 10
    max_bootstrapped_demos: int = 5
    max_labeled_demos: int = 5
    init_temperature: float = 1.0
    auto: t.Optional[t.Literal["light", "medium", "heavy"]] = "light"
    num_threads: t.Optional[int] = None
    max_errors: t.Optional[int] = None
    seed: int = 9
    verbose: bool = False
    track_stats: bool = True
    log_dir: t.Optional[str] = None
    metric_threshold: t.Optional[float] = None
    cache: t.Optional[CacheInterface] = field(default=None, repr=False)
    lm: t.Optional[t.Any] = field(default=None, repr=False)
    _dspy: t.Optional[t.Any] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            import dspy

            self._dspy = dspy
        except ImportError as e:
            raise ImportError(
                "DSPy optimizer requires dspy. Install with:\n"
                "  uv add 'ragas-ng[dspy]'  # or: pip install 'ragas-ng[dspy]'\n"
            ) from e

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate optimizer parameters."""
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")

        if self.max_bootstrapped_demos < 0:
            raise ValueError("max_bootstrapped_demos must be non-negative")

        if self.max_labeled_demos < 0:
            raise ValueError("max_labeled_demos must be non-negative")

        if self.init_temperature <= 0:
            raise ValueError("init_temperature must be positive")

        if self.auto not in ["light", "medium", "heavy", None]:
            raise ValueError("auto must be 'light', 'medium', 'heavy', or None")

        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive if specified")

        if self.max_errors is not None and self.max_errors < 0:
            raise ValueError("max_errors must be non-negative if specified")

        if self.metric_threshold is not None and (
            self.metric_threshold < 0 or self.metric_threshold > 1
        ):
            raise ValueError("metric_threshold must be between 0 and 1")

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
        """
        Optimize metric prompts using DSPy MIPROv2.

        Steps, for each prompt the dataset annotates:

        1. Convert the prompt to a DSPy Signature
        2. Create DSPy Module with signature
        3. Convert the prompt's annotations to DSPy Examples
        4. Run MIPROv2 optimization
        5. Extract the optimized instruction and demonstrations
        6. Return a copy of the prompt with them

        Parameters
        ----------
        dataset : SingleMetricAnnotation
            Annotated dataset with ground truth scores.
        loss : Loss
            Loss function to optimize.
        config : Mapping[str, Any], optional
            Additional settings: ``num_trials`` (without ``auto``).
        run_config : RunConfig, optional
            ``max_retries`` and ``timeout`` for the ``dspy.LM`` built from
            ``llm`` (its ``num_retries`` and ``timeout``). A DSPy LM passed as
            ``lm`` keeps its own settings.
        batch_size : None
            DSPy sizes its own minibatches; any other value raises
            ``NotImplementedError``.
        callbacks : None
            DSPy traces its own calls; ragas callbacks raise
            ``NotImplementedError``.
        raise_exceptions : bool
            Without ``max_errors``, True (the default) keeps DSPy's own error
            limit (``dspy.settings.max_errors``, 10 by default), so a few bad
            replies don't end a long, paid run; False never stops on errors.
            DSPy logs each error; at the limit it raises.

        Returns
        -------
        Dict[str, BasePrompt]
            For each annotated prompt, a copy of the metric's prompt with the
            optimized instruction and demonstrations as its examples.
        """
        if self.metric is None:
            raise ValueError("No metric provided for optimization.")

        if self.llm is None and self.lm is None:
            raise ValueError("No llm provided for optimization.")

        if self._dspy is None:
            raise RuntimeError("DSPy module not loaded.")

        if batch_size is not None:
            raise NotImplementedError(
                "DSPyOptimizer can't honor batch_size: MIPROv2 sizes its own "
                "minibatches. Leave it unset."
            )
        if callbacks:
            raise NotImplementedError(
                "DSPyOptimizer can't honor ragas callbacks: DSPy runs and "
                "traces its own calls. Leave callbacks unset."
            )
        # None leaves DSPy's own limit (dspy.settings.max_errors) in place.
        max_errors = self.max_errors
        if max_errors is None and not raise_exceptions:
            max_errors = sys.maxsize

        metric = self.metric
        config = dict(config or {})
        if self.cache is not None:
            cache_key = self._generate_cache_key(
                dataset, loss, config, max_errors=max_errors
            )
            if self.cache.has_key(cache_key):
                logger.info(f"Cache hit for DSPy optimization of metric: {metric.name}")
                return self._from_cacheable(self.cache.get(cache_key))

        logger.info(f"Starting DSPy optimization for metric: {metric.name}")

        from ragas.optimizers.dspy_adapter import (
            create_dspy_metric,
            dspy_demos_to_examples,
            prompt_to_dspy_signature,
            ragas_dataset_to_dspy_examples,
            setup_dspy_llm,
        )

        setup_dspy_llm(
            self._dspy, self.lm if self.lm is not None else self.llm, run_config
        )

        prompts = metric.get_prompts()  # copies
        optimized_prompts: t.Dict[str, BasePrompt] = {}

        for prompt_name, prompt in prompts.items():
            examples = ragas_dataset_to_dspy_examples(dataset, prompt_name)
            if not examples:
                logger.info(f"No annotations for prompt {prompt_name}; skipped.")
                continue
            logger.info(f"Optimizing prompt: {prompt_name}")

            signature = prompt_to_dspy_signature(prompt)
            module = self._dspy.Predict(signature)

            metric_fn = create_dspy_metric(loss, _score_field(prompt, dataset.name))

            # MIPROv2 takes the metric here, not in compile(). With an auto
            # level it sizes the search itself and rejects num_candidates;
            # without one it needs num_trials.
            teleprompter = self._dspy.MIPROv2(
                metric=metric_fn,
                num_candidates=None if self.auto else self.num_candidates,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                init_temperature=self.init_temperature,
                auto=self.auto,
                num_threads=self.num_threads,
                max_errors=max_errors,
                seed=self.seed,
                verbose=self.verbose,
                track_stats=self.track_stats,
                log_dir=self.log_dir,
                metric_threshold=self.metric_threshold,
            )

            compile_kwargs: t.Dict[str, t.Any] = {"requires_permission_to_run": False}
            if not self.auto:
                compile_kwargs["num_trials"] = config.get(
                    "num_trials", self.num_candidates
                )
            optimized = teleprompter.compile(
                module,
                trainset=examples,
                **compile_kwargs,
            )

            optimized_instruction = self._extract_instruction(optimized)
            if optimized_instruction:
                prompt.instruction = optimized_instruction
            demos = getattr(optimized, "demos", None)
            if isinstance(demos, (list, tuple)) and demos:
                prompt.examples = dspy_demos_to_examples(prompt, demos)
            optimized_prompts[prompt_name] = prompt

            logger.info(
                f"Optimized prompt for {prompt_name}: {prompt.instruction[:100]}..."
            )

        if self.cache is not None:
            cache_key = self._generate_cache_key(
                dataset, loss, config, max_errors=max_errors
            )
            self.cache.set(cache_key, self._to_cacheable(optimized_prompts))
            logger.info("Cached optimization results")

        return optimized_prompts

    @staticmethod
    def _to_cacheable(prompts: t.Mapping[str, BasePrompt]) -> t.Dict[str, t.Any]:
        return {
            name: {
                "instruction": prompt.instruction,
                "examples": [
                    [inp.model_dump(), out.model_dump()] for inp, out in prompt.examples
                ],
            }
            for name, prompt in prompts.items()
        }

    def _from_cacheable(self, cached: t.Mapping[str, t.Any]) -> t.Dict[str, BasePrompt]:
        """Cached results as copies of the metric's prompts."""
        prompts = self._metric.get_prompts()
        result: t.Dict[str, BasePrompt] = {}
        for name, entry in cached.items():
            prompt = prompts[name]
            prompt.instruction = entry["instruction"]
            prompt.examples = [
                (
                    prompt.input_model.model_validate(inp),
                    prompt.output_model.model_validate(out),
                )
                for inp, out in entry["examples"]
            ]
            result[name] = prompt
        return result

    def _extract_instruction(self, optimized_module: t.Any) -> str:
        """
        Extract the optimized instruction from DSPy module.

        Parameters
        ----------
        optimized_module : Any
            The optimized DSPy module from MIPROv2.

        Returns
        -------
        str
            The optimized instruction string.
        """
        if hasattr(optimized_module, "signature"):
            sig = optimized_module.signature
            if hasattr(sig, "instructions"):
                return sig.instructions
            elif hasattr(sig, "__doc__"):
                return sig.__doc__ or ""

        if hasattr(optimized_module, "extended_signature"):
            return str(optimized_module.extended_signature)

        return ""

    def _generate_cache_key(
        self,
        dataset: SingleMetricAnnotation,
        loss: Loss,
        config: t.Dict[t.Any, t.Any],
        max_errors: t.Optional[int] = None,
    ) -> str:
        """
        Generate a unique cache key for optimization results.

        Parameters
        ----------
        dataset : SingleMetricAnnotation
            Annotated dataset with ground truth scores.
        loss : Loss
            Loss function to optimize.
        config : Dict[Any, Any]
            Additional configuration parameters.
        max_errors : int, optional
            The error limit the run uses (default: the optimizer's).

        Returns
        -------
        str
            SHA256 hash of the optimization parameters.
        """
        if self.metric is None:
            raise ValueError("Metric must be set to generate cache key")

        cache_data = {
            "metric_name": self.metric.name,
            # the prompts being optimized: a changed definition or instruction
            # must not hit an entry made from the old one
            "prompts": {
                name: str(hash(prompt))
                for name, prompt in self.metric.get_prompts().items()
            },
            "dataset_hash": hashlib.sha256(
                json.dumps(dataset.model_dump(), sort_keys=True).encode()
            ).hexdigest(),
            "loss_name": loss.__class__.__name__,
            "num_candidates": self.num_candidates,
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
            "max_labeled_demos": self.max_labeled_demos,
            "init_temperature": self.init_temperature,
            "auto": self.auto,
            "num_threads": self.num_threads,
            "max_errors": self.max_errors if max_errors is None else max_errors,
            "seed": self.seed,
            "verbose": self.verbose,
            "track_stats": self.track_stats,
            "log_dir": self.log_dir,
            "metric_threshold": self.metric_threshold,
            "config": config,
        }

        key_string = json.dumps(cache_data, sort_keys=True, default=str)
        cache_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()
        return cache_key
