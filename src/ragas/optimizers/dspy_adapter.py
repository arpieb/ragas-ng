"""Conversions between ragas prompts, annotations and LLMs and their DSPy forms."""

from __future__ import annotations

import logging
import typing as t

from pydantic import BaseModel, ValidationError

from ragas.dataset_schema import SingleMetricAnnotation
from ragas.losses import Loss

if t.TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.prompt.metrics.base_prompt import BasePrompt

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "DSPy optimizer requires dspy-ai. Install with:\n"
    "  uv add 'ragas[dspy]'  # or: pip install 'ragas[dspy]'\n"
)


def _import_dspy() -> t.Any:
    try:
        import dspy
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return dspy


def _router_of(llm: t.Any) -> t.Optional[t.Any]:
    """The ``litellm.Router`` an ``llm_factory()`` LLM calls through, if any."""
    fn = getattr(getattr(llm, "client", None), "create_fn", None)
    for _ in range(10):  # wrappers: instructor's patch, rate limit, partial
        if fn is None:
            return None
        owner = getattr(fn, "__self__", None)
        if owner is not None and hasattr(owner, "model_list"):
            return owner
        fn = getattr(fn, "__wrapped__", None) or getattr(fn, "func", None)
    return None


def dspy_lm_from_llm(dspy: t.Any, llm: "InstructorBaseRagasLLM") -> t.Any:
    """A ``dspy.LM`` calling the same LiteLLM model as ``llm``.

    DSPy calls models through LiteLLM itself, and every ``llm_factory()`` LLM
    is a LiteLLM model name plus call arguments, so the LM is built from
    those: the model, the arguments you passed to ``llm_factory`` (such as
    ``api_key``, ``api_base``, ``temperature``) and, for an ``OpenAI`` client,
    its API key and base URL. DSPy's own cache and retries apply to its calls,
    not ragas' (nor ``rate_limit``).

    Raises:
        ValueError: for an LLM this can't express as a ``dspy.LM`` (one not
            built by ``llm_factory()``, or over a ``litellm.Router`` or Azure
            client). Pass ``DSPyOptimizer(lm=dspy.LM(...))`` for those.
    """
    base_lm = getattr(dspy, "BaseLM", None)
    if base_lm is not None and isinstance(llm, base_lm):
        return llm

    hint = " Pass DSPyOptimizer(lm=dspy.LM(...)) to choose DSPy's model yourself."
    if not getattr(llm, "_litellm", False):
        raise ValueError(
            "DSPyOptimizer runs DSPy through LiteLLM, so it needs an "
            f"llm_factory() LLM; got {type(llm).__name__}.{hint}"
        )
    if _router_of(llm) is not None:
        raise ValueError(
            f"DSPyOptimizer can't route DSPy through a litellm.Router.{hint}"
        )

    model_args = getattr(llm, "model_args", {}) or {}
    explicit = getattr(llm, "_explicit_args", set()) or set()
    # Only what the caller chose: ragas' deterministic defaults (temperature
    # 0.01, top_p 0.1) would stop MIPROv2 from sampling varied proposals.
    kwargs = {k: model_args[k] for k in explicit if k in model_args}

    client = getattr(llm, "source_client", None)
    if client is not None:
        if getattr(llm, "provider", None) == "azure":
            raise ValueError(
                f"DSPyOptimizer can't take credentials from an Azure client.{hint}"
            )
        if getattr(client, "api_key", None):
            kwargs.setdefault("api_key", client.api_key)
        if getattr(client, "base_url", None):
            kwargs.setdefault("api_base", str(client.base_url))

    return dspy.LM(llm.model, **kwargs)  # type: ignore[attr-defined]


def setup_dspy_llm(dspy: t.Any, llm: t.Any) -> None:
    """Configure DSPy to call the model ``llm`` calls (see ``dspy_lm_from_llm``).

    ``llm`` may also be a DSPy LM, used as it is.
    """
    dspy.settings.configure(lm=dspy_lm_from_llm(dspy, llm))


def prompt_to_dspy_signature(prompt: "BasePrompt") -> t.Type[t.Any]:
    """
    Convert a ragas prompt to a DSPy Signature.

    The fields are the prompt's input and output model fields, with their
    types and descriptions; the instruction becomes the signature's
    instructions.

    Parameters
    ----------
    prompt : BasePrompt
        The ragas prompt to convert.

    Returns
    -------
    Type[dspy.Signature]
        A DSPy Signature class.
    """
    dspy = _import_dspy()

    fields: t.Dict[str, t.Any] = {}
    annotations: t.Dict[str, t.Any] = {}

    for name, field_info in prompt.input_model.model_fields.items():
        fields[name] = dspy.InputField(desc=field_info.description or "")
        annotations[name] = field_info.annotation or str

    for name, field_info in prompt.output_model.model_fields.items():
        fields[name] = dspy.OutputField(desc=field_info.description or "")
        annotations[name] = field_info.annotation or str

    signature_class = type(
        f"{prompt.__class__.__name__}Signature",
        (dspy.Signature,),
        {"__doc__": prompt.instruction, "__annotations__": annotations, **fields},
    )

    return signature_class


def ragas_dataset_to_dspy_examples(
    dataset: SingleMetricAnnotation,
    prompt_name: str,
) -> t.List[t.Any]:
    """
    Convert Ragas annotated dataset to DSPy examples.

    Parameters
    ----------
    dataset : SingleMetricAnnotation
        The annotated dataset with ground truth scores.
    prompt_name : str
        The name of the prompt to extract examples for.

    Returns
    -------
    List[dspy.Example]
        List of DSPy examples for training.
    """
    dspy = _import_dspy()

    examples = []

    for sample in dataset:
        if not sample["is_accepted"]:
            continue

        prompt_data = sample["prompts"].get(prompt_name)

        if prompt_data is None:
            continue

        prompt_input = prompt_data["prompt_input"]
        prompt_output = (
            prompt_data["edited_output"]
            if prompt_data["edited_output"]
            else prompt_data["prompt_output"]
        )

        example_dict = {**prompt_input}
        if isinstance(prompt_output, dict):
            example_dict.update(prompt_output)
        else:
            example_dict["output"] = prompt_output

        input_keys = list(prompt_input.keys())
        example = dspy.Example(**example_dict).with_inputs(*input_keys)
        examples.append(example)

    return examples


def dspy_demos_to_examples(
    prompt: "BasePrompt", demos: t.Iterable[t.Any]
) -> t.List[t.Tuple[BaseModel, BaseModel]]:
    """DSPy demonstrations as ``(input, output)`` examples of ``prompt``.

    A demo that doesn't validate against the prompt's input and output
    models is skipped.
    """
    examples = []
    for demo in demos:
        if hasattr(demo, "toDict"):
            data = demo.toDict()
        elif isinstance(demo, dict):
            data = demo
        else:
            data = dict(getattr(demo, "__dict__", {}))
        try:
            example_input = prompt.input_model.model_validate(
                {k: data[k] for k in prompt.input_model.model_fields if k in data}
            )
            example_output = prompt.output_model.model_validate(
                {k: data[k] for k in prompt.output_model.model_fields if k in data}
            )
        except ValidationError as e:
            logger.debug("Skipping a DSPy demo that doesn't fit %s: %s", prompt.name, e)
            continue
        examples.append((example_input, example_output))
    return examples


def create_dspy_metric(
    loss: Loss, metric_name: str
) -> t.Callable[[t.Any, t.Any], float]:
    """
    Convert Ragas Loss function to DSPy metric.

    DSPy expects a metric function with signature: metric(example, prediction) -> float
    where higher is better.

    Parameters
    ----------
    loss : Loss
        The Ragas loss function.
    metric_name : str
        Name of the metric being optimized.

    Returns
    -------
    Callable[[Any, Any], float]
        A DSPy-compatible metric function.
    """

    def dspy_metric(example: t.Any, prediction: t.Any, trace: t.Any = None) -> float:
        ground_truth = getattr(example, metric_name, None)
        predicted = getattr(prediction, metric_name, None)

        if ground_truth is None or predicted is None:
            return 0.0

        # DSPy maximises its metric, and Loss.fitness is higher-is-better.
        return loss.fitness([predicted], [ground_truth])

    return dspy_metric
