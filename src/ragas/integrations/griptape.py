import typing as t

from ragas.backends.base import BaseBackend
from ragas.dataset import Dataset
from ragas.dataset_schema import SingleTurnSample

try:
    from griptape.engines.rag import RagContext  # type: ignore
except ImportError:
    raise ImportError(
        "griptape is not installed. Please install it using `pip install griptape` "
        "to use the Griptape integration."
    )


def transform_to_ragas_dataset(
    grip_tape_rag_contexts: t.List[RagContext],  # type: ignore
    reference_contexts: t.Optional[t.List[str]] = None,
    references: t.Optional[t.List[str]] = None,
    rubrics: t.Optional[t.List[t.Dict[str, str]]] = None,
    *,
    name: str = "griptape",
    backend: t.Union[BaseBackend, str] = "inmemory",
    **backend_kwargs: t.Any,
) -> Dataset:
    """Griptape RAG contexts as a ``Dataset`` of ``SingleTurnSample`` rows.

    ``name``, ``backend`` and ``backend_kwargs`` make the ``Dataset``
    (in memory by default; call ``save()`` to write it). Score it with
    ``metrics_experiment``.
    """
    # Collect only the non-None lists
    provided_lists = {
        "grip_tape_rag_context": grip_tape_rag_contexts or [],
        "reference_contexts": reference_contexts or [],
        "references": references or [],
        "rubrics": rubrics or [],
    }

    # Find the maximum length among provided lists
    max_len = max(len(lst) for lst in provided_lists.values())

    # Ensure all provided lists have the same length
    for key, lst in provided_lists.items():
        if lst and len(lst) != max_len:
            raise ValueError(
                f"Inconsistent length for {key}: expected {max_len}, got {len(lst)}"
            )

    # Create samples while handling missing values
    samples = []
    for i in range(max_len):
        sample = {
            "user_input": grip_tape_rag_contexts[i].query,
            "retrieved_contexts": (
                [
                    rag_context.to_text() if rag_context else ""
                    for rag_context in grip_tape_rag_contexts[i].text_chunks
                ]
            ),
            "reference_contexts": reference_contexts[i] if reference_contexts else None,
            "response": (
                "\n".join(
                    o.to_text() if o else "" for o in grip_tape_rag_contexts[i].outputs
                )
                if grip_tape_rag_contexts
                else None
            ),
            "reference": references[i] if references else None,
            "rubrics": rubrics[i] if rubrics else None,
        }
        samples.append(sample)

    return Dataset(
        name,
        backend,
        data_model=SingleTurnSample,
        data=[SingleTurnSample(**sample) for sample in samples],
        **backend_kwargs,
    )
