import typing as t

import pytest

from ragas.dataset_schema import (
    HumanMessage,
    MultiTurnSample,
    SingleTurnSample,
    sample_type_for,
)
from ragas.optimizers.annotations import (
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


def create_sample_annotation(metric_output):
    return SampleAnnotation(
        metric_input={
            "response": "",
            "reference": "",
            "user_input": "",
        },
        metric_output=metric_output,
        prompts={
            "single_turn_aspect_critic_prompt": PromptAnnotation(
                prompt_input={
                    "response": "",
                    "reference": "",
                    "user_input": "",
                },
                prompt_output={"reason": "", "verdict": 1},
                edited_output=None,
            )
        },
        is_accepted=True,
        target=None,
    )


def test_loader_sample():
    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    test_dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    sample = test_dataset.sample(2)
    assert len(sample) == 2

    sample = test_dataset.sample(2, stratify_key="metric_output")
    assert len(sample) == 2
    assert sum(item["metric_output"] for item in sample) == 1


def test_loader_batch():
    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    batches = dataset.batch(batch_size=2)
    assert all([len(item) == 2 for item in batches])

    batches = dataset.stratified_batches(batch_size=2, stratify_key="metric_output")
    assert all(sum([item["metric_output"] for item in batch]) == 1 for batch in batches)


def test_base_eval_sample():
    from ragas.dataset_schema import BaseSample

    class FakeSample(BaseSample):
        user_input: str
        response: str
        reference: t.Optional[str] = None

    fake_sample = FakeSample(user_input="What is X", response="Y")
    assert fake_sample.to_dict() == {"user_input": "What is X", "response": "Y"}
    assert fake_sample.get_features() == ["user_input", "response"]


def test_multiturn_sample_validate_user_input_invalid_type():
    """Test that MultiTurnSample validation correctly rejects invalid message types."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Hello"),
                "invalid_string",  # This should be rejected by Pydantic
            ]
        )


def test_multiturn_sample_validate_user_input_valid_types():
    """Test that MultiTurnSample validation accepts valid message types."""
    from ragas.messages import AIMessage

    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
    )
    assert len(sample.user_input) == 2
    assert isinstance(sample.user_input[0], HumanMessage)
    assert isinstance(sample.user_input[1], AIMessage)


def test_sample_type_for_dispatches_on_user_input():
    assert sample_type_for({"user_input": "q"}) is SingleTurnSample
    assert sample_type_for({"response": "a"}) is SingleTurnSample
    messages = [{"content": "hi", "type": "human"}]
    assert sample_type_for({"user_input": messages}) is MultiTurnSample


@pytest.mark.parametrize("backend", ["local/jsonl", "local/csv"])
@pytest.mark.parametrize("eval_sample", samples)
def test_samples_round_trip_through_a_dataset_backend(tmp_path, backend, eval_sample):
    """What EvaluationDataset.to_jsonl/to_csv covered, through Dataset (#258)."""
    from ragas.dataset import Dataset

    saved = Dataset(
        "ds",
        backend,
        data_model=type(eval_sample),
        data=[eval_sample, eval_sample],
        root_dir=str(tmp_path),
    )
    saved.save()
    loaded = Dataset.load(
        "ds", backend, data_model=type(eval_sample), root_dir=str(tmp_path)
    )
    assert len(loaded) == 2
    assert loaded[0].user_input == eval_sample.user_input


def test_single_turn_sample_metadata_round_trips_through_jsonl(tmp_path):
    from ragas.dataset import Dataset

    sample = SingleTurnSample(
        user_input="Q",
        response="A",
        reference_contexts=["ctx"],
        persona_name="Researcher",
        query_style="FORMAL",
        query_length="SHORT",
    )
    Dataset(
        "ds",
        "local/jsonl",
        data_model=SingleTurnSample,
        data=[sample],
        root_dir=str(tmp_path),
    ).save()
    (loaded,) = Dataset.load(
        "ds", "local/jsonl", data_model=SingleTurnSample, root_dir=str(tmp_path)
    )
    assert loaded == sample
