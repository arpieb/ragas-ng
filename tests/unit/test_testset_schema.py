import pytest

from ragas.dataset import Dataset
from ragas.dataset_schema import (
    HumanMessage,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.testset.synthesizers.testset_schema import (
    Testset as RagasTestset,
    TestsetSample as RagasTestsetSample,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


def _testset(eval_sample) -> RagasTestset:
    sample = RagasTestsetSample(eval_sample=eval_sample, synthesizer_name="test")
    return RagasTestset(samples=[sample, sample])


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_to_dataset(eval_sample):
    dataset = _testset(eval_sample).to_dataset("ts")
    assert isinstance(dataset, Dataset)
    assert dataset.name == "ts"
    assert dataset.data_model is type(eval_sample)
    assert list(dataset) == [eval_sample, eval_sample]
    # rows are copies: changing one leaves the testset alone
    assert dataset[0] is not eval_sample


@pytest.mark.parametrize("backend", ["local/csv", "local/jsonl"])
@pytest.mark.parametrize("eval_sample", samples)
def test_testset_to_dataset_round_trips_through_a_file_backend(
    tmp_path, backend, eval_sample
):
    dataset = _testset(eval_sample).to_dataset("ts", backend, root_dir=str(tmp_path))
    dataset.save()
    loaded = Dataset.load(
        "ts", backend, data_model=type(eval_sample), root_dir=str(tmp_path)
    )
    if backend == "local/jsonl":
        assert list(loaded) == [eval_sample, eval_sample]
    else:
        # CSV can't tell None from "" in a str field (ragas.backends.flat)
        def blank_to_none(sample):
            return {k: (None if v == "" else v) for k, v in sample.model_dump().items()}

        assert [blank_to_none(s) for s in loaded] == [blank_to_none(eval_sample)] * 2


def test_testset_to_dataset_rejects_mixed_sample_types():
    testset = RagasTestset(
        samples=[
            RagasTestsetSample(eval_sample=s, synthesizer_name="test") for s in samples
        ]
    )
    with pytest.raises(ValueError, match="mixes SingleTurnSample and MultiTurnSample"):
        testset.to_dataset("ts")


def test_testset_has_no_evaluation_dataset_converters():
    for name in ("to_evaluation_dataset", "to_hf_dataset", "from_hf_dataset"):
        assert not hasattr(RagasTestset, name)


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_csv(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    testset.to_csv(tmpdir / "csvfile.csv")


@pytest.mark.parametrize("eval_sample", samples)
def test_testset_save_load_jsonl(tmpdir, eval_sample):
    testset_sample = RagasTestsetSample(
        eval_sample=eval_sample, synthesizer_name="test"
    )
    testset = RagasTestset(samples=[testset_sample, testset_sample])
    testset.to_jsonl(tmpdir / "jsonlfile.jsonl")
    loaded_testset = RagasTestset.from_jsonl(tmpdir / "jsonlfile.jsonl")
    assert loaded_testset == testset
