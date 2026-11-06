"""A backend named in @experiment / arun() gets its constructor arguments (#165).

Before #165 a name was created with no arguments, so only "inmemory" worked:
backend="local/jsonl" raised TypeError for the missing root_dir.
"""

import pytest
from pydantic import BaseModel

from ragas.backends import InMemoryBackend, LocalJSONLBackend
from ragas.dataset import Dataset
from ragas.experiment import experiment


class Row(BaseModel):
    q: str


def _dataset():
    return Dataset(name="qa", backend=InMemoryBackend(), data=[{"q": "a"}, {"q": "b"}])


def _echo():
    @experiment(Row)
    async def run(row):
        return Row(q=row["q"])

    return run


@pytest.mark.asyncio
async def test_arun_backend_name_with_kwargs(tmp_path):
    result = await _echo().arun(
        _dataset(),
        name="exp",
        backend="local/jsonl",
        backend_kwargs={"root_dir": str(tmp_path)},
    )
    assert isinstance(result.backend, LocalJSONLBackend)
    assert (tmp_path / "experiments" / "exp.jsonl").exists()
    assert len(result) == 2


@pytest.mark.asyncio
async def test_decorator_backend_name_with_kwargs(tmp_path):
    @experiment(Row, backend="local/csv", backend_kwargs={"root_dir": str(tmp_path)})
    async def run(row):
        return Row(q=row["q"])

    await run.arun(_dataset(), name="exp")
    assert (tmp_path / "experiments" / "exp.csv").exists()


@pytest.mark.asyncio
async def test_arun_kwargs_override_decorator_kwargs(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"

    @experiment(Row, backend="local/jsonl", backend_kwargs={"root_dir": str(first)})
    async def run(row):
        return Row(q=row["q"])

    await run.arun(_dataset(), name="exp", backend_kwargs={"root_dir": str(second)})
    assert (second / "experiments" / "exp.jsonl").exists()
    assert not (first / "experiments" / "exp.jsonl").exists()


@pytest.mark.asyncio
async def test_missing_constructor_args_error_says_how_to_pass_them():
    with pytest.raises(TypeError, match="backend_kwargs"):
        await _echo().arun(_dataset(), name="exp", backend="local/jsonl")


@pytest.mark.asyncio
async def test_backend_kwargs_with_an_instance_raises(tmp_path):
    with pytest.raises(ValueError, match="by name"):
        await _echo().arun(
            _dataset(),
            backend=LocalJSONLBackend(root_dir=str(tmp_path)),
            backend_kwargs={"root_dir": str(tmp_path)},
        )
    with pytest.raises(ValueError, match="by name"):
        experiment(Row, backend=InMemoryBackend(), backend_kwargs={"x": 1})


@pytest.mark.asyncio
async def test_backend_kwargs_without_a_backend_raises():
    with pytest.raises(ValueError, match="without a backend"):
        await _echo().arun(_dataset(), backend_kwargs={"root_dir": "."})


@pytest.mark.asyncio
async def test_function_kwargs_still_reach_the_function():
    @experiment(Row)
    async def run(row, suffix):
        return Row(q=row["q"] + suffix)

    result = await run.arun(_dataset(), name="exp", suffix="!")
    assert sorted(r.q for r in result) == ["a!", "b!"]
