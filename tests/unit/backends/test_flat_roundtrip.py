"""Flat (text-only) backends: rows that differ, and typed round trips.

Covers #100 (CSV crashed on rows with different keys) and the CSV backend
failing to load back typed datasets it saved: lists and dicts were written as
Python repr and None as an empty cell, which data_model validation rejected.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import typing as t
import uuid
from enum import Enum

import pytest
from pydantic import BaseModel

from ragas import Dataset, experiment
from ragas.backends.flat import decode_flat_row, encode_flat_rows
from ragas.backends.local_csv import LocalCSVBackend
from ragas.dataset_schema import SingleTurnSample


@pytest.fixture
def backend(tmp_path) -> LocalCSVBackend:
    return LocalCSVBackend(root_dir=str(tmp_path))


# --- #100: rows with different keys -------------------------------------------


def test_rows_with_extra_and_missing_keys_save_and_load(backend):
    rows = [{"a": 1}, {"a": 2, "b": "x"}, {"c": True}]

    backend.save_dataset("mixed", rows)

    loaded = backend.load_dataset("mixed")
    assert loaded == [
        {"a": "1", "b": "", "c": ""},
        {"a": "2", "b": "x", "c": ""},
        {"a": "", "b": "", "c": "True"},
    ]


def test_header_keeps_first_seen_order():
    header, _ = encode_flat_rows([{"z": 1, "a": 2}, {"m": 3, "a": 4}])
    assert header == ["z", "a", "m"]


@pytest.mark.asyncio
async def test_experiment_rows_that_differ_are_saved(tmp_path):
    backend = LocalCSVBackend(root_dir=str(tmp_path))
    dataset = Dataset(name="inputs", backend=backend)
    for q in ["one", "two"]:
        dataset.append({"q": q})

    @experiment(backend=backend)
    async def run(row):
        result = {"q": row["q"], "score": 1}
        if row["q"] == "two":
            result["note"] = "only on some rows"
        return result

    await run.arun(dataset, name="exp")

    loaded = backend.load_experiment("exp")
    notes = {r["q"]: r["note"] for r in loaded}  # rows arrive as they finish
    assert notes == {"one": "", "two": "only on some rows"}


def test_testset_to_csv_with_mixed_sample_fields(tmp_path):
    from ragas.testset import Testset, TestsetSample

    ds = Testset(
        samples=[
            TestsetSample(eval_sample=s, synthesizer_name="t")
            for s in (
                SingleTurnSample(user_input="q1", retrieved_contexts=["a", "b"]),
                SingleTurnSample(
                    user_input="q2",
                    reference_contexts=["c"],
                    reference_contexts_metadata=[{"source": "x.md"}],
                ),
            )
        ]
    )
    path = tmp_path / "ds.csv"

    ds.to_csv(path)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert json.loads(rows[0]["retrieved_contexts"]) == ["a", "b"]
    assert rows[0]["reference_contexts_metadata"] == ""
    assert json.loads(rows[1]["reference_contexts_metadata"]) == [{"source": "x.md"}]


# --- typed round trips --------------------------------------------------------


class Color(Enum):
    RED = "red"


class Inner(BaseModel):
    label: str
    weight: float


class Row(BaseModel):
    question: str
    contexts: t.List[str]
    meta: t.Dict[str, t.Any] = {}
    inner: Inner
    inners: t.List[Inner] = []
    score: t.Optional[float] = None
    passed: bool
    count: int
    day: dt.date
    at: dt.datetime
    run_id: uuid.UUID
    color: Color
    tags: t.Tuple[str, ...] = ()
    note: str = ""


def _row(**overrides) -> Row:
    values = dict(
        question="What is 2+2?",
        contexts=["a", "b, with a comma", 'and "quotes"'],
        meta={"k": 1, "nested": {"x": [1, 2]}},
        inner=Inner(label="x", weight=0.5),
        inners=[Inner(label="y", weight=1.0)],
        score=None,
        passed=False,
        count=7,
        day=dt.date(2026, 1, 2),
        at=dt.datetime(2026, 1, 2, 3, 4, 5),
        run_id=uuid.UUID(int=1),
        color=Color.RED,
        tags=("t1", "t2"),
    )
    values.update(overrides)
    return Row(**values)


def test_typed_dataset_round_trips_through_csv(tmp_path):
    original = [_row(), _row(score=0.25, passed=True, count=0, meta={})]
    ds = Dataset(
        name="typed", backend="local/csv", root_dir=str(tmp_path), data_model=Row
    )
    for row in original:
        ds.append(row)
    ds.save()

    loaded = Dataset.load(
        name="typed", backend="local/csv", root_dir=str(tmp_path), data_model=Row
    )

    assert list(loaded) == original


def test_reload_and_validate_with_decode_too(tmp_path):
    ds = Dataset(
        name="typed", backend="local/csv", root_dir=str(tmp_path), data_model=Row
    )
    ds.append(_row())
    ds.save()

    ds.reload()
    assert list(ds) == [_row()]

    untyped = Dataset.load(name="typed", backend="local/csv", root_dir=str(tmp_path))
    assert list(untyped.validate_with(Row)) == [_row()]


@pytest.mark.parametrize(
    "text", ["[draft] report", "2024-01-02", "{not json}", "007", ""]
)
def test_str_fields_are_never_changed(tmp_path, text):
    ds = Dataset(name="s", backend="local/csv", root_dir=str(tmp_path), data_model=Row)
    ds.append(_row(question=text, note=text))
    ds.save()

    [loaded] = Dataset.load(
        name="s", backend="local/csv", root_dir=str(tmp_path), data_model=Row
    )

    assert loaded.question == text
    assert loaded.note == text


def test_untyped_csv_loads_return_strings(tmp_path):
    ds = Dataset(
        name="typed", backend="local/csv", root_dir=str(tmp_path), data_model=Row
    )
    ds.append(_row())
    ds.save()

    [raw] = Dataset.load(name="typed", backend="local/csv", root_dir=str(tmp_path))

    assert all(isinstance(v, str) for v in raw.values())
    assert json.loads(raw["contexts"]) == _row().contexts  # JSON, not repr
    assert raw["score"] == ""


# --- the decoder --------------------------------------------------------------


class Optionals(BaseModel):
    required_opt: t.Optional[int]
    with_default: int = 5
    maybe_list: t.Optional[t.List[int]] = None
    union: t.Union[int, t.List[int]] = 0


def test_empty_cells_fall_back_to_defaults_or_none():
    decoded = decode_flat_row(
        {"required_opt": "", "with_default": "", "maybe_list": "", "union": ""},
        Optionals,
    )
    assert Optionals(**decoded) == Optionals(required_opt=None)


def test_structured_union_members_parse_json():
    decoded = decode_flat_row(
        {"required_opt": "3", "maybe_list": "[1, 2]", "union": "[4]"}, Optionals
    )
    assert Optionals(**decoded) == Optionals(
        required_opt=3, maybe_list=[1, 2], union=[4]
    )


def test_unknown_keys_and_non_strings_pass_through():
    decoded = decode_flat_row({"required_opt": 3, "extra": "[1]"}, Optionals)
    assert decoded == {"required_opt": 3, "extra": "[1]"}
