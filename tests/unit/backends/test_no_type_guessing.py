"""Backends load what was stored; types come from the data model.

Covers #101 (JSONL turned date-like strings into dates), the Google Drive
backend's number guessing, "None" cells and 26-column reads, malformed JSONL
lines being skipped silently, and #102 (annotation loaders leaked file
handles) with the other dataset_schema file I/O.
"""

from __future__ import annotations

import builtins
import datetime as dt
import json
import typing as t
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from ragas import Dataset
from ragas.backends.local_jsonl import LocalJSONLBackend
from ragas.dataset_schema import (
    EvaluationDataset,
    MetricAnnotation,
    SingleMetricAnnotation,
    SingleTurnSample,
)

DATE_LIKE = [
    "2024-01-02",
    "2024-01-02T10:30:00",
    "1999-12-31T23:59:59.5Z",
    "ab-cd-efgh",
]


@pytest.fixture
def jsonl(tmp_path) -> LocalJSONLBackend:
    return LocalJSONLBackend(root_dir=str(tmp_path))


# --- #101: JSONL -----------------------------------------------------------------


@pytest.mark.parametrize("text", DATE_LIKE)
def test_jsonl_date_like_strings_stay_strings(jsonl, text):
    jsonl.save_dataset("d", [{"note": text, "nested": {"also": [text]}}])

    [row] = jsonl.load_dataset("d")

    assert row == {"note": text, "nested": {"also": [text]}}


class Note(BaseModel):
    text: str
    day: dt.date
    at: dt.datetime
    when: t.Optional[dt.datetime] = None


@pytest.mark.parametrize("text", DATE_LIKE)
def test_typed_jsonl_str_field_keeps_date_like_text(tmp_path, text):
    row = Note(text=text, day=dt.date(2024, 1, 2), at=dt.datetime(2024, 1, 2, 3, 4))
    ds = Dataset(
        name="n", backend="local/jsonl", root_dir=str(tmp_path), data_model=Note
    )
    ds.append(row)
    ds.save()

    loaded = Dataset.load(
        name="n", backend="local/jsonl", root_dir=str(tmp_path), data_model=Note
    )

    assert list(loaded) == [row]  # dates parsed by the model, text untouched


def test_untyped_jsonl_dates_load_as_iso_strings(jsonl):
    jsonl.save_dataset(
        "d", [{"at": dt.datetime(2024, 1, 15, 10, 30), "day": dt.date(2024, 1, 16)}]
    )

    [row] = jsonl.load_dataset("d")

    assert row == {"at": "2024-01-15T10:30:00", "day": "2024-01-16"}


def test_malformed_jsonl_line_raises_with_its_line_number(jsonl, tmp_path):
    path = tmp_path / "datasets" / "bad.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"ok": 1}\n{"broken": json}\n{"ok": 3}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        jsonl.load_dataset("bad")


def test_blank_lines_are_still_skipped(jsonl, tmp_path):
    path = tmp_path / "datasets" / "blank.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"ok": 1}\n\n{"ok": 2}\n\n', encoding="utf-8")

    assert jsonl.load_dataset("blank") == [{"ok": 1}, {"ok": 2}]


# --- Google Drive ----------------------------------------------------------------

gdrive = pytest.importorskip("ragas.backends.gdrive_backend")


def _gdrive_backend():
    if not gdrive.GDRIVE_AVAILABLE:
        pytest.skip("Google Drive dependencies not available")
    with patch("ragas.backends.gdrive_backend.build"):
        with patch("ragas.backends.gdrive_backend.Credentials"):
            with patch("os.path.exists", return_value=True):
                backend = gdrive.GDriveBackend(
                    folder_id="f", service_account_path="/fake/path.json"
                )
    backend.datasets_folder_id = "datasets_folder"
    backend.experiments_folder_id = "experiments_folder"
    backend.drive_service.files().list.return_value.execute.return_value = {
        "files": [{"id": "sheet"}]
    }
    return backend


def _sheet_values(backend) -> t.List[t.List[str]]:
    update = backend.sheets_service.spreadsheets().values().update
    return update.call_args.kwargs["body"]["values"]


def test_gdrive_writes_none_as_empty_and_lists_as_json():
    backend = _gdrive_backend()

    backend.save_dataset("d", [{"a": None, "b": ["x", "y"], "c": {"k": 1}, "d": 3}])

    header, row = _sheet_values(backend)
    cells = dict(zip(header, row))
    assert cells == {"a": "", "b": '["x", "y"]', "c": '{"k": 1}', "d": "3"}


def test_gdrive_leaves_numeric_text_alone():
    backend = _gdrive_backend()
    get = backend.sheets_service.spreadsheets().values().get
    get.return_value.execute.return_value = {"values": [["code", "n"], ["007", "1.50"]]}

    assert backend.load_dataset("d") == [{"code": "007", "n": "1.50"}]


class Coded(BaseModel):
    code: str
    n: float
    tags: t.List[str] = []
    missing: t.Optional[int] = None


def test_gdrive_typed_load_uses_the_model():
    backend = _gdrive_backend()
    get = backend.sheets_service.spreadsheets().values().get
    get.return_value.execute.return_value = {
        "values": [["code", "n", "tags", "missing"], ["007", "1.50", '["a"]', ""]]
    }

    [row] = Dataset.load(name="d", backend=backend, data_model=Coded)

    assert row == Coded(code="007", n=1.5, tags=["a"], missing=None)


def test_gdrive_reads_and_clears_every_column():
    backend = _gdrive_backend()
    values = backend.sheets_service.spreadsheets().values()
    values.get.return_value.execute.return_value = {"values": []}

    backend.load_dataset("d")
    backend.save_dataset("d", [{"a": 1}])

    assert values.get.call_args.kwargs["range"] == "A:ZZZ"
    assert values.clear.call_args.kwargs["range"] == "A:ZZZ"


# --- #102 and dataset_schema file I/O ---------------------------------------------


class _OpenRecorder:
    def __init__(self):
        self.real_open = builtins.open
        self.handles: t.List[t.Any] = []
        self.encodings: t.List[t.Optional[str]] = []

    def __call__(self, file, mode="r", *args, **kwargs):
        self.encodings.append(kwargs.get("encoding"))
        handle = self.real_open(file, mode, *args, **kwargs)
        self.handles.append(handle)
        return handle


@pytest.fixture
def opens(monkeypatch) -> _OpenRecorder:
    rec = _OpenRecorder()
    monkeypatch.setattr(builtins, "open", rec)
    return rec


def test_annotation_loaders_close_their_files_and_read_utf8(tmp_path, opens):
    single = tmp_path / "single.json"
    single.write_text(json.dumps({"name": "m", "samples": []}), encoding="utf-8")
    multi = tmp_path / "multi.json"
    multi.write_text(json.dumps({"m": []}), encoding="utf-8")
    opens.handles.clear()
    opens.encodings.clear()

    SingleMetricAnnotation.from_json(str(single))
    MetricAnnotation.from_json(str(multi), metric_name=None)

    assert len(opens.handles) == 2
    assert all(h.closed for h in opens.handles)
    assert opens.encodings == ["utf-8", "utf-8"]


def test_dataset_files_are_utf8_and_round_trip_non_ascii(tmp_path, opens):
    ds = EvaluationDataset(
        samples=[SingleTurnSample(user_input="¿Qué tal? 日本語", response="ü")]
    )

    ds.to_jsonl(tmp_path / "d.jsonl")
    ds.to_csv(tmp_path / "d.csv")
    loaded = EvaluationDataset.from_jsonl(tmp_path / "d.jsonl")

    assert loaded.samples[0].user_input == "¿Qué tal? 日本語"
    assert opens.encodings == ["utf-8", "utf-8", "utf-8"]


def test_from_jsonl_ignores_a_trailing_blank_line(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"user_input": "q"}\n\n', encoding="utf-8")

    assert len(EvaluationDataset.from_jsonl(path)) == 1
