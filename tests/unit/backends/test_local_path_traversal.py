"""Dataset/experiment names must not escape a local backend's storage root."""

from pathlib import Path

import pytest

from ragas.backends.base import resolve_resource_path
from ragas.backends.local_csv import LocalCSVBackend
from ragas.backends.local_jsonl import LocalJSONLBackend

BAD_NAMES = [
    "../escape",
    "../../etc/cron.d/x",
    "sub/dir",
    "..\\escape",
    "/abs/path",
    "..",
    ".",
    "",
    "nul\x00byte",
]


@pytest.fixture(params=[LocalCSVBackend, LocalJSONLBackend], ids=["csv", "jsonl"])
def backend(request, tmp_path):
    return request.param(root_dir=str(tmp_path / "root"))


@pytest.mark.parametrize("name", BAD_NAMES)
def test_save_rejects_names_that_leave_the_root(backend, tmp_path, name):
    for save in (backend.save_dataset, backend.save_experiment):
        with pytest.raises(ValueError):
            save(name, [{"a": 1}])
    # nothing was written anywhere under tmp_path
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


@pytest.mark.parametrize("name", BAD_NAMES)
def test_load_rejects_names_that_leave_the_root(backend, tmp_path, name):
    # a file the traversal would hit, if it were allowed
    (tmp_path / "escape.csv").write_text("a\n1\n")
    (tmp_path / "escape.jsonl").write_text('{"a": 1}\n')
    for load in (backend.load_dataset, backend.load_experiment):
        with pytest.raises(ValueError):
            load(name)


@pytest.mark.parametrize(
    "name", ["my dataset", "v1.2-final", "...", "名前", "a..b", ".hidden"]
)
def test_ordinary_names_round_trip(backend, name):
    backend.save_dataset(name, [{"a": "1"}])
    assert backend.load_dataset(name)[0]["a"] in ("1", 1)
    assert name in backend.list_datasets()


def test_helper_keeps_file_directly_in_data_dir(tmp_path: Path):
    path = resolve_resource_path(tmp_path, "x", ".csv")
    assert path == tmp_path / "x.csv"
