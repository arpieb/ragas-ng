"""The gzip-aware JSON helpers shared by Prompt and DynamicFewShotPrompt (#217)."""

import gzip
import json

import pytest

from ragas.prompt.utils import read_json_file, write_json_file


@pytest.mark.parametrize("name", ["data.json", "data.json.gz"])
def test_round_trip(tmp_path, name):
    path = tmp_path / name
    write_json_file(path, {"a": [1, "é"]}, "thing")
    assert read_json_file(path, "thing") == {"a": [1, "é"]}


def test_gz_suffix_writes_gzip(tmp_path):
    path = tmp_path / "data.json.gz"
    write_json_file(str(path), {"a": 1}, "thing")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert json.load(f) == {"a": 1}


def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot load thing from"):
        read_json_file(tmp_path / "missing.json", "thing")


def test_invalid_json_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load thing from"):
        read_json_file(path, "thing")


def test_unwritable_path_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot save thing to"):
        write_json_file(tmp_path / "no" / "such" / "dir.json", {}, "thing")
