from __future__ import annotations

import typing as t


def test_import():
    import ragas
    from ragas.testset import TestsetGenerator

    assert TestsetGenerator is not None
    assert ragas is not None


def test_type_casting():
    t.cast(t.List[int], [1, 2, 3])
