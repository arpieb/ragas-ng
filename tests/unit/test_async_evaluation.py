import pytest


class TestAsyncUtilsControl:
    """run() runs on the shared loop; nest_asyncio is gone (#80)."""

    def test_run_returns_the_result(self):
        from ragas.async_utils import run

        async def test_func():
            return "test"

        assert run(test_func) == "test"

    def test_run_allow_nest_asyncio_is_deprecated_and_ignored(self):
        from ragas.async_utils import run

        async def test_func():
            return "test"

        with pytest.warns(DeprecationWarning, match="nest_asyncio"):
            assert run(test_func, allow_nest_asyncio=False) == "test"

    def test_apply_nest_asyncio_is_a_deprecated_no_op(self):
        from ragas.async_utils import apply_nest_asyncio

        with pytest.warns(DeprecationWarning):
            assert apply_nest_asyncio() is False
