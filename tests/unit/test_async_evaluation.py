import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestEvaluateAsyncControl:
    """The sync evaluate() runs aevaluate() on the shared background loop."""

    def _evaluate(self, **kwargs):
        def fake_run_sync(coro):
            # close the coroutine the mock won't run: left to the garbage
            # collector, its "never awaited" warning fires later, in
            # whichever test is running then (#207's CI failure)
            coro.close()
            return MagicMock()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )
            with patch(
                "ragas.async_utils.run_sync", side_effect=fake_run_sync
            ) as mock_run_sync:
                from ragas import evaluate

                evaluate(
                    dataset=MagicMock(),
                    metrics=[MagicMock()],
                    show_progress=False,
                    **kwargs,
                )
        return mock_run_sync

    def test_evaluate_uses_run_sync(self):
        self._evaluate().assert_called_once()

    @pytest.mark.parametrize("value", [True, False])
    def test_evaluate_allow_nest_asyncio_is_deprecated_and_ignored(self, value):
        with pytest.warns(DeprecationWarning, match="allow_nest_asyncio"):
            mock_run_sync = self._evaluate(allow_nest_asyncio=value)
        mock_run_sync.assert_called_once()


class TestAevaluateImport:
    """Test that aevaluate can be imported and is async."""

    def test_aevaluate_importable(self):
        """Test that aevaluate can be imported."""
        from ragas import aevaluate

        assert callable(aevaluate)
        assert asyncio.iscoroutinefunction(aevaluate)

    def test_evaluate_has_allow_nest_asyncio_param(self):
        """Test that evaluate function has the new parameter."""
        import inspect

        from ragas import evaluate

        sig = inspect.signature(evaluate)
        # kept for compatibility; deprecated and ignored
        assert "allow_nest_asyncio" in sig.parameters
        assert sig.parameters["allow_nest_asyncio"].default is None


class TestNestAsyncioNotAppliedInAevaluate:
    """Test that aevaluate doesn't apply nest_asyncio."""

    @pytest.mark.asyncio
    async def test_aevaluate_no_nest_asyncio_applied(self):
        """Test that aevaluate doesn't call apply_nest_asyncio."""
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about unawaited coroutines in tests
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*coroutine.*was never awaited",
            )

            # Mock all the dependencies to avoid actual API calls
            with patch("ragas.evaluation.EvaluationDataset"):
                with patch("ragas.evaluation.validate_required_columns"):
                    with patch("ragas.evaluation.validate_supported_metrics"):
                        with patch("ragas.evaluation.Executor") as mock_executor_class:
                            with patch("ragas.evaluation.new_group"):
                                with patch(
                                    "ragas.async_utils.apply_nest_asyncio"
                                ) as mock_apply:
                                    # Mock executor
                                    mock_executor = MagicMock()
                                    mock_executor.aresults = AsyncMock(
                                        return_value=[0.8]
                                    )
                                    mock_executor_class.return_value = mock_executor

                                    # Mock dataset
                                    mock_dataset_instance = MagicMock()
                                    mock_dataset_instance.get_sample_type.return_value = MagicMock()
                                    mock_dataset_instance.__iter__ = lambda x: iter([])

                                    from ragas import aevaluate

                                    try:
                                        await aevaluate(
                                            dataset=mock_dataset_instance,
                                            metrics=[],
                                            show_progress=False,
                                        )
                                    except Exception:
                                        pass

            # aevaluate should never call apply_nest_asyncio
            mock_apply.assert_not_called()


class TestAsyncIntegration:
    """Basic integration tests for async scenarios."""

    @pytest.mark.asyncio
    async def test_aevaluate_in_running_loop(self):
        """Test aevaluate can be called when an event loop is already running."""
        # This test runs with pytest-asyncio, so an event loop is running
        from ragas import aevaluate

        # Just test that the function can be called without RuntimeError
        # We'll mock everything to avoid API calls
        with patch("ragas.evaluation.EvaluationDataset"):
            with patch("ragas.evaluation.validate_required_columns"):
                with patch("ragas.evaluation.validate_supported_metrics"):
                    with patch("ragas.evaluation.Executor") as mock_executor_class:
                        with patch("ragas.evaluation.new_group"):
                            mock_executor = MagicMock()
                            mock_executor.aresults = AsyncMock(return_value=[])
                            mock_executor_class.return_value = mock_executor

                            try:
                                await aevaluate(
                                    dataset=MagicMock(),
                                    metrics=[],
                                    show_progress=False,
                                )
                                # Should not raise RuntimeError about event loop
                            except Exception as e:
                                # We expect other exceptions due to mocking, but not RuntimeError
                                assert "event loop" not in str(e).lower()
                                assert "nest_asyncio" not in str(e).lower()
