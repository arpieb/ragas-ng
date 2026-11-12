"""
Unit tests for the cancellation functionality.

The Executor (testset generation's ``return_executor``) and experiments:
cancelling the task that runs ``arun()`` cancels its running rows and saves
nothing.
"""

import asyncio
import threading

import pytest

from ragas import Dataset, metrics_experiment
from ragas.backends import InMemoryBackend
from ragas.dataset_schema import SingleTurnSample
from ragas.executor import Executor
from ragas.metrics.collections import ExactMatch


class TestExecutorCancellation:
    """Test cancellation functionality in Executor."""

    def test_executor_cancel_method_exists(self):
        """Test that Executor has cancel and is_cancelled methods."""
        executor = Executor()
        assert hasattr(executor, "cancel")
        assert hasattr(executor, "is_cancelled")
        assert callable(executor.cancel)
        assert callable(executor.is_cancelled)

    def test_executor_cancellation_state(self):
        """Test cancellation state management."""
        executor = Executor()

        # Initially not cancelled
        assert not executor.is_cancelled()

        # After cancel(), should be cancelled
        executor.cancel()
        assert executor.is_cancelled()

    def test_executor_cancel_idempotent(self):
        """Test that calling cancel() multiple times is safe."""
        executor = Executor()

        # Multiple calls should be safe
        executor.cancel()
        assert executor.is_cancelled()

        executor.cancel()  # Second call
        assert executor.is_cancelled()

    def test_executor_respects_cancellation(self):
        """Test that executor respects cancellation during execution."""
        executor = Executor(desc="Test Cancellation", show_progress=False)

        # Test basic cancellation without complex async scenarios
        # to avoid asyncio edge case warnings
        async def simple_task():
            return "completed"

        # Submit a task but don't execute it
        executor.submit(simple_task)

        # Cancel before execution
        executor.cancel()
        assert executor.is_cancelled()

        # The cancellation state should be preserved
        assert executor.is_cancelled()


class SlowExactMatch(ExactMatch):
    """ExactMatch that waits on an event, so a run is still going when cancelled."""

    def __init__(self, started: asyncio.Event, cancelled: list):
        super().__init__()
        self._started = started
        self._cancelled = cancelled

    async def ascore(self, reference: str, response: str):  # type: ignore[override]
        self._started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self._cancelled.append(response)
            raise
        return await super().ascore(reference=reference, response=response)


def _rows(backend: InMemoryBackend) -> Dataset:
    return Dataset(
        "rows",
        backend,
        data_model=SingleTurnSample,
        data=[SingleTurnSample(response=f"r{i}", reference="r0") for i in range(3)],
    )


class TestExperimentCancellation:
    """Cancelling an experiment: cancel the task running ``arun()``."""

    @pytest.mark.asyncio
    async def test_cancelling_the_task_cancels_running_rows_and_saves_nothing(self):
        backend = InMemoryBackend()
        started, cancelled = asyncio.Event(), []
        run = metrics_experiment([SlowExactMatch(started, cancelled)])

        task = asyncio.create_task(run.arun(_rows(backend), name="long"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert sorted(cancelled) == ["r0", "r1", "r2"]
        assert "long" not in backend.list_experiments()

    @pytest.mark.asyncio
    async def test_wait_for_bounds_a_run(self):
        backend = InMemoryBackend()
        started, cancelled = asyncio.Event(), []
        run = metrics_experiment([SlowExactMatch(started, cancelled)])

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(run.arun(_rows(backend), name="bounded"), 0.05)

        assert sorted(cancelled) == ["r0", "r1", "r2"]
        assert "bounded" not in backend.list_experiments()


class TestGeneratorCancellation:
    """Test cancellation functionality in TestsetGenerator."""

    def test_generate_with_docs_return_executor_parameter(self):
        """Test that generate_with_docs accepts return_executor parameter."""
        # This is mainly a signature test since full testing requires LLM/embeddings
        # Import locally to avoid pytest collection issues
        from ragas.testset.synthesizers.generate import TestsetGenerator

        generator = TestsetGenerator.__new__(
            TestsetGenerator
        )  # Create without __init__

        # Verify the method signature includes return_executor
        import inspect

        sig = inspect.signature(generator.generate_with_docs)
        assert "return_executor" in sig.parameters

        # Verify default value is False
        param = sig.parameters["return_executor"]
        assert param.default is False

    def test_generate_method_return_executor_parameter(self):
        """Test that generate method accepts return_executor parameter."""
        # Import locally to avoid pytest collection issues
        from ragas.testset.synthesizers.generate import TestsetGenerator

        generator = TestsetGenerator.__new__(TestsetGenerator)

        # Verify the method signature includes return_executor
        import inspect

        sig = inspect.signature(generator.generate)
        assert "return_executor" in sig.parameters

        # Verify default value is False
        param = sig.parameters["return_executor"]
        assert param.default is False


class TestCancellationIntegration:
    """Test integration scenarios with cancellation."""

    def test_cancellation_thread_safety(self):
        """Test that cancellation works safely across threads."""
        executor = Executor(show_progress=False)

        # Add a task
        async def simple_task():
            await asyncio.sleep(0.1)
            return "done"

        executor.submit(simple_task)

        # Cancel from another thread
        cancel_thread = threading.Thread(target=executor.cancel)
        cancel_thread.start()
        cancel_thread.join()

        # Should be cancelled
        assert executor.is_cancelled()

    def test_multiple_executors_isolation(self):
        """Test that cancelling one executor doesn't affect others."""
        executor1 = Executor(show_progress=False)
        executor2 = Executor(show_progress=False)
        executor3 = Executor(show_progress=False)

        # Cancel only executor2
        executor2.cancel()

        # Check isolation
        assert not executor1.is_cancelled()
        assert executor2.is_cancelled()
        assert not executor3.is_cancelled()

    def test_cancellation_with_empty_job_list(self):
        """Test cancellation with no submitted jobs."""
        executor = Executor(show_progress=False)

        # Cancel without any jobs
        executor.cancel()
        assert executor.is_cancelled()

        # Results should be empty
        results = executor.results()
        assert results == []
