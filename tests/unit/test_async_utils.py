import pytest

from ragas.async_utils import run_async_tasks


def test_is_event_loop_running_in_script():
    from ragas.async_utils import is_event_loop_running

    assert is_event_loop_running() is False


def test_run_function():
    from ragas.async_utils import run

    async def foo():
        return 42

    result = run(foo)
    assert result == 42


@pytest.fixture
def tasks():
    async def echo_order(index: int):
        return index

    return [echo_order(i) for i in range(1, 11)]


def test_run_async_tasks_unbatched(tasks):
    results = run_async_tasks(tasks)
    assert sorted(results) == sorted(range(1, 11))


def test_run_async_tasks_batched(tasks):
    results = run_async_tasks(tasks, batch_size=3)
    assert sorted(results) == sorted(range(1, 11))


def test_run_async_tasks_no_progress(tasks):
    results = run_async_tasks(tasks, show_progress=False)
    assert sorted(results) == sorted(range(1, 11))
