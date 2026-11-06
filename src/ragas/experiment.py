"""Experiments hold the results of an experiment against a dataset."""

__all__ = ["Experiment", "experiment", "version_experiment"]

import asyncio
import typing as t
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from ragas.backends.base import BaseBackend
from ragas.dataset import Dataset, DataTable
from ragas.utils import find_git_root, memorable_names


class Experiment(DataTable):
    DATATABLE_TYPE = "Experiment"


def version_experiment(
    experiment_name: str,
    commit_message: t.Optional[str] = None,
    repo_path: t.Union[str, Path, None] = None,
    create_branch: bool = True,
    stage_all: bool = False,
) -> str:
    """Version control the current state of the codebase for an experiment.

    This function requires GitPython to be installed. You can install it with:
        pip install ragas[git]
        # or
        uv pip install ragas[git]

    Args:
        experiment_name: Name for the experiment (used in branch name)
        commit_message: Custom commit message (defaults to "Experiment: {experiment_name}")
        repo_path: Path to git repository (defaults to current directory)
        create_branch: Whether to create a git branch for the experiment
        stage_all: Whether to stage untracked files (default: tracked files only)

    Returns:
        The commit hash of the versioned state
    """
    try:
        import git
    except ImportError as e:
        raise ImportError(
            "version_experiment() requires GitPython. Install it with:\n"
            "  pip install ragas[git]\n"
            "  # or\n"
            "  uv pip install ragas[git]\n\n"
            "Or install with full features:\n"
            "  pip install ragas[all]\n"
            "  # or\n"
            "  uv pip install ragas[all]"
        ) from e

    # Default to current directory if no repo path is provided
    if repo_path is None:
        repo_path = find_git_root()

    # Initialize git repo object
    repo = git.Repo(repo_path)

    # Check if there are any changes to the repo
    has_changes = False
    if stage_all and repo.is_dirty(untracked_files=True):
        print("Staging all changes")
        repo.git.add(".")
        has_changes = True
    elif repo.is_dirty(untracked_files=False):
        print("Staging changes to tracked files")
        repo.git.add("-u")
        has_changes = True

    # Check if there are uncommitted changes
    if has_changes:
        # Default commit message if none provided
        if commit_message is None:
            commit_message = f"Experiment: {experiment_name}"

        # Commit changes
        commit = repo.index.commit(commit_message)
        commit_hash = commit.hexsha
        print(f"Changes committed with hash: {commit_hash[:8]}")
    else:
        # No changes to commit, use current HEAD
        commit_hash = repo.head.commit.hexsha
        print("No changes detected, nothing to commit")

    # Format the branch/tag name
    version_name = f"ragas/{experiment_name}"

    # Create branch if requested
    if create_branch:
        repo.create_head(version_name, commit_hash)
        print(f"Created branch: {version_name}")

    return commit_hash


@t.runtime_checkable
class ExperimentProtocol(t.Protocol):
    async def __call__(self, *args, **kwargs) -> t.Any: ...
    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs,
    ) -> "Experiment": ...


class ExperimentWrapper:
    """Wrapper class that implements ExperimentProtocol for decorated functions."""

    def __init__(
        self,
        func: t.Callable,
        experiment_model: t.Optional[t.Type[BaseModel]] = None,
        default_backend: t.Optional[t.Union[BaseBackend, str]] = None,
        name_prefix: str = "",
        default_backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        _check_backend_kwargs(default_backend, default_backend_kwargs)
        self.func = func
        self.experiment_model = experiment_model
        self.default_backend = default_backend
        self.default_backend_kwargs = default_backend_kwargs
        self.name_prefix = name_prefix
        # Preserve function metadata
        self.__name__ = getattr(func, "__name__", "experiment_function")
        self.__doc__ = getattr(func, "__doc__", None)

    async def __call__(self, *args, **kwargs) -> t.Any:
        """Call the original function."""
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    async def arun(
        self,
        dataset: Dataset,
        name: t.Optional[str] = None,
        backend: t.Optional[t.Union[BaseBackend, str]] = None,
        *args,
        backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs,
    ) -> "Experiment":
        """Run the experiment against a dataset.

        Args:
            dataset: The rows to run the experiment function on.
            name: The experiment's name. Generated if omitted.
            backend: Where to save the results: a backend instance or a
                registered name such as ``"local/jsonl"``. Defaults to the
                decorator's ``backend``, then to the dataset's backend.
            *args: Passed to the experiment function after each row.
            backend_kwargs: Constructor arguments for a backend given by
                name, e.g. ``{"root_dir": "./data"}``.
            **kwargs: Passed to the experiment function. ``name``,
                ``backend`` and ``backend_kwargs`` are ``arun``'s own, so pass
                the function's arguments by keyword.
        """
        # Generate name if not provided
        if name is None:
            name = memorable_names.generate_unique_name()
        if self.name_prefix:
            name = f"{self.name_prefix}-{name}"

        # Resolve backend: arun's, else the decorator's, else the dataset's
        if backend is not None:
            experiment_backend, constructor_kwargs = backend, backend_kwargs
        else:
            experiment_backend = self.default_backend
            constructor_kwargs = (
                backend_kwargs
                if backend_kwargs is not None
                else self.default_backend_kwargs
            )
        _check_backend_kwargs(experiment_backend, constructor_kwargs)
        if experiment_backend is not None:
            resolved_backend = _resolve_experiment_backend(
                experiment_backend, constructor_kwargs
            )
        else:
            resolved_backend = dataset.backend

        # Create experiment
        experiment_view = Experiment(
            name=name,
            data_model=self.experiment_model,
            backend=resolved_backend,
        )

        # Create tasks for all items
        tasks = []
        for item in dataset:
            tasks.append(self(item, *args, **kwargs))

        progress_bar = None
        try:
            progress_bar = tqdm(total=len(dataset), desc="Running experiment")

            # Process all items
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    if result is not None:
                        experiment_view.append(result)
                except Exception as e:
                    # Log individual task failures but continue
                    print(f"Warning: Task failed with error: {e}")
                finally:
                    progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()

        # Save experiment
        experiment_view.save()

        return experiment_view


def _check_backend_kwargs(
    backend: t.Optional[t.Union[BaseBackend, str]],
    backend_kwargs: t.Optional[t.Dict[str, t.Any]],
) -> None:
    """``backend_kwargs`` only makes sense with a backend given by name."""
    if backend_kwargs is None:
        return
    if backend is None:
        raise ValueError(
            "backend_kwargs was given without a backend name; pass "
            "backend='<name>' as well, e.g. backend='local/jsonl'."
        )
    if not isinstance(backend, str):
        raise ValueError(
            "backend_kwargs applies only to a backend given by name; "
            f"got a {type(backend).__name__} instance, which is already "
            "constructed."
        )


def _resolve_experiment_backend(
    backend: t.Union[BaseBackend, str],
    backend_kwargs: t.Optional[t.Dict[str, t.Any]],
) -> BaseBackend:
    try:
        return Experiment._resolve_backend(backend, **(backend_kwargs or {}))
    except TypeError as e:
        if not isinstance(backend, str):
            raise
        # Before #165, names were created with no arguments at all, so the
        # error must say how to pass them.
        raise TypeError(
            f"{e} Pass the constructor arguments with backend_kwargs, e.g. "
            f"backend={backend!r}, backend_kwargs={{'root_dir': './data'}}, "
            "or pass a backend instance."
        ) from e


def experiment(
    experiment_model: t.Optional[t.Type[BaseModel]] = None,
    backend: t.Optional[t.Union[BaseBackend, str]] = None,
    name_prefix: str = "",
    *,
    backend_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
) -> t.Callable[[t.Callable], ExperimentProtocol]:
    """Decorator for creating experiment functions.

    Args:
        experiment_model: The Pydantic model type to use for experiment results
        backend: Optional backend to use for storing experiment results: an
            instance, or a registered name such as ``"local/jsonl"``
        name_prefix: Optional prefix for experiment names
        backend_kwargs: Constructor arguments for a backend given by name,
            e.g. ``{"root_dir": "./data"}``

    Returns:
        Decorator function that wraps experiment functions

    Example:
        @experiment(ExperimentDataRow)
        async def run_experiment(row: TestDataRow):
            # experiment logic here
            return ExperimentDataRow(...)
    """
    _check_backend_kwargs(backend, backend_kwargs)

    def decorator(func: t.Callable) -> ExperimentProtocol:
        wrapper = ExperimentWrapper(
            func=func,
            experiment_model=experiment_model,
            default_backend=backend,
            name_prefix=name_prefix,
            default_backend_kwargs=backend_kwargs,
        )
        return t.cast(ExperimentProtocol, wrapper)

    return decorator
