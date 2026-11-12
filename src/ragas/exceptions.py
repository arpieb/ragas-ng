from __future__ import annotations

import typing as t


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ExceptionInRunner(RagasException):
    """
    Exception raised when an exception is raised in the executor.
    """

    def __init__(self):
        msg = "The runner thread which was running the jobs raised an exeception. Read the traceback above to debug it. You can also pass `raise_exceptions=False` incase you want to show only a warning message instead."
        super().__init__(msg)


class RagasOutputParserException(RagasException):
    """
    Exception raised when the output parser fails to parse the output.
    """

    def __init__(self, detail: t.Optional[str] = None):
        msg = "The output parser failed to parse the output including retries."
        if detail:
            msg = f"{msg} Last error: {detail}"
        super().__init__(msg)


class LLMDidNotFinishException(RagasException):
    """
    Exception raised when the LLM did not finish.
    """

    def __init__(self):
        msg = "The LLM generation was not completed. Please increase the max_tokens and try again."
        super().__init__(msg)


class ImageLoadError(RagasException, ValueError):
    """An image context was refused or could not be loaded (#231).

    Raised by the multimodal metrics for a data URI, URL or image path in
    ``retrieved_contexts`` that their ``ImageSourcePolicy`` refuses or that
    fails to load, so a score is never computed from a judge that did not see
    the image. ``ImageSourcePolicy(on_blocked="warn")`` logs instead.

    Attributes:
        source: The context string (data URIs are shortened in the message).
        reason: Why it was not loaded.
        hint: The setting that would allow it, if any.
        context_index: 1-based position in ``retrieved_contexts``, when known.
    """

    def __init__(
        self,
        source: str,
        reason: str,
        hint: t.Optional[str] = None,
        context_index: t.Optional[int] = None,
    ):
        self.source = source
        self.reason = reason
        self.hint = hint
        self.context_index = context_index
        super().__init__(
            f"{self.describe()} To score without the image instead, use "
            'ImageSourcePolicy(on_blocked="warn").'
        )

    def describe(self) -> str:
        """The message without the ``on_blocked`` hint."""
        source = self.source
        shown = source if len(source) <= 120 else source[:60] + "..." + source[-20:]
        where = (
            f"Image context {self.context_index}"
            if self.context_index
            else "Image context"
        )
        message = f"{where} {shown!r} was not loaded: {self.reason}."
        return f"{message} {self.hint}" if self.hint else message

    def with_index(self, context_index: int) -> "ImageLoadError":
        return ImageLoadError(self.source, self.reason, self.hint, context_index)


# Exceptions migrated from experimental module
class RagasError(Exception):
    """Base class for all Ragas-related exceptions."""

    pass


class ValidationError(RagasError):
    """Raised when field validation fails."""

    pass


class DuplicateError(RagasError):
    """Exception raised when a duplicate resource is created."""

    pass


class NotFoundError(RagasError):
    """Exception raised when a resource is not found."""

    pass


class ResourceNotFoundError(NotFoundError):
    """Exception raised when a resource doesn't exist."""

    pass


class ProjectNotFoundError(ResourceNotFoundError):
    """Exception raised when a project doesn't exist."""

    pass


class DatasetNotFoundError(ResourceNotFoundError):
    """Exception raised when a dataset doesn't exist."""

    pass


class ExperimentNotFoundError(ResourceNotFoundError):
    """Exception raised when an experiment doesn't exist."""

    pass


class DuplicateResourceError(RagasError):
    """Exception raised when multiple resources exist with the same identifier."""

    pass


class DuplicateProjectError(DuplicateResourceError):
    """Exception raised when multiple projects exist with the same name."""

    pass


class DuplicateDatasetError(DuplicateResourceError):
    """Exception raised when multiple datasets exist with the same name."""

    pass


class DuplicateExperimentError(DuplicateResourceError):
    """Exception raised when multiple experiments exist with the same name."""

    pass


class ThresholdError(RagasError, AssertionError):
    """``ragas.check()`` found metrics outside their thresholds.

    An ``AssertionError``, so pytest reports it as a failed assertion. The
    message is a per-metric table; ``failures`` holds the same as
    ``(metric, problem)`` pairs.
    """

    def __init__(self, message: str, failures: "list[tuple[str, str]]"):
        super().__init__(message)
        self.failures = failures
