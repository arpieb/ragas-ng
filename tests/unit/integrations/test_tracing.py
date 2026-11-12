"""
Comprehensive test suite for tracing integrations.

Tests the MLflow integration with proper mocking to avoid external
dependencies. The removed Langfuse helper is covered by
test_langfuse_removed.py.
"""

import os
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest


class TestMLflowIntegration:
    """Test suite for MLflow tracing integration."""

    def test_mlflow_imports_with_missing_dependency(self):
        """Test that imports work gracefully when mlflow is not available."""
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.entities": None}):
            from ragas.integrations.tracing.mlflow import MLflowTrace, sync_trace

            assert MLflowTrace is not None
            assert callable(sync_trace)

    def test_mlflow_imports_with_dependency_available(self):
        """Test imports when mlflow is available."""
        mock_mlflow = MagicMock()
        mock_entities = MagicMock()

        with patch.dict(
            "sys.modules", {"mlflow": mock_mlflow, "mlflow.entities": mock_entities}
        ):
            from ragas.integrations.tracing.mlflow import MLflowTrace

            assert MLflowTrace is not None

    def test_mlflow_trace_initialization(self):
        """Test MLflowTrace initialization with mock trace."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock instead of trying to instantiate the real class
        mock_trace = MagicMock()
        mlflow_trace = MLflowTrace(mock_trace)
        assert mlflow_trace.trace == mock_trace

    def test_mlflow_trace_get_url_with_env(self):
        """Test MLflowTrace URL generation with MLFLOW_HOST set."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock for the trace object
        mock_trace = MagicMock()
        mock_trace.info = MagicMock()
        mock_trace.info.request_id = "test-request-id"
        mock_trace.info.experiment_id = "test-experiment-id"

        with patch.dict(os.environ, {"MLFLOW_HOST": "https://mlflow.example.com/"}):
            mlflow_trace = MLflowTrace(mock_trace)
            url = mlflow_trace.get_url()

        assert url == (
            "https://mlflow.example.com/#/experiments/test-experiment-id"
            "?compareRunsMode=TRACES&selectedTraceId=test-request-id"
        )
        parsed = urlparse(url)
        assert (parsed.scheme, parsed.netloc, parsed.path) == (
            "https",
            "mlflow.example.com",
            "/",
        )

    @pytest.mark.parametrize(
        "host, base",
        [
            ("http://localhost:5000", "http://localhost:5000"),
            ("https://example.com/mlflow/", "https://example.com/mlflow"),
        ],
    )
    def test_mlflow_trace_get_url_keeps_port_and_path_prefix(self, host, base):
        from ragas.integrations.tracing.mlflow import MLflowTrace

        mock_trace = MagicMock()
        mock_trace.info.request_id = "tr-1"
        mock_trace.info.experiment_id = "7"
        with patch.dict(os.environ, {"MLFLOW_HOST": host}):
            url = MLflowTrace(mock_trace).get_url()
        assert url == (
            f"{base}/#/experiments/7?compareRunsMode=TRACES&selectedTraceId=tr-1"
        )

    def test_mlflow_trace_get_url_encodes_ids(self):
        """IDs can't add query parameters or path segments (#243)."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        mock_trace = MagicMock()
        mock_trace.info.request_id = "tr-1&next=https://evil.example"
        mock_trace.info.experiment_id = "1/../2?x=y#z"
        with patch.dict(os.environ, {"MLFLOW_HOST": "https://mlflow.example.com"}):
            url = MLflowTrace(mock_trace).get_url()
        fragment = urlparse(url).fragment
        path, _, query = fragment.partition("?")
        assert path == "/experiments/1%2F..%2F2%3Fx%3Dy%23z"
        assert parse_qs(query) == {
            "compareRunsMode": ["TRACES"],
            "selectedTraceId": ["tr-1&next=https://evil.example"],
        }

    @pytest.mark.parametrize(
        "host",
        [
            "mlflow.example.com",
            "ftp://mlflow.example.com",
            "javascript:alert(1)",
            "https://",
            "https://mlflow.example.com/?a=b",
            "https://mlflow.example.com/#x",
        ],
    )
    def test_mlflow_trace_get_url_rejects_bad_hosts(self, host):
        from ragas.integrations.tracing.mlflow import MLflowTrace

        with patch.dict(os.environ, {"MLFLOW_HOST": host}):
            with pytest.raises(ValueError, match="MLFLOW_HOST must"):
                MLflowTrace(MagicMock()).get_url()

    def test_mlflow_trace_get_url_no_env(self):
        """Test MLflowTrace URL generation without MLFLOW_HOST."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock for the trace object
        mock_trace = MagicMock()
        mlflow_trace = MLflowTrace(mock_trace)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="MLFLOW_HOST environment variable is not set"
            ):
                mlflow_trace.get_url()

    def test_mlflow_trace_filter(self):
        """Test MLflowTrace span filtering."""
        from ragas.integrations.tracing.mlflow import MLflowTrace

        # Use MagicMock for both span and trace objects
        mock_span = MagicMock()
        mock_span.name = "test-span"

        mock_trace = MagicMock()
        mock_trace.search_spans = MagicMock(return_value=[mock_span])

        mlflow_trace = MLflowTrace(mock_trace)
        filtered_spans = mlflow_trace.get_filter("test-span")

        assert len(filtered_spans) == 1
        assert filtered_spans[0] == mock_span
        mock_trace.search_spans.assert_called_once_with(name="test-span")

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_success(self):
        """Test successful MLflow trace synchronization."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with (
            patch(
                "ragas.integrations.tracing.mlflow.get_last_active_trace_id"
            ) as mock_get_id,
            patch("ragas.integrations.tracing.mlflow.get_trace") as mock_get_trace,
        ):
            mock_get_id.return_value = "test-trace-id"
            mock_trace = MagicMock()
            mock_get_trace.return_value = mock_trace

            result = await sync_trace()

            assert result is not None
            assert result.trace == mock_trace
            mock_get_id.assert_called_once()
            mock_get_trace.assert_called_once_with("test-trace-id")

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_no_active_trace(self):
        """Test MLflow sync_trace when no active trace exists."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with patch(
            "ragas.integrations.tracing.mlflow.get_last_active_trace_id"
        ) as mock_get_id:
            mock_get_id.return_value = None

            with pytest.raises(ValueError, match="No active trace found"):
                await sync_trace()

    @pytest.mark.asyncio
    async def test_mlflow_sync_trace_not_found(self):
        """Test MLflow sync_trace when trace is not found."""
        from ragas.integrations.tracing.mlflow import sync_trace

        with (
            patch(
                "ragas.integrations.tracing.mlflow.get_last_active_trace_id"
            ) as mock_get_id,
            patch("ragas.integrations.tracing.mlflow.get_trace") as mock_get_trace,
        ):
            mock_get_id.return_value = "test-trace-id"
            mock_get_trace.return_value = None

            with pytest.raises(ValueError, match="Trace not found"):
                await sync_trace()


class TestTracingIntegrationInitModule:
    """Test the tracing integration __init__ module."""

    def test_lazy_import_mlflow_classes(self):
        """Test lazy imports for MLflow classes."""
        from ragas.integrations.tracing import MLflowTrace

        assert MLflowTrace is not None

    def test_invalid_attribute_access(self):
        """Test that accessing non-existent attributes raises AttributeError."""
        import ragas.integrations.tracing as tracing

        with pytest.raises(AttributeError, match="has no attribute 'non_existent'"):
            _ = tracing.non_existent


if __name__ == "__main__":
    pytest.main([__file__])
