"""
Tracing integrations for Ragas evaluation framework.

ragas emits OpenTelemetry spans (OpenInference conventions), so any
OTel-compatible backend -- Langfuse, Phoenix/Arize, Opik, MLflow, Jaeger,
Datadog -- receives ragas runs with no ragas-specific integration code.

Modules here:
- ``mlflow``: ``sync_trace()`` / ``MLflowTrace`` read back the last MLflow
  trace (requires the ``tracing`` extra).

The Langfuse helper (``ragas.integrations.tracing.langfuse``) was removed
(#97): it never fetched real traces. Use the Langfuse SDK directly; see the
Langfuse integration guide.

Example (MLflow):
    ```python
    from ragas.integrations.tracing.mlflow import sync_trace
    from ragas import metrics_experiment
    import mlflow

    with mlflow.start_run():
        exp = await metrics_experiment(metrics).arun(dataset)
        trace = await sync_trace()
        print(trace.get_url())
    ```
"""

# Type stubs for pyright - these won't execute but provide type information
if False:
    from .mlflow import MLflowTrace  # noqa: F401

_REMOVED_LANGFUSE = {
    "observe",
    "logger",
    "LangfuseTrace",
    "sync_trace",
    "add_query_param",
}


# Lazy imports to handle optional dependencies gracefully
def __getattr__(name: str):
    if name == "MLflowTrace":
        from .mlflow import MLflowTrace

        return MLflowTrace
    if name in _REMOVED_LANGFUSE:
        raise AttributeError(
            f"ragas.integrations.tracing.{name} was removed with the Langfuse "
            "helper (#97), which never fetched real traces. Use the Langfuse "
            "SDK directly (`from langfuse import observe, get_client`); ragas "
            "runs reach Langfuse as OpenTelemetry spans. See the Langfuse "
            "integration guide."
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
