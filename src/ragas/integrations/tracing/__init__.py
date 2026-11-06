"""
Tracing integrations for Ragas evaluation framework.

ragas emits OpenTelemetry spans (OpenInference conventions), so any
OTel-compatible backend -- Langfuse, Phoenix/Arize, Opik, MLflow, Jaeger,
Datadog -- receives ragas runs with no ragas-specific integration code.

Modules here:
- ``mlflow``: ``sync_trace()`` / ``MLflowTrace`` read back the last MLflow
  trace (requires the ``tracing`` extra).
- ``langfuse``: **deprecated** (#97). Use the Langfuse SDK directly; see the
  Langfuse integration guide.

Example (MLflow):
    ```python
    from ragas.integrations.tracing.mlflow import sync_trace
    from ragas import evaluate
    import mlflow

    with mlflow.start_run():
        result = evaluate(dataset, metrics)
        trace = await sync_trace()
        print(trace.get_url())
    ```
"""

# Type stubs for pyright - these won't execute but provide type information
if False:
    from .langfuse import (  # noqa: F401
        LangfuseTrace,
        add_query_param,
        logger,
        observe,
        sync_trace,
    )
    from .mlflow import MLflowTrace  # noqa: F401


# Lazy imports to handle optional dependencies gracefully
def __getattr__(name: str):
    if name in ["observe", "logger", "LangfuseTrace", "sync_trace", "add_query_param"]:
        from .langfuse import (
            LangfuseTrace,
            add_query_param,
            logger,
            observe,
            sync_trace,
        )

        if name == "observe":
            return observe
        elif name == "logger":
            return logger
        elif name == "LangfuseTrace":
            return LangfuseTrace
        elif name == "sync_trace":
            return sync_trace
        elif name == "add_query_param":
            return add_query_param
    elif name == "MLflowTrace":
        from .mlflow import MLflowTrace

        return MLflowTrace
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
