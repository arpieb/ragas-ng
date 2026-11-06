"""
Integrations module for Ragas evaluation framework.

This module provides integrations with various platforms, frameworks, and tools
to enhance the Ragas evaluation experience.

Available integrations:
- Tracing: MLflow trace read-back (ragas runs reach any OpenTelemetry backend,
  such as Langfuse or Phoenix, with no integration code)
- Frameworks: LlamaIndex, Griptape
- Observability: Helicone
- Platforms: Amazon Bedrock, R2R
- AI Systems: Swarm for multi-agent evaluation
- Protocols: AG-UI for event-based agent communication

Import tracing integrations:
```python
from ragas.integrations.tracing import MLflowTrace
```
"""

# Tracing integrations are available as a submodule
# Import them explicitly when needed to handle optional dependencies gracefully
