#!/usr/bin/env python3
"""
Evaluate a small RAG dataset with an OCI Generative AI model.

Ragas reaches OCI through LiteLLM's ``oci/`` provider, so the evaluator is a
plain ``llm_factory("oci/<model id>")``.

Prerequisites (see docs/howtos/integrations/oci_genai.md):
    export OCI_USER=ocid1.user.oc1..example
    export OCI_FINGERPRINT=your_fingerprint
    export OCI_TENANCY=ocid1.tenancy.oc1..example
    export OCI_KEY_FILE=~/.oci/private_key.pem
    export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..example
    export OCI_REGION=us-chicago-1
    export OCI_MODEL_ID=cohere.command-latest  # optional
"""

import os

from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, Faithfulness

SAMPLES = [
    {
        "user_input": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "retrieved_contexts": [
            "France is a country in Europe. Its capital is Paris. "
            "France is known for its culture and cuisine."
        ],
        "reference": "Paris",
    },
    {
        "user_input": "Who wrote Romeo and Juliet?",
        "response": "William Shakespeare wrote Romeo and Juliet.",
        "retrieved_contexts": [
            "Romeo and Juliet is a famous play written by William Shakespeare. "
            "It's a tragic love story."
        ],
        "reference": "William Shakespeare",
    },
    {
        "user_input": "What is the largest planet in our solar system?",
        "response": "Jupiter is the largest planet in our solar system.",
        "retrieved_contexts": [
            "Jupiter is the largest planet in our solar system. "
            "It's a gas giant with many moons."
        ],
        "reference": "Jupiter",
    },
]


def main():
    model_id = os.getenv("OCI_MODEL_ID", "cohere.command-latest")
    llm = llm_factory(f"oci/{model_id}")

    dataset = EvaluationDataset.from_list(SAMPLES)
    metrics = [Faithfulness(llm=llm), ContextPrecision(llm=llm)]

    result = evaluate(dataset, metrics=metrics)
    print(result)
    print(result.to_pandas()[[m.name for m in metrics]])


if __name__ == "__main__":
    if not os.getenv("OCI_COMPARTMENT_ID"):
        raise SystemExit(
            "Set the OCI_* environment variables first; "
            "see docs/howtos/integrations/oci_genai.md."
        )
    main()
