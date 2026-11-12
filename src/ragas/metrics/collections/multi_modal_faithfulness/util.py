"""Utility functions and prompt classes for MultiModalFaithfulness metric."""

import typing as t

from pydantic import BaseModel, Field

from ragas.metrics.collections.image_sources import (
    ImageSourcePolicy,
    process_image_to_base64,
)


class MultiModalFaithfulnessInput(BaseModel):
    """Input model for multimodal faithfulness evaluation."""

    response: str = Field(..., description="The response to evaluate for faithfulness")
    retrieved_contexts: t.List[str] = Field(
        ...,
        description="List of retrieved contexts (text or image paths/URLs)",
    )


class MultiModalFaithfulnessOutput(BaseModel):
    """Output model for multimodal faithfulness evaluation."""

    faithful: bool = Field(
        ...,
        description="True if the response is faithful to the contexts, False otherwise",
    )
    reason: str = Field(
        default="",
        description="Explanation for the faithfulness verdict",
    )


def build_multimodal_content(
    prompt_text: str,
    retrieved_contexts: t.List[str],
    closing_text: str,
    image_policy: t.Optional[ImageSourcePolicy] = None,
) -> t.List[t.Dict[str, t.Any]]:
    """OpenAI-format content parts: ``prompt_text``, each context, ``closing_text``.

    A context that resolves to an image (see ``process_image_to_base64``) is
    sent as an ``image_url`` part with a data URI, followed by an
    ``[Image context N]`` label; any other context as ``Context N: ...`` text.
    ``image_policy`` limits where images are loaded from (the default reads
    no local files and fetches public URLs only; see ``ImageSourcePolicy``).
    """
    content: t.List[t.Dict[str, t.Any]] = [{"type": "text", "text": prompt_text}]

    for i, ctx in enumerate(retrieved_contexts):
        image_data = process_image_to_base64(ctx, image_policy)
        if image_data:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['encoded_data']}"
                    },
                }
            )
            content.append({"type": "text", "text": f"[Image context {i + 1}]"})
        else:
            content.append({"type": "text", "text": f"Context {i + 1}: {ctx}"})

    content.append({"type": "text", "text": closing_text})
    return content


def build_multimodal_message_content(
    instruction: str,
    response: str,
    retrieved_contexts: t.List[str],
    image_policy: t.Optional[ImageSourcePolicy] = None,
) -> t.List[t.Dict[str, t.Any]]:
    """
    Build multimodal message content for the LLM.

    Args:
        instruction: The evaluation instruction
        response: The response to evaluate
        retrieved_contexts: List of contexts (text or image references)
        image_policy: Where images may be loaded from (``ImageSourcePolicy``)

    Returns:
        List of content blocks for the message
    """
    prompt_text = f"""{instruction}

Response to evaluate:
{response}

Retrieved contexts:
"""
    return build_multimodal_content(
        prompt_text,
        retrieved_contexts,
        "\n\nBased on the above contexts (both visual and textual), determine if the response is faithful. A response is faithful if all claims can be inferred from the provided contexts.",
        image_policy,
    )


# Instruction for the prompt
MULTIMODAL_FAITHFULNESS_INSTRUCTION = """You are evaluating whether a response is faithful to the provided context information.

A response is considered FAITHFUL if:
- All claims in the response can be directly inferred from the visual or textual context
- The response does not contain information that contradicts the context
- The response does not hallucinate facts not present in the context

A response is considered NOT FAITHFUL if:
- It contains claims that cannot be verified from the context
- It contradicts information in the context
- It makes up facts not supported by the context

You must evaluate faithfulness based on BOTH visual (images) and textual context if provided."""
