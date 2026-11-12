## MultiModalFaithfulness

`MultiModalFaithfulness` metric measures the factual consistency of the generated answer against both visual and textual context. It is calculated from the answer, retrieved textual context, and visual context. The answer is scaled to a (0,1) range, with higher scores indicating better faithfulness.

The generated answer is regarded as faithful if all the claims made in the answer can be inferred from either the visual or textual context provided. To determine this, the response is directly evaluated against the provided contexts, and the faithfulness score is either 0 or 1.

### Example

```python
from ragas.llms.base import llm_factory
from ragas.metrics.collections import ImageSourcePolicy, MultiModalFaithfulness

# Setup - use a vision-capable model
llm = llm_factory("gpt-6-luna")  # Vision-capable model required

# Create metric instance. Local image paths are read only from the
# directory you name; URLs and data URIs need no policy.
metric = MultiModalFaithfulness(
    llm=llm, image_policy=ImageSourcePolicy(local_dir="path/to")
)

# Evaluate faithfulness
result = await metric.ascore(
    response="The Tesla Model X is an electric SUV.",
    retrieved_contexts=[
        "path/to/tesla_image.jpg",  # Image context
        "Tesla manufactures electric vehicles.",  # Text context
    ],
)
print(f"Faithfulness Score: {result.value}")  # 1.0 (faithful) or 0.0 (not faithful)
```

### How It's Calculated

!!! example
    **Question**: What about the Tesla Model X?

    **Context (visual)**:
    - An image of the Tesla Model X (custom_eval/multimodal/images/tesla.jpg)

    **High faithfulness answer**: The Tesla Model X is an electric SUV manufactured by Tesla.

    **Low faithfulness answer**: Cats are cute.

Let's examine how faithfulness was calculated using the low faithfulness answer:

- **Step 1:** Evaluate the generated response against the given contexts.
    - Response: "Cats are cute."

- **Step 2:** Verify if the response can be inferred from the given context.
    - Response: No

- **Step 3:** Use the result to determine the faithfulness score.

    $$
    \text{Faithfulness} = 0
    $$

In this example, the response "Cats are cute" cannot be inferred from the image of the Tesla Model X, so the faithfulness score is 0.

### Supported Context Types

The metric supports multiple types of context inputs:

- **Text contexts**: Plain text strings
- **Image URLs**: HTTP/HTTPS URLs pointing to images on public hosts
- **Local image paths**: File paths to local images (jpg, png, gif, webp, bmp), read only from the directory given as `ImageSourcePolicy(local_dir=...)`
- **Base64 data URIs**: Inline base64-encoded images

A context that is not loaded as an image is sent to the LLM as text.

#### Where images are loaded from

Image references come from your evaluation data, so by default the metric:

- reads **no local files**. Pass `image_policy=ImageSourcePolicy(local_dir="images/")` to read images from that directory. Relative paths are resolved against it, and a path that leads outside it, through `..` or a symlink, is refused.
- fetches a URL only when its host resolves to **public addresses**. Private, loopback, link-local and cloud-metadata addresses are refused, and every redirect is checked again. For images on an internal server, pass `ImageSourcePolicy(allow_private_urls=True)`.
- accepts images up to 10 MB (`max_bytes=`), and only content that decodes as an image.

The address check resolves the host separately from the connection, so it does not stop DNS rebinding. If the data is untrusted, also limit outbound network access where the evaluation runs.

### Requirements

- A vision-capable LLM is required (e.g., `gpt-6-luna`, `claude-opus-5-5`, `gemini-3.6-flash`)
- Use `llm_factory` to create the LLM instance
