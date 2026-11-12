## MultiModalRelevance

`MultiModalRelevance` metric measures the relevance of the generated answer against both visual and textual context. It is calculated from the user input, response, and retrieved contexts (both visual and textual). The answer is scaled to a (0,1) range, with higher scores indicating better relevance.

The generated answer is regarded as relevant if it aligns with the visual or textual context provided. To determine this, the response is directly evaluated against the provided contexts, and the relevance score is either 0 or 1.

### Example

```python
from ragas.llms.base import llm_factory
from ragas.metrics.collections import ImageSourcePolicy, MultiModalRelevance

# Setup - use a vision-capable model
llm = llm_factory("gpt-6-luna")  # Vision-capable model required

# Create metric instance. Local image paths are read only from the
# directory you name; URLs and data URIs need no policy.
metric = MultiModalRelevance(
    llm=llm, image_policy=ImageSourcePolicy(local_dir="path/to")
)

# Evaluate relevance
result = await metric.ascore(
    user_input="What about the Tesla Model X?",
    response="The Tesla Model X is an electric SUV.",
    retrieved_contexts=[
        "path/to/tesla_image.jpg",  # Image context
        "Tesla manufactures electric vehicles.",  # Text context
    ],
)
print(f"Relevance Score: {result.value}")  # 1.0 (relevant) or 0.0 (not relevant)
```

### How It's Calculated

!!! example
    **Question**: What about the Tesla Model X?

    **Context (visual)**:
    - An image of the Tesla Model X (custom_eval/multimodal/images/tesla.jpg)

    **High relevance answer**: The Tesla Model X is an electric SUV manufactured by Tesla.

    **Low relevance answer**: Cats are cute.

Let's examine how relevance was calculated using the low relevance answer:

- **Step 1:** Evaluate the generated response against the given contexts.
    - Response: "Cats are cute."

- **Step 2:** Verify if the response aligns with the given context.
    - Response: No

- **Step 3:** Use the result to determine the relevance score.

    $$
    \text{Relevance} = 0
    $$

In this example, the response "Cats are cute" does not align with the image of the Tesla Model X, so the relevance score is 0.

### Supported Context Types

The metric supports multiple types of context inputs:

- **Text contexts**: Plain text strings
- **Image URLs**: HTTP/HTTPS URLs pointing to images on public hosts
- **Local image paths**: File paths ending in an image extension (jpg, jpeg, png, gif, webp, bmp), read only from the directory given as `ImageSourcePolicy(local_dir=...)`
- **Base64 data URIs**: Inline base64-encoded images

Any other context is sent to the LLM as text. An image reference that can't be loaded (refused by the policy below, missing, not an image, or a failed download) raises `ragas.exceptions.ImageLoadError` before the LLM is called, so you never get a score from a judge that didn't see the image. The message says which context failed and why. To score anyway, with the reference sent as text, pass `ImageSourcePolicy(on_blocked="warn")`, which logs a warning instead.

#### Where images are loaded from

Image references come from your evaluation data, so by default the metric:

- reads **no local files**. Pass `image_policy=ImageSourcePolicy(local_dir="images/")` to read images from that directory. Relative paths are resolved against it, and a path that leads outside it, through `..` or a symlink, is refused.
- fetches a URL only when its host resolves to **public addresses**. Private, loopback, link-local and cloud-metadata addresses are refused, and every redirect is checked again. For images on an internal server, pass `ImageSourcePolicy(allow_private_urls=True)`.
- accepts images up to 10 MB (`max_bytes=`), and only content that decodes as an image.

The address check resolves the host separately from the connection, so it does not stop DNS rebinding. If the data is untrusted, also limit outbound network access where the evaluation runs.

### Requirements

- A vision-capable LLM is required (e.g., `gpt-6-luna`, `claude-opus-5-5`, `gemini-3.6-flash`)
- Use `llm_factory` to create the LLM instance
