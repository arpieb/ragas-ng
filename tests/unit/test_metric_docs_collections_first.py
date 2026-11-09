"""Metric pages show the collections API first (#168).

Each metric page leads with ``ragas.metrics.collections`` examples and keeps
legacy examples (``single_turn_ascore`` / ``multi_turn_ascore``) under a heading
that says "legacy". A legacy-only example elsewhere on the page reads as the
recommended API, as it did for context recall and the general purpose metrics.
"""

import re
from pathlib import Path

METRIC_DOCS = (
    Path(__file__).resolve().parents[2] / "docs/concepts/metrics/available_metrics"
)

LEGACY_CALL = re.compile(r"\b(?:single|multi)_turn_ascore\(")
HEADING = re.compile(r"^(#{1,6})\s+(.*)")


def legacy_examples_outside_legacy_headings(page_text):
    """``(line, heading path)`` of legacy examples not under a legacy heading."""
    headings = {}
    hits = []
    in_fence, block, start = False, [], 0
    for number, line in enumerate(page_text.splitlines(), 1):
        if line.strip().startswith("```"):
            if in_fence:
                path = " / ".join(headings[level] for level in sorted(headings))
                if LEGACY_CALL.search("\n".join(block)) and "legacy" not in (
                    path.lower()
                ):
                    hits.append((start, path))
                block = []
            else:
                start = number
            in_fence = not in_fence
            continue
        if in_fence:
            block.append(line)
            continue
        m = HEADING.match(line)
        if m:
            level = len(m.group(1))
            headings = {k: v for k, v in headings.items() if k < level}
            headings[level] = m.group(2)
    return hits


def test_legacy_metric_examples_are_under_legacy_headings():
    hits = [
        f"{page.name}:{line}: {path}"
        for page in sorted(METRIC_DOCS.glob("*.md"))
        for line, path in legacy_examples_outside_legacy_headings(
            page.read_text(encoding="utf-8")
        )
    ]
    assert not hits, (
        "Legacy metric examples outside a 'Legacy' heading. Show the "
        "ragas.metrics.collections example first and move the legacy one under "
        "the page's legacy heading (or say on the page that no port exists):\n  "
        + "\n  ".join(hits)
    )


def test_detector():
    page = """# Metric

## Example

```python
await scorer.single_turn_ascore(sample)
```

## Legacy Metrics API

### Example with SingleTurnSample

```python
await scorer.single_turn_ascore(sample)
```

## Other metric

### Thing (legacy)

```python
await scorer.multi_turn_ascore(sample)
```

### Collections

```python
await scorer.ascore(user_input="q")
```
"""
    assert legacy_examples_outside_legacy_headings(page) == [(5, "Metric / Example")]
