# Behavior changes from ragas

ragas-ng fixes bugs in legacy ragas even where code that depended on the old
behavior has to change. This page lists those changes, apart from score
changes, which are listed in [Metric score changes](score_changes.md). Each
entry says what changed and what, if anything, you need to do.

Entries marked **⚠ silent** don't raise an error, so read them first. Your
code keeps running, but it may now do something different.

## Running sync APIs: no more nest_asyncio ([#80](https://github.com/arpieb/ragas-ng/issues/80))

ragas no longer depends on or imports `nest_asyncio`. Every sync entry point
(`evaluate()`, `metric.score()`, `batch_score()`, `@metric` metrics, the
executor and testset transforms) runs on one shared background event loop. They
work inside Jupyter and async frameworks without patching the running loop.
Repeated calls also reuse one loop, so async clients such as `AsyncOpenAI`
stay bound to it. A second `score()` used to fail with "bound to a different
event loop".

| Before | Now | What to do |
|---|---|---|
| `evaluate(..., allow_nest_asyncio=False)` | accepted, ignored, `DeprecationWarning` | remove the argument |
| `ragas.async_utils.apply_nest_asyncio()` | no-op returning `False`, `DeprecationWarning` | remove the call |
| `score()` inside a running loop raised "Cannot call sync score() from an async context" | works | nothing |

**One case to avoid:** in a notebook, if you `await` an async client yourself
on the notebook's loop and then pass the same client to sync `evaluate()`, the
client is bound to the notebook's loop, not ragas'. Use `await aevaluate(...)`
in async code.

## evaluate() leaves your objects alone ([#76](https://github.com/arpieb/ragas-ng/issues/76))

`evaluate()` works on copies of your metrics and never sets anything on your
LLM. The `run_config` and token counting apply only to calls made during that
run, including jobs from `return_executor=True`.

- **⚠ silent:** `metric.llm` and `metric.embeddings` stay as you left them
  (often `None`) after `evaluate(llm=...)`. Code that read `metric.llm` after a
  run to find the LLM it used now gets `None`.
- **⚠ silent:** without `run_config=`, `evaluate()` leaves your LLM's own
  `set_run_config(...)` in effect. It used to overwrite it with defaults.
- Two concurrent `evaluate()` calls on shared metrics or a shared LLM no longer
  mix up their LLMs, settings or token counts.
- `TestsetGenerator` works the same way, and no longer leaves its token
  counter attached to your LLM.

## evaluate() inputs and results ([#77](https://github.com/arpieb/ragas-ng/issues/77), [#78](https://github.com/arpieb/ragas-ng/issues/78), [#79](https://github.com/arpieb/ragas-ng/issues/79))

| Input or call | Before | Now |
|---|---|---|
| `dataset=[SingleTurnSample(...), ...]` or a list of dicts | failed later | converted to an `EvaluationDataset` |
| an empty list, or a list mixing types | failed later | `ValueError` / `TypeError` before the run |
| a raw provider client as `llm=` / `embeddings=` | failed during the run | `TypeError` before the run, pointing to `llm_factory()` / `embedding_factory()` |
| `metrics=(m1, m2)` (a tuple) | rejected | accepted |
| a prompt that runs more than once per sample in traces | only the last run was kept | `name`, `name (2)`, `name (3)`, ... |
| `total_tokens()` / `total_cost()` with no usage recorded | `IndexError` | zero, with a warning |

## Testset generation: size, failures and personas ([#81](https://github.com/arpieb/ragas-ng/issues/81), [#82](https://github.com/arpieb/ragas-ng/issues/82), [#84](https://github.com/arpieb/ragas-ng/issues/84))

- **⚠ silent: `testset_size` is an upper bound.** Testsets used to overshoot
  it, for example 12 samples for 10, or 3 for 1. The size is now split exactly
  across the synthesizers. You can get fewer samples if the knowledge graph
  is too small, and a warning then gives the count and the reason.
- With `raise_exceptions=False`, a failed scenario or sample is left out, with
  a warning. It used to crash generation and lose all the completed work.
- Persona names returned by the LLM match ignoring case and spacing. Names
  that match no persona are skipped instead of raising `KeyError`.
- Personas generated on the first `generate()` call are still saved on the
  generator as `persona_list` and reused, but a list you supplied is no longer
  shuffled in place. New: `testset.personas` holds the personas used.
- `with_debugging_logs=True` now turns on the `ragas.testset.*` loggers. It
  had no effect before.

## Testset generation: building the knowledge graph ([#83](https://github.com/arpieb/ragas-ng/issues/83), [#85](https://github.com/arpieb/ragas-ng/issues/85)–[#88](https://github.com/arpieb/ragas-ng/issues/88))

- **⚠ silent: every document gets a chunk.** `HeadlineSplitter` turns a
  document it doesn't split into one CHUNK node. It used to add the document
  a second time under the same id, and that document never got a chunk to
  generate questions from. If you load a knowledge graph saved with an older
  version, its unsplit documents still have no chunks.
- `default_transforms` handles mixed-length corpora:
  - The long-document pipeline no longer crashes on documents under 500
    tokens.
  - The medium pipeline is used whenever any document is over 100 tokens.
  - An empty list raises `ValueError`.
  - If every document is 100 tokens or shorter, the error points to
    `generate_with_chunks()`.
- `CustomNodeFilter` scores chunks you supplied yourself using their own
  summary. It used to skip them with a warning.
- `apply_transforms` stops at the first failure. It cancels the step's
  remaining tasks, doesn't run later steps, and raises the error. It used to
  run every task in the step first. There is no rollback, so retry on a fresh
  `KnowledgeGraph`.

## Testset generation: cost and metadata ([#90](https://github.com/arpieb/ragas-ng/issues/90), [#91](https://github.com/arpieb/ragas-ng/issues/91))

- **⚠ silent:** `testset.total_tokens()` / `total_cost()` from the
  `generate_with_*` methods now include building the knowledge graph, usually
  most of the cost, so the totals are higher. The transforms also get the
  run's `RunConfig` timeouts and retries.
- `generate_with_llamaindex_docs` without `run_config=` keeps your LLM's own
  settings. It used to apply defaults.
- New field `SingleTurnSample.reference_contexts_metadata`: the source
  document's metadata for each reference context, kept as JSON-safe values.
  It is a new column in `to_pandas()` output. If you write CSV from datasets
  where only some samples have the field, `to_csv()` can fail on the mixed
  columns.

## Few-shot examples ([#92](https://github.com/arpieb/ragas-ng/issues/92), [#93](https://github.com/arpieb/ragas-ng/issues/93))

- **⚠ silent: the similarity threshold applies.**
  `FewShotPydanticPrompt.threshold_for_examples`, which
  `DemonstrationConfig.threshold` sets during `metric.train(...)`, never
  reached the example store, so its default of 0.7 always applied. It does
  now.
- **⚠ silent:** `DynamicFewShotPrompt` shows no examples when none reach
  `similarity_threshold`. It used to fall back to the most recent examples.
  See [Metric score changes](score_changes.md#dynamicfewshotprompt-no-examples-below-the-threshold-93).
- Asking for 0 examples or history entries returns none. This applies to
  `top_k=0`, `max_similar_examples=0` and DSPy `inspect_history(0)`. `[-0:]`
  used to return them all. A negative count raises `ValueError`.
- `metric.train(...)` with annotations whose output fails validation no longer
  shifts the remaining examples. Before, each later input was paired with the
  previous annotation's output.
- **Custom example stores:** the abstract `ExampleStore.get_examples` and
  `SimpleExampleStore.get_examples` now take `threshold`. A store without that
  parameter still works, but the threshold is ignored and a warning says so
  once. Add `threshold: float = 0.7` to your store to use it.
