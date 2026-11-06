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
- **Repeatable personas across runs.** Persona generation runs at
  temperature 1.0 by design, so each new generator or script run gets a
  different set (see [Prompt output handling](#prompt-output-handling-94)). To
  get the same personas, and so the same mix of questions, every time, save
  them once and pass them back:

    ```python
    import json
    from ragas.testset.persona import Persona

    # first run
    with open("personas.json", "w") as f:
        json.dump([p.model_dump() for p in testset.personas], f)

    # later runs
    with open("personas.json") as f:
        personas = [Persona(**p) for p in json.load(f)]
    generator = TestsetGenerator(llm=llm, embedding_model=emb, persona_list=personas)
    ```
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

## Prompt output handling ([#94](https://github.com/arpieb/ragas-ng/issues/94))

This applies to `PydanticPrompt` and `ImageTextPrompt`: the prompt classes
behind testset generation, the private legacy metric modules, and metrics with
customised prompts.

- **Error type: parse failures raise `RagasOutputParserException`.** This
  covers a `ValidationError` or `JSONDecodeError`, and instructor's
  retries-exhausted errors with an `llm_factory()` LLM. Some of these used to
  escape as their raw types. The original is kept as `__cause__`, and the
  message includes it. If you caught `pydantic.ValidationError` around a
  prompt call, catch `RagasOutputParserException` instead. Provider errors
  such as timeouts, rate limits and connection errors still pass through
  unchanged.
- Every failure is reported to callbacks with `on_chain_error`, so traces no
  longer leave a failed prompt run open.
- **`retries_left` is the number of fix attempts.** When the LLM's output
  doesn't parse, ragas asks the LLM to fix it up to `retries_left` times
  (default 3). Each fixed output is parsed like the original, including JSON
  inside code fences or surrounding text. Before, there was one fix attempt,
  and its result failed if it came back fenced.
- Output that is already valid JSON is parsed as is. JSON with a fenced block
  inside one of its string values used to be cut down to that inner block.
- **⚠ silent: `llm_factory()` LLMs run prompts concurrently.** The prompt
  called the blocking `generate()` for an LLM on a sync client, so
  concurrent prompts, such as every testset transform, ran one at a time. It
  now awaits `agenerate()`, which runs sync clients in a thread. Expect
  faster testset generation and more parallel requests to your provider. Set
  `RunConfig(max_workers=...)` if you need to limit them.
- **⚠ silent: `n` and `temperature` apply with `llm_factory()` LLMs.**
  - `generate_multiple(n=3)` makes three calls at a sampling temperature,
    0.3 unless you pass one. It used to make one call and warn.
  - A `temperature=` passed to a prompt is now used. It used to be ignored.
  - **Testset personas vary between runs again.** Persona generation has
    asked for temperature 1.0 since it was added, so that personas come out
    varied. That worked with the legacy LangChain-style LLM wrappers, which
    honour the temperature. The code path for `llm_factory()` LLMs, added
    later, dropped it, so those users got nearly the same personas every
    time without anyone choosing that. The original design applies again: a
    new generator or script run gets a different set of personas. To keep
    them fixed, see
    [Repeatable personas across runs](#testset-generation-size-failures-and-personas-81-82-84).
  - See [Metric score changes](score_changes.md#legacy-prompts-with-llm_factory-llms-n-and-temperature-apply-94).

## `@metric` function arguments ([#95](https://github.com/arpieb/ragas-ng/issues/95))

This applies to metrics created with `@discrete_metric`, `@numeric_metric`,
`@ranking_metric` or `create_metric_decorator()`.

- **⚠ silent: arguments arrive as the objects you passed.** The decorator
  validates inputs with pydantic and then called your function with
  `model_dump()`, which turned pydantic-model arguments, such as a
  `SingleTurnSample` or your own `BaseModel`, into plain dicts. Your function
  now receives the objects themselves. Coercion still applies, so `"2.5"`
  becomes `2.5` for a `float` parameter. If your function worked around this
  by indexing a dict (`sample["user_input"]`), switch to attribute access
  (`sample.user_input`).
- **⚠ silent: concurrent calls no longer share inputs.** The validated inputs
  were stored on the metric object between validation and the call. Two calls
  on one metric from different event loops, such as `aevaluate()` on your loop
  alongside `score()`, could run with each other's inputs and return the wrong
  score. Each call now keeps its own inputs.
- If the inputs can't be validated, for example because of a type hint
  pydantic can't resolve, the function is called with your arguments and a
  warning is issued once. Before, it either crashed `score()` or called the
  function with no arguments, and the error was returned as
  `MetricResult(value=None)`.

## CSV files: every row's columns, JSON cells, typed loads ([#100](https://github.com/arpieb/ragas-ng/issues/100))

This applies to `LocalCSVBackend` (`"local/csv"`) and
`EvaluationDataset.to_csv()`.

- **Rows may differ.** The header is the union of every row's keys, in
  first-seen order, and a missing value is an empty cell. Before, any row with
  a key the first row lacked raised `ValueError: dict contains fields not in
  fieldnames`. Because `@experiment` saves its results at the end, that lost
  the whole run's results.
- **⚠ silent: lists and dicts are JSON text.** Cells used to hold Python
  `repr`, such as `['a', 'b']`. They now hold JSON, such as `["a", "b"]`. If
  you parse these cells yourself with `ast.literal_eval`, use `json.loads`.
- **Typed loads restore types.** `Dataset.load(..., data_model=Row)`,
  `reload()` and `validate_with(Row)` decode each cell using the field's type
  before validation, so lists, dicts, nested models, numbers, booleans, dates
  and empty optional fields load back. Before, any list, dict or
  `Optional[...] = None` field failed validation.
- `str` fields are never changed. CSV can't tell `""` from `None`, so an
  `Optional[str]` saved as `None` loads as `""`. Without a `data_model`, CSV
  loads still return strings. Use `"local/jsonl"` to keep types without a
  model.
- `Dataset.save()` writes model rows with `model_dump(mode="json")`, so
  values such as UUIDs, enums and paths are saved as strings, and the model
  parses them back. Before, they could fail to save to JSONL.

## Backends load what was stored ([#101](https://github.com/arpieb/ragas-ng/issues/101), [#102](https://github.com/arpieb/ragas-ng/issues/102))

Backends no longer guess types on load. Types come from the `data_model` you
load with.

- **⚠ silent: JSONL keeps strings as strings.** `LocalJSONLBackend`
  (`"local/jsonl"`) turned any string that looked like an ISO date, such as
  `"2024-01-02"`, into a `date` or `datetime` on load. That changed your
  text, and a model's `str` field then rejected it. Now:
    - Loading with a `data_model`: `date` / `datetime` fields parse their ISO
      strings and `str` fields keep their text, so typed datasets round-trip
      exactly.
    - Loading without one: dates you saved come back as ISO strings, such as
      `"2024-01-15T10:30:00"`. Use `datetime.fromisoformat(...)`, or load with
      a model.
- **A malformed JSONL line raises `ValueError`** naming the line and file.
  It used to be skipped with a `print()`, so rows disappeared silently. Blank
  lines are still skipped.
- **⚠ silent: Google Drive cells stay text.** `GDriveBackend` turned
  numeric-looking cells into numbers on load, so `"007"` became `7`. Cells
  are now strings unless a `data_model` types them, as with CSV. Also:
    - `None` is saved as an empty cell instead of the text `"None"`.
    - Lists and dicts are saved as JSON text.
    - Loads read every column. Only columns A to Z were read before, so wider
      datasets silently lost columns.
- **Dataset files are UTF-8.** `to_csv()`, `to_jsonl()`, `from_jsonl()`,
  `MetricAnnotation.from_json()` and `SingleMetricAnnotation.from_json()` read
  and write UTF-8 and close their files. They used the platform encoding,
  which breaks non-ASCII text on Windows. `from_jsonl()` ignores blank lines.
