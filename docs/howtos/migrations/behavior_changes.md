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

## Message converters: AG-UI snapshots and OpenAI-format histories ([#96](https://github.com/arpieb/ragas-ng/issues/96), [#98](https://github.com/arpieb/ragas-ng/issues/98))

- **⚠ silent: AG-UI `MessagesSnapshotEvent` tool calls keep their name and
  arguments.** `convert_messages_snapshot()` read `name` / `args` from the
  wrong place: AG-UI nests them under `function`, with arguments as a JSON
  string. So every snapshot tool call came out as `unknown_tool` with `{}`,
  and tool-call metrics such as `ToolCallAccuracy` scored those calls as
  wrong. Scores for snapshot-based samples change. Arguments that aren't
  valid JSON are kept as `{"raw_args": "..."}`, as in the streaming path.
- AG-UI message content: `None` is `""`. It used to be the text `"None"`. A
  list of content parts keeps its text parts, joined by newlines; image,
  audio, video, document and binary parts are skipped. It used to be the
  pydantic repr of the whole list.
- `ragas.integrations.swarm.convert_to_ragas_messages()` takes any
  OpenAI-format chat history, not only Swarm's:
    - An assistant message without `tool_calls` (or with `None`) no longer
      raises `KeyError`.
    - Arguments that aren't a JSON object are kept as `raw_args` instead of
      raising.
    - List-of-parts content keeps its text parts.
    - `system` and `developer` messages are skipped instead of raising
      `ValueError`.
    - An unknown or missing role still raises.

## Langfuse helper removed ([#97](https://github.com/arpieb/ragas-ng/issues/97))

`ragas.integrations.tracing.langfuse` is removed, and importing it raises
`ModuleNotFoundError`. It never worked in ragas-ng: `sync_trace()` returned a
placeholder trace with no observations, and `LangfuseTrace.filter()` always
returned `[]`. Langfuse's own SDK and OpenTelemetry cover both of its jobs
without vendor-specific code in ragas, so it is removed rather than ported to
the Langfuse SDK.

| Before | Now |
|---|---|
| `from ragas.integrations.tracing.langfuse import observe` (or `from ragas.integrations.tracing import observe`) | `from langfuse import observe` |
| `await sync_trace(trace_id)` / `LangfuseTrace(...)` | `from langfuse import get_client; get_client().api.trace.get(trace_id)` |
| `trace.filter("span_name")` | `[o for o in trace.observations if o.name == "span_name"]` on the Langfuse SDK's trace |
| `add_query_param(url, name, value)` | `urllib.parse`: `urlencode` / `urlunparse` |
| `pip install "ragas[tracing]"` installed `langfuse` | it installs only `mlflow`; `pip install langfuse` if you use it |

To see ragas runs in Langfuse, export ragas' OpenTelemetry spans to it. The
Langfuse SDK's default export filter drops spans from unknown
instrumentations, `ragas` among them, so allow the `ragas` scope with
`should_export_span`. The [Langfuse guide](../integrations/langfuse.md) shows
both set-ups and how to score your app's traces with ragas.

The MLflow helper, `ragas.integrations.tracing.mlflow`, is unchanged.

## Adapted testset prompts write in the target language ([#106](https://github.com/arpieb/ragas-ng/issues/106))

`adapt_prompts(language, ...)` recorded the target language on each prompt
but never put it in the rendered prompt. An adapted prompt carried only a
translated instruction (with `adapt_instruction=True`) and translated
examples, so the generated queries followed the language of the documents
and personas. A French corpus adapted to `"spanish"` produced a mix of
French, English and Spanish queries.

- The single-hop and multi-hop query/answer generation prompts now tell the
  model to write their output in the adapted language. Testset queries and
  reference answers come out in that language, whatever the corpus's
  language is.
- This is opt-in per prompt, through the `states_output_language` class
  attribute on `PydanticPrompt`. Other prompts don't get the line: persona
  matching, entity extraction and theme combination must copy values from
  their input exactly, and translating those values (for example, a persona
  named "curious student" coming back as "Estudiante curioso") drops every
  scenario.
- **Metric prompts are unchanged.** Adapting a metric still translates its
  instruction and examples. It doesn't promise the language of the metric's
  output (reasons, extracted statements), and scores are unaffected.
- Prompts in English, and prompts never adapted, render exactly as before.

## Experiment backends by name ([#165](https://github.com/arpieb/ragas-ng/issues/165))

`@experiment(backend=...)` and `arun(..., backend=...)` accepted a backend name
such as `"local/jsonl"`, but created it with no constructor arguments, so it
raised `TypeError` for the missing `root_dir`. Only `"inmemory"` worked by
name. Both now take `backend_kwargs`:

```python
@experiment(Result, backend="local/jsonl", backend_kwargs={"root_dir": "./data"})
async def run(row): ...


await run.arun(dataset, backend="local/csv", backend_kwargs={"root_dir": "./data"})
```

- `backend_kwargs` given to `arun` replaces the decorator's.
- `backend_kwargs` with a backend instance, or with no backend, raises
  `ValueError`. Before, there was no such parameter, so this breaks nothing.
- `backend_kwargs` is keyword-only and reserved, like `name` and `backend`. An
  experiment function with a parameter of that name can't receive it through
  `arun`.

## Experiments: concurrency, threads, order and failures ([#117](https://github.com/arpieb/ragas-ng/issues/117))

`ExperimentWrapper.arun` (`await my_experiment.arun(dataset)`) changed in four ways:

- **At most 16 rows run at once by default.** Before, every row started at
  once, so a large dataset sent all its LLM calls together and hit provider
  rate limits. Set `@experiment(max_workers=...)` or
  `arun(..., max_workers=...)`. `-1` restores the old unlimited behavior.
  Runs with many fast rows may take longer than before.
- **Sync experiment functions run in worker threads,** up to `max_workers`
  at once. Before, they ran one after another on the event loop, blocking
  it. A function that isn't thread-safe needs `max_workers=1`. Calling the
  decorated function directly (`await my_experiment(row)`) still runs it
  inline.
- **⚠ silent: results are in dataset order.** Before, they were stored in
  the order rows finished, so row *i* of the results was usually not row *i*
  of the dataset.
- **Failed rows are logged.** A row whose function raises is still left out
  of the results. It's now a warning on the `ragas.experiment` logger, with
  the row index and error, instead of a `print()`, and a final warning gives
  "N of M rows failed and were not saved".

`max_workers` is a keyword-only parameter of `arun`, like `name`, `backend`
and `backend_kwargs`, so an experiment function can't receive a parameter of
that name through `arun`.

## Retired model IDs replaced ([#176](https://github.com/arpieb/ragas-ng/issues/176))

Docs, examples, `ragas quickstart` templates and docstrings named models that their providers have shut down, or will shut down soon:
- `gpt-3.5-turbo`, `gpt-4`, `gpt-4.1-nano` and `gemini-2.0-flash`, which fail today;
- `gpt-5-mini` and `gpt-5-nano` (2026-12-11), and `claude-sonnet-4-5` (2026-11-30);
- `gpt-4o` and `gpt-4o-mini` (2027-01-14).

They now use each provider's listed replacement:

| Was | Now |
|---|---|
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` | `gpt-6-luna` |
| `gpt-3.5-turbo`, `gpt-5-mini` | `gpt-5.6-terra` |
| `gpt-4.1-nano`, `gpt-5-nano` | `gpt-5.6-luna` |
| `gpt-4`, `gpt-4-turbo`, `gpt-5`, `o1`/`o3` | `gpt-5.6-sol` |
| `claude-sonnet-4-5`, Claude 3.x Sonnet | `claude-sonnet-5-5` |
| `gemini-2.0-flash`, `gemini-1.5-flash` | `gemini-3.6-flash` |
| `gemini-1.5-pro`, `gemini-2.5-pro` | `gemini-3.8-flash` |
| `text-embedding-004` (Gemini) | `gemini-embedding-2` |

- **The default judge model changed:** see [Score changes](score_changes.md#the-default-judge-model-is-gpt-6-luna-176).
- **The new OpenAI models are all reasoning models.** `llm_factory()` already maps parameters for them. Code that calls the OpenAI client directly can't send `temperature` (other than 1) or `max_tokens` (use `max_completion_tokens`). On Chat Completions, `gpt-6-luna` supports tool calls only with `reasoning_effort="none"`. The `workflow_eval` and `agent_evals` templates were updated accordingly.
- **Docs output:** pages that show LLM output from earlier runs say so at the top. That output was produced with the old models.
- **Your own code:** if it names any of the old IDs, switch it as well. See your provider's deprecations page.

## Claude: sampling parameters ([#177](https://github.com/arpieb/ragas-ng/issues/177))

- `llm_factory(..., provider="anthropic", client=Anthropic())` no longer sends ragas' default `temperature` / `top_p`. Every call with anthropic SDK 2.6 used to fail with `TypeError: ... unexpected keyword argument 'temperature'`.
- A `temperature` or `top_p` **you** pass (`llm_factory(..., temperature=0.2)`) is still sent. On current Claude models it fails, and the provider's error says why: only `temperature=1` is accepted on older API versions, and none on API `2026-10-21`.
- A per-call temperature ragas chooses for sampling (`strictness`, legacy prompts with `n > 1`) is not sent to Claude. Through LiteLLM it is sent with `drop_params=True`, so LiteLLM drops it where a model rejects it.
- The planned move of every call onto LiteLLM's parameter handling is #182.

## Claude: JSON mode by default ([#184](https://github.com/arpieb/ragas-ng/issues/184))

Without a `mode`, `llm_factory()` now asks Claude for structured output as JSON:
- `Mode.ANTHROPIC_JSON` with an Anthropic client (`provider="anthropic", client=Anthropic()`);
- `Mode.JSON` for Claude through LiteLLM (`"anthropic/claude-..."`, including Bedrock's `anthropic.claude-...` IDs).

Instructor's defaults (`ANTHROPIC_TOOLS` / `TOOLS`) force a tool call. `claude-sonnet-5-5` rejects that with `400 tool_choice: type "tool" and "any" are not supported for this model`, so every structured call failed on both paths.

- Verified live with `claude-sonnet-5-5` and `claude-haiku-5-5` on both paths: generation, `AspectCritic` (with and without `strictness`), and `AnswerAccuracy`.
- A `mode` you pass is kept, so `mode=instructor.Mode.ANTHROPIC_TOOLS` restores the old behavior for models that accept it.
- A LiteLLM `Router` or module passed as `client` keeps instructor's default, because the model isn't known when it's wrapped. Pass `mode=` there.

## NVIDIA collections metrics raise on LLM errors ([#185](https://github.com/arpieb/ragas-ng/issues/185))

- `AnswerAccuracy`, `ResponseGroundedness` and `ContextRelevance` used to catch every exception from the judge call and turn it into `nan`. A misconfigured judge (wrong model, bad key, a provider rejecting a parameter) scored `nan` and the error was lost.
- Provider and transport errors now propagate, as in other collections metrics. With `evaluate(raise_exceptions=False)` the row is still missing, and the error is logged.
- Only an **invalid rating** (outside the metric scale after `max_retries` attempts) still gives `nan`.
- `DataCompyScore` is unchanged: its `except` covers CSV parsing, not an LLM call.

## Model names in examples and hints ([#181](https://github.com/arpieb/ragas-ng/issues/181))

- **Example templates** name each model once, near the top of the module that uses it:
  - `RAGAS_EXAMPLE_APP_MODEL` sets the model the app or agent under test calls;
  - `RAGAS_EXAMPLE_JUDGE_MODEL` sets the model that judges.

  Unset, each template uses the same default as before.
- **Error messages and deprecation hints** that suggest a model use `ragas.llms.base.DEFAULT_LLM_MODEL`.
- **Metric docstrings** show `llm_factory("<model>", client=client)` instead of a specific model.
- **Model retirements** are tracked in `scripts/model_registry.toml`, and a unit test fails when a retired model is named in code, docs or examples. See `scripts/README.md`.
- `text2sql/analyze_errors.py` used `gpt-5` (shutting down 2026-12-11) and now uses `gpt-5.6-sol`. A source comment and two doc lines still named `gpt-5` or the shut-down `o1`; they were updated.

## One path for every LLM: LiteLLM ([#182](https://github.com/arpieb/ragas-ng/issues/182))

`llm_factory()` returns an `InstructorLLM`, and every call goes through LiteLLM. ragas no longer wraps each provider's own SDK.

**Breaking:** native SDK clients of providers other than OpenAI are rejected with `ValueError`. That covers `Anthropic()`, `genai.Client()` and `GenerativeModel`, `Mistral()`, `Groq()`, `Cohere()`, boto3 Bedrock and xAI, and instructor-patched clients. The error names the model string to use:

| Before | Now |
|---|---|
| `llm_factory("claude-sonnet-5-5", provider="anthropic", client=Anthropic())` | `llm_factory("anthropic/claude-sonnet-5-5")` |
| `llm_factory("gemini-3.6-flash", provider="google", client=genai.Client(...))` | `llm_factory("gemini/gemini-3.6-flash")` |
| a google-genai Vertex client | `llm_factory("vertex_ai/gemini-3.6-flash", vertex_project=..., vertex_location=...)` |
| Mistral / Groq / Cohere clients | `llm_factory("mistral/<m>")` / `"groq/<m>"` / `"cohere_chat/<m>"` |
| a boto3 Bedrock client | `llm_factory("bedrock/<model id>", aws_region_name=...)` |
| `mode=instructor.Mode.ANTHROPIC_JSON` / `ANTHROPIC_TOOLS` (and other native modes) | `Mode.JSON` / `Mode.TOOLS`; native modes raise |

The provider's key comes from its environment variable (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY`, ...), or pass `api_key=`. To keep your own instructor client, construct `InstructorLLM(client=..., model=..., provider=...)` directly. Its per-provider parameter rules are unchanged.

**Unchanged, but routed differently:**
- `OpenAI` / `AsyncOpenAI` and `AzureOpenAI` / `AsyncAzureOpenAI` clients still work, through LiteLLM, as `openai/<model>` or `azure/<deployment>`.
  - LiteLLM gets a **copy** of the client, because it writes `max_retries` and `organization` onto the object it receives. Your client is left untouched.
  - `evaluate()` still builds its default embeddings from your client.
- A `litellm.Router` as `client` works as before.

**Deprecated:**
- `llm_factory(adapter=...)` is ignored, with a `DeprecationWarning`.
- `ragas.llms.adapters` (`get_adapter`, `auto_detect_adapter`, `InstructorAdapter`, `LiteLLMAdapter`) still imports, and both adapters call `llm_factory()`.
- Constructing `LiteLLMStructuredLLM` warns. `llm_factory()` no longer returns it.

All three are removed in the next release.

**Parameters** (one set of rules for every `llm_factory()` LLM):
- ragas' defaults `temperature=0.01` and `top_p=0.1` are sent with LiteLLM's `drop_params`, so LiteLLM drops them for models that reject them.
- Values you set are sent as given.
- `max_tokens` is sent only if you set it. The default of 1024 used to truncate long and reasoning outputs on the client path.
- The OpenAI reasoning-model and Claude rules are unchanged (see [LLMs (llm_factory)](../llm-adapters.md#model-arguments)).

**Token usage:** responses are OpenAI-shaped for every provider, so use `get_token_usage_for_openai`.

**Embeddings:** the default embeddings for a Gemini LLM are now ragas' default (OpenAI's `text-embedding-ada-002` through LiteLLM). They used to be Google's, when you passed a google-genai client. Pass `GoogleEmbeddings(...)` explicitly for a Gemini-only setup. `GoogleEmbeddings` still takes a google-genai client.

## Tracing: metrics, LLM calls and experiments ([#196](https://github.com/arpieb/ragas-ng/issues/196))

Collections metrics, `llm_factory()` LLMs and `@experiment` take no `callbacks`, so they used to emit no OpenTelemetry spans and add nothing to the run tree. Now:

- **Metrics:** every metric's `score()` / `ascore()` is an `EVALUATOR` span. That covers `ragas.metrics.collections` metrics and `DiscreteMetric`, `NumericMetric`, `RankingMetric` and decorator metrics. A metric called on its own is a root span, or a child of your application's active span. Under `evaluate()` it is the metric span `evaluate()` opens, not a second one.
- **LLM calls:** every call to an `llm_factory()` LLM is an `LLM` span named after its response model. It records the prompt, the validated response, `llm.model_name`, the messages, the invocation parameters and token counts. It nests under the metric, legacy prompt or experiment row that made it.
- **Prompts reach your tracing service, truncated by default.** LLM spans carry the judge's prompt, which holds your evaluation data. Their messages, like every span's `input.value` / `output.value`, are cut to 4,096 characters. Set `RAGAS_TRACE_FULL_CONTENT=true` to send them in full. See [Observability](../observability.md#what-leaves-your-process).
- **Experiments:** `arun()` is an `experiment <name>` span, with one `row N` span per row. A failed row is an error span. Metrics scored in the experiment function nest under their row.
- **`result.traces` has each LLM call's input and output again.** Since metrics imported from `ragas.metrics` started handing off to collections, `result.traces[i][metric]` had been empty for `llm_factory()` LLMs. It now maps each call to `{"input": {"prompt": ...}, "output": <response model>}`, keyed by the response model's name, e.g. `AspectCriticOutput`.
- A legacy prompt run without `callbacks` nests under the run in progress instead of starting a new trace.
- Span payloads render pydantic models as JSON fields instead of `repr()`.

Without an OpenTelemetry SDK, spans are still no-ops. The LLM call is unchanged unless a usage collector or a recording span needs the provider's raw response for token counts.

## `FaithfulnesswithHHEM` loads a pinned model commit ([#195](https://github.com/arpieb/ragas-ng/issues/195))

The HHEM model is loaded with `trust_remote_code=True`, which runs the model repo's `modeling_hhem_v2.py`. Both `FaithfulnesswithHHEM` metrics (collections and legacy) used to load it from the repo's `main`, so a new push there changed the code ragas ran with no ragas change. They now load commit `de9a5e6` ("Support transformers 5", 2026-10-06), which was reviewed. The collections metric takes `revision=` to load another. Scores are unchanged: on the model card's examples, this commit gives the card's published scores under both transformers 4.57 and 5.17.

The reported 0.0 for "The first superbowl was held on Jan 15, 1967" is HHEM working as designed. The example's context names only the "First AFL–NFL World Championship Game", and HHEM, a strict entailment model, does not credit what the context leaves to outside knowledge. The docs example now uses a context that states it.

## Metric reasons: the verb agrees with the count ([#200](https://github.com/arpieb/ragas-ng/issues/200))

The first line of `MetricResult.reason` for `Faithfulness`, `FaithfulnesswithHHEM`, `FactualCorrectness`, `NoiseSensitivity` and `ContextRecall` always said "are" after the count, as in "1 of 1 statement in the response are entailed". The verb is now "is" when one item is counted or there is only one ("1 of 2 statements … is supported", "0 of 1 statement … is"), and "are" otherwise. Scores are unchanged. Code that matches on the exact reason text may need updating.

## `evaluate()`: default embeddings follow a LiteLLM LLM's provider ([#190](https://github.com/arpieb/ragas-ng/issues/190))

When a metric needs embeddings and none are passed, `evaluate()` used to give every client-less `llm_factory()` LLM OpenAI's `text-embedding-ada-002`, so `llm_factory("gemini/...")` with `AnswerRelevancy` or `AnswerCorrectness` needed `OPENAI_API_KEY`. The embedding model now comes from the LLM's prefix:

- `gemini/...` gets `gemini/gemini-embedding-2`, using the LLM's `api_key` if it was given one.
- `vertex_ai/...` gets `vertex_ai/gemini-embedding-2`, with the LLM's `vertex_project`, `vertex_location` and `vertex_credentials`.
- Bare and `openai/` names keep `text-embedding-ada-002`.
- Other providers (`ollama/`, `anthropic/`, `bedrock/`, ...) also keep it, but now with a `UserWarning` that says to pass `embeddings=`.

**⚠ Scores change** for Gemini and Vertex users of embedding metrics who relied on the default, since a different embedding model gives different similarities. Pass `embeddings=` to keep the old model. LLMs built with an OpenAI or Azure client are unchanged: they still take embeddings from that client.

## Experiments: `raise_exceptions` stops on the first failed row ([#170](https://github.com/arpieb/ragas-ng/issues/170))

`@experiment(raise_exceptions=True)` or `arun(..., raise_exceptions=True)` makes `arun` stop at the first row whose function raises, like `evaluate(raise_exceptions=True)`. The rows still running are cancelled, rows not yet started never run, **nothing is saved**, and that row's exception is raised from `arun`, with a note naming the row index. The default, `False`, is unchanged: the row is logged and left out, and the run goes on.

Add `save_partial=True` (decorator or `arun`) to save the rows that finished before the stop instead; the error is still raised, and its note says how many rows were saved. Without `raise_exceptions` it has no effect.

`raise_exceptions` and `save_partial` are now `arun`'s own keyword arguments, like `name`, `backend`, `backend_kwargs` and `max_workers`. An experiment function that took a keyword argument of either name through `arun` no longer receives it.

## Experiments: a sync `run()` ([#172](https://github.com/arpieb/ragas-ng/issues/172))

`@experiment` functions now have `run(dataset, ...)`, the synchronous form of `await arun(dataset, ...)`. It takes the same arguments and returns the same `Experiment`, so a script no longer needs `asyncio.run(fn.arun(...))`. Like `evaluate()` and `metric.score()`, it runs on ragas' shared background event loop (#74, #80), so it also works inside a running loop such as Jupyter. Ctrl-C cancels the rows still running.

## `evaluate()` / `Executor`: stopping stops the jobs ([#207](https://github.com/arpieb/ragas-ng/issues/207))

- **`raise_exceptions=True` no longer leaves jobs running.** The first error used to be raised while the other jobs ran on in the background: for `evaluate()`, LLM calls still being made, and paid for, after the error. Now the jobs still running are cancelled and awaited before the error propagates, and no job starts after the failing one (with `max_workers=1`, the rest never run).
- **`Executor.cancel()` takes effect within about 50 ms.** It used to wait for the next job to finish, then cancel only what hadn't finished yet. Jobs still running are cancelled; `results()` returns what had finished.
- The cancel path no longer leaves a coroutine un-awaited (`RuntimeWarning: coroutine ... was never awaited`).

`ragas.async_utils.as_completed` is unchanged apart from that warning; ragas itself now uses `concurrent_results`.

## Requests and tokens per minute: `RateLimit` ([#171](https://github.com/arpieb/ragas-ng/issues/171))

New: `llm_factory(..., rate_limit=RateLimit(rpm=..., tpm=...))` (from `ragas.rate_limit`) paces every request the LLM sends, waiting until the limit has room instead of hitting a 429. `max_workers` only limited jobs at once. Pass the same `RateLimit` to every LLM that shares an API key; `LiteLLMEmbeddings(rate_limit=...)` takes one too. Without `rate_limit` nothing changes. It adds one dependency, [`pyrate-limiter`](https://pypi.org/project/pyrate-limiter/) (pure Python, no dependencies of its own). See [Requests and tokens per minute](../customizations/run_config.md#requests-and-tokens-per-minute).

## `EvaluationResult.check()`: thresholds for CI ([#169](https://github.com/arpieb/ragas-ng/issues/169))

New: `result.check(min_mean=..., max_mean=..., allow_missing=0)` raises `ThresholdError` (an `AssertionError`, from `ragas.exceptions`) unless each metric's mean is within its bounds and no row is unscored. The message is a per-metric table of mean, bounds, count and missing rows. A bound for a metric the result doesn't have raises `ValueError`. See [Adding to your CI pipeline](../applications/add_to_ci.md).

## Copying LLMs, embeddings and metrics ([#211](https://github.com/arpieb/ragas-ng/issues/211))

`copy.deepcopy()` of an `llm_factory()` LLM raised `RecursionError`, and of `LiteLLMEmbeddings`, of a metric holding either, or of a string-distance metric (`NonLLMStringSimilarity`, `NonLLMContextRecall`, ...) raised `TypeError: cannot pickle 'module' object`. They now copy. A copy has its own settings (`model_args`, `system_prompt`, run config, ...) and **shares** the provider clients, the cache and the `RateLimit`: a copied LLM goes to the same provider, under the same rate limit.

A copy (deep or shallow) of an LLM or embeddings with a `cache` used to keep the original's cached `generate` / `embed_text`, so its calls ran on the original object, with the original's settings. They now run on the copy.

`distance_measure_map` on the string-distance metrics and `OverlapScoreBuilder` is now a read-only property rather than an instance attribute.

## Prompt optimizers rank losses by their direction ([#216](https://github.com/arpieb/ragas-ng/issues/216))

`Loss` now declares its direction with a `higher_is_better` class attribute: `False` for `MSELoss` (an error), `True` for `BinaryMetricLoss` (accuracy or F1). `Loss.fitness(predicted, actual)` returns a value where higher is always better, and both optimizers rank by it.

- **⚠ silent:** `GeneticOptimizer` kept the prompt with the *highest* MSE, and passed `(actual, predicted)` to the loss. It now keeps the best prompt and passes `(predicted, actual)`. Cross-over paired every candidate with the first one, including the first with itself; each candidate now crosses with its nearest other candidate. Optimized prompts will differ.
- **⚠ silent:** the DSPy metric negated every loss, so under `BinaryMetricLoss` `DSPyOptimizer` minimised accuracy. It now maximises it. Under `MSELoss` it still returns the negated error.
- `GeneticOptimizer.evaluate_fitness()` now returns fitness values (higher is better), not raw losses.
- `MSELoss(reduction="sum")`, `BinaryMetricLoss(metric="f1_score")` and `InstructionConfig(loss=MSELoss())` raised `TypeError` and now work. An unknown option raises `ValueError`.
- A custom `Loss` that is higher-is-better must set `higher_is_better = True`; the default is lower-is-better.
- Each `InstructionConfig` gets its own copy of `optimizer_config`, so editing one no longer changes the others.

## Collections metric constructors: unknown arguments raise ([#216](https://github.com/arpieb/ragas-ng/issues/216))

- Any `ragas.metrics.collections` metric now raises `TypeError` for a keyword argument it doesn't take. They used to be dropped, so `AspectCritic(..., strictnes=3)` silently built a metric with `strictness=1`. Fix the spelling or remove the argument.
- `ContextPrecision(llm, name="x")` and `ContextUtilization(llm, name="x")` raised `TypeError: multiple values for keyword argument 'name'`. They now accept `name`; the defaults are unchanged.
- A custom `BaseMetric` subclass that doesn't define `ascore` can no longer be instantiated (`TypeError`). It used to score `0.0` from a placeholder.
- `AnswerCorrectness(embeddings=...)` now checks that the embeddings are a modern `BaseRagasEmbedding`, as every other collections metric does; anything else raises `ValueError`. `embeddings=None` with `weights=[1.0, 0.0]` still works.

## Testset generation: scenarios, filters and personas ([#216](https://github.com/arpieb/ragas-ng/issues/216))

- `SingleHopScenario` is now defined once. The class exported from `ragas.testset.synthesizers.single_hop` was a different class from the one the synthesizer checks for, so passing a scenario you built yourself raised `TypeError: scenario type should be SingleHopScenario`.
- **⚠ silent:** `JaccardSimilarityBuilder` now honours `filter_nodes` in a transform pipeline. It compared every node in the graph, so with `filter_nodes` set you now get fewer relationships, limited to the nodes you selected. `SummaryCosineSimilarityBuilder` now links only document nodes in a pipeline, as documented; it linked every node with a summary embedding, chunks included. Neither is in the default transforms.
- `OverlapScoreBuilder().name` is now `"OverlapScoreBuilder"`. It was an empty string.
- **⚠ silent:** `NodeFilter.transform()` now removes every node its filter flags, as the pipeline does. It removed only the first.
- **⚠ silent:** `generate_personas_from_kg()` now groups similar summaries by true cosine similarity, comparing each summary with its own node's embedding. It used an unnormalised dot product, and when some nodes had no summary it could compare a summary with another node's embedding. You may get different personas from the same graph. A node with a summary but no `summary_embedding` now raises a clear `ValueError`.

## One deprecation warning, Vertex clients used, LiteLLM batch sizes per provider ([#216](https://github.com/arpieb/ragas-ng/issues/216))

- `evaluate()` now emits one `DeprecationWarning`, pointing at your call. It added a second one from `aevaluate()` that pointed inside ragas. `aevaluate()` called directly still warns once.
- `GoogleEmbeddings(use_vertex=True, client=...)` now uses the client you pass. A `TextEmbeddingModel` is called directly, and a google-genai `genai.Client(vertexai=True, ...)` goes through `models.embed_content`. Any other client loads `model` once and reuses it; `TextEmbeddingModel.from_pretrained` used to run on every call, ignoring the client.
- **⚠ silent:** `LiteLLMEmbeddings` without `batch_size=` now picks its batch size from the model's provider: 100 for OpenAI, 96 for Cohere, 5 for Vertex AI, 32 for Hugging Face, and 10 otherwise. Every model got 10. OpenAI and Cohere users see fewer, larger embedding requests. Pass `batch_size=` to keep the old behaviour.

## The LLM takes chat messages, and the multimodal metrics use them ([#214](https://github.com/arpieb/ragas-ng/issues/214))

`generate()` and `agenerate()` on `llm_factory()` LLMs now accept a list of OpenAI-format chat messages as well as a prompt string; message content can be a list of text and `image_url` parts. A messages list takes the same path as a string: retries, cache, rate limit, tracing, token usage, provider parameter mapping, and sync or async clients. The LLM's `system_prompt` is prepended unless the list already has a system message. String prompts are sent and cached exactly as before, so existing caches still hit.

`MultiModalFaithfulness` and `MultiModalRelevance` (collections) called the provider client directly, so they skipped retries, the cache, tracing, usage recording and `system_prompt`, and failed with a sync client such as `OpenAI()`. They now go through `llm.agenerate()`, so all of these apply and a sync client works. Scores are unchanged, except that a configured `system_prompt` is now sent and can change the verdict.

## One prompt class ([#214](https://github.com/arpieb/ragas-ng/issues/214))

`ragas.prompt.metrics.BasePrompt`, the prompt class collections metrics use, is now ragas' one structured-prompt implementation, and `ragas.prompt.PydanticPrompt` is a subclass that only adds support for legacy `BaseRagasLLM` wrappers. Rendered prompts are byte-for-byte unchanged for both, so cache keys and scores don't move.

- `BasePrompt` gains what `PydanticPrompt` had: `generate()` / `generate_multiple()` through an `llm_factory()` LLM (with `n` concurrent samples, `process_input` / `process_output` hooks and a traced prompt run), `name`, content-based `==` and `hash()`, and `save()` / `load()`.
- **⚠ silent:** `PydanticPrompt.adapt()` with an `llm_factory()` LLM now translates with the same prompt collections metrics use, so translations of adapted testset prompts can differ. With a legacy LLM it still uses `TranslateStatements`.
- `BasePrompt.adapt(..., adapt_instruction=True)` now translates the instruction even when the prompt has no example strings. It returned without translating it.
- `save()` logs the saved path at INFO instead of printing it.

## evaluate() runs modern metrics natively ([#214](https://github.com/arpieb/ragas-ng/issues/214))

`evaluate()` and `aevaluate()` no longer wrap `ragas.metrics.collections` metrics as legacy metrics. They call the metric's `ascore()` with the sample's fields by name. Scores, reasons, traces and token usage are the same as before.

Metrics that take the LLM per call now work in `evaluate()`: `NumericMetric` (and other prompt-template metrics) and `@numeric_metric`-decorated functions. They get the `llm` passed to `evaluate()`, or the default LLM if none was passed; every row used to score NaN. Passing a legacy `BaseRagasLLM` to such a metric now raises `TypeError` before the run starts, and a decorated function whose required parameters are not sample fields raises `ValueError` up front instead of scoring NaN on every row.

**⚠ silent:** `evaluate(metrics=None)` now runs the collections `AnswerRelevancy`, `ContextPrecision`, `Faithfulness` and `ContextRecall`, built with the run's LLM and embeddings, instead of the `ragas.metrics` hand-off classes. Scores should not change, since the hand-off already scored through collections. With a legacy LLM or legacy embeddings you still get the old defaults.

## Prompt optimization moves to collections metrics ([#214](https://github.com/arpieb/ragas-ng/issues/214))

`metric.train()`, `GeneticOptimizer` and `DSPyOptimizer` now tune the metrics in `ragas.metrics.collections`. The legacy `ragas.metrics` classes are being removed (#215), and their `train()` is removed with this change. `InstructionConfig` and `DemonstrationConfig` now take only `llm_factory()` LLMs and `embedding_factory()` embeddings, and raise a `ValidationError` for legacy wrappers.

- Collections metrics gain `get_prompts()` and `set_prompts(**prompts)`. Prompts are keyed by attribute name (`prompt` for `AspectCritic`; `statement_generator_prompt` and `nli_statement_prompt` for `Faithfulness`). `get_prompts()` returns copies, so to change a prompt you edit the copy and pass it to `set_prompts()`. `set_prompts()` copies what it is given, rejects unknown names with `ValueError`, and rejects a non-`BasePrompt`, or a prompt with different input or output models, with `TypeError`.
- Annotation files key each sample's `prompts` by those names. Legacy names such as `single_turn_aspect_critic_prompt` raise a `ValueError` that lists the metric's names. The sample file for the Vertex AI alignment guide uses the new names.
- `metric.train(path, demonstration_config, instruction_config, *, dataset=None, run_config=None, batch_size=None, raise_exceptions=True)` also accepts the annotations as an object through `dataset=`. It no longer takes `callbacks` or `with_debugging_logs`. A config with `enabled=False` is now skipped; it used to run anyway.
- **⚠ silent:** `train()` no longer modifies the metric's prompt objects in place. Optimized prompts replace them through `set_prompts()`, and candidates are scored on a copy of the metric. In-place edits used to reach every metric sharing a prompt object.
- `Optimizer.optimize()` now returns optimized copies of the prompts (`{name: BasePrompt}`) for `set_prompts()`, rather than `{name: instruction}`. `DSPyOptimizer`'s copies also carry the demonstrations MIPROv2 chose, which it used to drop.
- Each collections metric has an `output_type`, which picks the default loss: binary for `AspectCritic`, `AgentGoalAccuracyWithReference`/`WithoutReference` and `SQLSemanticEquivalence`; discrete for the rubric metrics and `SimpleCriteriaScore`; continuous otherwise.
- Few-shot demonstrations (`DemonstrationConfig`) make each annotated prompt a copy of itself that, per input, renders the `top_k` annotated examples at least `threshold` similar to it. `technique="random"` now works: a fixed random sample of `top_k` examples, with no embedding calls. It was ignored.
- `DSPyOptimizer` now drives DSPy through a `dspy.LM` built from the LLM's LiteLLM model name and credentials, rather than wrapping a legacy LLM. Pass `DSPyOptimizer(lm=dspy.LM(...))` for an LLM over a `litellm.Router` or an Azure client, which raise `ValueError`. DSPy's own cache and retries apply to its calls, and `llm_factory(rate_limit=...)` doesn't.
- `DSPyOptimizer` failed with every current DSPy: MIPROv2 takes its metric in the constructor and rejects `num_candidates` with an `auto` level. It now passes them that way, runs `num_trials` (from `optimizer_config`, default `num_candidates`) when `auto=None`, and doesn't stop to ask for confirmation. DSPy signatures now carry the prompt fields' types, and only prompts with annotations are optimized.
- **⚠ silent:** DSPy's metric compared an output field named after the metric, which no prompt has, so every prediction scored 0. It now compares that field if present, else the prompt's first numeric output field, such as `verdict`.
- `GeneticOptimizer`'s feedback mutation reads each prompt's input and output from the traced LLM calls of the metric. A custom LLM that doesn't trace its calls gets no feedback, with a warning, and keeps the candidate's instruction.

## Testset generation takes modern LLMs, embeddings and prompts only ([#214](https://github.com/arpieb/ragas-ng/issues/214))

Testset generation (`TestsetGenerator`, the synthesizers, the transforms, `default_transforms()` and `generate_personas_from_kg()`) now takes only an `llm_factory()` LLM and a modern embedding model (`BaseRagasEmbedding`, such as `embedding_factory()` output). A legacy `BaseRagasLLM` or `BaseRagasEmbeddings` wrapper raises `TypeError` when you build the object or call the function, with a message that names what to use instead.

- `TestsetGenerator.from_llama_index()` and `generate_with_llamaindex_docs()` are removed. Build the generator with `llm_factory()` and `embedding_factory()`, and pass LlamaIndex documents to `generate_with_docs()` with `document_adapter=lambda d: Document(page_content=d.text, metadata=d.metadata)`. See [LlamaIndex](../integrations/llamaindex.md).
- New: `generate_with_docs()` and `generate_with_chunks()` take `document_adapter=`, a function that turns each of your documents into one with `page_content` and `metadata`, so documents from any framework work without ragas depending on it. `generate_with_docs()` also accepts plain strings now, as `generate_with_chunks()` did. `generate_with_langchain_docs()` stays as a deprecated alias of `generate_with_docs()`.
- **⚠ silent:** `generate_with_docs()` now skips documents with no text, as `generate_with_chunks()` and the removed LlamaIndex method did. Blank documents used to become empty graph nodes and counted toward the document-length shares that pick the default transforms, so a corpus with blank documents can now get a different pipeline.
- The testset prompts (extractors, filters, synthesizers, personas) are now `ragas.prompt.metrics.BasePrompt` subclasses, not `PydanticPrompt`. Their rendered text is byte-for-byte unchanged, so cache keys don't move. Their `generate()` no longer takes `stop=` or `retries_left=`. `get_prompts()` and `set_prompts()` on testset classes find and take any `BasePrompt`, including a `PydanticPrompt`.
- `EmbeddingExtractor` always calls `aembed_text()`. With an embedding model that had no `is_async` attribute, such as `LiteLLMEmbeddings`, it ran the sync `embed_text()` on a worker thread and warned (`UserWarning`); it now makes the async call and doesn't warn. The embeddings are the same.

## The `dspy` extra installs DSPy 3 with optuna

`pip install "ragas[dspy]"` now installs `dspy[optuna]>=3` instead of `dspy-ai>=2.4.0`. `DSPyOptimizer` runs MIPROv2, which on DSPy 3 needs `optuna`; the old extra left it out, so the optimizer failed on import of a fresh install. DSPy 2.x is no longer supported: `DSPyOptimizer` already needed DSPy 3's MIPROv2 constructor ([#214](https://github.com/arpieb/ragas-ng/issues/214)).

## Dead code removed: `ragas evals`, `as_completed`, `find_indirect_clusters` ([#215](https://github.com/arpieb/ragas-ng/issues/215))

These had no callers in ragas, and the CLI command could not run.

- The `ragas evals` CLI command is removed. It called a `run_async` method that experiments don't have and a `Project` class that no longer exists, so every run failed. Run an experiment from Python instead: `Dataset.load(...)`, then `my_experiment.run(dataset)` or `await my_experiment.arun(dataset)`. `ragas quickstart` and `ragas hello-world` are unchanged, except that the `hello_world/evals.py` script that `ragas hello-world` writes is now a working `@experiment` you run with `python hello_world/evals.py`. It used to be meant for `ragas evals` and could not run either.
- `ragas.async_utils.as_completed()` and `ragas.async_utils.process_futures()` are removed. Use `concurrent_results()`, which ragas has used since #207 and which cancels what is still running when you stop iterating, or `asyncio.as_completed()`.
- `KnowledgeGraph.find_indirect_clusters()` is removed. Testset generation uses `find_n_indirect_clusters()`, which stays. With it goes ragas' last use of `scikit-network` (its Leiden clustering) and of `networkx`.
- The module `ragas.backends.utils` is removed. Its `MemorableNames` was a copy of `ragas.utils.MemorableNames`; import it from there.

## The legacy layer is removed ([#215](https://github.com/arpieb/ragas-ng/issues/215))

The legacy metrics, LLM and embedding wrappers and prompt classes are deleted, not deprecated. Everything they did has a modern replacement, which #214 moved testset generation, the optimizers and `evaluate()` onto.

| Removed | Use instead |
|---|---|
| The legacy metrics: `from ragas.metrics import Faithfulness`, `faithfulness`, `RubricsScore`, …, and the private modules `ragas.metrics._*` | `from ragas.metrics.collections import Faithfulness`; `Faithfulness(llm=llm)`. Collections metrics take their models at construction and score with `await metric.ascore(user_input=..., response=...)`. |
| Renamed metrics | `LLMContextRecall` → `ContextRecall`, `ResponseRelevancy` → `AnswerRelevancy`, `RubricsScore` → `DomainSpecificRubrics`, `AnswerSimilarity` → `SemanticSimilarity`, `ChrfScore` → `CHRFScore`, `InstanceRubrics` → `InstanceSpecificRubrics`, `LLMContextPrecisionWith[out]Reference` → `ContextPrecisionWith[out]Reference`, `LLMSQLEquivalence` → `SQLSemanticEquivalence`, `SummarizationScore` → `SummaryScore`, `TopicAdherenceScore` → `TopicAdherence` |
| `ragas.metrics.quoted_spans` | `ragas.metrics.collections.QuotedSpansAlignment` |
| `Metric`, `MetricWithLLM`, `MetricWithEmbeddings`, `SingleTurnMetric`, `MultiTurnMetric`, `ModeMetric`, `ensembler` (`ragas.metrics.base`) | Subclass `ragas.metrics.collections.BaseMetric`, or use `DiscreteMetric` / `NumericMetric` / `RankingMetric` |
| `BaseRagasLLM`, `LlamaIndexLLMWrapper`, `HaystackLLMWrapper`, `OCIGenAIWrapper` / `oci_genai_factory`, `LiteLLMStructuredLLM`, the `ragas.llms.adapters` package, `ragas.llms.output` (`LLMResult`, `Generation`) | `llm_factory()`, which reaches providers through LiteLLM model prefixes (e.g. `llm_factory("anthropic/...")`) |
| `BaseRagasEmbeddings`, `HuggingfaceEmbeddings`, `LlamaIndexEmbeddingsWrapper`, `HaystackEmbeddingsWrapper` | `BaseRagasEmbedding` and its providers: `embedding_factory()`, `HuggingFaceEmbeddings`, `LiteLLMEmbeddings`, `OpenAIEmbeddings`, `GoogleEmbeddings` |
| `ragas.run_config.add_retry` / `add_async_retry`, which retried every exception, a 400 included | `llm_factory()` LLMs retry transient errors per `RunConfig` themselves |
| `PydanticPrompt`, `StringPrompt`, the legacy `ragas.prompt.base.BasePrompt`, `FewShotPydanticPrompt`, `ExampleStore` / `InMemoryExampleStore`, `ImageTextPrompt` / `ImageTextPromptValue`, `RagasOutputParser`, `FixOutputFormat`, `TranslateStatements`, `ragas.prompt.value`, `ragas.prompt.utils.extract_json` | `ragas.prompt.BasePrompt`, which is now `ragas.prompt.metrics.BasePrompt` (subclass it the way you subclassed `PydanticPrompt`); `DynamicFewShotPrompt` for few-shot examples; messages-based `llm.agenerate` for images |
| `ragas.validation.validate_required_columns` / `validate_supported_metrics`, `ragas.utils.get_required_columns_v1`, `get_metric_language`, `DeprecationHelper` | Nothing: they only served the legacy metrics |
| The `ai-frameworks` (Haystack) and `oci` extras | `llm_factory()` over LiteLLM |

- `from ragas.metrics import <old name>` raises `ImportError` with the replacement import, e.g. `'RubricsScore' was removed from 'ragas.metrics' along with the legacy metrics. Use: from ragas.metrics.collections import DomainSpecificRubrics`. It is a hint, not a shim: nothing is re-exported. `DiscreteMetric`, `NumericMetric`, `RankingMetric`, the `*_metric` decorators, `MetricResult`, `BaseMetric`, `LLMMetric`, `MetricType` and `MetricOutputType` stay in `ragas.metrics`.
- `evaluate()` and `aevaluate()` take only `SimpleBaseMetric` metrics (collections metrics and the `DiscreteMetric` family), an `llm_factory()` LLM and a `BaseRagasEmbedding`; anything else raises `TypeError` before the run starts. `evaluate(metrics=None)` always runs the collections defaults.
- `ragas.integrations.llama_index.evaluate()` no longer wraps LlamaIndex LLMs and embeddings. Its `llm` and `embeddings` are now ragas models (`llm_factory()`, `embedding_factory()`), and its `metrics` are collections metrics.
- `PromptMixin.get_prompts()` / `set_prompts()` (testset synthesizers and transforms) find and take any `ragas.prompt.BasePrompt`; the testset-only override is folded into it.
- `ragas.metrics._collections_bridge` is gone; the sample-to-`ascore()` mapping `evaluate()` uses is `ragas.metrics.sample_inputs`.
- `DistanceMeasure` for `OverlapScoreBuilder` comes from `ragas.metrics.collections`; the values are unchanged.

## Multimodal metrics restrict where context images are loaded from ([#229](https://github.com/arpieb/ragas-ng/issues/229))

`MultiModalFaithfulness` and `MultiModalRelevance` load the images named in `retrieved_contexts`. They used to open any local image path and fetch any http(s) URL, following redirects to any host, internal ones included. Upstream's checks for this (vibrantlabsai/ragas#1991) were only in the legacy `ImageTextPrompt`. They now apply to both metrics, configured through the new `ragas.metrics.collections.ImageSourcePolicy`, passed as `image_policy=`:

- Local image paths are no longer read by default; such a context raises `ImageLoadError` ([#231](https://github.com/arpieb/ragas-ng/issues/231)). To read local images, pass `image_policy=ImageSourcePolicy(local_dir="images/")`. Paths are resolved against that directory, and a path that resolves outside it is refused, symlinks included.
- A URL whose host resolves to a non-public address (private, loopback, link-local, shared, reserved or multicast, IPv4 or IPv6) is not fetched, and neither is a redirect to one; it raises `ImageLoadError` (#231). For an internal image server, pass `ImageSourcePolicy(allow_private_urls=True)`.
- URLs with credentials, a backslash in the host part, or a host that `urllib` and `requests` parse differently are refused. At most 5 redirects are followed.
- The size limit (10 MB) is now `ImageSourcePolicy(max_bytes=...)`.
- Data URIs are unchanged.
- The image helpers `process_image_to_base64()` and `is_image_path_or_url()` moved from the `multi_modal_faithfulness.util` and `multi_modal_relevance.util` modules to `ragas.metrics.collections.image_sources`. `process_image_to_base64()` takes the policy as its second argument.

## Multimodal metrics raise when an image context can't be loaded ([#231](https://github.com/arpieb/ragas-ng/issues/231))

`MultiModalFaithfulness` and `MultiModalRelevance` used to send an image reference they could not load to the judge as text, such as `Context 1: path/to/tesla.jpg`, and still return a score. That covered missing files, failed downloads, non-image content, and everything #229 refuses. The judge never saw the image, so the score looked valid but was not.

- An image reference that is refused or fails to load now raises `ragas.exceptions.ImageLoadError`, a `ValueError`, before the LLM is called. The message gives the context's position, the reason, and the setting that would allow it. Under `evaluate()` this follows `raise_exceptions`: it is raised, or the score is NaN and the error is logged.
- `ImageSourcePolicy(on_blocked="warn")` restores the old fallback: a warning is logged and the reference is sent as text.
- An image reference is an image data URI, an http(s) URL, or a path ending in an image extension (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`). Any other context is text, as before.
- **⚠ silent:** under `local_dir`, a file without an image extension is no longer opened. It is sent as text, even if it holds an image.
- Other URL schemes, such as `ftp://`, raise instead of being treated as file paths.

## Testset generation and optimizers: one code path per job ([#217](https://github.com/arpieb/ragas-ng/issues/217))

`TestsetGenerator`'s three entry points now share one graph builder and one check of their inputs.

- `generate_with_docs()` and `generate_with_chunks()` take `num_personas` and `batch_size`, as `generate()` does. They used to always use 3 personas and no batching.
- An LLM and an embedding model are needed only to build the default transforms. With your own `transforms`, `generate_with_docs()` and `generate_with_chunks()` no longer raise `ValueError` when the generator has no embedding model.
- `generate_with_docs(transforms=[])` applies no transforms, as `generate_with_chunks(transforms=[])` already did. It used to replace an empty list with the default transforms.
