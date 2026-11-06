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
`should_export_span`. The [Langfuse guide](../integrations/_langfuse.md) shows
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
