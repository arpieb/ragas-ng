# Metric score changes

ragas-ng fixes scoring bugs in the `ragas.metrics.collections` metrics, even when that means its scores **no longer match legacy ragas**. Scores from the legacy package are already published. These metrics are for getting correct numbers going forward.

If you compare runs across versions, check this page first. Each entry lists the metric, what changed, and which inputs score differently.

## The short imports score through collections

`from ragas.metrics import BleuScore` (and every other deprecated metric name and pre-built instance such as `faithfulness`) returns the legacy class, but **its score comes from the matching `ragas.metrics.collections` metric**. So every fix on this page applies whichever import you use. The legacy interface is unchanged: the constructor arguments, `evaluate()` injecting `llm` and `embeddings`, `single_turn_ascore`, `get_prompts`, `adapt_prompts` and `train`.

It falls back to the legacy implementation, and logs a warning saying so, when the collections metric can't honour the configuration:

| Situation | Example |
|---|---|
| The LLM isn't an `InstructorBaseRagasLLM` from `llm_factory()` | OCI GenAI, Haystack or LlamaIndex wrappers |
| The embeddings aren't a modern `BaseRagasEmbedding` | a custom legacy embeddings class |
| A prompt was customised | `set_prompts(...)`, `adapt_prompts(...)`, `train(...)` |
| A legacy-only option was changed | `FactualCorrectness(language=...)`, `AnswerSimilarity(is_cross_encoder=True)`, a `distance_measure` that isn't a `NonLLMStringSimilarity` |

Only the private modules (`from ragas.metrics._bleu_score import BleuScore`) still give the unmodified legacy class.

## FactualCorrectness: recall and f1 ([#48](https://github.com/arpieb/ragas-ng/issues/48), [#49](https://github.com/arpieb/ragas-ng/issues/49))

`ragas.metrics.collections.FactualCorrectness`

**Recall was computed with the wrong numerator.** Two claim sets are judged:

- the response's claims, checked against the reference (precision direction)
- the reference's claims, checked against the response (recall direction)

Recall used the *precision-direction* supported count over the precision-direction supported count plus the recall-direction misses. It was a mix of the two, and it moved with how many claims the *response* made. f1 combined the same mixed counts.

| | Before | Now |
|---|---|---|
| precision | supported response claims / response claims | unchanged |
| recall | supported response claims / (supported response claims + unsupported reference claims) | supported reference claims / reference claims |
| f1 (f-beta) | `fbeta_score(tp, fp, fn)` over the mixed counts | f-beta of the precision and recall above |
| no claims extracted | `0.0` | `NaN` (undefined) |

**Who sees a difference:** every `mode="recall"` and `mode="f1"` (the default) score where the response and reference produce different claim counts or coverage. Precision scores are unchanged, except for the no-claims case.

**Example:** the response makes 4 claims, all supported. The reference makes 4 claims, of which the response covers 1. Recall was `4/7 = 0.57` and is now `1/4 = 0.25`. F1 was `0.73` and is now `0.40`.

**Also:** `mode="recall"` no longer makes the precision-direction LLM calls it didn't need, so it's two calls instead of four.

## BleuScore: whole text, not first sentence ([#50](https://github.com/arpieb/ragas-ng/issues/50))

`ragas.metrics.collections.BleuScore`

Response and reference were split on `". "` and passed to `sacrebleu.corpus_bleu` so that each reference sentence became its own reference *stream* for only the first response sentence. **Everything after the response's first sentence was ignored.**

| | Before | Now |
|---|---|---|
| segments scored | first response sentence, against any one reference sentence | the whole response against the whole reference |

**Who sees a difference:** any multi-sentence response. A response whose first sentence matches the reference and whose remaining sentences are wrong used to score `1.0`. It now scores about `0.33`.

## Context precision: perfect ranking is exactly 1.0 ([#57](https://github.com/arpieb/ragas-ng/issues/57))

`ContextPrecision`, `ContextUtilization`, `ContextPrecisionWithReference`, `ContextPrecisionWithoutReference`, `NonLLMContextPrecisionWithReference` (collections)

Average precision divided by `relevant_count + 1e-10` unconditionally, so a perfect ranking scored `0.9999999999` and never `1.0`. The epsilon now applies only where it was needed: no relevant contexts gives `0.0`.

**Who sees a difference:** every non-zero score rises by a relative ~1e-10. It's invisible at normal precision but breaks `score == 1.0` checks and CI thresholds set at 1.0.

## SummaryScore: no crash, and a questions-based denominator ([#60](https://github.com/arpieb/ragas-ng/issues/60))

`ragas.metrics.collections.SummaryScore`

The QA component divided the number of `"1"` answers by the number of answers the judge *returned*:

| | Before | Now |
|---|---|---|
| no questions generated | `ZeroDivisionError` (after an extra LLM call) | `NaN`, answer call skipped |
| judge returns a different number of answers than questions | ratio over whatever came back | `NaN`, with a warning |
| `" 1 "` (whitespace) | counted as unanswered | counted as answered |

**Who sees a difference:** samples whose answer count didn't match the question count now score NaN instead of an over- or under-stated ratio. Samples that used to crash now return NaN.

## CHRFScore: missing reference is NaN, not 0.0 ([#61](https://github.com/arpieb/ragas-ng/issues/61))

`ragas.metrics.collections.CHRFScore`

| Input | Before | Now |
|---|---|---|
| reference empty, whitespace, `None` or not a string | `0.0` | `NaN` |
| response not a string | `0.0` | `NaN` |
| response empty, valid reference | `0.0` | `0.0` (unchanged) |

A missing reference is a gap in the dataset, not a model failure. Scoring it `0.0` pulled the mean down for something the model didn't do. As with other undefined scores, NaN rows are left out of the mean.

## AnswerRelevancy: clipped to 0–1 ([#56](https://github.com/arpieb/ragas-ng/issues/56))

`ragas.metrics.collections.AnswerRelevancy`

The score is the mean cosine similarity between the user's question and questions generated from the response. Cosine similarity can be slightly negative for unrelated embeddings, so the metric could return values like `-0.02` despite being documented as 0–1. Scores are now clipped to `[0, 1]`.

**Who sees a difference:** only samples that scored below 0. They now score `0.0`.

## Collections metrics reject a string where a list is expected ([#54](https://github.com/arpieb/ragas-ng/issues/54))

All `ragas.metrics.collections` metrics

A Python `str` is iterable, so passing `retrieved_contexts="some text"` instead of `["some text"]` was read as a list of single characters:

- ContextPrecision made one judge call per character.
- Metrics that join their contexts scored the characters as if they were contexts.
- QuotedSpansAlignment returned `0.0`.

Every list-typed `ascore()` input now raises `ListInputError` (a subclass of both `TypeError` and `ValueError`) for a bare string, before any LLM call. That covers `retrieved_contexts`, `reference_contexts`, the context-ID lists, `reference_topics`, and multi-turn `user_input` / `reference_tool_calls`.

`evaluate()` users aren't affected: `SingleTurnSample` already rejects a string there.

## One verdict per statement, or no score ([#52](https://github.com/arpieb/ragas-ng/issues/52))

`Faithfulness`, `FactualCorrectness`, `NoiseSensitivity`, `TopicAdherence` (collections)

These metrics send the judge a list (statements, claims, topics) and score the fraction it marks supported. They divided by however many verdicts *came back*. If the judge dropped an unsupported statement, faithfulness went **up**.

Now each submitted item needs exactly one verdict:

1. If the counts match, the verdicts are taken in order.
2. If the counts differ, verdicts are matched to items by the statement text the judge echoes back. Extra verdicts are ignored.
3. If that fails, the judge is asked once more.
4. If it still fails, the sample scores `NaN` with a warning. The NaN shows up as a missing row in `EvaluationResult.summary()` and the repr.

| Metric | Before | Now |
|---|---|---|
| Faithfulness, FactualCorrectness | ratio over the returned verdicts | ratio over submitted statements, or NaN |
| NoiseSensitivity | ragged verdict matrix (crash or nonsense) | NaN |
| TopicAdherence | missing classifications padded with "not a reference topic", extras truncated | NaN |

**Who sees a difference:** only samples where the judge returned the wrong number of verdicts. Others are unchanged.

ContextRecall and AnswerCorrectness are not covered. Their judge splits the reference into statements itself, so there is no submitted list to check against.

## strictness actually samples ([#55](https://github.com/arpieb/ragas-ng/issues/55))

`AnswerRelevancy`, `AspectCritic`, `SimpleCriteriaScore` (collections)

`strictness=N` is documented as N independent judgements: AnswerRelevancy averages over N generated questions, and the criteria metrics take a majority vote. The N calls were identical, at the deterministic defaults (temperature `0.01`, top_p `0.1`), so the samples barely differed. **With a cache enabled, all N calls returned the same cached response,** so one judgement was counted N times.

| `strictness` | Before | Now |
|---|---|---|
| 1 (default) | one call with the configured settings | unchanged |
| N > 1 | N identical calls, run one after another; one call with a cache | N calls run concurrently at temperature `0.3` (the legacy sampling value), `top_p` dropped, each cached separately |

**Who sees a difference:** anyone using `strictness > 1`. AnswerRelevancy averages over genuinely different questions, and criteria verdicts are real majority votes. Scores can move in either direction. Reasoning models (o-series, gpt-5+) still use the only temperature they accept, 1.0.

`agenerate()` / `generate()` on `InstructorLLM` and `LiteLLMStructuredLLM` gain optional `temperature` and `sample_index` arguments. `sample_index` only distinguishes cache entries and is never sent to the provider.

## ToolCallAccuracy: missing or extra calls are a coverage penalty, not 0 ([#62](https://github.com/arpieb/ragas-ng/issues/62))

`ragas.metrics.collections.ToolCallAccuracy`

The docstring promised a proportional coverage penalty for length mismatches, but alignment compared the full sequences, so **any** length difference scored `0` and the penalty code never ran. Extra calls also went unpenalized whenever the lengths happened to match up.

| Case | Before | Now |
|---|---|---|
| 2 of 3 expected calls, in order | `0.0` | `0.67` |
| all expected calls plus 1 extra (3 vs 2) | `0.0` | `0.67` |
| a call to a tool not in the reference | `0.0` | `0.0` |
| right tools, wrong order (strict) | `0.0` | `0.0` |
| equal lengths | average argument accuracy | unchanged |

Rule: the shorter list must be contained in the longer one, as an in-order subsequence under `strict_order=True` or as a multiset of tool names otherwise. The matched calls' argument accuracy is divided by the longer list's length. In flexible mode, when several calls share a tool name, each is paired with the counterpart whose arguments match best.

## Non-LLM context recall: threshold is inclusive ([#64](https://github.com/arpieb/ragas-ng/issues/64))

`ragas.metrics.collections.NonLLMContextRecall`

Recall counted a reference context as recalled when similarity was `> threshold`; precision used `>=`. Both now use `>=`. **Who sees a difference:** only contexts whose similarity equals the threshold exactly. In practice that means `threshold=1.0`, where an exact match counted for precision but scored `0` for recall.

## ROUGE, BLEU and quoted spans on non-English text ([#51](https://github.com/arpieb/ragas-ng/issues/51))

`RougeScore`, `BleuScore`, `QuotedSpansAlignment` (collections)

These assumed space-separated Latin text:

| | Before | Now |
|---|---|---|
| **RougeScore**, non-ASCII text | `rouge_score`'s tokenizer drops everything outside `a-z0-9`: identical Russian, Greek, Arabic or Chinese text scored `0.0`, and `très` became `tr s` | Unicode-aware tokenizer; each CJK character is a token. **ASCII-only text is tokenized exactly as before.** |
| **BleuScore**, Chinese / Japanese | the space-based `13a` tokenizer saw one "word" per sentence, so a one-character difference scored `0.0` | `tokenize="zh"` for Chinese, `"char"` for Japanese, unless you pass `tokenize` |
| **QuotedSpansAlignment** | only `"…"`, `'…'`, `“…”`, `‘…’`; quoted CJK sentences counted as one word, so they fell below `min_span_words` | also `«…»`, `‹…›`, `„…“`, `「…」`, `『…』`; CJK characters count individually |

**Who sees a difference:** anyone scoring text that isn't plain ASCII. For English, ROUGE and BLEU scores are unchanged. The legacy `ragas.metrics.quoted_spans` module had lost its curly-quote characters and matched straight quotes only; it now uses the same pattern.

## SemanticSimilarity: clipped to 0–1 ([#107](https://github.com/arpieb/ragas-ng/issues/107))

`ragas.metrics.collections.SemanticSimilarity`

The score is the cosine similarity of the two embeddings, which can be slightly negative for unrelated texts, despite the metric being documented as 0–1. It is now clipped to `[0, 1]`, the same as AnswerRelevancy. **Who sees a difference:** only samples that scored below 0; they now score `0.0`. With `threshold` set, the score is binary as before.

## AnswerCorrectness: no statements is undefined, not a perfect match ([#136](https://github.com/arpieb/ragas-ng/issues/136))

`ragas.metrics.collections.AnswerCorrectness`

When the response or the reference yielded no statements, the factuality part was set to `1.0` ("assume perfect match"). An evasive or one-word answer then got full factuality credit and scored about `0.75` or higher with the default weights. The sample now scores `NaN`, the same as FactualCorrectness ([#49](https://github.com/arpieb/ragas-ng/issues/49)), and shows as a missing row in `EvaluationResult.summary()`. **Who sees a difference:** only samples where statement extraction came back empty.

## ContextRecall: one classification per reference sentence ([#52](https://github.com/arpieb/ragas-ng/issues/52))

`ragas.metrics.collections.ContextRecall`

The judge used to split the reference into statements itself, and recall was the fraction of whatever classifications it returned. A sentence it dropped silently left the denominator, and dropping an unsupported sentence *raised* recall.

The reference is now split into sentences in code: on `. ! ?` before a capitalised word or digit, and after `。！？`, but not after common abbreviations like "Dr." or "e.g.". The judge gets that list and must return one classification per sentence, under the same matching, retry and NaN rule as Faithfulness above.

**Who sees a difference:** scores move where the judge used to split differently, for example breaking one long sentence into several statements, or where it dropped sentences. Recall is now always out of the reference's sentence count.

## evaluate() with no metrics: the collections defaults ([#76](https://github.com/arpieb/ragas-ng/issues/76))

`ragas.evaluate`

`evaluate(dataset)` with no `metrics=` runs AnswerRelevancy, ContextPrecision,
Faithfulness and ContextRecall. It used to run the plain legacy objects from
the private modules, so it skipped the hand-off that
`from ragas.metrics import faithfulness` gets, and scored with the legacy
implementations. It now builds the hand-off classes fresh for each call. With an
LLM from `llm_factory()` (the default), they score through
`ragas.metrics.collections`, so every collections fix in this list applies.
**Who sees a difference:** only callers that leave `metrics=` out; passing
metrics explicitly is unchanged.

## ContextEntityRecall: a perfect match is 1.0

`ragas.metrics.collections.ContextEntityRecall`

The denominator carried a `1e-8` epsilon, so finding every reference entity
scored 0.99999999, not 1.0. A reference with no entities scored 0.0. Recall is
undefined there, and it is now NaN. **Who sees a difference:** every sample
moves by at most 1e-8, and samples whose reference has no entities now show as
missing rows in `EvaluationResult.summary()`.

## DynamicFewShotPrompt: no examples below the threshold ([#93](https://github.com/arpieb/ragas-ng/issues/93))

`ragas.prompt.DynamicFewShotPrompt` (the prompts of aligned `DiscreteMetric`,
`NumericMetric` and `RankingMetric` metrics)

When no stored example reached `similarity_threshold`, the prompt included the
most recent examples anyway, presented as similar ones. It now includes no
examples in that case, which matches `FewShotPydanticPrompt`. **Who sees a
difference:** aligned metrics with an embedding model, on inputs unlike any of
their stored examples. The judge now sees no examples for those inputs, so its
verdicts can change. Lower `similarity_threshold` to include more examples.

## Legacy prompts with llm_factory() LLMs: n and temperature apply ([#94](https://github.com/arpieb/ragas-ng/issues/94))

`ragas.prompt.PydanticPrompt` with an `InstructorBaseRagasLLM`

The legacy prompt path made one call whatever `n` asked for, and ignored the
`temperature` it was given. It now makes `n` concurrent calls at a sampling
temperature (0.3 unless given), each with its own cache entry, matching the
collections metrics' [strictness fix](#strictness-actually-samples-55). A
temperature passed to a prompt is used.

**Who sees a difference:** only metrics that run on the legacy implementation
with an `llm_factory()` LLM. That means the private modules, such as
`from ragas.metrics._answer_relevance import AnswerRelevancy`, or a metric
whose prompts were customised, where the hand-off falls back to legacy (see
[above](#the-short-imports-score-through-collections)).

- `AnswerRelevancy(strictness=N)` now averages over N generated questions
  instead of one.
- The NVIDIA metrics' fixed `temperature=0.1` judge calls now run at 0.1
  rather than the LLM's configured temperature.

## The default judge model is gpt-6-luna ([#176](https://github.com/arpieb/ragas-ng/issues/176))

`evaluate()`, testset generation and the testset transforms fall back to `ragas.llms.base.DEFAULT_LLM_MODEL` when you don't pass an `llm`. That was `gpt-4o-mini`, which OpenAI shuts down on 2027-01-14. It is now **`gpt-6-luna`**, OpenAI's listed replacement and its cost tier ($0.10 / $0.50 per 1M input/output tokens, vs. $0.15 / $0.60).

- **Scores change** for any run that relied on the default. It's a different judge.
- `gpt-6-luna` is a reasoning model, so ragas doesn't send it the low default `temperature` or `top_p`. The default goes through LiteLLM, which sends no sampling parameters, so the provider's default applies. With an OpenAI client, ragas sends `temperature=1`. Output is capped only if you set `max_tokens` (see [LLM adapters](../llm-adapters.md)). The default judge is therefore no longer near-deterministic at `temperature=0.01`, and runs vary more from one to the next.
- OpenAI publishes no dated snapshot of `gpt-6-luna`, so the default can't be pinned. To keep scores comparable across runs and upgrades, pass your own `llm=llm_factory("<model>", ...)` with the exact model you want, and record it with your results.

## Claude judges: no sampling parameters ([#177](https://github.com/arpieb/ragas-ng/issues/177))

ragas no longer sends `temperature` or `top_p` to Claude unless you set them yourself. Current Claude models (verified with `claude-haiku-5-5` and `claude-sonnet-5-5`) reject both, and anthropic SDK 2.6 removed them from `messages.create()`. So `llm_factory(model, provider="anthropic", client=Anthropic())` failed on every call, and `strictness` sampling failed through LiteLLM.

- **Claude judges now run at the provider's default sampling.** Before, the native-client path used `temperature=0.01` with older SDKs and models. Scores from those setups can change, and repeated runs vary more.
- **`strictness > 1`** still draws several samples. They now differ because of Claude's default sampling, not a ragas-chosen `temperature=0.3`. ragas logs this once per LLM.
- On the LiteLLM path, a temperature ragas chooses for sampling is sent with LiteLLM's `drop_params`. LiteLLM then drops it for models that reject it, and other providers still get it.
