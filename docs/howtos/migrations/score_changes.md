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
