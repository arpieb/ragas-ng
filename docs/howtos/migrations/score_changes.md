# Metric score changes

ragas-ng fixes scoring bugs in the `ragas.metrics.collections` metrics, even when that means its scores **no longer match legacy ragas**. Scores from the legacy package are already published. These metrics are for getting correct numbers going forward.

If you compare runs across versions, check this page first. Each entry lists the metric, what changed, and which inputs score differently.

Legacy metrics (`ragas.metrics._*`, deprecated) are generally left as they were unless stated otherwise.

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
