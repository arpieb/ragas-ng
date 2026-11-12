# Scoring a dataset

`metrics_experiment()` scores metrics over a `Dataset` as an experiment; `summarize()` and `check()` read the per-metric means and gate CI on them. They replace `evaluate()`: see [From `evaluate()` to experiments](../howtos/migrations/evaluate_to_experiment.md).

```python
from ragas import check, metrics_experiment, summarize

run = metrics_experiment([Faithfulness(llm=llm)])
exp = run.run(dataset, name="nightly")  # await run.arun(...) in async code
summarize(exp)  # {'faithfulness': {'mean': 0.86, 'count': 48, 'missing': 2}}
check(exp, min_mean={"faithfulness": 0.8})
```

## metrics_experiment()

::: ragas.scoring.metrics_experiment

## MetricsExperiment

::: ragas.scoring.MetricsExperiment
    options:
        inherited_members: true
        members:
            - arun
            - run
            - metric_names

## summarize()

::: ragas.scoring.summarize

## check()

::: ragas.scoring.check
