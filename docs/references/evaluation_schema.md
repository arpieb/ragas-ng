# Schemas

Samples, messages, datasets and experiments. To score a `Dataset` of samples, see [Scoring a dataset](evaluate.md).

::: ragas.dataset_schema
    options:
        members_order: "source"
        members:
            - BaseSample
            - SingleTurnSample
            - MultiTurnSample
            - sample_from_dict
            - sample_type_for

::: ragas.messages
    options:
        members_order: "source"

::: ragas.dataset.Dataset
    options:
        show_root_heading: True

::: ragas.experiment.Experiment
    options:
        show_root_heading: True
