# Storage backends

A `Dataset` and the results of an `@experiment` are saved through a **backend**: local CSV or JSONL files, memory, or Google Sheets. You can also write your own. This guide covers choosing a backend, using it, what each one keeps, and adding new ones.

## Choosing a backend

| Name | Class | Stores | Constructor arguments | Use it for |
|---|---|---|---|---|
| `local/csv` | `LocalCSVBackend` | `.csv` files | `root_dir` | Files you open in a spreadsheet or review in git. |
| `local/jsonl` | `LocalJSONLBackend` | `.jsonl` files | `root_dir` | Keeping numbers, lists and nested values without a data model. |
| `inmemory` | `InMemoryBackend` | Python objects | none | Tests, notebooks and intermediate splits. Nothing persists. |
| `gdrive` | `GDriveBackend` | Google Sheets | `folder_id`, plus credentials | Sharing datasets and results with a team. Needs `pip install "ragas-ng[gdrive]"`. |

List what's registered in your environment:

```python
from ragas.backends import get_registry

print(sorted(get_registry().keys()))
```

```
['gdrive', 'inmemory', 'local/csv', 'local/jsonl']
```

`gdrive` is listed even without the extra installed. Creating it then raises an `ImportError` that lists the packages to install.

## Using a backend

Pass the backend's name together with its constructor arguments, or pass an instance:

```python
from ragas import Dataset

dataset = Dataset(name="qa", backend="local/csv", root_dir="./data")
dataset.append({"question": "What is ragas?", "tags": ["eval", "llm"]})
dataset.append({"question": "What is a testset?", "expected": "Generated QA pairs"})
dataset.save()
```

The local backends keep datasets and experiments in separate folders under `root_dir`, one file per name:

```
data/
├── datasets/
│   └── qa.csv
└── experiments/
```

`save()` overwrites the stored dataset with the current rows. Load it back by name:

```python
loaded = Dataset.load(name="qa", backend="local/csv", root_dir="./data")
print(loaded[0])
```

```
{'question': 'What is ragas?', 'tags': '["eval", "llm"]', 'expected': ''}
```

The same dataset through an instance, here JSONL:

```python
from ragas.backends import LocalJSONLBackend

backend = LocalJSONLBackend(root_dir="./data")
Dataset(
    name="qa", backend=backend, data=[{"question": "q", "score": 0.5, "tags": ["a"]}]
).save()

print(Dataset.load(name="qa", backend=backend)[0])
print(backend.list_datasets())
```

```
{'question': 'q', 'score': 0.5, 'tags': ['a']}
['qa']
```

### Experiments

An experiment saves its results through the dataset's backend unless you choose another. Results go to `experiments/<name>`:

```python
from pydantic import BaseModel

from ragas import experiment


class Result(BaseModel):
    question: str
    answer: str


def my_app(question: str) -> str:
    return "..."  # call your application here


@experiment(Result)
async def run(row):
    return Result(question=row["question"], answer=my_app(row["question"]))


results = await run.arun(loaded, name="baseline")  # data/experiments/baseline.csv
```

--8<--
running_async.md
--8<--

To save results somewhere else, pass `backend` to `arun(...)` or to `@experiment(...)`. Give a backend by name with its constructor arguments in `backend_kwargs`, or pass an instance:

```python
results = await run.arun(
    loaded,
    name="baseline",
    backend="local/jsonl",
    backend_kwargs={"root_dir": "./data"},
)  # data/experiments/baseline.jsonl

# the same, with an instance
results = await run.arun(
    loaded, name="baseline", backend=LocalJSONLBackend(root_dir="./data")
)


# or as the default for every run of this experiment
@experiment(Result, backend="local/jsonl", backend_kwargs={"root_dir": "./results"})
async def run_to_jsonl(row): ...
```

`backend_kwargs` is separate because `arun`'s other keyword arguments go to your experiment function. A `backend_kwargs` passed to `arun` replaces the decorator's.

## What each backend keeps

CSV stores every value as text. JSONL stores JSON types. Load with a Pydantic `data_model` to get your types back from either:

```python
from typing import List, Optional


class Row(BaseModel):
    question: str
    tags: List[str] = []
    expected: Optional[str] = None


typed = Dataset.load(name="qa", backend="local/csv", root_dir="./data", data_model=Row)
print(typed[0])
```

```
question='What is ragas?' tags=['eval', 'llm'] expected=''
```

| | `local/csv` | `local/jsonl` | `inmemory` | `gdrive` |
|---|---|---|---|---|
| Without a `data_model` | Strings; lists and dicts as JSON text | JSON types; dates as ISO strings | Exactly what you saved | Strings |
| With a `data_model` | Types restored from the model | Types restored from the model | Exactly what you saved | Types restored from the model |
| Rows with different columns | Header is the union of all rows' keys, in first-seen order | Yes | Yes | Header is the union of all rows' keys, sorted |
| `None` vs `""` | Both load as `""` | Kept | Kept | Both load as `""` |

!!! warning "A missing text value comes back from CSV as a blank string"
    CSV and Google Sheets store `None` and `""` the same way, as an empty cell. When you load with a `data_model`, an empty cell in an optional text field such as `reference` comes back as `""`, not `None`.

    For scoring, that matters: `metrics_experiment` treats `""` as a value. A row whose `reference` was missing is scored against an empty reference, usually a low score, instead of raising the missing-column error a `None` gives. `summarize()` doesn't count it as missing either.

    Save datasets you'll score, such as `Testset.to_dataset(...)` output, with `local/jsonl` or `inmemory`, which keep `None`. If you must use CSV, check for blank required fields before scoring, e.g. `[i for i, row in enumerate(dataset) if row.reference == ""]`.

Backends don't guess types: a string that looks like a number or a date stays a string unless your model says otherwise. See [Behavior changes](../migrations/behavior_changes.md#csv-files-every-rows-columns-json-cells-typed-loads-100) for what changed from ragas.

## In-memory

`InMemoryBackend` keeps rows as Python objects for as long as the instance lives. Each instance has its own storage, so keep a reference to it and load through the same one:

```python
from ragas.backends import InMemoryBackend

mem = InMemoryBackend()
Dataset(name="scratch", backend=mem, data=[{"x": 1}]).save()
print(Dataset.load(name="scratch", backend=mem)[0])
```

```
{'x': 1}
```

## Google Drive

`GDriveBackend` stores each dataset and experiment as a Google Sheet in a Drive folder you choose. It creates `datasets/` and `experiments/` subfolders there.

1. Install the extra: `pip install "ragas-ng[gdrive]"`.
2. In a Google Cloud project, enable the **Google Drive API** and **Google Sheets API**.
3. Create credentials. Either:
    - **a service account** (recommended for scripts and CI): download its JSON key, and share the Drive folder with the service account's email address as an editor; or
    - **an OAuth client** (desktop app) for interactive use: download its client JSON. The first run opens a browser to sign in and saves a token for later runs.

    See Google's [Create access credentials](https://developers.google.com/workspace/guides/create-credentials).
4. Copy the folder ID from the folder's URL: `https://drive.google.com/drive/folders/<folder_id>`.

```python
dataset = Dataset(
    name="qa",
    backend="gdrive",
    folder_id="1AbC...xyz",
    service_account_path="service-account.json",
    # or, for OAuth:
    # credentials_path="credentials.json",
    # token_path="token.json",
)
```

Instead of passing the paths, you can set `GDRIVE_SERVICE_ACCOUNT_PATH`, `GDRIVE_CREDENTIALS_PATH` and `GDRIVE_TOKEN_PATH` (default `token.json`). A folder the credentials can't access raises `ValueError` when the backend is created.

## Adding a backend

A backend is a subclass of `BaseBackend` with three methods. Each takes `kind`, which is `"datasets"` or `"experiments"` (the two are stored apart), and reads or writes a list of dicts by name. `BaseBackend` builds the public `load_dataset`, `save_experiment`, `list_datasets` and so on from them:

```python
import typing as t

from ragas.backends import BaseBackend, register_backend


class DictBackend(BaseBackend):
    def __init__(self):
        self.store: t.Dict[str, t.List[t.Dict[str, t.Any]]] = {}

    def _load(self, kind, name):
        key = f"{kind}/{name}"
        if key not in self.store:
            raise FileNotFoundError(f"{kind[:-1]} {name!r} not found")
        return self.store[key]

    def _save(self, kind, name, data, data_model):
        self.store[f"{kind}/{name}"] = list(data)  # overwrite

    def _list(self, kind):
        return sorted(
            k.split("/", 1)[1] for k in self.store if k.startswith(f"{kind}/")
        )


register_backend("dict", DictBackend, aliases=["my-dict"])
dataset = Dataset(name="demo", backend="my-dict", data=[{"q": "hi"}])
dataset.save()
print(dataset.backend.list_datasets())
```

```
['demo']
```

Loads return plain dicts; `Dataset` validates them with the `data_model`. A load of a name that doesn't exist should raise `FileNotFoundError`, as the built-in backends do.

For a backend that keeps one local file per dataset, subclass `FileBackend` instead. It takes `root_dir`, stores `root_dir/datasets/<name><suffix>` and `root_dir/experiments/<name><suffix>`, rejects names that would escape `root_dir`, and lists by suffix; you set `suffix` and write `_read(path)` and `_write(path, data)`:

```python
import json

from ragas.backends import FileBackend


class JSONBackend(FileBackend):
    suffix = ".json"

    def _read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")
```

To make a backend available by name whenever your package is installed, register it as an entry point in the `ragas.backends` group instead of calling `register_backend()`:

```toml
# pyproject.toml of your package
[project.entry-points."ragas.backends"]
"my/backend" = "my_package.backend:MyBackend"
```

The [backend architecture notes](https://github.com/arpieb/ragas-ng/blob/main/src/ragas/backends/README.md) in the source tree cover the design and testing in more detail.
