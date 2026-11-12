# Backend Architecture Guide

For using backends (choosing one, what each keeps, Google Drive setup), see the user guide: `docs/howtos/customizations/storage_backends.md`. This file covers the design, for contributors.

Simple plugin architecture for data storage backends. Implement one abstract class, register via entry points.

## Architecture

```
Registry (dict-like) → Backend (implements BaseBackend) → Storage
```

**Key Files:**
- `base.py` - `BaseBackend` (three abstract methods) and `FileBackend` (one local file per name)
- `registry.py` - Plugin discovery & dict-like access
- `local_csv.py`, `local_jsonl.py` - `FileBackend` reference implementations
- `inmemory.py`, `gdrive_backend.py` - `BaseBackend` reference implementations

## Quick Start

**1. Implement BaseBackend:**
```python
from ragas.backends.base import BaseBackend, Kind


class MyBackend(BaseBackend):
    def __init__(self, connection_string: str):
        self.conn = connection_string

    def _load(self, kind: Kind, name: str) -> List[Dict[str, Any]]:
        # kind is "datasets" or "experiments"; raise FileNotFoundError if missing
        return [{"id": 1, "text": "example"}]

    def _save(self, kind: Kind, name: str, data: List[Dict], data_model):
        # Overwrite whatever is stored under (kind, name)
        pass

    def _list(self, kind: Kind) -> List[str]:
        return []
```

`BaseBackend` provides `load_dataset`, `load_experiment`, `save_dataset`,
`save_experiment`, `list_datasets` and `list_experiments` on top of these.

**2. Register via entry points:**
```toml
# pyproject.toml
[project.entry-points."ragas.backends"]
"my_backend" = "my_package.backend:MyBackend"
```

**3. Use:**
```python
from ragas.backends import get_registry

registry = get_registry()
backend = registry["my_backend"](connection_string="...")
```

## Required Methods

**BaseBackend (3 abstract methods):**
```python
def _load(kind: Kind, name: str) -> List[Dict[str, Any]]   # FileNotFoundError if missing, [] if empty
def _save(kind: Kind, name: str, data: List[Dict], data_model: Optional[Type[BaseModel]]) -> None
def _list(kind: Kind) -> List[str]                          # sorted names
```

`Kind` is `Literal["datasets", "experiments"]`.

**FileBackend (local files):** set `suffix` and implement two methods. The base
handles name validation (`resolve_resource_path`), the `datasets/` and
`experiments/` directories, missing-file errors and listing.
```python
class MyFileBackend(FileBackend):
    suffix = ".ext"

    def _read(self, path: Path) -> List[Dict[str, Any]]: ...
    def _write(
        self, path: Path, data: List[Dict[str, Any]]
    ) -> None: ...  # [] -> empty file
```

## Registry Usage

**Dict-like interface:**
```python
from ragas.backends import get_registry

registry = get_registry()
print(registry)  # {'local/csv': <class 'LocalCSVBackend'>, ...}

# Access backend classes
backend_class = registry["local/csv"]
backend = backend_class(root_dir="./data")

# Check availability
if "my_backend" in registry:
    backend = registry["my_backend"]()
```

## Reference Implementations

**LocalCSVBackend** (`local_csv.py`):
- **Pattern:** File-based storage with CSV format
- **Init:** `LocalCSVBackend(root_dir="./data")`
- **Storage:** `{root_dir}/datasets/{name}.csv`, `{root_dir}/experiments/{name}.csv`
- **Features:** Directory auto-creation, UTF-8 encoding, proper CSV escaping

**LocalJSONLBackend** (`local_jsonl.py`):
- **Pattern:** File-based storage with JSONL format  
- **Init:** `LocalJSONLBackend(root_dir="./data")`
- **Storage:** `{root_dir}/datasets/{name}.jsonl`, `{root_dir}/experiments/{name}.jsonl`
- **Features:** Handles complex nested data, preserves types

**GDriveBackend** (`gdrive_backend.py`, see `gdrive_backend.md`):
- **Pattern:** Cloud storage with Google Sheets format
- **Init:** `GDriveBackend(folder_id, service_account_path=...)`
- **Storage:** Google Drive folder with sheets for datasets/experiments
- **Features:** Collaborative editing, cloud sync, multiple auth methods

## Implementation Notes

- `data_model` is context only: `Dataset` validates rows itself before saving
  and after loading. Text-only stores (CSV, Sheets) use `flat.py` to encode
  cells and `Dataset` decodes them for the model on load.
- Return copies if the store is in memory, so callers can't mutate it.

## Testing Your Backend

```python
def test_backend():
    backend = MyBackend(config="test")
    
    # Test save/load cycle
    test_data = [{"id": 1, "text": "test"}]
    backend.save_dataset("test_dataset", test_data, None)
    loaded = backend.load_dataset("test_dataset")
    assert loaded == test_data
    
    # Test listing
    datasets = backend.list_datasets()
    assert "test_dataset" in datasets
```

## Plugin Development

**Full plugin structure:**
```
my-backend-plugin/
├── pyproject.toml              # Entry point configuration
├── src/my_backend/
│   ├── __init__.py            # Export backend class
│   └── backend.py             # Backend implementation
└── tests/
    └── test_backend.py        # Integration tests
```

**Entry point registration:**
```toml
[project.entry-points."ragas.backends"]
"s3" = "my_backend.backend:S3Backend"
"postgres" = "my_backend.backend:PostgresBackend"
```

**Install & use:**
```bash
pip install my-backend-plugin
python -c "from ragas.backends import get_registry; print(get_registry())"
```

## Registry Internals

**Discovery process:**
1. Registry loads entry points from group `"ragas.backends"`  
2. Each entry point maps `name -> backend_class`
3. Lazy loading - backends loaded on first access
4. Dict-like interface for easy access

**Debugging:**
```python
from ragas.backends import get_registry

registry = get_registry()

# Check what's available
print(f"Available backends: {list(registry.keys())}")

# Get backend info
for name in registry:
    backend_class = registry[name]
    print(f"{name}: {backend_class.__module__}.{backend_class.__name__}")
```

## Design Decisions

**Why BaseBackend instead of separate Project/DataTable backends?**
- Simpler: One interface to implement vs. two
- Clearer: Backend owns both storage and operations
- Flexible: Backends can optimize cross-operation concerns

**Why entry points vs. manual registration?**
- Extensible: Third-party backends without code changes
- Standard: Follows Python packaging conventions  
- Discoverable: Automatic registration on install

**Why dict-like registry?**
- Intuitive: Familiar `registry["name"]` access pattern
- Debuggable: Shows available backends in repr
- Flexible: Supports `in`, `keys()`, iteration

---

**Quick Start:** subclass `FileBackend` for a local file format (see `local_jsonl.py`) or `BaseBackend` otherwise (see `inmemory.py`), then add an entry point.