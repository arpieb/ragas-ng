# scripts

## Model retirements

Providers shut down model IDs on a schedule. A retired ID fails on the first call, so a doc page, quickstart template or docstring that names one is broken for every reader (#176, #181).

- **`model_registry.toml`**: every model ID ragas' code, docs and examples may name, with the provider's shutdown date, listed replacement and source page.
- **`tests/unit/test_retired_models.py`**: reads the registry.
  - It **fails** on any retired ID in `src`, `examples` or `docs` code (not recorded output).
  - It **warns** 60 days before a named model's shutdown date.
  - It checks that every replacement is registered and still alive.
- **`replace_model_ids.py`** (with `model_ids.py`): applies the registry's replacements.

### When a provider announces a retirement

1. **Update the registry.** Add or edit the row: set `shutdown`, `replacement` and `source`. Add the replacement model as its own row if it's new.
2. **Run the check:**
   ```bash
   uv run python scripts/replace_model_ids.py --check --within 60
   ```
   It lists every file and ID it would change, plus two kinds of spot that need a human:
   - **Collisions:** two old IDs now map to the same new one in a file. A comparison (baseline vs. candidate, model A vs. B) may now compare a model with itself.
   - **Check these lines:** `temperature` / `max_tokens` / `tools` next to a replaced model. Reasoning models reject temperature (other than 1) and `max_tokens`, and some only allow tool calls with `reasoning_effort="none"`.
3. **Apply,** then fix those spots by hand:
   ```bash
   uv run python scripts/replace_model_ids.py --apply --within 60
   ```
   Prose that names a model family ("the o1 series") usually needs rewording, not a swapped ID.
4. **Regenerate** the `_*.md` pages the report lists from their notebooks.
5. **If the library default changed,** update `DEFAULT_LLM_MODEL` in `src/ragas/llms/factory.py` and add a `score_changes.md` entry: a different judge gives different scores. Error messages and hints build their suggestion from `DEFAULT_LLM_MODEL`, and metric docstrings use `"<model>"`, so nothing else in `src` needs editing.
6. **Check:**
   ```bash
   uv run pytest tests/unit/test_retired_models.py -W default
   ```
   plus `make check-docs`.

What the script leaves alone:
- recorded output (`text` fences, notebook outputs, nbconvert's indented output) and results filenames, which keep the names they were produced with;
- tiktoken encoding names (`src/ragas/tokenizers.py` and its reference page);
- the migration notes.

The list is `ALLOWED` in `model_ids.py`.

**Not matched:** Bedrock-style IDs that wrap a model name in a vendor prefix and version suffix (`anthropic.claude-3-5-sonnet-20241022-v2:0`). Their replacement isn't simply the new model name, so search for them by hand:
```bash
grep -rnE "\.claude-[0-9a-z.-]+-v[0-9]+:[0-9]" src examples docs
```

Example templates read `RAGAS_EXAMPLE_APP_MODEL` / `RAGAS_EXAMPLE_JUDGE_MODEL`, and their defaults are the literals the script updates.
