"""Model-ID retirement tooling (#181).

Shared by ``scripts/replace_model_ids.py`` and
``tests/unit/test_retired_models.py``, so both agree on what is retired and
on which parts of a file are code rather than recorded output.

Usage:
    uv run python scripts/replace_model_ids.py --check   # report only
    uv run python scripts/replace_model_ids.py --apply   # edit files
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "scripts" / "model_registry.toml"

# Where model IDs live that someone would run or copy.
SCAN_DIRS = ("src", "examples", "docs")
SCAN_FILES = ("README.md", "CLAUDE.md")
SCAN_SUFFIXES = (".py", ".md", ".ipynb", ".txt")
# Also updated by the replace script (they call real providers), but not
# guarded: e2e tests are opt-in and may pin a model on purpose.
EXTRA_REPLACE_DIRS = ("tests/e2e", "tests/docs")

# Files where an old ID is correct as written.
ALLOWED = {
    # tiktoken encoding names; no API call, and tiktoken doesn't know new IDs
    "src/ragas/tokenizers.py",
    "docs/references/tokenizers.md",
    # historical notes about what changed
    "docs/howtos/migrations/behavior_changes.md",
    "docs/howtos/migrations/score_changes.md",
}

WARN_DAYS = 60


@dataclass(frozen=True)
class Model:
    id: str
    provider: str
    shutdown: t.Optional[dt.date] = None
    retired: bool = False
    replacement: t.Optional[str] = None
    source: str = ""
    note: str = ""

    def is_retired(self, today: dt.date) -> bool:
        return self.retired or (self.shutdown is not None and self.shutdown <= today)

    def retires_within(self, today: dt.date, days: int) -> bool:
        return (
            not self.is_retired(today)
            and self.shutdown is not None
            and (self.shutdown - today).days <= days
        )


def load_registry(path: Path = REGISTRY) -> t.List[Model]:
    with open(path, "rb") as f:
        rows = tomllib.load(f)["model"]
    return [Model(**row) for row in rows]


def id_pattern(ids: t.Iterable[str]) -> re.Pattern:
    """Match whole model IDs.

    Not preceded by a word char, ``.`` or ``-`` (so ``gpt-4`` doesn't match
    inside ``openai.gpt-4`` prefixes' neighbours or ``my-gpt-4``), and not
    followed by a word char, ``-``, ``.<digit>`` (``gpt-4`` vs ``gpt-4.1``) or
    a results-file suffix (``gpt-4o.csv`` is a filename from a past run).
    """
    alternatives = sorted(set(ids), key=len, reverse=True)
    return re.compile(
        r"(?<![\w.-])("
        + "|".join(re.escape(a) for a in alternatives)
        + r")(?![\w-]|\.\d|\.csv|\.json)"
    )


# --- which parts of a file are code ---------------------------------------

_OUTPUT_LANGS = ("", "text", "output", "console", "plaintext")


def _is_generated(path: Path) -> bool:
    """nbconvert pages (``_name.md``): outputs are 4-space indented."""
    return path.suffix == ".md" and path.name.startswith("_")


def code_segments(path: Path, text: str) -> t.List[t.Tuple[int, int]]:
    """``(start, end)`` character ranges of ``text`` that are code or prose.

    Excluded: fenced blocks with no language or ``text`` (recorded output),
    nbconvert's indented output in generated pages, and notebook outputs
    (notebooks are handled cell by cell in ``replace_in_notebook``).
    """
    if path.suffix != ".md":
        return [(0, len(text))]
    segments, pos, in_fence, output = [], 0, False, False
    generated = _is_generated(path)
    for line in text.splitlines(keepends=True):
        end = pos + len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                output = stripped[3:].strip() in _OUTPUT_LANGS
            else:
                in_fence = False
        elif not (in_fence and output) and not (
            generated
            and not in_fence
            and line.startswith("    ")
            and not stripped.startswith(("-", "*", "|"))
        ):
            segments.append((pos, end))
        pos = end
    return segments


def code_text(path: Path) -> str:
    """The code and prose of ``path``, with recorded output removed."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        nb = json.loads(text)
        return "\n".join("".join(c["source"]) for c in nb["cells"])
    return "".join(text[a:b] for a, b in code_segments(path, text))


def scan_paths(
    dirs: t.Sequence[str] = SCAN_DIRS, files: t.Sequence[str] = SCAN_FILES
) -> t.Iterator[Path]:
    for d in dirs:
        for path in sorted((ROOT / d).rglob("*")):
            if (
                path.is_file()
                and path.suffix in SCAN_SUFFIXES
                and not any(part.endswith(".egg-info") for part in path.parts)
                and "__pycache__" not in path.parts
            ):
                yield path
    for name in files:
        if (ROOT / name).exists():
            yield ROOT / name


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# --- replacing ------------------------------------------------------------

# Direct provider calls whose parameters a reasoning model rejects; worth a
# human look when the model next to them changes (#179).
_RISKY_PARAM = re.compile(r"\b(temperature|max_tokens|top_p|tools|tool_choice)\s*=")


@dataclass
class FileReport:
    path: Path
    replacements: t.Dict[t.Tuple[str, str], int] = field(default_factory=dict)
    collisions: t.Dict[str, t.Set[str]] = field(default_factory=dict)
    risky_lines: t.List[t.Tuple[int, str]] = field(default_factory=list)
    new_text: t.Optional[str] = None

    @property
    def count(self) -> int:
        return sum(self.replacements.values())


def _replace(text: str, mapping: t.Dict[str, str], report: FileReport) -> str:
    pattern = id_pattern(mapping)

    def sub(m: re.Match) -> str:
        old = m.group(1)
        new = mapping[old]
        report.replacements[(old, new)] = report.replacements.get((old, new), 0) + 1
        return new

    return pattern.sub(sub, text)


def _replace_segments(path: Path, text: str, mapping, report) -> str:
    out, pos = [], 0
    for a, b in code_segments(path, text):
        out.append(text[pos:a])
        out.append(_replace(text[a:b], mapping, report))
        pos = b
    out.append(text[pos:])
    return "".join(out)


def _replace_notebook(path: Path, text: str, mapping, report) -> str:
    nb = json.loads(text)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            new = _replace_segments(Path("cell.md"), src, mapping, report)
        else:
            new = _replace(src, mapping, report)
        if new != src:
            lines = new.split("\n")
            cell["source"] = [line + "\n" for line in lines[:-1]] + (
                [lines[-1]] if lines[-1] else []
            )
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def plan_file(path: Path, mapping: t.Dict[str, str]) -> FileReport:
    """What replacing ``mapping`` in ``path`` would change (nothing written)."""
    report = FileReport(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        new = _replace_notebook(path, text, mapping, report)
    else:
        new = _replace_segments(path, text, mapping, report)
    if not report.count:
        return report
    report.new_text = new
    # two distinct old IDs collapsing onto one new ID in a file: a comparison
    # (baseline vs candidate, model A vs B) may now compare a model with itself
    by_new: t.Dict[str, t.Set[str]] = {}
    for old, new_id in report.replacements:
        by_new.setdefault(new_id, set()).add(old)
    report.collisions = {
        n: olds
        for n, olds in by_new.items()
        if len({re.sub(r"-20\d\d-\d\d-\d\d$|-0\d\d$", "", o) for o in olds}) > 1
    }
    new_ids = set(by_new)
    lines = new.splitlines()
    for i, line in enumerate(lines):
        if _RISKY_PARAM.search(line):
            window = lines[max(0, i - 6) : i + 7]
            if any(any(n in w for n in new_ids) for w in window):
                report.risky_lines.append((i + 1, line.strip()))
    return report


def replacement_map(
    models: t.Sequence[Model], today: dt.date, within_days: int = 0
) -> t.Dict[str, str]:
    """Old ID -> final replacement, for models retired (or retiring within
    ``within_days``), following chains (A -> B -> C) to a live model."""
    by_id = {m.id: m for m in models}

    def final(model_id: str, seen: t.Tuple[str, ...] = ()) -> str:
        m = by_id.get(model_id)
        if m is None or not m.replacement or model_id in seen:
            return model_id
        if not (m.is_retired(today) or m.retires_within(today, within_days)):
            return model_id
        return final(m.replacement, seen + (model_id,))

    return {
        m.id: final(m.id)
        for m in models
        if m.replacement
        and (m.is_retired(today) or m.retires_within(today, within_days))
    }
