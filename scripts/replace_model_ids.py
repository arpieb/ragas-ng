"""Replace retired model IDs across src, examples, docs and e2e tests (#181).

Reads ``scripts/model_registry.toml``. Replaces only in code and prose:
recorded output (``text`` fences, notebook outputs, nbconvert output) and the
files in ``model_ids.ALLOWED`` keep the IDs they were produced with.

    uv run python scripts/replace_model_ids.py --check            # report
    uv run python scripts/replace_model_ids.py --apply            # edit
    uv run python scripts/replace_model_ids.py --check --within 60
        # also models whose shutdown is within 60 days

After --apply, fix by hand what the report lists under "collisions" and
"check these lines", regenerate the listed notebook pages, then run
``uv run pytest tests/unit/test_retired_models.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_ids  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report only")
    mode.add_argument("--apply", action="store_true", help="edit files")
    parser.add_argument(
        "--within",
        type=int,
        default=0,
        metavar="DAYS",
        help="also replace models shutting down within DAYS (default 0)",
    )
    parser.add_argument("--today", type=dt.date.fromisoformat, default=dt.date.today())
    args = parser.parse_args(argv)

    models = model_ids.load_registry()
    mapping = model_ids.replacement_map(models, args.today, args.within)
    paths = list(model_ids.scan_paths()) + list(
        model_ids.scan_paths(model_ids.EXTRA_REPLACE_DIRS, ())
    )
    reports = [
        model_ids.plan_file(p, mapping)
        for p in paths
        if model_ids.rel(p) not in model_ids.ALLOWED
    ]
    reports = [r for r in reports if r.count]

    total = sum(r.count for r in reports)
    print(f"{total} replacements in {len(reports)} files")
    for r in reports:
        pairs = ", ".join(
            f"{old} -> {new} x{n}" for (old, new), n in sorted(r.replacements.items())
        )
        print(f"  {model_ids.rel(r.path)}: {pairs}")

    collisions = [r for r in reports if r.collisions]
    if collisions:
        print("\nCollisions (two old IDs now map to one new ID; check comparisons):")
        for r in collisions:
            for new, olds in r.collisions.items():
                print(f"  {model_ids.rel(r.path)}: {', '.join(sorted(olds))} -> {new}")

    risky = [r for r in reports if r.risky_lines]
    if risky:
        print(
            "\nCheck these lines (sampling/tool parameters next to a replaced "
            "model; reasoning models reject temperature and max_tokens):"
        )
        for r in risky:
            for lineno, line in r.risky_lines:
                print(f"  {model_ids.rel(r.path)}:{lineno}: {line[:100]}")

    notebooks = [r for r in reports if r.path.suffix == ".ipynb"]
    if notebooks:
        print("\nRegenerate these pages from their notebooks after --apply:")
        for r in notebooks:
            page = r.path.with_name("_" + r.path.stem + ".md")
            print(f"  {model_ids.rel(page) if page.exists() else '(no page)'}")

    if args.apply:
        for r in reports:
            assert r.new_text is not None
            r.path.write_text(r.new_text, encoding="utf-8")
        print(f"\nWrote {len(reports)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
