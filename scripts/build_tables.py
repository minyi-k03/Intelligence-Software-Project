"""Build CSV and Markdown evaluation tables from scored JSONL runs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.eval.tables import Row, build_tables

TITLES = {
    "main": "본 비교 (3라운드)",
    "ablation": "절제 (1라운드 측정, 같은 묶음의 M 대비)",
    "side_effects": "부작용 지표",
    "mcnemar": "M 대 P1 양측 McNemar 정확 검정",
}


def write_table(directory: Path, name: str, rows: list[Row], *, verification: bool) -> None:
    """Write one table as CSV and Markdown using the same ordered columns."""
    columns = list(rows[0])
    with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(value: object) -> str:
        """Render one table cell without discarding small nonzero p-values."""
        if value is None:
            return "—"
        return format(value, ".6g") if isinstance(value, float) else str(value)

    def line(values: list[str]) -> str:
        """Join a sequence of cell strings into one Markdown table row."""
        return "| " + " | ".join(values) + " |"

    title = TITLES[name] + (" · 검증 모드" if verification else "")
    lines = [f"# {title}", "", line(columns), line(["---"] * len(columns))]
    lines.extend(line([cell(row[column]) for column in columns]) for row in rows)
    (directory / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Parse local paths, validate all input runs, and write the four tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    parser.add_argument(
        "--verification", action="store_true",
        help="Allow the same evaluation subset in every run. Requires --output-dir.",
    )
    args = parser.parse_args()
    if args.verification and args.output_dir.resolve() == (ROOT / "results/tables").resolve():
        parser.error("use --output-dir outside results/tables for verification")
    try:
        tables = build_tables(
            args.runs_dir, ROOT / "data/frozen/eval.jsonl", verification=args.verification
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            write_table(args.output_dir, name, rows, verification=args.verification)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"build_tables: {exc}\n")


if __name__ == "__main__":
    main()
