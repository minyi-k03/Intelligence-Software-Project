"""Verify result tables end to end with scored mock logs."""

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.eval.stats import mcnemar_exact
from src.eval.tables import MAIN, build_tables, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/frozen/eval.jsonl"


def write_mock_runs(root: Path, questions: list[dict]) -> None:
    """Create paired runs with reversed row order, a warmup, and varying scores."""
    for bundle, round_no, names in [
        ("main", 1, MAIN),
        ("main", 2, MAIN),
        ("main", 3, MAIN),
        ("ablation", 1, (
            "m", "m_minus_router", "m_minus_judge", "m_minus_requery",
            "m_minus_cap", "m_minus_compress", "m_minus_loop_breaker",
        )),
    ]:
        round_name = f"r{round_no}"
        folder = root / bundle / round_name
        folder.mkdir(parents=True)
        (folder / "RUN_META.json").write_text(
            json.dumps({"max_vram_mib": 1000 + round_no, "max_vram_mib_by_pipeline": {"m": 800}}),
            encoding="utf-8",
        )
        for name in names:
            rows = []
            for question in reversed(questions) if name == "p1" else questions:
                warmup = question["qid"] == questions[0]["qid"]
                unanswerable = question["answer_kind"] == "unanswerable"
                rows.append({
                    **{key: question[key] for key in ("qid", "type", "answer_kind")},
                    "bundle": bundle,
                    "round": round_name,
                    "pipeline": name,
                    "is_warmup": warmup,
                    "timeout": False,
                    "correct": name == "m" and round_no != 3,
                    "f1": 0.5 if question["answer_kind"] == "text" else None,
                    "latency_s": 99 if warmup else (round_no if name == "m" else 2),
                    "max_input_tokens": 99 if warmup else 10,
                    "n_calls": 3 if warmup else 1,
                    "n_retries": int(name != "m"),
                    "calls": [
                        {"input_tokens": 99 if warmup else 10, "output_tokens": 99 if warmup else round_no,
                         "truncated": False, "parse_ok": None},
                    ] * ((3 if warmup else 1) + int(name != "m")),
                    "abstained": unanswerable or name == "p1",
                    "budget_exhausted": False,
                    "loop_breaks": 0,
                    "evidence_kept": False if question["type"] == "hotpotqa" else None,
                })
            (folder / f"{name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )


class TestTablesStats(unittest.TestCase):
    """Exercise the paired statistics and the report entry point."""

    def temp_path(self) -> Path:
        """Allocate a temporary directory and clean it up after the test."""
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def test_mcnemar_exact(self) -> None:
        """Check concordant, one-sided imbalance, and mixed discordant counts."""
        assert mcnemar_exact([(True, True)])["p_value"] == 1
        assert mcnemar_exact([(True, False)] * 6)["p_value"] == 0.03125
        assert mcnemar_exact([(True, False)] * 8 + [(False, True)])["p_value"] == 20 / 512
        assert mcnemar_exact([(False, True)] * 6)["p_value"] == 0.03125

    def test_full_tables(self) -> None:
        """Verify majority accuracy, paired samples, side effects, and same-bundle ratios."""
        tmp_path = self.temp_path()
        write_mock_runs(tmp_path, list(read_jsonl(EVAL)))
        tables = build_tables(tmp_path, EVAL)
        m = tables["main"][-1]
        assert m["accuracy"] == m["unanswerable_accuracy"] == 1
        assert m["text_f1"] == 0.5
        assert m["output_tokens_per_question"] == m["median_latency_s"] == 2
        assert m["mean_latency_s"] == 2
        assert m["budget_calls_per_question"] == m["total_calls_per_question"] == 1
        assert m["input_tokens_per_question"] == m["max_input_tokens"] == 10
        assert m["max_vram_mib"] == 800
        assert m["bundle_max_vram_mib"] == 1003
        router = next(row for row in tables["ablation"] if row["pipeline"] == "m_minus_router")
        assert router["time_ratio"] == router["total_calls_ratio"] == 2
        assert router["accuracy_delta"] == -1
        assert router["strategy"] == "②"
        assert [row["n"] for row in tables["mcnemar"]] == [120, 40, 40, 40]
        assert [row["m_only"] for row in tables["mcnemar"]] == [120, 40, 40, 40]
        p1 = tables["side_effects"][2]
        assert p1["over_abstention_rate"] == 1
        assert p1["n_answerable"] == 300
        assert p1["n_evidence_measured"] == 120
        assert p1["evidence_kept_rate"] == 0
        assert p1["parse_failure_count"] is None

    def test_warmup_only(self) -> None:
        """A single verified question still has a score, but no measured costs."""
        runs = self.temp_path()
        write_mock_runs(runs, list(read_jsonl(EVAL))[:1])
        tables = build_tables(runs, EVAL, verification=True)
        m = tables["main"][-1]
        assert m["n"] == m["accuracy"] == tables["mcnemar"][0]["m_only"] == 1
        assert m["text_f1"] == 0.5
        for key in ("total_calls_per_question", "median_latency_s", "max_input_tokens"):
            assert m[key] is None
        assert tables["ablation"][0]["time_ratio"] is None

    def test_incomplete_ablation(self) -> None:
        """A partially measured ablation bundle must not produce official tables."""
        runs = self.temp_path()
        write_mock_runs(runs, list(read_jsonl(EVAL)))
        (runs / "ablation/r1/m_minus_judge.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing core variants.*m_minus_judge"):
            build_tables(runs, EVAL)

    def test_qid_pairing(self) -> None:
        """Pair mixed outcomes by qid across differently ordered pipeline logs."""
        runs = self.temp_path()
        questions = list(read_jsonl(EVAL))[:10]
        write_mock_runs(runs, questions)
        for name, correct in (("m", questions[:7]), ("p1", questions[5:8])):
            correct_ids = {q["qid"] for q in correct}
            for path in (runs / "main").glob(f"r*/{name}.jsonl"):
                rows = list(read_jsonl(path))
                for row in rows:
                    row["correct"] = row["qid"] in correct_ids
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        overall, hotpot, squad, gsm = build_tables(runs, EVAL, verification=True)["mcnemar"]
        for row in (overall, hotpot):
            self.assertEqual(
                [row[key] for key in ("both_correct", "m_only", "p1_only", "both_wrong")],
                [2, 5, 1, 2],
            )
            self.assertEqual(row["p_value"], 14 / 64)
        for row in (squad, gsm):
            self.assertEqual(row["n"], 0)
            self.assertIsNone(row["p_value"])

    def test_verification_cli(self) -> None:
        """Run the public command on three mock questions and verify all artifacts."""
        tmp_path = self.temp_path()
        questions = list(read_jsonl(EVAL))
        runs = tmp_path / "runs"
        write_mock_runs(runs, questions[::40])
        output = tmp_path / "tables"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_tables.py"), "--runs-dir", str(runs),
             "--output-dir", str(output), "--verification"],
            cwd=tmp_path, check=True,
        )
        for name in ("main", "ablation", "side_effects", "mcnemar"):
            assert "검증 모드" in (output / f"{name}.md").read_text(encoding="utf-8")
            assert (output / f"{name}.csv").is_file()
        with (output / "mcnemar.csv").open(newline="", encoding="utf-8") as stream:
            assert [int(row["n"]) for row in csv.DictReader(stream)] == [3, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, "question mismatch"):
            build_tables(runs, EVAL)

        path = runs / "main/r1/p1.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(rows[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "question mismatch"):
            build_tables(runs, EVAL, verification=True)
