"""Aggregate scored JSONL runs into evaluation tables."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from statistics import mean, median
from typing import Any

from .stats import mcnemar_exact

TYPES = ("hotpotqa", "squad2", "gsm8k")
MAIN = ("p0_raw", "p0", "p1", "m")
CORE_ABLATIONS = {
    "m_minus_router": "②",
    "m_minus_requery": "③",
    "m_minus_judge": "③",
    "m_minus_compress": "③",
    "m_minus_cap": "①",
    "m_minus_loop_breaker": "①",
}
Row = dict[str, Any]
Run = dict[str, Row]


def read_jsonl(path: Path) -> Iterator[Row]:
    """Yield JSON records, with their source location in decoding errors."""
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            yield row


def _load_run(path: Path, questions: Run) -> Run:
    """Index all scored rows and verify question identity and run metadata."""
    run = {}
    for row in read_jsonl(path):
        qid = row["qid"]
        if qid in run:
            raise ValueError(f"{path}: duplicate question {qid}")
        question = questions.get(qid)
        if question is None or any(
            row[field] != question[field] for field in ("type", "answer_kind")
        ):
            raise ValueError(f"{path}: unexpected question or metadata for {qid}")
        expected = (path.parent.parent.name, path.parent.name, path.stem)
        if (row["bundle"], row["round"], row["pipeline"]) != expected:
            raise ValueError(f"{path}: wrong bundle, round or pipeline for {qid}")
        if type(row["correct"]) is not bool or (row["timeout"] and row["correct"]):
            raise ValueError(f"{path}: invalid correctness for {qid}")
        run[qid] = row
    if not run:
        raise ValueError(f"{path}: no scored questions")
    return run


def _mean(values: Iterable[float]) -> float | None:
    """Average observed values or return None when there are none."""
    observed = list(values)
    return mean(observed) if observed else None


def _ratio(value: float | None, baseline: float | None) -> float | None:
    """Compute the ratio when the baseline is nonzero."""
    return value / baseline if value is not None and baseline else None


def _scores(runs: list[Run]) -> dict[str, bool]:
    """Return one correctness value per question by majority vote."""
    return {
        qid: sum(run[qid]["correct"] for run in runs) > len(runs) // 2
        for qid in runs[0]
    }


def _summarize(runs: list[Run], questions: Run) -> Row:
    """Aggregate accuracy and costs for one pipeline across its rounds."""
    scores = _scores(runs)
    rows = [row for run in runs for row in run.values()]
    cost_rows = [row for row in rows if not row["is_warmup"]]
    times = [row["latency_s"] for row in cost_rows]
    return {
        "n": len(scores),
        "accuracy": mean(scores.values()),
        **{
            kind: _mean(ok for qid, ok in scores.items() if questions[qid]["type"] == kind)
            for kind in TYPES
        },
        "text_f1": _mean(
            row["f1"] for row in rows if row["answer_kind"] == "text" and row["f1"] is not None
        ),
        "unanswerable_accuracy": _mean(
            ok for qid, ok in scores.items() if questions[qid]["answer_kind"] == "unanswerable"
        ),
        "budget_calls_per_question": _mean(row["n_calls"] for row in cost_rows),
        "total_calls_per_question": _mean(row["n_calls"] + row["n_retries"] for row in cost_rows),
        "input_tokens_per_question": _mean(
            sum(call["input_tokens"] for call in row["calls"]) for row in cost_rows
        ),
        "output_tokens_per_question": _mean(
            sum(call["output_tokens"] for call in row["calls"]) for row in cost_rows
        ),
        "mean_latency_s": _mean(times),
        "median_latency_s": median(times) if times else None,
        "max_input_tokens": max((row["max_input_tokens"] for row in cost_rows), default=None),
    }


def _side_effects(runs: list[Run]) -> Row:
    """Aggregate intervention counts and applicable side-effect rates."""
    rows = [row for run in runs for row in run.values()]
    calls = [call for row in rows for call in row["calls"]]
    parsed = [call["parse_ok"] for call in calls if call["parse_ok"] is not None]
    answerable = [row for row in rows if row["answer_kind"] != "unanswerable"]
    evidence = [
        row["evidence_kept"] for row in rows
        if row["type"] == "hotpotqa" and row.get("evidence_kept") is not None
    ]
    return {
        "n_executions": len(rows),
        "n_answerable": len(answerable),
        "over_abstention_rate": _mean(row["abstained"] for row in answerable),
        "budget_exhausted_count": sum(row["budget_exhausted"] for row in rows),
        "truncated_count": sum(call["truncated"] for call in calls),
        "n_parse_checked": len(parsed),
        "parse_failure_count": parsed.count(False) if parsed else None,
        "loop_break_count": sum(row["loop_breaks"] for row in rows),
        "n_evidence_measured": len(evidence),
        "evidence_kept_rate": _mean(evidence),
    }


def _vram(rounds: list[Path], pipeline: str) -> Row:
    """Return pipeline and round peak VRAM where all rounds recorded them."""
    metadata = []
    for folder in rounds:
        path = folder / "RUN_META.json"
        metadata.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else {})
    pipeline_values = [(meta.get("max_vram_mib_by_pipeline") or {}).get(pipeline) for meta in metadata]
    round_values = [meta.get("max_vram_mib") for meta in metadata]
    return {
        "max_vram_mib": max(pipeline_values) if None not in pipeline_values else None,
        "bundle_max_vram_mib": max(round_values) if None not in round_values else None,
    }


def build_tables(runs_dir: Path, eval_set: Path, *, verification: bool = False) -> dict[str, list[Row]]:
    """Read main and ablation runs and return four tables keyed by output stem."""
    questions = {row["qid"]: row for row in read_jsonl(eval_set)}
    if len(questions) != 120 or any(
        sum(row["type"] == kind for row in questions.values()) != 40 for kind in TYPES
    ):
        raise ValueError(f"{eval_set}: expected 120 questions, 40 per type")

    main_rounds = [runs_dir / "main" / f"r{index}" for index in range(1, 4)]
    ablation_round = runs_dir / "ablation" / "r1"
    variants = sorted(path.stem for path in ablation_round.glob("m_minus_*.jsonl"))
    if not variants:
        raise ValueError(f"{ablation_round}: expected at least one ablation variant")
    missing_variants = CORE_ABLATIONS.keys() - variants
    if missing_variants and not verification:
        raise ValueError(f"{ablation_round}: missing core variants: {sorted(missing_variants)}")
    main = {
        name: [_load_run(folder / f"{name}.jsonl", questions) for folder in main_rounds]
        for name in MAIN
    }
    ablation = {
        name: [_load_run(ablation_round / f"{name}.jsonl", questions)]
        for name in ("m", *variants)
    }
    expected = main["m"][0].keys() if verification else questions.keys()
    for bundle, pipelines in (("main", main), ("ablation", ablation)):
        for name, runs in pipelines.items():
            for index, run in enumerate(runs, 1):
                if run.keys() != expected:
                    missing = sorted(expected - run.keys())
                    extra = sorted(run.keys() - expected)
                    raise ValueError(
                        f"{bundle}/r{index}/{name}: question mismatch "
                        f"(missing={missing}, extra={extra})"
                    )

    main_rows = [
        {"pipeline": name, **_summarize(runs, questions), **_vram(main_rounds, name)}
        for name, runs in main.items()
    ]
    ablation_rows = [
        {
            "pipeline": name,
            "strategy": CORE_ABLATIONS.get(name),
            **_summarize(runs, questions),
            **_vram([ablation_round], name),
        }
        for name, runs in ablation.items()
    ]
    baseline = ablation_rows[0]
    for row in ablation_rows:
        row["accuracy_delta"] = row["accuracy"] - baseline["accuracy"]
        row["total_calls_ratio"] = _ratio(
            row["total_calls_per_question"], baseline["total_calls_per_question"]
        )
        row["time_ratio"] = _ratio(row["mean_latency_s"], baseline["mean_latency_s"])

    m_scores, p1_scores = _scores(main["m"]), _scores(main["p1"])
    return {
        "main": main_rows,
        "ablation": ablation_rows,
        "side_effects": [
            {"bundle": bundle, "pipeline": name, **_side_effects(runs)}
            for bundle, pipelines in (("main", main), ("ablation", ablation))
            for name, runs in pipelines.items()
        ],
        "mcnemar": [
            {
                "type": kind,
                **mcnemar_exact(
                    (m_scores[qid], p1_scores[qid])
                    for qid in m_scores if kind == "overall" or questions[qid]["type"] == kind
                ),
            }
            for kind in ("overall", *TYPES)
        ],
    }
