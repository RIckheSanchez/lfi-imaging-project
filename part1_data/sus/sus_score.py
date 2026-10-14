"""
sus_score.py
Compute SUS scores and summarise task completion

Usage:
    python3 sus_score.py responses.csv
    python3 sus_score.py responses.csv --tasks task_log.csv --out sus_report.md

responses.csv columns: participant_id, q1..q10 (each 1-5)
task_log.csv columns: participant_id, task_id, success (yes/partial/no), time_s, help_requests, notes

scoring(Brooke 1996):
    odd items: score-1; even items: 5-score; sum * 2.5 → 0-100
    68 is the common industry average benchmark (Sauro 2011)
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SUS_BENCHMARK = 68.0
QUESTIONS = [f"q{i}" for i in range(1, 11)]


def score_one(answers: list[int]) -> float:
    if len(answers) != 10:
        raise ValueError(f"need 10 answers, got {len(answers)} ")
    for i, a in enumerate(answers, 1):
        if a not in (1, 2, 3, 4, 5):
            raise ValueError(f"q{i} answer must be int 1-5, got {a!r}")
    total = sum((a - 1) if i % 2 == 1 else (5 - a) for i, a in enumerate(answers, 1))
    return total * 2.5


def read_responses(path: str) -> dict[str, float]:
    scores = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [q for q in ["participant_id", *QUESTIONS] if q not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        for row in reader:
            pid = row["participant_id"].strip()
            if not pid:
                continue
            try:
                answers = [int(row[q]) for q in QUESTIONS]
            except ValueError as e:
                raise ValueError(f"participant {pid} answers are not integers: {e}") from e
            if pid in scores:
                raise ValueError(f"participant {pid} appears more than once")
            scores[pid] = score_one(answers)
    if not scores:
        raise ValueError(f"{path} contains no responses")
    return scores


def _t_critical_95(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:
        return 1.96


def summarise(scores: dict[str, float]) -> dict:
    values = list(scores.values())
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = _t_critical_95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "ci95": (mean - half, mean + half),
        "above_benchmark": mean > SUS_BENCHMARK,
    }


def read_tasks(path: str) -> dict:
    per_task = defaultdict(lambda: {"n": 0, "yes": 0, "partial": 0, "no": 0,
                                    "times": [], "help": 0})
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("participant_id", "").strip():
                continue
            t = per_task[row["task_id"].strip()]
            t["n"] += 1
            result = row["success"].strip().lower()
            if result not in ("yes", "partial", "no"):
                raise ValueError(f"success must be yes/partial/no, got {result!r}")
            t[result] += 1
            if row.get("time_s", "").strip():
                t["times"].append(float(row["time_s"]))
            if row.get("help_requests", "").strip():
                t["help"] += int(row["help_requests"])
    return dict(sorted(per_task.items()))


def render_report(scores: dict, summary: dict, tasks: dict | None) -> str:
    lo, hi = summary["ci95"]
    lines = [
        "# SUS evaluation results",
        "",
        f"n participants: {summary['n']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mean score | {summary['mean']:.1f} |",
        f"| std | {summary['sd']:.1f} |",
        f"| Median | {summary['median']:.1f} |",
        f"| Range | {summary['min']:.1f} - {summary['max']:.1f} |",
        f"| 95% confidence interval | {lo:.1f} - {hi:.1f} |",
        f"| vs benchmark {SUS_BENCHMARK:.0f}  | {'above' if summary['above_benchmark'] else 'not above'}benchmark |",
        "",
        "## eachparticipant",
        "",
        "| participant_id | SUS |",
        "|---|---|",
        *[f"| {pid} | {s:.1f} |" for pid, s in scores.items()],
    ]
    if summary["n"] < 5:
        lines += ["", f"Note: only {summary['n']} participants; CI is wide - interpret cautiously."]

    if tasks:
        lines += [
            "",
            "## taskdonestatus",
            "",
            "| task | n | ok | partial | failed | completion | mean duration (s) | help requests |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for tid, t in tasks.items():
            rate = t["yes"] / t["n"] * 100 if t["n"] else 0
            avg = f"{statistics.mean(t['times']):.0f}" if t["times"] else "-"
            lines.append(
                f"| {tid} | {t['n']} | {t['yes']} | {t['partial']} | {t['no']} | "
                f"{rate:.0f}% | {avg} | {t['help']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute SUS scores")
    parser.add_argument("responses", help="questionnaire responses CSV")
    parser.add_argument("--tasks", help="task log CSV (optional)")
    parser.add_argument("--out", help="write the report to this Markdown file")
    args = parser.parse_args(argv)

    scores = read_responses(args.responses)
    report = render_report(scores, summarise(scores), read_tasks(args.tasks) if args.tasks else None)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
