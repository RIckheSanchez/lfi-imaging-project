"""
test_sus_score.py
Usage: python3 test_sus_score.py
"""

import tempfile
from pathlib import Path

from sus_score import read_responses, read_tasks, render_report, score_one, summarise


def test_known_scores():
    assert score_one([3] * 10) == 50.0
    assert score_one([5, 1] * 5) == 100.0
    assert score_one([1, 5] * 5) == 0.0
    # odd items 4, even items 2 → contribution 3 each → 75
    assert score_one([4, 2] * 5) == 75.0


def test_invalid_answers_rejected():
    for bad in ([3] * 9, [0] + [3] * 9, [6] + [3] * 9):
        try:
            score_one(bad)
            raise AssertionError(f"should reject {bad}")
        except ValueError:
            pass


def test_csv_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        resp = Path(tmp) / "r.csv"
        resp.write_text(
            "participant_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10\n"
            "P1,5,1,5,1,5,1,5,1,5,1\n"
            "P2,3,3,3,3,3,3,3,3,3,3\n",
            encoding="utf-8",
        )
        tasks = Path(tmp) / "t.csv"
        tasks.write_text(
            "participant_id,task_id,success,time_s,help_requests,notes\n"
            "P1,T1,yes,60,0,\n"
            "P2,T1,partial,120,1,\n",
            encoding="utf-8",
        )
        scores = read_responses(str(resp))
        assert scores == {"P1": 100.0, "P2": 50.0}
        s = summarise(scores)
        assert s["mean"] == 75.0 and s["above_benchmark"]
        t = read_tasks(str(tasks))
        assert t["T1"]["yes"] == 1 and t["T1"]["partial"] == 1 and t["T1"]["help"] == 1
        report = render_report(scores, s, t)
        assert "75.0" in report and "| T1 | 2 | 1 | 1 | 0 | 50% | 90 | 1 |" in report


if __name__ == "__main__":
    test_known_scores()
    test_invalid_answers_rejected()
    test_csv_pipeline()
    print("✅ SUS scoring tests all passed")
