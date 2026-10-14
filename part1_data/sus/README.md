# SUS usability evaluation - run plan

Everything needed is in this folder; you still need participants.

| File | Purpose |
|---|---|
| `tasks.md` | Participant tasks (answer key at the end is for the facilitator only) |
| `questionnaire.md` | SUS 10-item questionnaire + 3 open questions |
| `responses_template.csv` | Enter questionnaire scores |
| `task_log_template.csv` | Enter per-task completion and timing |
| `sus_score.py` | Compute SUS scores and write a report |

## Before you start

- **Ethics:** Running a questionnaire with people may need UQ human research ethics (often low-risk). Ask Karl whether this project is already covered or course-assessment exempt. **Do not collect formal data until confirmed.**
- **Participants:** 5-8 people who know a little Python but did not help write this tool (teammates, classmates, lab members). Use P1, P2, ... - no real names.
- **Setup:** Same prepared machine for everyone, deps installed, `data/` containing Chip `borad1.hdf5`, a Skin sample, and `2026_Vegetables/frame-20.hdf5` + `frame-21.hdf5`.

## Per participant (~30 minutes)

1. State purpose: evaluate the tool, not the person; they may withdraw anytime. (2 min)
2. Give the task section of `tasks.md` and `INTEGRATION.md`; run T1→T5. Facilitator times and logs help counts; no unsolicited hints. (20 min)
3. Fill `questionnaire.md` immediately. (5 min)
4. Ask open questions; record verbatim answers. (3 min)

## Scoring

Copy the CSV templates, enter data, then:

```bash
cd sus
python3 sus_score.py responses.csv --tasks task_log.csv --out sus_report.md
```

The report includes mean, std, 95% CI, comparison to the 68 benchmark, and per-task completion rate / mean duration.

Coordinate with **Person 4 (He Zhenyu)** per the thesis appendix - do not run a separate SUS track in isolation.
