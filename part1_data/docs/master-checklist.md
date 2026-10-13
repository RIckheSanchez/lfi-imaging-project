# REIT4841 master checklist (19 items)

Sources: Thesis Plan (Aug) + Haobo feedback + Karl/contact replies + 2026-10 runs.  
**Do not repeat disproven Aug Plan wording in Report/Poster** (see bottom).

## Tonight - local tooling (1-8)

| # | Item | Acceptance | Status |
|---|---|---|---|
| 1 | Leaf `scan_mode=time_series`, Karl note in notes | metadata has time_series; notes mention Karl/time series | done |
| 2 | RDM dataset inventory (incl. Sound) | see `RDM_DATASETS.md` | done (fill RDM counts yourself) |
| 3 | Sound `.mat` load | `MAT_LAYOUTS["Sound"]` + `sound_unconfirmed` | skeleton; need real Sound file |
| 4 | Metadata sidecar + field table + JSON/HDF5 attrs | `write_metadata_sidecar` / `write_metadata_hdf5_attrs`; `METADATA_FIELDS.md` | done |
| 5 | pytest + invalid inputs | `tests/` | done |
| 6 | benchmark: Leaf / float32 / batch / memory | `benchmark.py` | done |
| 7 | frame-1 / frame-3(1) provenance | `PROVENANCE.md` | waiting on your RDM folder IDs |
| 8 | README + requirements.txt | includes "how to add a format" | done |

## Tonight - you (9-10)

| # | Item | Acceptance |
|---|---|---|
| 9 | Send zip to Haobo; team git + feature-branch PR for his review | PR URL + review |
| 10 | Confirm poster deadline | calendar / this checklist |

## Tomorrow (11-15)

| # | Item | Notes |
|---|---|---|
| 11 | Old vs new workflow evaluation | Plan 6.4; higher priority than SUS |
| 12 | Confirm Haobo actually calls the interface | script name + function |
| 13 | SUS (coordinate with He Zhenyu / Person 4) | appendix assigns Person 4 |
| 14 | Poster + demo script | after assets ready |
| 15 | Code walkthrough | for demo |

## Thu/Fri (16)

| # | Item |
|---|---|
| 16 | Karl: Chip/Marble units, Skin formula, PCB formula; keep email trail |

## Also (17-18)

| # | Item |
|---|---|
| 17 | AI use statement: local tooling/AI assistant wrote code / helped analysis; do not reuse Aug "all my own code" |
| 18 | Update Report PDF TOC fields (remove template placeholders) |

## Do not repeat in submitted materials

1. "All six datasets load as (70, 1400, 400)" - shapes differ by dataset.  
2. "Leaf was already tested at Plan time" - first Leaf `.mat` load was **2026-10-11**.  
3. "Person 2 interface in active use" - Haobo still had a temp loader on 10/5; only claim after he wires this API.  
4. Full CSV/TXT loader - Plan 4.1 deprioritized; primary path is HDF5 (+ Leaf/Sound `.mat`).
