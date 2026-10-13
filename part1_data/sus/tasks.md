# Participant task sheet

You will complete 5 short tasks with the LFI data loader. We are evaluating the tool, not you.
Failing a task is useful information. You may skip after ~5 minutes stuck.
Think aloud. You may read `INTEGRATION.md` and docstrings.

Prep: open Python in the project root (`python3`). Data is under `data/`.

---

**T1 - Load a z-stack**  
Load Chip file `borad1.hdf5`. How many z slices? Image size per slice (H × W)?

**T2 - Get phase in radians**  
Load `data/2026_Vegetables/frame-21.hdf5`, get phase in radians, report min and max.

**T3 - Understand rejection**  
Do the same for the Skin sample `2025-11-28_10-14-26.594507929.hdf5`. The tool should refuse.
In your own words: why? Which phase encoding?

**T4 - Choose a channel**  
Load frame 21 **Right** channel; confirm the laser channel in metadata changed.

**T5 - Batch-export metadata**  
Export metadata for all of `data/2026_Vegetables` to a new folder.
In the index, which frame is marked anomalous?

---

## Facilitator answer key (do not show participants)

| Task | Expected approach | Success |
|---|---|---|
| T1 | `load_single_scan(...)` then `n_slices` / `height` / `width` or `summary()` | 70 slices, 1400 × 400 |
| T2 | `get_phase_radians(load_single_scan(...))` | about −3.142 / 3.141 |
| T3 | `get_phase_radians` raises `PhaseNotConvertedError` | mention `skin_unconfirmed` / formula unconfirmed |
| T4 | `load_single_scan(..., side='Right')` | `metadata['laser_channel'] == 'QCL1/Right'` |
| T5 | `batch_export(dir, out)` or CLI `python3 output_interface.py dir --out out` | frame 20 in `metadata_index.csv` |

Log in `task_log_template.csv`: `yes` = independent; `partial` = needed a hint or incomplete; `no` = abandoned/timeout.
Each question to you counts as one `help_requests`.
