# Real-data loader run results

Data directory: `/Users/wangzeyu/Desktop/REIT4841/data`

Total 12 files: ok 12, failed 0, skip 0

| file | target_type | status | shape | frame_ids | anomalous_frame_ids | original_dtype | phase_encoding_used | phase_converted | error |
|---|---|---|---|---|---|---|---|---|---|
| 2026_Vegetables/frame-20.hdf5 | Vegetables | OK | 1x400x1300 | [20] | [20] | int16 | vegetables_int16 | False |  |
| 2026_Vegetables/frame-21.hdf5 | Vegetables | OK | 1x400x1300 | [21] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-22.hdf5 | Vegetables | OK | 1x400x1300 | [22] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-23.hdf5 | Vegetables | OK | 1x400x1300 | [23] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-24.hdf5 | Vegetables | OK | 1x400x1300 | [24] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-25.hdf5 | Vegetables | OK | 1x400x1300 | [25] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-26.hdf5 | Vegetables | OK | 1x400x1300 | [26] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-27.hdf5 | Vegetables | OK | 1x400x1300 | [27] | [] | int16 | vegetables_int16 | True |  |
| 2026_Vegetables/frame-28.hdf5 | Vegetables | OK | 1x400x1300 | [28] | [] | int16 | vegetables_int16 | True |  |
| Leaf/leaf1.mat | Leaf | OK | 475x1200x400 | [0, 1, ..., 474] (475) | [] | int16 | leaf_wrap_4096 | True |  |
| Marble/maeble10%_0.hdf5 | Marble | OK | 1x1400x400 | [1] | [] | float64 | zstack_unconfirmed | False |  |
| Marble/maeble10%_1.hdf5 | Marble | OK | 1x1400x400 | [1] | [] | float64 | zstack_unconfirmed | False |  |

Notes:
- Phase formula from dataset identity (folder name) lookup; do not infer from value ranges.
- `phase_converted=False` means phase kept as raw codes: formula unconfirmed or file has anomalous frames.
- Non-empty `anomalous_frame_ids` (e.g. Vegetables 20) were loaded but flagged; must not be treated as normal data.
- HDF5 structure dumps for failed files: `structure_*.txt`.
