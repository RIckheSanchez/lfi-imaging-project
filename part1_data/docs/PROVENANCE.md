# Provenance of loose local files (confirm yourself)

The loader picks phase formulas by **folder identity**. Files whose path lacks `Vegetables` / `PCB` / `Skin` / ... may be mis-tagged by heuristics.

| Local file | Size | Structure summary | Current heuristic | Confirm RDM folder |
|---|---|---|---|---|
| `~/Downloads/frame-3.hdf5` | ~320 MB | frames; `scan_identifier=CORN06052026` | PCB | 2026_PCB? Vegetables (corn)? |
| `~/Downloads/frame-3 (1).hdf5` | **byte-identical** to frame-3 | same | same | same (duplicate download) |
| `~/Downloads/frame-1.hdf5` | ~345 MB | frames/1; empty scan_id; QCL1 L/R | PCB (filename heuristic) | 2026_PCB / megabugs / other? |
| `~/Downloads/borad1.hdf5` | ~188 MB | z-stack 70 slices | Chip (`borad` heuristic / explicit) | Chip |
| `maeble10%_*.hdf5` | ~2.5 MB | z-stack 1 slice | under `data/Marble/` → Marble | confirm from Marble |

**Your action:** find these on RDM, fill the last column, tell the teammate. Do not convert phase for frame-1 / frame-3 until confirmed.
