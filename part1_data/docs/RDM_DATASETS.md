# RDM dataset inventory (use real numbers in the thesis report)

Root: `02 Data Sets/` (UQ RDM).  
**Fill the "RDM count" column after you log in.** "Locally verified" means actually loaded on this machine.

| Dataset | RDM path (approx.) | Main format | Typical shape (local) | Phase encoding | RDM count | Locally verified | Notes |
|---|---|---|---|---|---|---|---|
| Chip | `.../Chip/` | HDF5 z-stack | `(70, 1400, 400)` | `zstack_unconfirmed` (likely degrees) | _fill_ | yes (borad1) | units TBD with Karl |
| Marble | `.../Marble/` | HDF5 z-stack | `(1, 1400, 400)` | same | _fill_ | yes (maeble10%_*) | filename typo "maeble" |
| Leaf | `.../Leaf/` | MATLAB `.mat` | `(475, 1200, 400)` | `leaf_wrap_4096` | _fill_ | yes (leaf1.mat, 2026-10-11) | Karl: 3rd axis = **time series** |
| Sound | `.../Sound/` | MATLAB `.mat` (Plan) | _TBD_ | `sound_unconfirmed` | _fill_ | **no** | Thesis Plan commitment; confirm vars |
| Skin | `.../Skin/` | HDF5 frames | `(1, 400, 1300)` | `skin_unconfirmed` | _fill_ | 1 local sample | contact: no more for public use |
| Vegetables | `.../2026 Semester 2/2026_Vegetables/` | HDF5 v2026 per-frame files | `(1, 400, 1300)` / volume `(n,400,1300)` | `vegetables_int16` | ~136 (Haobo) | frames 20-28 | use `load_volume`; need `experiment_details.txt` |
| PCB | `.../2026_PCB/` | HDF5 frames / ? | _per file_ | `pcb_unconfirmed` | _fill_ | local frame-3 identity doubtful | CORN06052026 |
| Megabugs | `.../2026_megabugs/` | ? | _TBD_ | mapped as PCB | _fill_ | no | alias → PCB |
| Mushroom / Tomography / Beam Profile / Spectrum | respective folders | mostly HDF5; Spectrum once seen 0 KB | _TBD_ | no table → `unknown_unconfirmed` | _fill_ | no | CSV/TXT: Plan deprioritized |

## Shape note (do not write "everything is 70×1400×400")

- Chip z-stack: `(70, 1400, 400)`  
- Marble: single-slice z-stack `(1, 1400, 400)`  
- Skin / Vegetables single frame: `(1, 400, 1300)`  
- Leaf: `(475, 1200, 400)` time series  

## Sound

1. Download one `.mat` from RDM `Sound/`.  
2. `python3 -c "import scipy.io; print(scipy.io.whosmat('Sound/....mat'))"`  
3. Write names into `loader.MAT_LAYOUTS["Sound"]`, then confirm phase formula before leaving `sound_unconfirmed`.
