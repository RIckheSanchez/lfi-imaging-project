# Friday meeting questions (for supervisor)

Some items can be answered from files; the rest need confirmation in person.

## Data semantics / geometry

1. **Leaf third axis (~475 frames): time or depth?** DONE  
   Contact reply (via Haobo): **time series**.  
   Loader sets Leaf `scan_mode=time_series` and records this in notes.

2. **Physical pixel size for each dataset?** (µm/pixel or equivalent)  
   Chip / Leaf / Skin / Vegetables / PCB - same optical calibration?  
   Contact reply: original Skin/Vegetables samples are lost; use **pixel** units or measure the window in the lab.

3. **How is the Leaf ROI / reference region chosen?**  
   Any fixed coordinates or annotation files?

## Skin / acquisition

4. **Skin acquisition parameters:** drive current, modulation frequency, integration time, scan speed?  
   Local Skin sample has little of this. Lab notebook / RDM docs?  
   Contrast: 2026 Vegetables `frame-XX.hdf5` has `thz/info/` (environment, `laser_freq`, ~150 `mcp` registers) already loaded into `metadata['acquisition']`. Which registers belong in the thesis?

5. **Are both QCL Left/Right channels required?**  
   Skin sample scripts used `QCL0/Left`; PCB used `QCL1/Left`. Is Right required? Physical difference?

6. **Skin phase code → radians formula?**  
   Marked `skin_unconfirmed`. Local sample `IMAGE_PHASE` is often **uint16**; do not reuse the Vegetables int16 formula.

6b. **Is PCB / Megabugs phase the same as 2026 Vegetables?**  
   PCB is fixed to `pcb_unconfirmed` (never apply Vegetables by value range alone).

## RDM data

7. **How many Vegetables frames are enough?**  
   Locally ran frames 20-28. Haobo reports ~136 files in one scan. Full set or a representative contiguous range?  
   Loader supports `load_volume()` with numeric frame order and `experiment_details.txt` → `depth_mm`. Confirm field names (Start position / Step size) or send a sample file.

8. **Any Skin acquisition notes on RDM?**  
   Contact reply: **no more Skin data for public use**; own scans need Karl + possible ethics.

## Findings from real runs

9. **Frames 21→28 amplitude max rises (2689→4053) - normal?**  
   Warm-up, sample change, or settings drift?

10. **Is local `frame-3.hdf5` PCB or Vegetables?**  
    `scan_identifier = CORN06052026` looks like corn (2026-05-06). Which RDM folder?

11. **What processing produced Leaf `L_P_U_M`?**  
    `leaf1.mat` has `L_A`/`L_P` (int16, 1200×400×475) and `L_P_U`/`L_P_U_M` (float64, 1200×400×91).  
    `L_P_U` re-wraps to match `wrap(L_P×π/4096−π)` pixel-wise (independent confirmation of Leaf formula).  
    `L_P_U_M` looks spatially detrended vs `L_P_U`, but not a simple mean/frame-0 subtract. Why only 91 frames? Use raw `L_P` or `L_P_U_M` in the thesis?

11b. **Are Chip / Marble z-stack phases in degrees?** (priority)  
    `borad1.hdf5` and Marble files: float64 in [-180, 179.956], step 360/8192 - looks like degrees.  
    Loader previously treated Chip as radians (wrong); now `zstack_unconfirmed`. Candidate: `phase_rad = value · π/180`.

13. **Which RDM folder is `frame-1.hdf5` from?**  
    Frames layout; empty `scan_identifier`. Identity must come from the folder.

## SUS

12. **Does running SUS with 5-8 people need ethics approval?**  
    Materials in `sus/`. Coordinate with Person 4 (He Zhenyu) per thesis appendix.
