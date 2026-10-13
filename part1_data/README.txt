REIT4841 / LFI data loader - team share pack v3.1
=================================================

Start with INTEGRATION.md: how the downstream pipeline should call the API,
data/metadata contracts, and phase rules.

Please do after unpacking
-------------------------
1. Without downloading data you can still run (Marble samples ~2.5 MB each are included):
       python3 test_output_interface.py
       python3 output_interface.py data/Marble --out exported_check
2. From RDM, put Vegetables frames under data/2026_Vegetables/
   **and copy experiment_details.txt into the same folder**;
   put Leaf (leaf1.mat) under data/Leaf/. Then:
       python3 test_loader.py
       # volume example:
       python3 -c "from output_interface import load_volume, get_phase_radians; \
v=load_volume('data/2026_Vegetables'); print(v.amplitude.shape, v.z_indices[:5], (v.metadata.get('depth_mm') or [None])[:5])"
3. In your pipeline, use only output_interface (load_volume / iter_scans / get_phase_radians).
   Reply with: which script, which function, any errors or missing fields.
4. results_real/ and exported/ are my verification outputs, not raw data.

v3.1 (Haobo feedback)
---------------------
- [BUG] Frame files sorted by numeric frame id; lexical 100,110,...,21 would corrupt depth.
- New load_volume(data_dir): stack Vegetables frame-XX.hdf5 into one volume.
- Reads experiment_details.txt start/step → metadata['depth_mm'].
- include_waveform=True loads RX_LINE_OF_BEST_FIT (default off).
- Leaf supports frame_ids=[...] subset (v5 still peaks high; v7.3 can stream).

v3 (real leaf1.mat + Marble)
----------------------------
- Important fix: Chip/Marble z-stack phase looks like degrees [-180,180), step 360/8192,
  not radians. Chip was wrongly treated as radians before. Now zstack_unconfirmed;
  get_phase_radians() rejects until units are confirmed (MEETING_QUESTIONS 11b).
- Leaf leaf1.mat verified: L_A/L_P int16 (1200,400,475), ~15 s load.
  L_P_U (first 91 unwrapped frames) matches wrap(L_P·π/4096−π) after re-wrap.
  Leaf also enforces int16 + [-8192, 8191].
- Leaf conversion is in-place to avoid OOM on 8 GB machines.
- results_real/ and exported/ include Leaf and Marble.

v2 highlights
-------------
- 2026 Vegetables format (data/IMAGE_*/frame_<id>_qcl_<n>_<side>), L/R channels.
  Real frames 20-28 ran (see results_real/).
- Vegetables frame 20 anomalous: excluded by default; loadable as raw + flagged.
- Reads scan_identifier and thz/info acquisition metadata.
- MATLAB .mat (v5/v7/v7.3) for Leaf; names in loader.MAT_LAYOUTS.
- output_interface.py: iter_scans / get_phase_radians / export_scan /
  load_exported / batch_export + CLI metadata export.
- SUS kit under sus/.

How to run
----------
Deps: numpy, h5py, scipy (.mat), matplotlib (previews only).

    python3 test_loader.py
    python3 test_output_interface.py
    python3 output_interface.py data/2026_Vegetables --out exported

Tests use local sample paths (see tops of test files); missing files are skipped.
This pack omits large raw .hdf5/.mat; download from RDM into data/<DatasetFolder>/.
Do not rename dataset folders (identity comes from folder names).
