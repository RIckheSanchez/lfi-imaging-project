# Data readiness and next experiment, 11 October 2026

## Observed shared-drive inventory

The `02 Data Sets` directory visibly contains 11 folders: `2026 Semester 2`,
`2026_megabugs`, `2026_PCB`, `Beam Profile`, `Chip`, `Leaf`, `Marble`,
`Mushroom`, `Sound`, `Spectrum`, `Tomography`. There is no Skin-labelled folder
at this level. The user reports the team cannot locate skin data. This is not
an exhaustive content search and does not prove that no skin data exist anywhere.
Skin validation remains unavailable; other samples are not labelled as skin.

`Tomography` contains `corn_110826`, `corn_240926`, `phantom_180826`,
`corn2.hdf5`, `IC.hdf5`, `load_data_example.py`, `processing.py`, `S1.mat`,
and `S1b.mp4`. These are promising sources for non-skin validation, subject to
schema, physical coordinate and sample-identity checks.

## New evidence and conflicts

The downloaded `load_data_example.py` reads amplitude frame `0055` but phase
frame `0022`. These are different keys and must not be paired in validation.
The example establishes a second schema: `/amplitude/FRAME` and `/phase/FRAME`.

The downloaded `processing.py` converts phase from degrees to radians in some
loading paths. This is evidence of an older file convention, not proof that the
new Vegetables int16 data use degrees. Its `quick_load_h5` also assigns a rescaled
amplitude array to the phase variable. Do not use this helper as ground truth.
The files were read as text only, not executed or incorporated into the project.
The processing script is marked proprietary and is not copied into tracked code.

The note in `Tomography/phantom_180826/experiment_details.txt` says:

- "scan of corn for Jari's students"
- start z 17.17 mm; stop z 23.69 mm; focused position 28.17 mm
- increments 0.1 mm

The folder name and sample description conflict; the listed focus lies outside
the listed scan range. This needs interpretation by the acquisition owner, not
an automatic correction. A directory called phantom does not by itself supply
known geometric ground truth. The 151.9 MB `phantom.hdf5` download did not finish
within the browser tool timeout; its payload has not been inspected in this run.

## Implemented continuation

`python -m part3_reconstruction audit data/vegetables --output outputs/vegetables_audit.json`

The audit recursively checks local HDF5 files, matches channel keys, excludes BMP
derivatives and waveform payloads, records shapes/dtypes/ranges/missing pixels,
and flags constant images or values at storage maximum. Flags are observations,
not diagnoses of saturation or sample validity. Corrupt files are reported while
the rest of the directory continues. Exit code 2 indicates at least one error.
Memory use is per 2D image rather than the entire scan series.

`DataLoader.load_grouped_frame(path, frame_id, axis_order=..., phase_unit=...)`
loads the same frame key from both groups, rejects unpaired channel collections,
and requires an explicit axis order and phase unit (`rad`, `degrees`, or
`raw_counts`). Degrees are converted only when explicitly declared. No depth
coordinates are invented from frame keys. Numeric frame sorting and ID gaps are
checked by the audit; gaps are not proof of missing acquisitions without a manifest.

## Next evidence needed

Validation of this continuation: 34 tests passed. The batch audit of the two
downloaded Vegetables files found four matched channel pairs and no structural
errors; frame 20 was flagged for review because amplitude values reach the uint16
storage maximum. Its left amplitude image also contains about 64.9% zeros.
Neither observation alone identifies the acquisition cause.

1. Confirm which tomography file has a trusted sample identity and per-frame
   positions; obtain its units and orientation. Inspect a complete paired sequence.
2. Obtain a known-size phantom description or a reference processing output if
   quantitative physical accuracy is a thesis claim. A rendering alone is not truth.
3. Agree Person 2's dual-channel contract and connect Person 4's GUI. Test the
   end-to-end pipeline using a named non-skin case while skin access is unresolved.
4. Obtain skin data through the team/supervisor, or agree a thesis-scope change.
   Keep biomedical motivation separate from validation performed on other samples.

No shared-drive content, thesis claims, or remote GitHub files were modified.
