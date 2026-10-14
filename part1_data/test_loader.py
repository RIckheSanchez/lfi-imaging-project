"""
test_loader.py
Quick tests for loader (z-stack + frames + phase config)

Usage:
    python test_loader.py
"""

import os
import warnings
from pathlib import Path

from loader import (
    AnomalousOnlyError,
    load_single_scan,
    load_volume,
    get_z_slices,
    _infer_target_type,
    sort_data_paths,
    parse_experiment_details,
    depth_mm_for_frames,
)
from phase_encoding import get_phase_encoding, convert_phase
import numpy as np

# Optional local samples via env (never hardcode machine paths in the repo).
#   export LFI_CHIP_FILE=/path/to/chip.hdf5
#   export LFI_SKIN_FILE=/path/to/skin.hdf5
#   export LFI_PCB_FILE=/path/to/frame-3.hdf5
CHIP_FILE = os.environ.get("LFI_CHIP_FILE", "")
SKIN_FILE = os.environ.get("LFI_SKIN_FILE", "")
PCB_FILE = os.environ.get("LFI_PCB_FILE", "")
_HERE = os.path.dirname(os.path.abspath(__file__))
VEG_FRAME_20 = os.path.join(_HERE, 'data', '2026_Vegetables', 'frame-20.hdf5')
VEG_FRAME_21 = os.path.join(_HERE, 'data', '2026_Vegetables', 'frame-21.hdf5')
MARBLE_FILE = os.path.join(_HERE, 'data', 'Marble', 'maeble10%_0.hdf5')
LEAF_FILE = os.path.join(_HERE, 'data', 'Leaf', 'leaf1.mat')
# ───────────────────────────────────────────────────────────────


def test_chip_zstack():
    print("\n" + "=" * 50)
    print("test 1: Chip z-stack")
    print("=" * 50)

    if not CHIP_FILE or not os.path.exists(CHIP_FILE):
        print("  skip: set LFI_CHIP_FILE to a local chip/z-stack hdf5")
        return

    data = load_single_scan(CHIP_FILE, target_type='Chip')
    print(data.summary())

    assert data.metadata['file_format'] == 'hdf5_zstack'
    assert data.metadata['phase_encoding_used'] == 'zstack_unconfirmed'
    assert data.metadata['phase_converted'] is False
    assert data.phase.min() >= -180.0 and data.phase.max() < 180.0
    assert data.amplitude.shape == data.phase.shape
    assert len(data.z_indices) == data.n_slices
    assert data.metadata['original_dtype']
    assert 'frame_ids' in data.metadata
    assert 'rdm_source_path' in data.metadata
    assert 'phase_encoding_used' in data.metadata

    print("\n✅ test_chip_zstack passed!")


def test_skin_frames_unconfirmed_phase():
    print("\n" + "=" * 50)
    print("test 2: Skin frames(phase unconfirmed → keep raw codes)")
    print("=" * 50)

    if not SKIN_FILE or not os.path.exists(SKIN_FILE):
        print("  skip: set LFI_SKIN_FILE to a local skin hdf5")
        return

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        data = load_single_scan(SKIN_FILE, target_type='Skin', qcl='QCL0', side='Left')

    assert data.metadata['file_format'] == 'hdf5_frames'
    assert data.metadata['phase_encoding_used'] == 'skin_unconfirmed'
    assert data.metadata['phase_converted'] is False
    assert data.n_slices >= 1
    assert data.amplitude.shape == data.phase.shape
    assert any('unconfirmed' in str(x.message) for x in w)

    # explicitly forbid inventing a Skin conversion
    enc = get_phase_encoding('skin_unconfirmed')
    try:
        enc.convert(data.phase)
        raise AssertionError('skin_unconfirmed.convert should raise')
    except RuntimeError:
        pass

    print(data.summary())
    print("\n✅ test_skin_frames_unconfirmed_phase passed!")


def test_pcb_frames():
    print("\n" + "=" * 50)
    print("test 3: PCB frames(identity lookup → pcb_unconfirmed; do not reuse Vegetables formula by value range)")
    print("=" * 50)

    if not PCB_FILE or not os.path.exists(PCB_FILE):
        print("  skip: set LFI_PCB_FILE to a local PCB/frame hdf5")
        return

    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        data = load_single_scan(PCB_FILE, target_type='PCB', qcl='QCL1', side='Left')
    print(data.summary())

    assert data.metadata['file_format'] == 'hdf5_frames'
    # even if original_dtype=int16 and values fall in the Vegetables range, do not auto-apply vegetables_int16
    assert data.metadata['phase_encoding_used'] == 'pcb_unconfirmed'
    assert data.metadata['phase_converted'] is False
    assert data.metadata['original_dtype'] == 'int16'
    assert 'frame_ids' in data.metadata and data.metadata['frame_ids']
    assert data.metadata['scan_identifier'] == 'CORN06052026'
    assert 'environment' in data.metadata['acquisition']

    z = get_z_slices(PCB_FILE)
    assert len(z) >= 1
    print(
        f"  frame ids: {z} | phase_encoding={data.metadata['phase_encoding_used']} "
        f"converted={data.metadata['phase_converted']} "
        f"original_dtype={data.metadata['original_dtype']}"
    )

    print("\n✅ test_pcb_frames passed!")


def test_phase_formulas_unit():
    print("\n" + "=" * 50)
    print("test 4: phase formulas (no HDF5)")
    print("=" * 50)

    # Vegetables
    codes = np.array([-8192, 0, 8191], dtype=np.int16)
    rad, enc, ok = convert_phase(codes, 'vegetables_int16')
    assert ok and enc.name == 'vegetables_int16'
    assert np.isclose(rad[0], -np.pi)
    assert np.isclose(rad[1], 0.0)
    assert np.isclose(rad[2], 8191 / 8192 * np.pi)

    # range check (use int32 so 9000 is not truncated as int16)
    bad = np.array([9000], dtype=np.int32)
    try:
        convert_phase(bad, 'vegetables_int16')
        raise AssertionError('out-of-range should fail')
    except ValueError:
        pass

    # selection: lookup by target_type only; independent of value range
    from phase_encoding import resolve_encoding_for_phase
    lookalike = np.array([-100, 0, 8191], dtype=np.int16)
    assert resolve_encoding_for_phase('PCB', lookalike) == 'pcb_unconfirmed'
    assert resolve_encoding_for_phase('Vegetables', lookalike) == 'vegetables_int16'
    assert resolve_encoding_for_phase('Skin', lookalike) == 'skin_unconfirmed'

    # Leaf wrap
    # L_P=0 → wrap(0 - π) = -π?  wrap((-π)+π)%2π - π = 0%2π - π = -π
    # L_P=4096 → wrap(π - π) = 0
    leaf_codes = np.array([0, 4096], dtype=np.int16)
    leaf_rad, _, ok = convert_phase(leaf_codes, 'leaf_wrap_4096')
    assert ok
    assert np.isclose(leaf_rad[0], -np.pi) or np.isclose(leaf_rad[0], np.pi)
    assert np.isclose(leaf_rad[1], 0.0)

    print("\n✅ test_phase_formulas_unit passed!")


def test_get_z_slices_both():
    print("\n" + "=" * 50)
    print("test 5: get_z_slices works for both layouts")
    print("=" * 50)

    if not CHIP_FILE or not os.path.exists(CHIP_FILE) or not SKIN_FILE or not os.path.exists(SKIN_FILE):
        print("  skip: set LFI_CHIP_FILE and LFI_SKIN_FILE")
        return

    z_chip = get_z_slices(CHIP_FILE)
    z_skin = get_z_slices(SKIN_FILE)
    assert z_chip[0] == 530 and z_chip[-1] == 599
    assert len(z_skin) >= 1
    print(f"  Chip: {z_chip[0]} → {z_chip[-1]} ({len(z_chip)})")
    print(f"  Skin frames: {z_skin}")

    print("\n✅ test_get_z_slices_both passed!")


def test_infer_target_type_path_identity():
    print("\n" + "=" * 50)
    print("test 6: _infer_target_type path identity (2026_Vegetables vs frame-XX)")
    print("=" * 50)

    # RDM-style: folder variant + frame filename → must be Vegetables, not PCB
    veg = (
        "/Volumes/RDM/02 Data Sets/2026 Semester 2/"
        "2026_Vegetables/frame-20.hdf5"
    )
    assert _infer_target_type(veg) == "Vegetables", _infer_target_type(veg)

    # loose sample (no type identity in parents → frame-N stem → PCB)
    assert _infer_target_type("/tmp/local_samples/frame-3.hdf5") == "PCB"

    # exact folder name
    assert _infer_target_type("/data/Chip/borad1.hdf5") == "Chip"
    assert _infer_target_type("/data/Skin/2025-11-28_10-14-26.hdf5") == "Skin"

    # Megabugs alias → PCB
    assert _infer_target_type("/data/Megabugs/frame-1.hdf5") == "PCB"

    print("\n✅ test_infer_target_type_path_identity passed!")


def test_vegetables_real_normal_frame():
    print("\n" + "=" * 50)
    print("test 7: real Vegetables frame 21 (2026 layout, converted)")
    print("=" * 50)

    if not os.path.exists(VEG_FRAME_21):
        print(f"  skip: missing {VEG_FRAME_21}")
        return

    data = load_single_scan(VEG_FRAME_21)
    m = data.metadata
    assert data.target_type == 'Vegetables'
    assert m['file_format'] == 'hdf5_data_v2026'
    assert m['phase_encoding_used'] == 'vegetables_int16'
    assert m['phase_converted'] is True
    assert m['original_dtype'] == 'int16'
    assert m['frame_ids'] == [21]
    assert m['anomalous_frame_ids'] == []
    assert m['laser_channel'] == 'QCL1/Left'
    assert data.amplitude.shape == (1, 400, 1300)
    assert float(np.nanmax(np.abs(data.phase))) <= np.pi + 1e-9
    assert 'environment' in m['acquisition'] and 'mcp' in m['acquisition']

    right = load_single_scan(VEG_FRAME_21, side='Right')
    assert right.metadata['laser_channel'] == 'QCL1/Right'
    assert get_z_slices(VEG_FRAME_21) == [21]

    print("\n✅ test_vegetables_real_normal_frame passed!")


def test_vegetables_real_anomalous_frame():
    print("\n" + "=" * 50)
    print("test 8: real Vegetables frame 20 (anomalous: excluded by default; loadable flagged)")
    print("=" * 50)

    if not os.path.exists(VEG_FRAME_20):
        print(f"  skip: missing {VEG_FRAME_20}")
        return

    # default exclude: file only has frame 20; must raise clearly instead of returning silently
    try:
        load_single_scan(VEG_FRAME_20)
        raise AssertionError("frame 20 should be excluded by default (AnomalousOnlyError)")
    except AnomalousOnlyError:
        pass

    # explicit load: keep raw codes, flag, do not convert
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        data = load_single_scan(VEG_FRAME_20, exclude_anomalous=False)
    m = data.metadata
    assert m['anomalous_frame_ids'] == [20]
    assert m['phase_converted'] is False
    assert data.phase.dtype == np.int16

    print("\n✅ test_vegetables_real_anomalous_frame passed!")


def test_marble_real():
    print("\n" + "=" * 50)
    print("test 9: real Marble (filename typo maeble; identity from folder)")
    print("=" * 50)

    if not os.path.exists(MARBLE_FILE):
        print(f"  skip: missing {MARBLE_FILE}")
        return

    data = load_single_scan(MARBLE_FILE)
    m = data.metadata
    assert data.target_type == 'Marble'
    assert m['file_format'] == 'hdf5_zstack'
    assert m['phase_encoding_used'] == 'zstack_unconfirmed'
    assert m['phase_converted'] is False
    assert data.amplitude.shape == (1, 1400, 400)

    print("\n✅ test_marble_real passed!")


def test_leaf_real():
    print("\n" + "=" * 50)
    print("test 10: real Leaf leaf1.mat (~15 s, ~3.6 GB RAM)")
    print("=" * 50)

    if not os.path.exists(LEAF_FILE):
        print(f"  skip: missing {LEAF_FILE}")
        return

    data = load_single_scan(LEAF_FILE)
    m = data.metadata
    assert data.target_type == 'Leaf'
    assert m['file_format'] == 'mat'
    assert m['scan_mode'] == 'time_series'
    assert 'time series' in (m.get('notes') or '').lower()
    assert m['phase_encoding_used'] == 'leaf_wrap_4096'
    assert m['phase_converted'] is True
    assert m['original_dtype'] == 'int16'
    assert data.amplitude.shape == data.phase.shape == (475, 1200, 400)
    sample = data.phase[::25]
    assert sample.min() >= -np.pi and sample.max() < np.pi

    print("\n✅ test_leaf_real passed!")


def test_numeric_frame_sort():
    print("\n" + "=" * 50)
    print("test 11: frame files sorted numerically (not lexically)")
    print("=" * 50)

    names = [
        "frame-100.hdf5",
        "frame-21.hdf5",
        "frame-110.hdf5",
        "frame-140.hdf5",
        "frame-22.hdf5",
    ]
    got = [p.name for p in sort_data_paths(names)]
    assert got == [
        "frame-21.hdf5",
        "frame-22.hdf5",
        "frame-100.hdf5",
        "frame-110.hdf5",
        "frame-140.hdf5",
    ], got
    # lexical sort would be wrong - ensure we did not use it
    assert got != sorted(names)
    print("\n✅ test_numeric_frame_sort passed!")


def test_experiment_details_and_volume():
    print("\n" + "=" * 50)
    print("test 12: load_volume + experiment_details → depth_mm")
    print("=" * 50)

    need = [os.path.join(_HERE, "data", "2026_Vegetables", f"frame-{i}.hdf5") for i in (21, 22, 28)]
    if not all(os.path.exists(p) for p in need):
        print("  skip: missing frame-21/22/28")
        return

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "2026_Vegetables"
        sub.mkdir()
        # symlink in scrambled order; sort must use frame numbers
        for src in (need[2], need[0], need[1]):
            os.symlink(src, sub / Path(src).name)
        (sub / "experiment_details.txt").write_text(
            "Start position: 1.0 mm\n"
            "Step size: 0.1 mm\n"
            "First frame: 21\n",
            encoding="utf-8",
        )
        details = parse_experiment_details(sub / "experiment_details.txt")
        assert details["z_start_mm"] == 1.0
        assert details["z_step_mm"] == 0.1
        assert depth_mm_for_frames([21, 22, 28], details) == [1.0, 1.1, 1.7]

        vol = load_volume(str(sub), frame_ids=[28, 21, 22])
        assert vol.z_indices == [21, 22, 28], vol.z_indices
        assert vol.amplitude.shape == (3, 400, 1300)
        assert vol.metadata["phase_converted"] is True
        assert vol.metadata["depth_mm"] == [1.0, 1.1, 1.7]
        assert vol.metadata["file_format"] == "volume_from_frames"

    print("\n✅ test_experiment_details_and_volume passed!")


def test_waveform_optional():
    print("\n" + "=" * 50)
    print("test 13: optional RX_LINE_OF_BEST_FIT waveform")
    print("=" * 50)

    if not os.path.exists(VEG_FRAME_21):
        print(f"  skip: missing {VEG_FRAME_21}")
        return

    plain = load_single_scan(VEG_FRAME_21)
    assert plain.waveform is None

    data = load_single_scan(VEG_FRAME_21, include_waveform=True)
    assert data.waveform is not None
    assert data.waveform.shape == (1, 400, 1300, 184)
    assert data.waveform.dtype == np.int16
    assert "waveform" in data.metadata["data_channels"]

    print("\n✅ test_waveform_optional passed!")


def test_leaf_partial_frames():
    print("\n" + "=" * 50)
    print("test 14: Leaf partial frame load")
    print("=" * 50)

    if not os.path.exists(LEAF_FILE):
        print(f"  skip: missing {LEAF_FILE}")
        return

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = load_single_scan(LEAF_FILE, frame_ids=[0, 10, 20])
    assert data.amplitude.shape == (3, 1200, 400)
    assert data.z_indices == [0, 10, 20]
    assert data.metadata["mat_total_frames"] == 475
    assert data.metadata["phase_converted"] is True
    assert any("v5/v7" in str(x.message) for x in w)

    print("\n✅ test_leaf_partial_frames passed!")


if __name__ == '__main__':
    test_phase_formulas_unit()
    test_infer_target_type_path_identity()
    test_chip_zstack()
    test_get_z_slices_both()
    test_skin_frames_unconfirmed_phase()
    test_pcb_frames()
    test_vegetables_real_normal_frame()
    test_vegetables_real_anomalous_frame()
    test_marble_real()
    test_leaf_real()
    test_numeric_frame_sort()
    test_experiment_details_and_volume()
    test_waveform_optional()
    test_leaf_partial_frames()
    print("\n🎉 All tests passed!")
