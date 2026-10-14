"""experiment_details.txt aliases + depth origin when frame-20 is skipped."""

from __future__ import annotations

import warnings

import h5py
import numpy as np

from loader import (
    depth_mm_for_frames,
    load_volume,
    parse_experiment_details,
)


def _write_veg_frame(path, frame_id: int, h: int = 4, w: int = 5) -> None:
    name = f"frame_{frame_id}_qcl_0_left"
    amp = np.ones((h, w), dtype=np.float64)
    phase = np.linspace(-50, 50, h * w, dtype=np.int16).reshape(h, w)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.create_group("IMAGE_AMPLITUDE").create_dataset(name, data=amp)
        data.create_group("IMAGE_PHASE").create_dataset(name, data=phase)


def test_parse_start_z_position_and_imcrements_typo(tmp_path):
    """Real Vegetables wording + common typo 'imcrements'."""
    fp = tmp_path / "experiment_details.txt"
    fp.write_text(
        "start z position (mm): 30.05\n"
        "imcrements (mm): 0.1\n",
        encoding="utf-8",
    )
    d = parse_experiment_details(fp)
    assert d["z_start_mm"] == 30.05
    assert d["z_step_mm"] == 0.1
    assert d["first_frame_id"] is None


def test_depth_uses_catalog_min_when_frame20_skipped(tmp_path):
    """
    If first_frame_id is omitted and load_volume skips frame 20, depth for
    frame 21 must still use ref=20 → start+0.1, not start (ref=21).
    """
    veg = tmp_path / "2026_Vegetables"
    veg.mkdir()
    for fid in (20, 21, 22):
        _write_veg_frame(veg / f"frame-{fid}.hdf5", fid)
    (veg / "experiment_details.txt").write_text(
        "start z position (mm) = 30.05\n"
        "imcrements (mm) = 0.1\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = load_volume(str(veg))

    assert vol.z_indices == [21, 22]
    assert vol.metadata["depth_reference_frame_id"] == 20
    assert vol.metadata["depth_mm"] == [30.15, 30.25]


def test_depth_mm_for_frames_explicit_reference():
    details = {"z_start_mm": 30.05, "z_step_mm": 0.1, "first_frame_id": None}
    # Without hint: wrongly uses min(batch)=21
    assert depth_mm_for_frames([21, 22], details) == [30.05, 30.15]
    # With catalog hint including skipped 20:
    assert depth_mm_for_frames(
        [21, 22], details, reference_frame_id=20
    ) == [30.15, 30.25]


def test_depth_correct_when_frame_ids_subset(tmp_path):
    """
    Folder has frames 20..23; load_volume(..., frame_ids=[22,23]) must still
    use catalog min 20 as depth origin → [30.25, 30.35], not [30.05, 30.15].
    """
    veg = tmp_path / "2026_Vegetables"
    veg.mkdir()
    for fid in (20, 21, 22, 23):
        _write_veg_frame(veg / f"frame-{fid}.hdf5", fid)
    (veg / "experiment_details.txt").write_text(
        "start z position (mm) = 30.05\n"
        "imcrements (mm) = 0.1\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vol = load_volume(str(veg), frame_ids=[22, 23])

    assert vol.z_indices == [22, 23]
    assert vol.metadata["depth_reference_frame_id"] == 20
    assert vol.metadata["depth_mm"] == [30.25, 30.35]
    assert any(
        "no first_frame_id" in str(x.message)
        and "depth_reference_frame_id=20" in str(x.message)
        for x in w
    )
