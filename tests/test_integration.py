import types
import numpy as np
import pytest
from matplotlib.figure import Figure
from part3_reconstruction import ImagingData, VolumeReconstructor
from part3_reconstruction.integration import (from_viewer_volume, render_rgb, orthogonal_views,
                                              figure_to_png, to_reconstructed_image)


def viewer_like(nz=3, ny=4, nx=5):
    a = np.arange(nz*ny*nx, dtype=float).reshape(nz, ny, nx)
    return types.SimpleNamespace(amplitude=a, phase=a*10, z_indices=[530, 531, 532][:nz],
                                 target_type="Leaf", file_path="scan.hdf5")


def test_viewer_volume_keeps_frames_as_metadata_not_depth():
    d = from_viewer_volume(viewer_like())
    assert d.amplitude.shape == (3, 4, 5)
    assert d.metadata["phase_unit"] == "raw_counts" and d.metadata["frame_indices"] == [530, 531, 532]
    assert d.metadata["target_type"] == "Leaf"
    assert "z_coordinates" not in d.metadata


def test_viewer_volume_degrees_and_axis_order():
    obj = viewer_like()
    obj.phase = np.full((3, 4, 5), 180.)
    d = from_viewer_volume(obj, phase_unit="degrees", axis_order="zxy")
    assert d.amplitude.shape == (3, 5, 4)
    np.testing.assert_allclose(d.phase, np.pi)
    with pytest.raises(TypeError): from_viewer_volume(object())
    with pytest.raises(ValueError): from_viewer_volume(obj, axis_order="zy")


def test_render_rgb_transparency_flip_and_ranges():
    a = np.array([[0., 1.], [2., np.nan]])
    d = ImagingData(a, a, {"phase_unit": "rad"})
    rgba, info = render_rgb(d, "amplitude", norm="linear")
    assert rgba.shape == (2, 2, 4) and rgba.dtype == np.uint8
    assert rgba[1, 1, 3] == 0 and rgba[0, 0, 3] == 255
    assert info["display_min"] == 0 and info["display_max"] == 2 and info["missing_fraction"] == .25
    flipped, _ = render_rgb(d, "amplitude", norm="linear", flip_y=True)
    np.testing.assert_array_equal(flipped, rgba[::-1])
    _, pinfo = render_rgb(d, "phase", phase_mode="cyclic")
    assert pinfo["unit"] == "rad (wrapped)" and pinfo["display_min"] == pytest.approx(-np.pi)


def test_cyclic_phase_periodic_and_requires_radians():
    p = np.array([[0., 1.], [2., 3.]])
    one, _ = render_rgb(ImagingData(p, p, {"phase_unit": "rad"}), "phase", phase_mode="cyclic")
    two, _ = render_rgb(ImagingData(p, p + 2*np.pi, {"phase_unit": "rad"}), "phase", phase_mode="cyclic")
    assert np.abs(one.astype(int) - two.astype(int)).max() <= 1
    with pytest.raises(ValueError):
        render_rgb(ImagingData(p, p, {"phase_unit": "raw_counts"}), "phase", phase_mode="cyclic")


def test_orthogonal_views_and_png():
    d = from_viewer_volume(viewer_like())
    v = orthogonal_views(d, (1, 2, 3))
    assert v["xy"].shape == (4, 5) and v["xz"].shape == (3, 5) and v["yz"].shape == (3, 4)
    assert v["xz"][1, 3] == v["xy"][2, 3] == v["yz"][1, 2]
    with pytest.raises(ValueError): orthogonal_views(d, (3, 0, 0))
    fig = Figure(); fig.subplots().plot([0, 1])
    assert figure_to_png(fig, dpi=50)[:8] == b"\x89PNG\r\n\x1a\n"


def test_reconstructed_image_contract():
    d = from_viewer_volume(viewer_like())
    vol = VolumeReconstructor(d, spacing_xy=(.1, .2), z_coordinates=[2., 0., 1.])
    r = to_reconstructed_image(vol)
    assert r.z_levels == [0., 1., 2.] and r.resolution == (.1, .2)
    assert len(r.slices_3d) == 3 and r.image_2d.shape == (4, 5)
    assert r.quality_score == 0.0 and r.target_type == "Leaf"
