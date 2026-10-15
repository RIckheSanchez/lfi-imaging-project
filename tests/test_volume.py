import numpy as np
import nibabel as nib
import pytest
from part3_reconstruction import ImagingData, VolumeReconstructor
from part3_reconstruction.validation import dice_score


def volume(a, z, phase=None):
    return VolumeReconstructor(ImagingData(a, a if phase is None else phase, {"phase_state": "unwrapped"}), spacing_xy=(1, 1), z_coordinates=z)


def test_irregular_reversed_depth_linear_field():
    z = np.array([5., 2., 0.])
    zz, yy, xx = np.meshgrid(z, np.arange(4), np.arange(6), indexing="ij")
    v = volume(2*zz + 3*yy + xx, z).resample_isotropic(1)
    zz, yy, xx = np.meshgrid(np.arange(6), np.arange(4), np.arange(6), indexing="ij")
    np.testing.assert_allclose(v.data.amplitude, 2*zz+3*yy+xx, atol=1e-6)
    np.testing.assert_equal(v.get_orthoslice("xz", 2), v.data.amplitude[:, 2, :])
    np.testing.assert_equal(v.get_orthoslice("yz", 3), v.data.amplitude[:, :, 3])


def test_nifti_axis_and_world_roundtrip(tmp_path):
    a = np.arange(60).reshape(3, 4, 5)
    v = VolumeReconstructor(ImagingData(a, -a), spacing_xy=(.2, .3), z_coordinates=[10, 12, 14], origin_xy=(4, 7))
    path = v.export_nifti(tmp_path / "phase.nii.gz", "phase")
    img = nib.load(path)
    np.testing.assert_equal(img.get_fdata(), (-a).transpose(2, 1, 0))
    np.testing.assert_allclose(img.affine @ [4, 3, 2, 1], [4.8, 7.9, 14, 1])
    assert img.header.get_xyzt_units()[0] == "mm"


def test_sphere_geometry():
    z = np.arange(-10, 11, 2)
    zz, yy, xx = np.meshgrid(z, np.arange(21)-10, np.arange(21)-10, indexing="ij")
    v = volume((xx*xx+yy*yy+zz*zz <= 49).astype(float), z).resample_isotropic(1)
    zz, yy, xx = np.meshgrid(np.arange(-10, 11), np.arange(21)-10, np.arange(21)-10, indexing="ij")
    # Binary source sampling loses boundary information; record the observed
    # baseline rather than claiming arbitrary >.95 accuracy on this coarse grid.
    assert dice_score(v.data.amplitude >= .5, xx*xx+yy*yy+zz*zz <= 49) == pytest.approx(.9366336633663367)


def test_continuous_sphere_geometry():
    z = np.arange(-10, 11, 2)
    zz, yy, xx = np.meshgrid(z, np.arange(21)-10, np.arange(21)-10, indexing="ij")
    v = volume(49-xx*xx-yy*yy-zz*zz, z).resample_isotropic(1)
    zz, yy, xx = np.meshgrid(np.arange(-10, 11), np.arange(21)-10, np.arange(21)-10, indexing="ij")
    assert dice_score(v.data.amplitude >= 0, xx*xx+yy*yy+zz*zz <= 49) > .98


def test_translation_registration():
    from scipy.ndimage import shift
    rng = np.random.default_rng(1)
    a = rng.normal(size=(32, 32))
    b = shift(a, (3, -2), order=0, mode="constant")
    v = volume(np.stack([a, b]), [0, 1]).register_translation(upsample_factor=1)
    np.testing.assert_allclose(v.registration_shifts[1], [-3, 2])
    np.testing.assert_allclose(v.data.amplitude[1, 2:27, 4:29], a[2:27, 4:29], atol=1e-6)
    assert np.isnan(v.data.amplitude[1]).any()


def test_invalid_geometry_and_memory(tmp_path):
    a = np.ones((3, 4, 5))
    with pytest.raises(ValueError): volume(a, [0, 0, 1])
    with pytest.raises(ValueError): volume(a, [0, 1])
    with pytest.raises(ValueError): volume(a, [0, 1, 3]).export_nifti(tmp_path / "x.nii")
    with pytest.raises(ValueError): volume(a, [0, 1, 2]).resample_isotropic(.01, max_voxels=100)
    unknown = VolumeReconstructor(ImagingData(a, a), spacing_xy=(1, 1), z_coordinates=[0, 1, 2])
    with pytest.raises(ValueError): unknown.resample_isotropic()


def test_resampled_slice_geometry_and_index():
    v = volume(np.ones((3, 4, 5)), [0, 2, 4]).resample_isotropic(.5)
    assert v.data.slice(1).metadata["spacing_xy"] == (.5, .5)
    assert v.data.metadata["z_coordinates"] == v.z.tolist()
    with pytest.raises(ValueError): v.get_orthoslice("xy", -1)
    with pytest.raises(ValueError): v.get_orthoslice("xy", len(v.z))


def test_planewise_resampling_matches_3d_trilinear_reference():
    from scipy.interpolate import RegularGridInterpolator
    rng = np.random.default_rng(7)
    z = np.array([4.0, 0.0, 1.0, 2.5])                       # irregular and unsorted
    a = rng.standard_normal((4, 9, 11)); a[rng.random(a.shape) < .05] = np.nan
    data = ImagingData(a, rng.standard_normal((4, 9, 11)), {"phase_state": "unwrapped"})
    v = VolumeReconstructor(data, spacing_xy=(.7, .4), z_coordinates=z, origin_xy=(1, -2)).resample_isotropic(.3)
    order = np.argsort(z)
    ref = RegularGridInterpolator((z[order], -2 + np.arange(9) * .4, 1 + np.arange(11) * .7), a[order],
                                  bounds_error=False, fill_value=np.nan)
    zz, yy, xx = np.meshgrid(v.z, -2 + np.arange(v.data.amplitude.shape[1]) * .3,
                             1 + np.arange(v.data.amplitude.shape[2]) * .3, indexing="ij")
    np.testing.assert_allclose(v.data.amplitude, ref(np.stack((zz, yy, xx), -1)), rtol=1e-6, atol=1e-6, equal_nan=True)
