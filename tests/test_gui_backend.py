import numpy as np
import pytest
from part4_ui.lfi_core.data_types import LFIRawData
from part4_ui.lfi_core.backend import register_reconstructor, get_reconstructor_registration
from part3_reconstruction.gui_backend import DepthReconstructor, VEGETABLES_2026, frame_depths


def _stack(frame_ids, depth=None, ny=3, nx=4):
    n = len(frame_ids)
    amplitude = np.arange(n, dtype=float)[:, None, None] * np.ones((n, ny, nx))
    phase = (np.arange(n, dtype=float) * 10)[:, None, None] * np.ones((n, ny, nx))
    metadata = {"phase_converted": True} if depth is None else {"phase_converted": True, "depth_mm": depth}
    return LFIRawData(amplitude=amplitude, phase=phase, z_indices=list(frame_ids),
                      target_type="Vegetables", file_path="x", metadata=metadata)


def test_vegetables_parameters_give_focus_depths():
    data = _stack([21, 22, 23, 33])
    depth, source = frame_depths(data, VEGETABLES_2026)
    np.testing.assert_allclose(depth, [30.15, 30.25, 30.35, 31.35])
    assert source.startswith("parameters")


def test_reconstruct_resamples_amplitude_and_takes_nearest_phase():
    data = _stack([21, 22, 23, 33])
    out = DepthReconstructor().reconstruct(data, {**VEGETABLES_2026, "output_spacing_mm": 0.1})
    assert isinstance(out, LFIRawData) and out.amplitude.shape == (13, 3, 4)
    np.testing.assert_allclose(out.z_indices[0], 30.15)
    np.testing.assert_allclose(out.z_indices[-1], 31.35)
    grid = np.asarray(out.metadata["depth_mm"])
    np.testing.assert_allclose(out.amplitude[:, 0, 0], np.interp(grid, [30.15, 30.25, 30.35, 31.35], [0, 1, 2, 3]), atol=1e-5)
    # Phase is never interpolated: every value is one of the measured slices.
    assert set(np.unique(out.phase)) <= {0.0, 10.0, 20.0, 30.0}
    assert out.metadata["part3_reconstruction"]["phase_source_slice_ids"][6] in (23, 33)
    assert out.metadata["phase_converted"] is True and data.amplitude.shape == (4, 3, 4)  # input untouched


def test_metadata_depth_used_when_no_parameters_and_reverse_order():
    data = _stack([3, 2, 1], depth=[2.0, 1.0, 0.0])
    out = DepthReconstructor().reconstruct(data, {"output_spacing_mm": 0.5})
    assert out.z_indices == [0.0, 0.5, 1.0, 1.5, 2.0]
    np.testing.assert_allclose(out.amplitude[:, 0, 0], [2, 1.5, 1, 0.5, 0])
    assert out.metadata["part3_reconstruction"]["depth_source"] == "metadata depth_mm"


@pytest.mark.parametrize("params, message", [
    ({}, "depths unknown"),
    ({"z_start_mm": 1.0}, "together"),
    ({**VEGETABLES_2026, "output_spacing_mm": 0.001, "max_slices": 50}, "max_slices"),
    ({**VEGETABLES_2026, "output_spacing_mm": 0}, "positive"),
])
def test_reconstruct_rejects_bad_input(params, message):
    with pytest.raises(ValueError, match=message):
        DepthReconstructor().reconstruct(_stack([21, 22, 30]), params)
    with pytest.raises(ValueError, match="two slices"):
        DepthReconstructor().reconstruct(_stack([21]), VEGETABLES_2026)


def test_registers_with_the_viewer():
    try:
        register_reconstructor(DepthReconstructor(), {"output_spacing_mm": 0.1})
        registration = get_reconstructor_registration()
        assert isinstance(registration.implementation, DepthReconstructor)
        assert registration.params == {"output_spacing_mm": 0.1}
    finally:
        register_reconstructor(None)


def test_viewer_loader_reads_part2_volume(tmp_path):
    from part3_reconstruction.viewer import load_for_viewer
    z = np.array([30.15, 30.25, 31.25])
    amp = np.ones((3, 2, 5), np.float32)
    path = tmp_path / "vegetables_volume.npz"
    np.savez(path, amplitude_corrected=amp, raw_amplitude=amp, raw_phase=amp * 0, z_mm=z,
             frame_ids=np.array([21, 22, 32]), signal_mask=amp > 0)
    data = load_for_viewer(path, fallback=lambda p: pytest.fail("fallback used"))
    assert data.z_indices == [21, 22, 32] and data.metadata["depth_mm"] == list(z)
    out = DepthReconstructor().reconstruct(data, {"output_spacing_mm": 0.1})
    assert out.amplitude.shape[0] == 12
    sentinel = object()
    assert load_for_viewer(tmp_path / "scan.hdf5", fallback=lambda p: sentinel) is sentinel
