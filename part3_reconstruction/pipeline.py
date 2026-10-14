"""In-memory Part 1 -> Part 2 -> Part 3, without handoff files.

    from part3_reconstruction.pipeline import run_part1_to_part3
    data = run_part1_to_part3("frame-21.hdf5")      # ImagingData, ready for ImageGenerator

``slice_from_part2`` reads the ``spatial_products`` that Part 2's
``preprocess_part1`` attaches to its output, the same channels as the handoff
NPZ files. Phase stays in raw codes when Part 1 could not confirm its encoding.
"""
from .handoff import FRAME_ASPECT, frame_from_products, kind_from_target


def slice_from_part2(output, index=0, *, amplitude_channel="amplitude_corrected", frame_aspect=None):
    """One slice of a Part 2 output as 2D ImagingData."""
    products = output.metadata.get("spatial_products")
    if products is None:
        raise ValueError("Part 2 output has no spatial_products; is spatial correction enabled?")
    if not 0 <= index < len(output.z_indices):
        raise IndexError(f"slice index {index} outside 0..{len(output.z_indices) - 1}")
    kind = kind_from_target(output.target_type)
    upstream = output.metadata.get("upstream_metadata", {})
    if frame_aspect is None:
        frame_aspect = upstream.get("aspect_ratio_computed") or FRAME_ASPECT.get(kind)
    phase_unit = "rad" if output.metadata.get("phase_units") == "radians" else "raw_counts"
    data = frame_from_products(products, index, kind=kind, amplitude_channel=amplitude_channel,
                               phase_unit=phase_unit, frame_aspect=frame_aspect,
                               source=output.metadata.get("source_path", ""),
                               frame_id=int(output.z_indices[index]))
    data.metadata["provenance"] = "Part 1 load -> Part 2 preprocess_part1 (in memory)"
    return data


def run_part1_to_part3(path, index=0, *, amplitude_channel="amplitude_corrected", config_overrides=None):
    """Load an HDF5 scan with Part 1, pre-process it with Part 2, return a Part 3 slice."""
    from part1_data import load_single_scan
    from part2_preprocess.pipeline import preprocess_part1
    output = preprocess_part1(load_single_scan(str(path), include_waveform=False),
                              config_overrides=config_overrides)
    return slice_from_part2(output, index, amplitude_channel=amplitude_channel)
