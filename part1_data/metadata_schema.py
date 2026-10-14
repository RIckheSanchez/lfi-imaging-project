"""
LFI Experiment Metadata Schema
Defines the standard metadata fields for all LFI datasets.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import json


@dataclass
class LFIMetadata:
    """
    Standard metadata schema for LFI experimental datasets.
    Captures scan settings, device configuration, and provenance.
    """

    # --- Experiment Identity ---
    experiment_id:   str = ""          # Unique ID e.g. "2025-11-28_skin_001"
    target_type:     str = ""          # "Chip", "Skin", "Leaf", etc.
    timestamp:       str = ""          # ISO format: "2025-11-28T10:14:26"
    operator:        str = ""          # Researcher name

    # --- Scan Configuration ---
    scan_mode:       str = ""          # "z_stack" or "single_frame" / "frames"
    n_z_slices:      int = 0           # Number of z-depth layers (or frame count)
    z_start:         Optional[float] = None   # Start depth (mm)
    z_end:           Optional[float] = None   # End depth (mm)
    x_pixels:        int = 0           # Image width
    y_pixels:        int = 0           # Image height

    # --- Device Configuration ---
    laser_type:      str = ""          # e.g. "QCL", "THz"
    laser_channel:   str = ""          # e.g. "QCL0/Left"
    wavelength_um:   Optional[float] = None   # Laser wavelength (micrometers)

    # --- Data Properties ---
    data_channels:   list = field(default_factory=list)  # ["amplitude", "phase", "peak_to_peak"]
    dtype:           str = ""          # Loaded array dtype after processing, e.g. "float64"
    file_format:     str = ""          # "hdf5_zstack", "hdf5_frames", "hdf5_data_v2026", "mat"

    # --- Provenance / teammate-required fields ---
    raw_file_path:   str = ""          # Original file path on local disk
    rdm_source_path: str = ""          # Path/URL on Research Data Manager (if known)
    frame_ids:       list = field(default_factory=list)  # Frame ids as loaded (int or str)
    original_dtype:  str = ""          # On-disk phase dtype before conversion, e.g. "int16"
    phase_encoding_used: str = ""      # Key from phase_encoding.PHASE_ENCODINGS
    anomalous_frame_ids: list = field(default_factory=list)  # e.g. Vegetables frame 20
    phase_converted: bool = False      # True iff radians conversion was applied
    notes:           str = ""          # Free text notes

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for storage."""
        import dataclasses
        return dataclasses.asdict(self)

    def to_json(self, filepath: str):
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Metadata saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str):
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_load_result(
        cls,
        *,
        target_type: str,
        file_path: str,
        file_format: str,
        amplitude_shape: tuple,
        frame_ids: list,
        original_dtype: str,
        phase_encoding_used: str,
        phase_converted: bool,
        anomalous_frame_ids: Optional[list] = None,
        laser_channel: str = "",
        rdm_source_path: str = "",
        notes: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> "LFIMetadata":
        """Build metadata from a successful loader call."""
        n = int(amplitude_shape[0]) if amplitude_shape else 0
        y = int(amplitude_shape[1]) if len(amplitude_shape) > 1 else 0
        x = int(amplitude_shape[2]) if len(amplitude_shape) > 2 else 0
        if file_format == "hdf5_zstack":
            scan_mode = "z_stack"
        elif file_format == "volume_from_frames":
            scan_mode = "volume"
        elif file_format == "mat":
            # Karl, email, 2026-10-14: Leaf third axis is a time series
            scan_mode = "time_series" if target_type == "Leaf" else "stack_unconfirmed"
        else:
            scan_mode = "frames"
        meta = cls(
            target_type=target_type,
            scan_mode=scan_mode,
            n_z_slices=n,
            x_pixels=x,
            y_pixels=y,
            laser_channel=laser_channel,
            data_channels=["amplitude", "phase"],
            dtype="float64" if phase_converted else original_dtype,
            file_format=file_format,
            raw_file_path=file_path,
            rdm_source_path=rdm_source_path,
            frame_ids=list(frame_ids),
            original_dtype=original_dtype,
            phase_encoding_used=phase_encoding_used,
            anomalous_frame_ids=list(anomalous_frame_ids or []),
            phase_converted=phase_converted,
            notes=notes,
        )
        if extra:
            # stash unknown HDF5 attrs into notes rather than inventing fields
            pass
        return meta


# --- Example: Chip dataset metadata ---
chip_meta = LFIMetadata(
    experiment_id  = "chip_borad1_001",
    target_type    = "Chip",
    timestamp      = "2024-06-01T00:00:00",
    scan_mode      = "z_stack",
    n_z_slices     = 70,
    z_start        = 530,
    z_end          = 599,
    x_pixels       = 400,
    y_pixels       = 1400,
    laser_type     = "THz",
    data_channels  = ["amplitude", "phase"],
    dtype          = "float64",
    file_format    = "hdf5_zstack",
    raw_file_path  = "/DataSets/Chip/borad1.hdf5",
    original_dtype = "float64",
    phase_encoding_used = "zstack_unconfirmed",
    phase_converted = False,
)

# --- Example: Skin dataset metadata ---
skin_meta = LFIMetadata(
    experiment_id  = "skin_20251128_001",
    target_type    = "Skin",
    timestamp      = "2025-11-28T10:14:26",
    scan_mode      = "frames",
    n_z_slices     = 1,
    x_pixels       = 1300,
    y_pixels       = 400,
    laser_type     = "QCL",
    laser_channel  = "QCL0/Left",
    data_channels  = ["amplitude", "phase", "peak_to_peak"],
    dtype          = "uint16",
    file_format    = "hdf5_frames",
    raw_file_path  = "/DataSets/Skin/2025-11-28_10-14-26.hdf5",
    original_dtype = "uint16",
    phase_encoding_used = "skin_unconfirmed",
    phase_converted = False,
    notes = "Skin phase formula unconfirmed - raw codes retained",
)


if __name__ == "__main__":
    chip_meta.to_json("chip_metadata.json")
    skin_meta.to_json("skin_metadata.json")
    print("\n=== Chip Metadata ===")
    print(json.dumps(chip_meta.to_dict(), indent=2))
    print("\n=== Skin Metadata ===")
    print(json.dumps(skin_meta.to_dict(), indent=2))
