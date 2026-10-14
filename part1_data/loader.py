"""
loader.py
HDF5 loaders - Part 1 core module

Provides:
    load_single_scan()    load a single scan file
    load_volume()         stack multi-file frames in numeric order into one volume
    load_multiple_scans() batch load (one LFIRawData per file; prefer iter_scans when many)
    get_z_slices()        list z-slice / frame ids in a file
    parse_experiment_details()  read experiment_details.txt (start/step → depth_mm)
    sort_data_paths()     sort paths by numeric frame id (not lexical)

Formats:
    hdf5_zstack     - top-level /amplitude/<z>, /phase/<z>
    hdf5_frames     - /frames/<id>/Image Data/<QCL>/<Side>/IMAGE_*
    hdf5_data_v2026 - /data/IMAGE_*/frame_<id>_qcl_<n>_<left|right>(2026 Vegetables)
    mat             - MATLAB .mat (v5/v7/v7.3); variable names in MAT_LAYOUTS (Leaf)

Downstream should use output_interface.py; do not depend on this module's internals.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np

try:
    from .data_types import LFIRawData
    from .metadata_schema import LFIMetadata
    from .phase_encoding import (
        convert_phase,
        get_phase_encoding,
        resolve_encoding_for_phase,
    )
except ImportError:  # running as scripts inside part1_data/
    from data_types import LFIRawData
    from metadata_schema import LFIMetadata
    from phase_encoding import (
        convert_phase,
        get_phase_encoding,
        resolve_encoding_for_phase,
    )


FrameId = Union[int, str]


class AnomalousOnlyError(ValueError):
    """All selected frames are anomalous and were excluded (nothing left to load)."""


DATA_SUFFIXES = (".hdf5", ".h5", ".mat")
_FRAME_IN_NAME = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"(\d+)")
_EXPERIMENT_DETAILS_NAMES = (
    "experiment_details.txt",
    "Experiment_Details.txt",
    "experiment_details.TXT",
)


# ── internal helpers ────────────────────────────────────────────────

# Canonical target names for the phase table. Aliases → canonical.
_TARGET_ALIASES: dict[str, str] = {
    "chip": "Chip",
    "leaf": "Leaf",
    "skin": "Skin",
    "marble": "Marble",
    "mushroom": "Mushroom",
    "sound": "Sound",
    "tomography": "Tomography",
    "beam profile": "Beam Profile",
    "beam_profile": "Beam Profile",
    "spectrum": "Spectrum",
    "vegetables": "Vegetables",
    "pcb": "PCB",
    "megabugs": "PCB",
}


def _match_target_token(text: str) -> Optional[str]:
    """
    Case-insensitive substring match in a path/filename fragment.
    Longer tokens first to avoid short-token false hits.
    """
    low = text.casefold().replace("_", " ")
    # also scan underscore form on raw casefold
    low_raw = text.casefold()
    candidates = sorted(_TARGET_ALIASES.keys(), key=len, reverse=True)
    for token in candidates:
        tok = token.casefold()
        if tok in low or tok in low_raw:
            return _TARGET_ALIASES[token]
    return None


def _infer_target_type(filepath: str) -> str:
    """
    Infer target type from path identity only (never from values).

    Priority:
      1. walk parents nearest-first (substring match; finds 2026_Vegetables)
      2. file stem contains a type keyword
      3. weak heuristics only when path identity is unknown (local loose samples)
         - frame-XX.hdf5 → PCB(only when ancestor paths do not contain Vegetables)
    """
    path = Path(filepath).resolve() if os.path.exists(filepath) else Path(filepath)

    # 1) ancestor folders: .../2026_Vegetables/frame-20.hdf5 → Vegetables
    for parent in path.parents:
        hit = _match_target_token(parent.name)
        if hit is not None:
            return hit

    # 2) stem itself contains a type token
    stem_hit = _match_target_token(path.stem)
    if stem_hit is not None:
        return stem_hit

    # 3) weak heuristics (path has no Vegetables/PCB identity)
    low = path.stem.casefold()
    if "borad" in low or "board" in low:
        return "Chip"
    # Note: real Vegetables files are also named frame-XX.hdf5; folder match must win first.
    # Here only covers Downloads PCB sample frame-3.hdf5 with no folder identity.
    if low.startswith("frame-") or low.startswith("frame"):
        return "PCB"
    stem = path.stem
    if len(stem) >= 19 and stem[4] == "-" and stem[7] == "-" and "_" in stem:
        return "Skin"
    return "Unknown"


def _attrs_to_dict(attrs) -> dict:
    out = {}
    for k, v in attrs.items():
        try:
            if hasattr(v, "tolist"):
                out[k] = v.tolist()
            else:
                out[k] = v.decode() if isinstance(v, (bytes, bytearray)) else v
        except Exception:
            out[k] = str(v)
    return out


def _sort_frame_keys(keys) -> list[str]:
    def key_fn(x: str):
        s = str(x)
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted((str(k) for k in keys), key=key_fn)


def frame_number_from_name(name: Union[str, Path]) -> Optional[int]:
    """Extract frame number from names like frame-21.hdf5 / frame_21_qcl_1_left."""
    stem = Path(name).stem if isinstance(name, (str, Path)) else str(name)
    m = _FRAME_IN_NAME.search(stem)
    if m:
        return int(m.group(1))
    nums = _ANY_NUMBER.findall(stem)
    return int(nums[-1]) if nums else None


def _path_sort_key(path: Union[str, Path]) -> tuple:
    """
    Numeric frame sort key. Lexical sort yields 100,110,...,140,21,22... and corrupts depth order.
    """
    p = Path(path)
    n = frame_number_from_name(p)
    if n is not None:
        return (0, n, p.name.casefold())
    return (1, 0, p.name.casefold())


def sort_data_paths(paths: Sequence[Union[str, Path]]) -> list[Path]:
    """Sort paths by numeric frame id in the filename."""
    return sorted((Path(p) for p in paths), key=_path_sort_key)


def find_data_files(data_dir: Union[str, Path], *, recursive: bool = True) -> list[Path]:
    """List data files under a directory, sorted by numeric frame id."""
    root = Path(data_dir)
    it = root.rglob("*") if recursive else root.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in DATA_SUFFIXES]
    return sort_data_paths(files)


def find_experiment_details(start: Union[str, Path]) -> Optional[Path]:
    """Search up to two parent levels for experiment_details.txt from a file or folder."""
    p = Path(start)
    dirs = [p if p.is_dir() else p.parent]
    if not p.is_dir():
        dirs.append(p.parent.parent)
    else:
        dirs.append(p.parent)
    seen = set()
    for d in dirs:
        d = d.resolve()
        if d in seen or not d.exists():
            continue
        seen.add(d)
        for name in _EXPERIMENT_DETAILS_NAMES:
            cand = d / name
            if cand.is_file():
                return cand
    return None


def _normalize_details_key(key: str) -> str:
    """Casefold, drop parenthetical units like (mm), unify spaces/underscores."""
    k = key.strip().casefold().replace("-", " ")
    k = re.sub(r"\([^)]*\)", " ", k)  # "start z position (mm)" → "start z position"
    k = re.sub(r"\s+", " ", k).strip()
    return k


def parse_experiment_details(path: Union[str, Path]) -> dict:
    """
    Parse experiment_details.txt.

    Detect start/step (mm) fields; keys case-insensitive; supports `key: value` / `key = value`.
    Recognises lab wording such as ``start z position (mm)`` and the common typo
    ``imcrements (mm)`` (= increments).
    Returns e.g. {"z_start_mm": 0.0, "z_step_mm": 0.05, "raw": {...}, "source_path": "..."}.
    """
    fp = Path(path)
    text = fp.read_text(encoding="utf-8", errors="replace")
    raw: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
        elif "=" in line:
            k, v = line.split("=", 1)
        else:
            continue
        key = _normalize_details_key(k)
        key_us = key.replace(" ", "_")
        val = v.strip()
        raw[key] = val
        raw[key_us] = val

    aliases = {
        "z_start_mm": (
            "z_start_mm", "z_start", "start", "start_mm", "start position",
            "start z position", "starting z position", "start_z_position",
            "starting position", "start_position", "starting_position",
            "z0", "z_0", "initial position", "initial_position",
        ),
        "z_step_mm": (
            "z_step_mm", "z_step", "step", "step_mm", "step size", "step_size",
            "dz", "delta_z", "increment", "increments", "imcrements",  # typo seen on RDM
            "z_increment", "z increments", "z imcrements", "increment size",
        ),
        "first_frame_id": (
            "first_frame_id", "first_frame", "start_frame", "frame_start",
            "reference_frame", "first frame", "starting frame",
        ),
        "n_frames": (
            "n_frames", "num_frames", "number of frames", "frames", "n_steps",
        ),
    }

    def _as_float(key_group: str) -> Optional[float]:
        for alias in aliases[key_group]:
            if alias in raw:
                try:
                    return float(raw[alias].split()[0].replace(",", ""))
                except (ValueError, IndexError):
                    continue
        return None

    def _as_int(key_group: str) -> Optional[int]:
        v = _as_float(key_group)
        return int(v) if v is not None else None

    out = {
        "source_path": str(fp.resolve()),
        "raw": raw,
        "z_start_mm": _as_float("z_start_mm"),
        "z_step_mm": _as_float("z_step_mm"),
        "first_frame_id": _as_int("first_frame_id"),
        "n_frames": _as_int("n_frames"),
    }
    return out


def depth_mm_for_frames(
    frame_ids: Sequence[FrameId],
    details: Optional[dict],
    *,
    reference_frame_id: Optional[int] = None,
) -> Optional[list[float]]:
    """
    depth_mm[i] = z_start_mm + (frame_id - reference) * z_step_mm.

    Reference priority:
      1) details['first_frame_id'] from experiment_details.txt
      2) reference_frame_id (e.g. min frame id present on disk, including skipped
         anomalous frames such as Vegetables 20)
      3) min(frame_ids) — last resort; wrong if anomalous frames were dropped
         from this batch before calling
    """
    if not details:
        return None
    start, step = details.get("z_start_mm"), details.get("z_step_mm")
    if start is None or step is None:
        return None
    ids = [int(f) for f in frame_ids]
    ref = details.get("first_frame_id")
    if ref is None:
        ref = reference_frame_id
    if ref is None:
        ref = min(ids) if ids else 0
    return [round(float(start + (fid - int(ref)) * step), 12) for fid in ids]


def _attach_depth_metadata(
    metadata: dict,
    frame_ids: Sequence[FrameId],
    search_from: Union[str, Path],
    *,
    reference_frame_id: Optional[int] = None,
) -> dict:
    """If experiment_details.txt is found, set z_start_mm / z_step_mm / depth_mm."""
    details_path = find_experiment_details(search_from)
    if details_path is None:
        metadata.setdefault("depth_mm", None)
        metadata.setdefault("experiment_details_path", "")
        return metadata
    details = parse_experiment_details(details_path)
    metadata["experiment_details_path"] = details["source_path"]
    metadata["z_start_mm"] = details.get("z_start_mm")
    metadata["z_step_mm"] = details.get("z_step_mm")
    metadata["depth_mm"] = depth_mm_for_frames(
        frame_ids, details, reference_frame_id=reference_frame_id
    )
    # Record which reference was used when the file omits first_frame_id
    if details.get("first_frame_id") is not None:
        metadata["depth_reference_frame_id"] = int(details["first_frame_id"])
    elif reference_frame_id is not None:
        metadata["depth_reference_frame_id"] = int(reference_frame_id)
        warnings.warn(
            f"[loader] experiment_details.txt has no first_frame_id; "
            f"depth origin assumed to be min frame on disk "
            f"(depth_reference_frame_id={int(reference_frame_id)})",
            stacklevel=3,
        )
    if metadata["depth_mm"] is None:
        metadata["notes"] = (
            (metadata.get("notes") or "")
            + "; found experiment_details.txt but could not parse z_start_mm/z_step_mm"
        ).lstrip("; ")
    return metadata


def _apply_phase_encoding(
    phase_raw: np.ndarray,
    enc_name: str,
    target_type: str,
    convert_phase_flag: bool,
    original_dtype: str,
    has_anomalous: bool = False,
    inplace: bool = False,
):
    """
    Convert with the chosen encoding name; keep raw codes if unconfirmed.

    has_anomalous: Skip conversion when anomalous frames are present. Their codes are outside
                   the valid range (e.g. Vegetables frame 20 is [-32768, 32755]),
                   forcing conversion only trips range checks or yields nonsense radians.
    """
    if has_anomalous and convert_phase_flag:
        enc = get_phase_encoding(enc_name)
        warnings.warn(
            f"[loader] target={target_type}: contains anomalous frames; phase kept as raw codes "
            f"(dtype={original_dtype}), not converted with '{enc.name}' conversion.",
            stacklevel=2,
        )
        return np.asarray(phase_raw), enc, False

    phase, enc, converted = convert_phase(
        phase_raw, enc_name, convert=convert_phase_flag, inplace=inplace
    )
    if not converted and not enc.confirmed:
        warnings.warn(
            f"[loader] target={target_type}: phase encoding '{enc.name}' unconfirmed, "
            f"keep raw codes (dtype={original_dtype}), not converted to radians.",
            stacklevel=2,
        )
    return phase, enc, converted


def _detect_format(f: h5py.File) -> str:
    keys = set(f.keys())
    if "amplitude" in keys and "phase" in keys:
        return "hdf5_zstack"
    if "frames" in keys:
        return "hdf5_frames"
    if "data" in keys and isinstance(f["data"], h5py.Group) and "IMAGE_PHASE" in f["data"]:
        return "hdf5_data_v2026"
    raise ValueError(
        f"Unrecognized HDF5 layout. Need top-level amplitude+phase, frames,"
        f"or data/IMAGE_PHASE.\nActual keys: {list(keys)}"
    )


def _select_frame_ids(
    selected: list[str],
    target_type: str,
    exclude_anomalous: bool,
) -> tuple[list[str], list[int]]:
    """
    Flag anomalous frames and optionally exclude them.

    Returns:
        (load_ids, marked_anomalous)  marked_anomalous anomalous frames present in selected
    """
    # Anomalous frames (e.g. Vegetables 20) apply only for Vegetables identity
    anomalous: set[int] = set()
    if target_type == "Vegetables":
        veg = get_phase_encoding("vegetables_int16")
        anomalous = {int(x) for x in veg.anomalous_frame_ids}

    marked_anomalous = []
    load_ids = []
    for fid in selected:
        if fid.isdigit() and int(fid) in anomalous:
            marked_anomalous.append(int(fid))
            if exclude_anomalous:
                warnings.warn(
                    f"[loader] skipanomalous/startup frame frame_id={fid} (target={target_type})",
                    stacklevel=3,
                )
                continue
            warnings.warn(
                f"[loader] frame_id={fid} marked as anomalous/startup frame,"
                f" but still loaded because exclude_anomalous=False",
                stacklevel=3,
            )
        load_ids.append(fid)

    if not load_ids:
        raise AnomalousOnlyError(
            f"nothing left to load (all selected frames anomalous and excluded). "
            f"selected={selected}, anomalous={sorted(anomalous)}; "
            f"pass exclude_anomalous=False to inspect anomalous frames"
        )
    return load_ids, marked_anomalous


def _read_file_metadata(f: h5py.File) -> dict:
    """File-embedded metadata: root attrs, scan_identifier, thz/info acquisition."""
    extra = _attrs_to_dict(f.attrs)
    if "scan_identifier" in f:
        sid = f["scan_identifier"]
        if isinstance(sid, h5py.Dataset):
            v = sid[()]
            extra["scan_identifier"] = v.decode(errors="replace") if isinstance(v, bytes) else str(v)
        else:
            extra["scan_identifier"] = _attrs_to_dict(sid.attrs)
    if "thz" in f and "info" in f["thz"]:
        extra["acquisition"] = _read_scalar_tree(f["thz"]["info"])
    return extra


def _read_scalar_tree(group: h5py.Group) -> dict:
    """Recursively read scalar datasets under a group into a nested dict."""
    out = {}
    for k, obj in group.items():
        if isinstance(obj, h5py.Group):
            out[k] = _read_scalar_tree(obj)
        elif obj.shape == ():
            v = obj[()]
            if isinstance(v, (bytes, bytearray)):
                v = v.decode(errors="replace")
            elif hasattr(v, "item"):
                v = v.item()
            out[k] = v
    return out


def _qcl_sort_key(name: str) -> tuple:
    """Sort QCL group names by numeric suffix (QCL0 before QCL1)."""
    digits = re.sub(r"\D", "", str(name))
    return (int(digits) if digits else 10**9, str(name))


def _pick_qcl_side(image_data_group: h5py.Group,
                   qcl: Optional[str],
                   side: Optional[str]) -> tuple[str, str, h5py.Group]:
    """
    Pick QCL / Left|Right leaf group automatically or from args.

    Default channel (same rule as v2026): lowest-numbered QCL present + Left
    when available. Pass qcl=/side= explicitly to override.
    """
    qcl_keys = list(image_data_group.keys())
    if not qcl_keys:
        raise ValueError("Image Data has no QCL group")

    if qcl is None:
        qcl = sorted(qcl_keys, key=_qcl_sort_key)[0]
    if qcl not in image_data_group:
        raise ValueError(f"QCL group not found '{qcl}', available: {qcl_keys}")

    side_group = image_data_group[qcl]
    side_keys = list(side_group.keys())
    if side is None:
        side = "Left" if "Left" in side_keys else side_keys[0]
    if side not in side_group:
        raise ValueError(f"side not found '{side}', available: {side_keys}")

    return qcl, side, side_group[side]


def _anomalous_ids_for_target(target_type: str) -> set[int]:
    """Frame ids marked anomalous for this dataset identity (empty if none)."""
    if target_type != "Vegetables":
        return set()
    enc = get_phase_encoding("vegetables_int16")
    return {int(x) for x in enc.anomalous_frame_ids}


def _read_hdf5_zstack(filepath: str, f: h5py.File, target_type: str,
                      phase_encoding_name: Optional[str],
                      convert_phase_flag: bool,
                      rdm_source_path: str) -> LFIRawData:
    amp_group = f["amplitude"]
    phase_group = f["phase"]

    z_keys = _sort_frame_keys(amp_group.keys())
    z_indices = [int(k) if str(k).isdigit() else k for k in z_keys]

    amp_slices = [amp_group[k][:] for k in z_keys]
    phase_slices = [phase_group[k][:] for k in z_keys]
    amplitude = np.stack(amp_slices, axis=0)
    phase_raw = np.stack(phase_slices, axis=0)
    original_dtype = str(phase_raw.dtype)

    enc_name = resolve_encoding_for_phase(
        target_type, phase_raw, explicit_name=phase_encoding_name
    )
    phase, enc, converted = _apply_phase_encoding(
        phase_raw,
        enc_name,
        target_type,
        convert_phase_flag,
        original_dtype,
    )

    file_attrs = _attrs_to_dict(f.attrs)
    meta = LFIMetadata.from_load_result(
        target_type=target_type,
        file_path=str(Path(filepath).resolve()),
        file_format="hdf5_zstack",
        amplitude_shape=amplitude.shape,
        frame_ids=list(z_indices),
        original_dtype=original_dtype,
        phase_encoding_used=enc.name,
        phase_converted=converted,
        anomalous_frame_ids=list(enc.anomalous_frame_ids),
        rdm_source_path=rdm_source_path,
        notes=enc.notes,
    )

    md = {**file_attrs, **meta.to_dict()}
    _attach_depth_metadata(md, z_indices, filepath)
    return LFIRawData(
        amplitude=np.asarray(amplitude, dtype=np.float64),
        phase=np.asarray(phase, dtype=np.float64 if converted else phase.dtype),
        z_indices=list(z_indices),
        target_type=target_type,
        file_path=str(Path(filepath).resolve()),
        metadata=md,
    )


def _read_hdf5_frames(
    filepath: str,
    f: h5py.File,
    target_type: str,
    phase_encoding_name: Optional[str],
    convert_phase_flag: bool,
    rdm_source_path: str,
    *,
    frame_ids: Optional[Sequence[FrameId]] = None,
    qcl: Optional[str] = None,
    side: Optional[str] = None,
    exclude_anomalous: bool = True,
) -> LFIRawData:
    """
    Read frames layout:
        frames/<id>/Image Data/<QCL>/<Side>/IMAGE_AMPLITUDE|IMAGE_PHASE
    """
    frames_group = f["frames"]
    all_ids = _sort_frame_keys(frames_group.keys())
    if not all_ids:
        raise ValueError(f"frames/ is empty: {filepath}")

    if frame_ids is None:
        selected = all_ids
    else:
        selected = [str(i) for i in frame_ids]
        missing = [s for s in selected if s not in frames_group]
        if missing:
            raise KeyError(f"requested frame_ids not found: {missing}")

    load_ids, marked_anomalous = _select_frame_ids(
        selected, target_type, exclude_anomalous
    )
    loaded_anomalous = [i for i in marked_anomalous if str(i) in load_ids]

    amp_list = []
    phase_list = []
    used_qcl = used_side = None
    original_dtype = None

    for fid in load_ids:
        image_data = frames_group[fid]["Image Data"]
        used_qcl, used_side, leaf = _pick_qcl_side(image_data, qcl, side)
        if "IMAGE_AMPLITUDE" not in leaf or "IMAGE_PHASE" not in leaf:
            raise ValueError(
                f"frames/{fid}/Image Data/{used_qcl}/{used_side} "
                f"missing IMAGE_AMPLITUDE or IMAGE_PHASE; got: {list(leaf.keys())}"
            )
        amp = leaf["IMAGE_AMPLITUDE"][:]
        ph = leaf["IMAGE_PHASE"][:]
        if original_dtype is None:
            original_dtype = str(ph.dtype)
        amp_list.append(amp)
        phase_list.append(ph)

    amplitude = np.stack(amp_list, axis=0)
    phase_raw = np.stack(phase_list, axis=0)

    enc_name = resolve_encoding_for_phase(
        target_type, phase_raw, explicit_name=phase_encoding_name
    )
    phase, enc, converted = _apply_phase_encoding(
        phase_raw,
        enc_name,
        target_type,
        convert_phase_flag,
        original_dtype or "",
        has_anomalous=bool(loaded_anomalous),
    )

    z_indices = [int(k) if k.isdigit() else k for k in load_ids]
    laser_channel = f"{used_qcl}/{used_side}" if used_qcl else ""

    file_attrs = _read_file_metadata(f)

    meta = LFIMetadata.from_load_result(
        target_type=target_type,
        file_path=str(Path(filepath).resolve()),
        file_format="hdf5_frames",
        amplitude_shape=amplitude.shape,
        frame_ids=list(z_indices),
        original_dtype=original_dtype or "",
        phase_encoding_used=enc.name,
        phase_converted=converted,
        anomalous_frame_ids=marked_anomalous,
        laser_channel=laser_channel,
        rdm_source_path=rdm_source_path,
        notes=enc.notes,
    )

    md = {**file_attrs, **meta.to_dict()}
    _attach_depth_metadata(md, z_indices, filepath)
    return LFIRawData(
        amplitude=np.asarray(amplitude, dtype=np.float64),
        phase=np.asarray(phase, dtype=np.float64 if converted else phase.dtype),
        z_indices=list(z_indices),
        target_type=target_type,
        file_path=str(Path(filepath).resolve()),
        metadata=md,
    )


_V2026_NAME = re.compile(r"^frame_(\d+)_qcl_(\d+)_(left|right)$")


def _index_v2026(phase_group: h5py.Group) -> dict:
    """{(frame_id, qcl_num, side): dataset_name}, skip *_BMP previews."""
    index = {}
    for name in phase_group.keys():
        m = _V2026_NAME.match(name)
        if m:
            index[(m.group(1), m.group(2), m.group(3))] = name
    return index


def _read_hdf5_data_v2026(
    filepath: str,
    f: h5py.File,
    target_type: str,
    phase_encoding_name: Optional[str],
    convert_phase_flag: bool,
    rdm_source_path: str,
    *,
    frame_ids: Optional[Sequence[FrameId]] = None,
    qcl: Optional[str] = None,
    side: Optional[str] = None,
    exclude_anomalous: bool = True,
    include_waveform: bool = False,
) -> LFIRawData:
    """
    Read 2026 export layout (e.g. 2026_Vegetables/frame-XX.hdf5):
        data/IMAGE_AMPLITUDE/frame_<id>_qcl_<n>_<left|right>
        data/IMAGE_PHASE/frame_<id>_qcl_<n>_<left|right>
        data/RX_LINE_OF_BEST_FIT/frame_<id>_<left|right>  (H, W, 184) optional
        thz/info/...   acquisition params (environment, laser freq, mcp registers)
    *_BMP are uint8 previews; skipped by default.
    """
    data_group = f["data"]
    if "IMAGE_AMPLITUDE" not in data_group:
        raise ValueError(f"data/ missing IMAGE_AMPLITUDE under: {list(data_group.keys())}")
    amp_group = data_group["IMAGE_AMPLITUDE"]
    phase_group = data_group["IMAGE_PHASE"]

    index = _index_v2026(phase_group)
    if not index:
        raise ValueError(
            f"data/IMAGE_PHASE has no frame_<id>_qcl_<n>_<side> datasets: "
            f"{list(phase_group.keys())}"
        )

    qcl_num = re.sub(r"\D", "", str(qcl)) if qcl is not None else None
    side_key = side.lower() if side is not None else None
    if qcl_num is None:
        qcl_num = sorted({k[1] for k in index}, key=int)[0]
    if side_key is None:
        side_key = "left"

    available = _sort_frame_keys({k[0] for k in index if k[1] == qcl_num and k[2] == side_key})
    if not available:
        channels = sorted({f"qcl_{k[1]}_{k[2]}" for k in index})
        raise ValueError(f"channel not found qcl_{qcl_num}_{side_key}, available: {channels}")

    if frame_ids is None:
        selected = available
    else:
        selected = [str(i) for i in frame_ids]
        missing = [s for s in selected if s not in available]
        if missing:
            raise KeyError(f"requested frame_ids not found: {missing}")

    load_ids, marked_anomalous = _select_frame_ids(
        selected, target_type, exclude_anomalous
    )
    loaded_anomalous = [i for i in marked_anomalous if str(i) in load_ids]

    amp_list = []
    phase_list = []
    wave_list = []
    wave_group = data_group.get("RX_LINE_OF_BEST_FIT") if include_waveform else None
    for fid in load_ids:
        name = index[(fid, qcl_num, side_key)]
        if name not in amp_group:
            raise ValueError(f"IMAGE_AMPLITUDE missing amplitude dataset matching phase: {name}")
        amp_list.append(amp_group[name][:])
        phase_list.append(phase_group[name][:])
        if wave_group is not None:
            wname = f"frame_{fid}_{side_key}"
            if wname not in wave_group:
                raise KeyError(
                    f"include_waveform=True but cannot find RX_LINE_OF_BEST_FIT/{wname}, "
                    f"available: {list(wave_group.keys())[:8]}..."
                )
            wave_list.append(wave_group[wname][:])

    amplitude = np.stack(amp_list, axis=0)
    phase_raw = np.stack(phase_list, axis=0)
    original_dtype = str(phase_raw.dtype)
    waveform = np.stack(wave_list, axis=0) if wave_list else None

    enc_name = resolve_encoding_for_phase(
        target_type, phase_raw, explicit_name=phase_encoding_name
    )
    phase, enc, converted = _apply_phase_encoding(
        phase_raw,
        enc_name,
        target_type,
        convert_phase_flag,
        original_dtype,
        has_anomalous=bool(loaded_anomalous),
    )

    z_indices = [int(k) for k in load_ids]
    laser_channel = f"QCL{qcl_num}/{side_key.capitalize()}"

    extra = _read_file_metadata(f)
    channels = ["amplitude", "phase"]
    if waveform is not None:
        channels.append("waveform")

    meta = LFIMetadata.from_load_result(
        target_type=target_type,
        file_path=str(Path(filepath).resolve()),
        file_format="hdf5_data_v2026",
        amplitude_shape=amplitude.shape,
        frame_ids=list(z_indices),
        original_dtype=original_dtype,
        phase_encoding_used=enc.name,
        phase_converted=converted,
        anomalous_frame_ids=marked_anomalous,
        laser_channel=laser_channel,
        rdm_source_path=rdm_source_path,
        notes=enc.notes,
    )
    md = {**extra, **meta.to_dict(), "data_channels": channels}
    if waveform is not None:
        md["waveform_shape"] = list(waveform.shape)
        md["waveform_dtype"] = str(waveform.dtype)
    _attach_depth_metadata(md, z_indices, filepath)

    return LFIRawData(
        amplitude=np.asarray(amplitude, dtype=np.float64),
        phase=np.asarray(phase, dtype=np.float64 if converted else phase.dtype),
        z_indices=list(z_indices),
        target_type=target_type,
        file_path=str(Path(filepath).resolve()),
        metadata=md,
        waveform=waveform,
    )


# .mat layouts: explicit by dataset identity; do not guess phase from array shape.
# Leaf verified on RDM leaf1.mat: L_A / L_P are int16 (1200, 400, 475).
# Same file also has L_P_U / L_P_U_M (float64, 1200×400×91, first 91 unwrapped frames and a processed variant),
# these are processed products and are not loaded as raw data.
# Sound: NOT listed — no real RDM file verified; callers must pass mat_vars=.
# stack_axis: frame/slice axis (MATLAB H×W×N → last axis).
MAT_LAYOUTS: dict[str, dict] = {
    "Leaf": {"amplitude": "L_A", "phase": "L_P", "stack_axis": -1},
}


def _mat_variables(filepath: str) -> dict:
    """{name: MATLAB-view shape} without loading data."""
    if h5py.is_hdf5(filepath):
        with h5py.File(filepath, "r") as f:
            return {
                k: tuple(reversed(v.shape))
                for k, v in f.items()
                if isinstance(v, h5py.Dataset) and not k.startswith("#")
            }
    import scipy.io
    return {name: tuple(shape) for name, shape, _ in scipy.io.whosmat(filepath)}


def _normalize_mat_frame_ids(
    frame_ids: Optional[Sequence[FrameId]],
    n_frames: int,
) -> Optional[list[int]]:
    """Normalize frame_ids to 0-based stack indices; None means all."""
    if frame_ids is None:
        return None
    out = []
    for fid in frame_ids:
        i = int(fid)
        if i < 0 or i >= n_frames:
            raise IndexError(
                f".mat frame_id={i} out of range [0, {n_frames - 1}]"
            )
        out.append(i)
    return out


def _load_mat_arrays(
    filepath: str,
    names: Sequence[str],
    *,
    stack_axis: int = -1,
    frame_ids: Optional[Sequence[int]] = None,
) -> dict:
    """
    Load named variables; transpose v7.3 (HDF5) back to MATLAB axis order.

    frame_ids when set:
      - v7.3: hyperslab along stack axis; peak RAM ≈ selected frames
      - v5/v7: scipy still loads the whole variable, then slice and free (peak still high; result small)
    """
    if h5py.is_hdf5(filepath):
        out = {}
        with h5py.File(filepath, "r") as f:
            for n in names:
                dset = f[n]
                # h5py stores MATLAB arrays transposed; MATLAB last axis ↔ h5py first axis
                if frame_ids is not None and dset.ndim == 3 and stack_axis in (-1, 2):
                    parts = [dset[i] for i in frame_ids]
                    # Each block shape (W, H) in h5py; MATLAB (H,W,N) → h5py (N,W,H)
                    # dset[i] is (W, H); .T → (H, W); stack → (n, H, W) after moveaxis later
                    # Keep MATLAB order (H,W,n): transpose each then stack on last axis
                    mats = [np.asarray(p).T for p in parts]  # each (H, W)
                    out[n] = np.stack(mats, axis=-1)
                elif frame_ids is not None and dset.ndim == 3:
                    full = dset[()].T
                    out[n] = np.take(full, frame_ids, axis=stack_axis)
                else:
                    out[n] = dset[()].T
                    if frame_ids is not None and out[n].ndim == 3:
                        out[n] = np.take(out[n], frame_ids, axis=stack_axis)
        return out

    import scipy.io
    loaded = scipy.io.loadmat(filepath, variable_names=list(names))
    if frame_ids is None:
        return {n: loaded[n] for n in names}
    out = {}
    for n in names:
        arr = np.asarray(loaded[n])
        if arr.ndim == 3:
            arr = np.take(arr, frame_ids, axis=stack_axis)
        out[n] = arr
        del loaded[n]
    del loaded
    return out


def _to_stack(arr: np.ndarray, stack_axis: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return np.moveaxis(arr, stack_axis, 0)
    raise ValueError(f".mat variable '{name}' has shape {arr.shape}; only 2D or 3D supported")


def _read_mat(
    filepath: str,
    *,
    target_type: Optional[str] = None,
    phase_encoding: Optional[str] = None,
    convert_phase_flag: bool = True,
    rdm_source_path: str = "",
    mat_vars: Optional[dict] = None,
    frame_ids: Optional[Sequence[FrameId]] = None,
) -> LFIRawData:
    """
    Read .mat (v5/v7 via scipy, v7.3 via h5py).

    mat_vars can override default layout, e.g. {"amplitude": "A", "phase": "P", "stack_axis": 0}.
    frame_ids: load only these 0-based stack indices. v7.3 can truly stream; v5 still briefly loads the whole variable.
    """
    tt = target_type or _infer_target_type(filepath)
    available = _mat_variables(filepath)
    layout = {**(MAT_LAYOUTS.get(tt) or {}), **(mat_vars or {})}

    if "amplitude" not in layout or "phase" not in layout:
        raise ValueError(
            f"target={tt} has no verified .mat layout; pass mat_vars="
            f"{{'amplitude': '...', 'phase': '...'}} "
            f"(Sound is unverified — do not guess variable names). "
            f"Variables in file: {available}"
        )
    missing = [layout[k] for k in ("amplitude", "phase") if layout[k] not in available]
    if missing:
        raise ValueError(
            f"variables not found in file {missing}(target={tt}  layout: {layout})."
            f"variables in file: {available}; use mat_vars= to set the correct names"
        )

    stack_axis = layout.get("stack_axis", -1)
    phase_shape = available[layout["phase"]]
    n_frames = 1 if len(phase_shape) == 2 else phase_shape[stack_axis]
    selected = _normalize_mat_frame_ids(frame_ids, int(n_frames))

    if selected is not None and not h5py.is_hdf5(filepath):
        warnings.warn(
            "[loader] Leaf .mat is v5/v7: scipy still loads the whole variable then slices;"
            "peak RAM is close to a full load, but the returned arrays keep only selected frames."
            "For truly low memory, save the file as MATLAB v7.3.",
            stacklevel=3,
        )

    arrays = _load_mat_arrays(
        filepath,
        [layout["amplitude"], layout["phase"]],
        stack_axis=stack_axis,
        frame_ids=selected,
    )
    amplitude = _to_stack(arrays[layout["amplitude"]], stack_axis, layout["amplitude"])
    phase_raw = _to_stack(arrays[layout["phase"]], stack_axis, layout["phase"])
    if amplitude.shape != phase_raw.shape:
        raise ValueError(
            f"amplitude {layout['amplitude']}{amplitude.shape} vs phase "
            f"{layout['phase']}{phase_raw.shape} shape mismatch"
        )
    original_dtype = str(phase_raw.dtype)

    # .mat a single variable can be multi-GB; discard raw codes after convert-in-place
    enc_name = resolve_encoding_for_phase(tt, phase_raw, explicit_name=phase_encoding)
    phase, enc, converted = _apply_phase_encoding(
        phase_raw, enc_name, tt, convert_phase_flag, original_dtype, inplace=True
    )
    del arrays, phase_raw

    z_indices = list(selected) if selected is not None else list(range(amplitude.shape[0]))
    notes = enc.notes
    if tt == "Leaf":
        notes = (
            f"{notes}; Karl, email, 2026-10-14: "
            "third axis is a time series, not depth"
        ).lstrip("; ")
    if selected is not None:
        notes = f"{notes}; loaded only {len(selected)}/{n_frames} frames".lstrip("; ")

    meta = LFIMetadata.from_load_result(
        target_type=tt,
        file_path=str(Path(filepath).resolve()),
        file_format="mat",
        amplitude_shape=amplitude.shape,
        frame_ids=z_indices,
        original_dtype=original_dtype,
        phase_encoding_used=enc.name,
        phase_converted=converted,
        rdm_source_path=rdm_source_path,
        notes=notes,
    )
    md = {
        "mat_variables": {k: list(v) for k, v in available.items()},
        "mat_layout": layout,
        "mat_total_frames": int(n_frames),
        **meta.to_dict(),
    }
    # Leaf third axis is time, not depth — never attach experiment_details depth_mm.
    if tt == "Leaf":
        md.setdefault("depth_mm", None)
        md.setdefault("experiment_details_path", "")
    else:
        _attach_depth_metadata(md, z_indices, filepath)

    return LFIRawData(
        amplitude=np.asarray(amplitude, dtype=np.float64),
        phase=np.asarray(phase, dtype=np.float64 if converted else phase.dtype),
        z_indices=z_indices,
        target_type=tt,
        file_path=str(Path(filepath).resolve()),
        metadata=md,
    )


def _read_hdf5(
    filepath: str,
    *,
    target_type: Optional[str] = None,
    phase_encoding: Optional[str] = None,
    convert_phase_flag: bool = True,
    rdm_source_path: str = "",
    frame_ids: Optional[Sequence[FrameId]] = None,
    qcl: Optional[str] = None,
    side: Optional[str] = None,
    exclude_anomalous: bool = True,
    include_waveform: bool = False,
) -> LFIRawData:
    """Dispatch to z-stack or frames reader by file structure."""
    tt = target_type or _infer_target_type(filepath)

    with h5py.File(filepath, "r") as f:
        fmt = _detect_format(f)
        if fmt == "hdf5_zstack":
            unsupported = [
                name
                for name, val in (
                    ("frame_ids", frame_ids),
                    ("qcl", qcl),
                    ("side", side),
                )
                if val is not None
            ]
            if unsupported:
                raise ValueError(
                    f"hdf5_zstack ignores channel/frame subsetting; "
                    f"do not pass {unsupported} (file={filepath}). "
                    f"Omit these kwargs, or use a frames/v2026 file."
                )
            return _read_hdf5_zstack(
                filepath, f, tt, phase_encoding, convert_phase_flag, rdm_source_path
            )
        if fmt == "hdf5_data_v2026":
            return _read_hdf5_data_v2026(
                filepath,
                f,
                tt,
                phase_encoding,
                convert_phase_flag,
                rdm_source_path,
                frame_ids=frame_ids,
                qcl=qcl,
                side=side,
                exclude_anomalous=exclude_anomalous,
                include_waveform=include_waveform,
            )
        return _read_hdf5_frames(
            filepath,
            f,
            tt,
            phase_encoding,
            convert_phase_flag,
            rdm_source_path,
            frame_ids=frame_ids,
            qcl=qcl,
            side=side,
            exclude_anomalous=exclude_anomalous,
        )


# ── Public API ──────────────────────────────────────────────────

def load_single_scan(
    filepath: str,
    *,
    target_type: Optional[str] = None,
    phase_encoding: Optional[str] = None,
    convert_phase: bool = True,
    rdm_source_path: str = "",
    frame_ids: Optional[Sequence[FrameId]] = None,
    qcl: Optional[str] = None,
    side: Optional[str] = None,
    exclude_anomalous: bool = True,
    mat_vars: Optional[dict] = None,
    include_waveform: bool = False,
) -> LFIRawData:
    """
    Load one scan (HDF5 auto-detects z-stack / frames / 2026; also .mat).

    Args:
        filepath: path to .hdf5 / .h5 / .mat
        target_type: override inferred target type (affects default phase encoding)
        phase_encoding: override encoding name; see phase_encoding.PHASE_ENCODINGS
        convert_phase: False keeps raw codes; no radian conversion
        rdm_source_path: RDM source path (stored in metadata)
        frame_ids: load only these frames; frames/v2026 use file ids; .mat uses 0-based stack indices
        qcl / side: frames / v2026 only; None = lowest-numbered QCL + Left.
                    Not valid for z-stack (raises ValueError if passed).
        exclude_anomalous: exclude anomalous frames listed in the encoding table (e.g. Vegetables 20)
        mat_vars: .mat only; required for Sound (no verified layout). Overrides MAT_LAYOUTS.
        include_waveform: True loads RX_LINE_OF_BEST_FIT (~H×W×184, large)

    Returns:
        LFIRawData
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    suffix = Path(filepath).suffix.lower()
    if suffix not in (".hdf5", ".h5", ".mat"):
        raise ValueError(f"Unsupported file format, please provide .hdf5 / .h5 / .mat files: {filepath}")

    print(f"[load_single_scan] Loading: {filepath}")
    if suffix == ".mat":
        data = _read_mat(
            filepath,
            target_type=target_type,
            phase_encoding=phase_encoding,
            convert_phase_flag=convert_phase,
            rdm_source_path=rdm_source_path,
            mat_vars=mat_vars,
            frame_ids=frame_ids,
        )
    else:
        data = _read_hdf5(
            filepath,
            target_type=target_type,
            phase_encoding=phase_encoding,
            convert_phase_flag=convert_phase,
            rdm_source_path=rdm_source_path,
            frame_ids=frame_ids,
            qcl=qcl,
            side=side,
            exclude_anomalous=exclude_anomalous,
            include_waveform=include_waveform,
        )
    enc = data.metadata.get("phase_encoding_used", "?")
    converted = data.metadata.get("phase_converted", False)
    print(
        f"[load_single_scan] Loaded → {data.amplitude.shape} | "
        f"format={data.metadata.get('file_format')} | "
        f"phase_encoding={enc} converted={converted}"
    )
    return data


def load_volume(
    data_dir: str,
    *,
    target_type: Optional[str] = None,
    frame_ids: Optional[Sequence[FrameId]] = None,
    exclude_anomalous: bool = True,
    include_waveform: bool = False,
    convert_phase: bool = True,
    qcl: Optional[str] = None,
    side: Optional[str] = None,
    phase_encoding: Optional[str] = None,
    rdm_source_path: str = "",
    recursive: bool = False,
) -> LFIRawData:
    """
    Stack many single-frame files in a folder into one volume by numeric frame id.

    For 2026 Vegetables: one depth scan split into frame-XX.hdf5; must not stack with lexical sort.

    Args:
        data_dir: folder of frame-*.hdf5 (may also contain experiment_details.txt)
        frame_ids: stack only these frame ids; None = all in dir (still numeric order)
        exclude_anomalous: skip Vegetables frame 20 by default
        include_waveform: whether to load RX_LINE_OF_BEST_FIT (huge RAM for ~136 frames)
        recursive: recurse into subfolders; default is this level only to avoid other datasets

    Returns:
        one LFIRawData with amplitude/phase shape (n_frames, H, W);
        metadata['depth_mm'] set when experiment_details.txt can be parsed.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = find_data_files(root, recursive=recursive)
    if not files:
        raise RuntimeError(f"No data files found under {data_dir}")

    # filter by frame number parsed from filenames
    file_by_frame: list[tuple[int, Path]] = []
    for fp in files:
        n = frame_number_from_name(fp)
        if n is None:
            continue
        file_by_frame.append((n, fp))
    if not file_by_frame:
        raise RuntimeError(
            f"{data_dir} filenames have no parseable frame id (expect frame-21.hdf5)"
        )

    # Depth origin must use the full folder catalog BEFORE frame_ids filtering
    # (and before anomalous skip). Otherwise load_volume(..., frame_ids=[22,23])
    # would wrongly treat 22 as z_start when first_frame_id is omitted.
    catalog_min_frame = min(n for n, _ in file_by_frame)

    if frame_ids is not None:
        want = {int(x) for x in frame_ids}
        file_by_frame = [(n, p) for n, p in file_by_frame if n in want]
        missing = want - {n for n, _ in file_by_frame}
        if missing:
            raise KeyError(f"these frame files are missing in the directory: {sorted(missing)}")

    # keep numeric order (find_data_files already sorted; preserve after filter)
    file_by_frame.sort(key=lambda t: t[0])

    # Resolve identity before the loop so anomalous single-frame files
    # (e.g. Vegetables frame-20.hdf5) can be skipped without calling load_single_scan.
    tt_guess = target_type or _infer_target_type(str(file_by_frame[0][1]))
    anomalous_skip = _anomalous_ids_for_target(tt_guess) if exclude_anomalous else set()

    amps, phases, waves = [], [], []
    z_indices = []
    anomalous_all = []
    skipped_anomalous: list[int] = []
    tt = None
    enc_name = None
    converted = None
    original_dtype = None
    laser_channel = ""
    source_files = []

    for n, fp in file_by_frame:
        if n in anomalous_skip:
            warnings.warn(
                f"[load_volume] skip anomalous/startup frame file frame_id={n} ({fp.name})",
                stacklevel=2,
            )
            skipped_anomalous.append(n)
            anomalous_all.append(n)
            continue
        try:
            data = load_single_scan(
                str(fp),
                target_type=target_type,
                phase_encoding=phase_encoding,
                convert_phase=convert_phase,
                rdm_source_path=rdm_source_path,
                qcl=qcl,
                side=side,
                exclude_anomalous=exclude_anomalous,
                include_waveform=include_waveform,
            )
        except AnomalousOnlyError as e:
            # Fallback if pre-skip missed a file (e.g. id not in filename).
            if not exclude_anomalous:
                raise
            warnings.warn(
                f"[load_volume] skip {fp.name}: {e}",
                stacklevel=2,
            )
            skipped_anomalous.append(n)
            anomalous_all.append(n)
            continue
        if data.n_slices != 1:
            # multi-slice single files are concatenated in file-internal order
            pass
        if tt is None:
            tt = data.target_type
            enc_name = data.metadata.get("phase_encoding_used")
            converted = data.metadata.get("phase_converted")
            original_dtype = data.metadata.get("original_dtype")
            laser_channel = data.metadata.get("laser_channel", "")
        elif data.target_type != tt:
            raise ValueError(
                f"folder mixes multiple target_type values: {tt} vs {data.target_type} ({fp})"
            )
        if amps and data.amplitude.shape[1:] != amps[0].shape[1:]:
            raise ValueError(
                f"image size mismatch: existing {amps[0].shape[1:]}, "
                f"{fp.name} is {data.amplitude.shape[1:]}"
            )
        amps.append(data.amplitude)
        phases.append(data.phase)
        if include_waveform:
            if data.waveform is None:
                raise ValueError(f"{fp.name} has no waveform; cannot stack waveform volume")
            waves.append(data.waveform)
        z_indices.extend(data.z_indices)
        anomalous_all.extend(data.metadata.get("anomalous_frame_ids") or [])
        source_files.append(str(fp.resolve()))

    if not amps:
        raise ValueError(
            f"nothing left to load under {data_dir} after skipping anomalous frames "
            f"{sorted(set(skipped_anomalous))}. "
            f"Pass exclude_anomalous=False to include them as raw codes."
        )

    amplitude = np.concatenate(amps, axis=0)
    phase = np.concatenate(phases, axis=0)
    waveform = np.concatenate(waves, axis=0) if waves else None

    meta = LFIMetadata.from_load_result(
        target_type=tt or "Unknown",
        file_path=str(root.resolve()),
        file_format="volume_from_frames",
        amplitude_shape=amplitude.shape,
        frame_ids=list(z_indices),
        original_dtype=original_dtype or "",
        phase_encoding_used=enc_name or "",
        phase_converted=bool(converted),
        anomalous_frame_ids=sorted(set(int(x) for x in anomalous_all)),
        laser_channel=laser_channel,
        rdm_source_path=rdm_source_path,
        notes=(
            "stacked from multiple single-frame files in numeric frame order"
            + (f"; skipped anomalous frames {sorted(set(skipped_anomalous))}" if skipped_anomalous else "")
        ),
    )
    md = {
        **meta.to_dict(),
        "scan_mode": "volume",
        "source_files": source_files,
        "n_source_files": len(source_files),
        "skipped_anomalous_frame_ids": sorted(set(skipped_anomalous)),
    }
    if waveform is not None:
        md["data_channels"] = ["amplitude", "phase", "waveform"]
        md["waveform_shape"] = list(waveform.shape)
        md["waveform_dtype"] = str(waveform.dtype)
    _attach_depth_metadata(
        md, z_indices, root, reference_frame_id=catalog_min_frame
    )

    print(
        f"[load_volume] done → {amplitude.shape} | frames={z_indices[0]}→{z_indices[-1]} "
        f"| depth_mm="
        f"{'yes' if md.get('depth_mm') else 'no (missing experiment_details or parse failed)'}"
    )
    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=list(z_indices),
        target_type=tt or "Unknown",
        file_path=str(root.resolve()),
        metadata=md,
        waveform=waveform,
    )


def load_multiple_scans(
    directory: str,
    target_type: Optional[str] = None,
) -> list:
    """
    Batch-load all .hdf5 / .h5 / .mat files under a directory (all in memory;
    for many files prefer output_interface.iter_scans).

    Results are sorted by numeric frame id (not lexical). Use load_volume() to stack into one volume.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    hdf5_files = find_data_files(dir_path, recursive=True)
    if not hdf5_files:
        raise RuntimeError(f"in {directory} found no .hdf5 / .h5 / .mat files")

    results = []
    failed = []

    for fp in hdf5_files:
        try:
            data = load_single_scan(str(fp), target_type=target_type)
            if target_type is None or data.target_type == target_type:
                results.append(data)
        except Exception as e:
            print(f"[warning] skipping file {fp.name}: {e}")
            failed.append(str(fp))

    print(f"\n[load_multiple_scans] done: ok {len(results)}, failed {len(failed)}")
    if failed:
        print(f"  failed files: {failed}")

    return results


def get_z_slices(filepath: str) -> list:
    """
    Quickly list z-slice or frame ids (without loading full images)

    Returns:
        id list (z keys for z-stack; frame ids for frames)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if Path(filepath).suffix.lower() == ".mat":
        tt = _infer_target_type(filepath)
        layout = MAT_LAYOUTS.get(tt)
        if layout is None:
            raise ValueError(f"target={tt} has no .mat layout configured")
        shape = _mat_variables(filepath).get(layout["phase"])
        if shape is None:
            raise ValueError(f"phase variable not found in file {layout['phase']}")
        n = 1 if len(shape) == 2 else shape[layout.get("stack_axis", -1)]
        return list(range(n))

    with h5py.File(filepath, "r") as f:
        fmt = _detect_format(f)
        if fmt == "hdf5_zstack":
            z_keys = _sort_frame_keys(f["amplitude"].keys())
            return [int(k) if k.isdigit() else k for k in z_keys]
        if fmt == "hdf5_data_v2026":
            index = _index_v2026(f["data"]["IMAGE_PHASE"])
            return [int(k) for k in _sort_frame_keys({k[0] for k in index})]
        frame_keys = _sort_frame_keys(f["frames"].keys())
        return [int(k) if k.isdigit() else k for k in frame_keys]
