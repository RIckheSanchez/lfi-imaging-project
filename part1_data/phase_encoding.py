"""
phase_encoding.py
Explicit phase conversion config - never infer phase range from observed min/max.

Teammate-confirmed (leave other items pending; do not invent):
  Vegetables: phase_rad = code / 8192 * π
              dtype int16, valid codes [-8192, 8191]
              frame 20 = anomalous/startup frame; must be flagged
  Leaf:       phase_rad = wrap(L_P * π/4096 - π)
              (error vs reference ~ 1.8e-14)
  Skin:       unconfirmed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PhaseEncoding:
    """One explicit phase conversion setting with range checks."""

    name: str
    description: str
    formula: str
    expected_dtype: Optional[str]  # e.g. "int16"; None = not enforced
    code_min: Optional[float]      # inclusive; None = no lower check
    code_max: Optional[float]      # inclusive; None = no upper check
    confirmed: bool
    notes: str = ""
    anomalous_frame_ids: tuple = ()  # frames that must not be treated as normal data

    def convert(self, codes: np.ndarray, *, inplace: bool = False) -> np.ndarray:
        """
        Convert raw phase codes to radians.

        inplace: overwrite codes when it is already a writable float64 array
                 (for Leaf-sized multi-GB variables, avoid extra temporary copies).

        Raises:
            RuntimeError: encoding unconfirmed (e.g. Skin)
            ValueError:   dtype / code-range check failed
        """
        if not self.confirmed:
            raise RuntimeError(
                f"Phase encoding '{self.name}' is unconfirmed; conversion forbidden."
                f" {self.notes or 'Confirm with supervisor/teammate before adding a formula.'}"
            )

        arr = np.asarray(codes)
        self._check_dtype(arr)
        self._check_range(arr)
        return self._apply(arr, inplace)

    def _check_dtype(self, arr: np.ndarray) -> None:
        if self.expected_dtype is None:
            return
        expected = np.dtype(self.expected_dtype)
        # Allow wider same-family types (e.g. int32 holding int16 range)
        if arr.dtype != expected and not np.can_cast(arr.dtype, expected, casting="safe"):
            if str(arr.dtype) != self.expected_dtype:
                # Hard rule: if int16 expected, reject float
                if expected.kind in "iu" and arr.dtype.kind == "f":
                    raise ValueError(
                        f"Phase encoding '{self.name}' expects dtype={self.expected_dtype}, "
                        f"got={arr.dtype}"
                    )

    def _check_range(self, arr: np.ndarray) -> None:
        if self.code_min is None and self.code_max is None:
            return
        amin = float(np.nanmin(arr)) if arr.size else 0.0
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if self.code_min is not None and amin < self.code_min:
            raise ValueError(
                f"Phase encoding '{self.name}' lower bound failed: "
                f"min={amin} < code_min={self.code_min}"
            )
        if self.code_max is not None and amax > self.code_max:
            raise ValueError(
                f"Phase encoding '{self.name}' upper bound failed: "
                f"max={amax} > code_max={self.code_max}"
            )

    def _apply(self, arr: np.ndarray, inplace: bool = False) -> np.ndarray:
        raise NotImplementedError


def _float_work_array(arr: np.ndarray, inplace: bool) -> np.ndarray:
    """Reuse writable float64 in place when possible; otherwise allocate one float64 copy."""
    if inplace and arr.dtype == np.float64 and arr.flags.writeable:
        return arr
    return np.array(arr, dtype=np.float64)


class _PassthroughEncoding(PhaseEncoding):
    """Confirmed already-radians float data; no conversion. For explicit_name only."""

    def _apply(self, arr: np.ndarray, inplace: bool = False) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64)


class _VegetablesEncoding(PhaseEncoding):
    """phase_rad = code / 8192 * π"""

    def _apply(self, arr: np.ndarray, inplace: bool = False) -> np.ndarray:
        out = _float_work_array(arr, inplace)
        out *= np.pi / 8192.0
        return out


class _LeafEncoding(PhaseEncoding):
    """phase_rad = wrap(L_P * π/4096 - π)"""

    def _apply(self, arr: np.ndarray, inplace: bool = False) -> np.ndarray:
        out = _float_work_array(arr, inplace)
        out *= np.pi / 4096.0
        out -= np.pi
        _wrap_to_pi_inplace(out)
        return out


class _UnconfirmedEncoding(PhaseEncoding):
    def _apply(self, arr: np.ndarray, inplace: bool = False) -> np.ndarray:
        raise RuntimeError(self.notes)


def _wrap_to_pi(rad: np.ndarray) -> np.ndarray:
    """Wrap radians to [-π, π)."""
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_to_pi_inplace(rad: np.ndarray) -> None:
    """Same as _wrap_to_pi without allocating a new array."""
    rad += np.pi
    np.remainder(rad, 2.0 * np.pi, out=rad)
    rad -= np.pi


# ── Registered encodings (single source of truth) ──────────────

PHASE_ENCODINGS: dict[str, PhaseEncoding] = {
    "passthrough_radians": _PassthroughEncoding(
        name="passthrough_radians",
        description="Data already in radians; no conversion",
        formula="phase_rad = code",
        expected_dtype=None,
        code_min=None,
        code_max=None,
        confirmed=True,
        notes="Only for files confirmed to store radians",
    ),
    "vegetables_int16": _VegetablesEncoding(
        name="vegetables_int16",
        description="Vegetables dataset int16 phase codes",
        formula="phase_rad = code / 8192 * π",
        expected_dtype="int16",
        code_min=-8192,
        code_max=8191,
        confirmed=True,
        notes="frame 20 is anomalous/startup; see anomalous_frame_ids",
        anomalous_frame_ids=(20,),
    ),
    "leaf_wrap_4096": _LeafEncoding(
        name="leaf_wrap_4096",
        description="Leaf phase codes (aligned with teammate reference)",
        formula="phase_rad = wrap(L_P * π/4096 - π)",
        expected_dtype="int16",
        code_min=-8192,
        code_max=8191,
        confirmed=True,
        notes=(
            "Error vs reference ~ 1.8e-14; L_P_U in leaf1.mat (first 91 unwrapped frames) "
            "matches this formula pixel-wise after re-wrapping"
        ),
    ),
    "skin_unconfirmed": _UnconfirmedEncoding(
        name="skin_unconfirmed",
        description="Skin phase encoding not yet confirmed",
        formula="UNCONFIRMED - DO NOT INVENT",
        expected_dtype=None,
        code_min=None,
        code_max=None,
        confirmed=False,
        notes=(
            "Unconfirmed. Do not infer from observed min/max. "
            "Write a formula only after RDM/supervisor confirmation."
        ),
    ),
    "pcb_unconfirmed": _UnconfirmedEncoding(
        name="pcb_unconfirmed",
        description="PCB / Megabugs phase encoding not confirmed by teammate/supervisor",
        formula="UNCONFIRMED - DO NOT INVENT",
        expected_dtype=None,
        code_min=None,
        code_max=None,
        confirmed=False,
        notes=(
            "PCB and 2026 Vegetables are different datasets. "
            "Even if IMAGE_PHASE happens to be int16 in [-8192, 8191], "
            "do not apply vegetables_int16. Wait for identity confirmation."
        ),
    ),
    "zstack_unconfirmed": _UnconfirmedEncoding(
        name="zstack_unconfirmed",
        description="Chip / Marble z-stack phase units not confirmed",
        formula="UNCONFIRMED - candidate: phase_rad = value * π / 180",
        expected_dtype=None,
        code_min=None,
        code_max=None,
        confirmed=False,
        notes=(
            "borad1.hdf5 and Marble files store float64 phase in [-180, 179.956], "
            "step 360/8192 - looks like degrees, not radians. Keep raw until confirmed."
        ),
    ),
    "sound_unconfirmed": _UnconfirmedEncoding(
        name="sound_unconfirmed",
        description="Sound dataset phase encoding not confirmed",
        formula="UNCONFIRMED - DO NOT INVENT",
        expected_dtype=None,
        code_min=None,
        code_max=None,
        confirmed=False,
        notes=(
            "Sound is .mat (Thesis Plan commitment). Variable names and formula "
            "need a real file check; keep raw codes until confirmed."
        ),
    ),
    "unknown_unconfirmed": _UnconfirmedEncoding(
        name="unknown_unconfirmed",
        description="Unknown target type; do not guess a phase formula",
        formula="UNCONFIRMED - DO NOT INVENT",
        expected_dtype=None,
        code_min=None,
        code_max=None,
        confirmed=False,
        notes="Keep raw codes when target_type cannot be inferred from the path.",
    ),
}


# target_type → encoding name
# Only selection basis: dataset identity (folder / target type). Never use observed min/max.
# Range checks run only inside convert() after a confirmed encoding is chosen.
TARGET_PHASE_ENCODING: dict[str, str] = {
    "Chip": "zstack_unconfirmed",
    "Marble": "zstack_unconfirmed",
    "Vegetables": "vegetables_int16",
    "Leaf": "leaf_wrap_4096",
    "Skin": "skin_unconfirmed",
    "PCB": "pcb_unconfirmed",
    "Sound": "sound_unconfirmed",
    "Unknown": "unknown_unconfirmed",
}


def resolve_encoding_for_phase(
    target_type: str,
    phase_raw: np.ndarray | None = None,
    *,
    explicit_name: Optional[str] = None,
) -> str:
    """
    Choose phase encoding by dataset identity.

    - explicit_name: caller override (must be a key in PHASE_ENCODINGS)
    - else look up TARGET_PHASE_ENCODING[target_type]
    - phase_raw is kept for API compatibility and is NEVER used to pick a formula
    """
    del phase_raw  # must not participate in the decision
    if explicit_name is not None:
        if explicit_name not in PHASE_ENCODINGS:
            raise KeyError(f"Unknown phase encoding name: {explicit_name!r}")
        return explicit_name
    return TARGET_PHASE_ENCODING.get(target_type, "unknown_unconfirmed")


def get_phase_encoding(name_or_target: str) -> PhaseEncoding:
    """
    Look up PhaseEncoding by encoding name or target_type.

    Args:
        name_or_target: key in PHASE_ENCODINGS, or target_type in TARGET_PHASE_ENCODING
    """
    if name_or_target in PHASE_ENCODINGS:
        return PHASE_ENCODINGS[name_or_target]
    if name_or_target in TARGET_PHASE_ENCODING:
        return PHASE_ENCODINGS[TARGET_PHASE_ENCODING[name_or_target]]
    raise KeyError(
        f"Unknown phase encoding / target type: {name_or_target!r}. "
        f"Known encodings: {list(PHASE_ENCODINGS)}; "
        f"known targets: {list(TARGET_PHASE_ENCODING)}"
    )


def convert_phase(
    codes: np.ndarray,
    encoding: str | PhaseEncoding,
    *,
    convert: bool = True,
    inplace: bool = False,
) -> tuple[np.ndarray, PhaseEncoding, bool]:
    """
    Convert phase; keep raw codes when encoding is unconfirmed.

    Returns:
        (phase_array, encoding_used, converted)
        converted=False means still raw codes
    """
    enc = encoding if isinstance(encoding, PhaseEncoding) else get_phase_encoding(encoding)
    if not convert:
        return np.asarray(codes), enc, False
    if not enc.confirmed:
        return np.asarray(codes), enc, False
    return enc.convert(codes, inplace=inplace), enc, True
