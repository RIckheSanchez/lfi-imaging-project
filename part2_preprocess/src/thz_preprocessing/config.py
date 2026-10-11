"""Configuration loading and merging for the pre-processing pipeline.

Implements the "structured parameter selection" requirement from
proposal §3.2.2: every processing step is driven by named, documented
parameters loaded from a plain-text TOML file, never hard-coded.

Resolution order, lowest precedence first:

1. :data:`DEFAULTS` — code-level fallback values, used if no TOML is
   loadable.
2. ``config/default.toml`` — project defaults shipped with the repo.
3. User-supplied ``path`` (optional, overrides 2).
4. ``[targets.<target_type>]`` overrides from whichever TOML was loaded
   (optional, overrides whatever was set above).
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover — only for Python 3.9 / 3.10
    import tomli as _toml  # type: ignore[no-redef]


DEFAULTS: dict[str, Any] = {
    "spatial_correction": {
        "enabled": False,
        "background_sigma_pixels": 16.0,
        "background_boundary_mode": "reflect",
        "background_truncate": 4.0,
        "stripe_axis": "rows",
        "stripe_window_lines": 15,
        "stripe_min_valid": 16,
        "mask": "none",
        "mask_threshold_scale": 1.0,
        "fft_notches": [],
        "notch_width_cycles_per_pixel": 0.005,
    },
    "noise_reduction": {
        "method": "median2d",
        "median_window": 3,
        "gaussian_sigma": 1.0,
        "savgol_window": 11,
        "savgol_polyorder": 3,
        "bandpass_low_norm": 0.02,
        "bandpass_high_norm": 0.45,
        "bandpass_order": 4,
    },
    "normalisation": {
        "method": "baseline",
        "robust": True,
        "baseline_percentile": 5.0,
    },
    "phase_unwrapping": {
        "enabled": False,
        "method": "skimage_2d",
        "rng_seed": 0,
        "mask": "none",
        "mask_threshold_scale": 1.0,
        "min_signal_fraction": 0.2,
        "max_residue_fraction": 0.001,
        "low_quality_action": "keep_wrapped",
    },
}


_PROJECT_DEFAULT_TOML = (
    Path(__file__).resolve().parents[2] / "config" / "default.toml"
)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return _toml.load(f)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a deep copy of ``base``."""
    out = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load(
    path: str | Path | None = None,
    target_type: str | None = None,
) -> dict[str, Any]:
    """Load a fully resolved configuration dict.

    Parameters
    ----------
    path
        Optional path to a user-supplied TOML file. If ``None``,
        ``config/default.toml`` shipped with the repo is used.
    target_type
        Optional ``target_type`` tag (e.g. ``"Skin"``). When present and
        a ``[targets.<target_type>]`` sub-table exists in the TOML, its
        contents are merged on top of the base sections so callers can
        switch defaults purely by selecting a target.

    Returns
    -------
    dict
        Configuration dict with at least the keys ``"noise_reduction"``,
        ``"normalisation"`` and ``"phase_unwrapping"`` populated.
    """
    cfg = deepcopy(DEFAULTS)

    toml_path: Path | None
    if path is not None:
        toml_path = Path(path)
    elif _PROJECT_DEFAULT_TOML.exists():
        toml_path = _PROJECT_DEFAULT_TOML
    else:
        toml_path = None

    if toml_path is not None:
        loaded = _read_toml(toml_path)
        overrides_root: dict[str, Any] = {
            k: v for k, v in loaded.items() if k != "targets"
        }
        cfg = _deep_merge(cfg, overrides_root)

        if target_type is not None:
            targets = loaded.get("targets", {})
            t_key = target_type.lower()
            target_overrides = targets.get(t_key) or targets.get(target_type)
            if target_overrides:
                cfg = _deep_merge(cfg, target_overrides)

    cfg.setdefault("_source", {})
    cfg["_source"]["toml_path"] = str(toml_path) if toml_path else None
    cfg["_source"]["target_type"] = target_type
    return cfg


__all__ = ["DEFAULTS", "load"]
