"""Configuration loading and defaults for the pre-processing pipeline."""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    "noise_reduction": {
        "method": "bandpass",
        "low_hz": 1e3,
        "high_hz": 1e5,
    },
    "normalisation": {
        "method": "baseline",
    },
    "phase_unwrapping": {
        "enabled": False,
        "method": "itoh_1d",
    },
}


def load(path: str | None = None) -> dict[str, Any]:
    """Load a config dict from a TOML file, or return defaults.

    Parameters
    ----------
    path : str, optional
        Path to a TOML config file. If None, DEFAULTS is returned.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if path is None:
        return {k: dict(v) for k, v in DEFAULTS.items()}

    # TODO: load and merge TOML on top of DEFAULTS
    raise NotImplementedError("TOML loading not yet implemented")
