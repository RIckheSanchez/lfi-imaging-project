"""
Part 2: Signal Pre-processing (Haobo Yu)

Entry point to the ``thz_preprocessing`` package (v0.6.1) in ``part2_preprocess/src``.
See ``part2_preprocess/README.md`` for the method and ``docs/handoff_v06`` for
the output format given to the image-generation stage.

    from part1_data import load_volume
    from part2_preprocess.pipeline import preprocess_part1

    out = preprocess_part1(load_volume("2026_Vegetables"))
    products = out.metadata["spatial_products"]   # amplitude_corrected, phase_corrected, signal_mask ...
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thz_preprocessing import api  # noqa: E402
from thz_preprocessing.shared_loader import from_upstream  # noqa: E402


def preprocess_part1(scan, config_path=None, config_overrides=None):
    """Run the pre-processing pipeline on an ``LFIRawData`` from ``part1_data``.

    Phase that Part 1 left as raw codes (unconfirmed encoding) is kept as is:
    the pipeline then withholds unwrapping and phase correction.
    """
    return api.run_pipeline(from_upstream(scan), config_path, config_overrides)


__all__ = ["api", "from_upstream", "preprocess_part1"]
