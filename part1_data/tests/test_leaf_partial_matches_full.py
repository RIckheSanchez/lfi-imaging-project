"""Partial Leaf frame_ids must match the same index from a full load."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from loader import load_single_scan

LEAF = Path(__file__).resolve().parents[1] / "data" / "Leaf" / "leaf1.mat"
FRAME = 237


@pytest.mark.skipif(not LEAF.exists(), reason="leaf1.mat not present")
def test_leaf_partial_frame_matches_full_load():
    # Slow (~full 1.3GB mat twice-ish); only run when real Leaf is available.
    if os.environ.get("LFI_SKIP_SLOW_LEAF") == "1":
        pytest.skip("LFI_SKIP_SLOW_LEAF=1")

    full = load_single_scan(str(LEAF))
    part = load_single_scan(str(LEAF), frame_ids=[FRAME])

    assert full.z_indices[FRAME] == FRAME
    assert part.z_indices == [FRAME]
    assert np.array_equal(full.amplitude[FRAME], part.amplitude[0])
    assert np.array_equal(full.phase[FRAME], part.phase[0])
