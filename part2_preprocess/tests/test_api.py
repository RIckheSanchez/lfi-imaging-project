"""Basic smoke tests for the pre-processing API."""

import numpy as np
import pytest

from thz_preprocessing import api, config


def test_config_defaults_load():
    cfg = config.load()
    assert "noise_reduction" in cfg
    assert "normalisation" in cfg
    assert "phase_unwrapping" in cfg


def test_run_pipeline_raises_not_implemented():
    """Until implementation lands, the pipeline raises NotImplementedError."""
    signal = np.zeros((10, 100))
    with pytest.raises(NotImplementedError):
        api.run_pipeline(signal)
