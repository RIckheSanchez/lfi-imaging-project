"""Team integration contracts. Frontends run these backends on worker threads.

Replace the identity implementations via dependency injection. Implementations
must return LFIRawData, preserve compatible axes, and raise readable exceptions
on invalid inputs. Identity adapters perform no filtering or reconstruction.
"""
from typing import Protocol
from .data_types import LFIRawData


class Preprocessor(Protocol):
    def preprocess(self, data: LFIRawData, params: dict) -> LFIRawData:
        """Return preprocessed scan data using team-defined parameters."""
        ...


class Reconstructor(Protocol):
    def reconstruct(self, data: LFIRawData, params: dict) -> LFIRawData:
        """Return reconstructed scan data using team-defined parameters."""
        ...


class IdentityPreprocessor:
    """Default placeholder: returns the same object without preprocessing."""
    def preprocess(self, data: LFIRawData, params: dict) -> LFIRawData:
        return data


class IdentityReconstructor:
    """Default placeholder: returns the same object without reconstruction."""
    def reconstruct(self, data: LFIRawData, params: dict) -> LFIRawData:
        return data
