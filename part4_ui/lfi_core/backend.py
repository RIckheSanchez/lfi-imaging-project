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



from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class BackendRegistration:
    """An implementation and its independent default parameter dictionary."""
    implementation: object
    params: dict = field(default_factory=dict)


_preprocessor_registration = None
_reconstructor_registration = None


def register_preprocessor(implementation, params=None):
    """Register before creating MainWindow; None/Identity disables the controls."""
    global _preprocessor_registration
    if implementation is None or isinstance(implementation, IdentityPreprocessor):
        _preprocessor_registration = None
    else:
        if not callable(getattr(implementation, 'preprocess', None)):
            raise TypeError('Preprocessor must implement preprocess(data, params).')
        _preprocessor_registration = BackendRegistration(implementation, deepcopy(params or {}))


def register_reconstructor(implementation, params=None):
    """Register before creating MainWindow; None/Identity disables the controls."""
    global _reconstructor_registration
    if implementation is None or isinstance(implementation, IdentityReconstructor):
        _reconstructor_registration = None
    else:
        if not callable(getattr(implementation, 'reconstruct', None)):
            raise TypeError('Reconstructor must implement reconstruct(data, params).')
        _reconstructor_registration = BackendRegistration(implementation, deepcopy(params or {}))


def get_preprocessor_registration():
    return _preprocessor_registration


def get_reconstructor_registration():
    return _reconstructor_registration
