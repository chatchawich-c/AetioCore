from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__: str = _pkg_version("aetiocore-sem-dl")
except PackageNotFoundError:
    __version__ = "0.1.6"

from ._core import (
    PyWhiteBoxModel,
    emit_lean4,
    emit_smt2,
    fit_whitebox_f32,
)
from .api import EdgeWhiteBoxSEM, fit_whitebox_sem, intervene_do

__all__ = [
    "__version__",
    "EdgeWhiteBoxSEM",
    "fit_whitebox_sem",
    "intervene_do",
    "PyWhiteBoxModel",
    "emit_lean4",
    "emit_smt2",
    "fit_whitebox_f32",
]
