from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__: str = _pkg_version("aetiocore-sem")
except PackageNotFoundError:
    __version__ = "0.1.6"

from ._core import (
    PyCbResult,
    PyPlsResult,
    emit_lean4,
    emit_smt2,
    fit_cb_f32,
    fit_cb_f64,
    fit_pls_f32,
    fit_pls_f64,
    validate_dsl,
)
from .api import SEM, fit_indices, fit_sem, parse_spec, propagate_risk

__all__ = [
    "__version__",
    "SEM",
    "parse_spec",
    "fit_sem",
    "fit_indices",
    "propagate_risk",
    "PyCbResult",
    "PyPlsResult",
    "emit_lean4",
    "emit_smt2",
    "fit_cb_f32",
    "fit_cb_f64",
    "fit_pls_f32",
    "fit_pls_f64",
    "validate_dsl",
]
