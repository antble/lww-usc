from __future__ import annotations

from ._numba_kernels import (
    NUMBA_AVAILABLE,
    curcalc_current as _numba_curcalc_current,
    curcalc_density as _numba_curcalc_density,
    fill_potential_banded as _numba_fill_potential_banded,
    fill_scattering_banded as _numba_fill_scattering_banded,
)

try:
    from . import _cpp_kernels as _cpp
except Exception:  # pragma: no cover - depends on optional compiled extension.
    _cpp = None
    CPP_AVAILABLE = False
    OPENMP_ENABLED = False
    OPENMP_THREADS = 1
else:
    CPP_AVAILABLE = True
    OPENMP_ENABLED = bool(_cpp.openmp_enabled())
    OPENMP_THREADS = int(_cpp.openmp_threads())


def resolve_backend(preferred: str = "auto", use_compiled: bool = True) -> str:
    if not use_compiled or preferred == "python":
        return "python"
    if preferred == "cpp":
        return "cpp" if CPP_AVAILABLE else "python"
    if preferred == "numba":
        return "numba" if NUMBA_AVAILABLE else "python"
    if preferred != "auto":
        raise ValueError("kernel backend must be one of: auto, cpp, numba, python")
    if CPP_AVAILABLE:
        return "cpp"
    if NUMBA_AVAILABLE:
        return "numba"
    return "python"


def fill_potential_banded(amx, rvs, sin_matrix, n: int, nx: int, backend: str) -> bool:
    if backend == "cpp" and CPP_AVAILABLE:
        _cpp.fill_potential_banded(amx, rvs, sin_matrix, n, nx)
        return True
    if backend == "numba" and NUMBA_AVAILABLE:
        _numba_fill_potential_banded(amx, rvs, sin_matrix, n, nx)
        return True
    return False


def fill_scattering_banded(S, b, n: int, nx: int, tcol: float, backend: str) -> bool:
    if backend == "cpp" and CPP_AVAILABLE:
        _cpp.fill_scattering_banded(S, b, n, nx, tcol)
        return True
    if backend == "numba" and NUMBA_AVAILABLE:
        _numba_fill_scattering_banded(S, b, n, nx, tcol)
        return True
    return False


def curcalc_density(B, rj, nx: int, n: int, coef: float, backend: str) -> bool:
    if backend == "cpp" and CPP_AVAILABLE:
        _cpp.curcalc_density(B, rj, nx, n, coef)
        return True
    if backend == "numba" and NUMBA_AVAILABLE:
        _numba_curcalc_density(B, rj, nx, n, coef)
        return True
    return False


def curcalc_current(B, rj, nx: int, n: int, cofj: float, backend: str) -> bool:
    if backend == "cpp" and CPP_AVAILABLE:
        _cpp.curcalc_current(B, rj, nx, n, cofj)
        return True
    if backend == "numba" and NUMBA_AVAILABLE:
        _numba_curcalc_current(B, rj, nx, n, cofj)
        return True
    return False
