"""
Optimized core routines for the 1D Lattice Weyl-Wigner / Wigner-Poisson model.

Implementation notes:
- solve the Wigner linear system directly with LAPACK band storage
- avoid converting band matrices to dense Nx*N by Nx*N arrays
- vectorize Fermi boundaries, current/density, Poisson charge sums, and potential coefficients
- cache kinetic stencil and sine matrix used in the non-local potential
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from typing import Optional

import numpy as np
import scipy.linalg as spla

from ._native_kernels import (
    CPP_AVAILABLE,
    NUMBA_AVAILABLE,
    OPENMP_ENABLED,
    OPENMP_THREADS,
    curcalc_current as _native_curcalc_current,
    curcalc_density as _native_curcalc_density,
    fill_potential_banded as _native_fill_potential_banded,
    fill_scattering_banded as _native_fill_scattering_banded,
    resolve_backend,
)


@dataclass(frozen=True)
class Params:
    # Grid
    Nx: int = 86
    N: int = 72

    # Physical constants
    c: float = 2997.924580
    rmo: float = 0.05685675170
    rm: float = 6.67e-2
    temp: float = 77.0
    rhc: float = 1973.29
    re: float = 2.0
    rno: float = 1.0e-6
    ri: float = 0.3

    # Geometry
    box: float = 550.0
    well: float = 50.0
    barrier: float = 30.0
    spacer: float = 30.0
    pot: float = 0.3

    # Other
    chemp: float = 0.0863814

    @property
    def rmass(self) -> float:
        return self.rm * self.rmo * (self.c**2)

    @property
    def bbeta(self) -> float:
        return 11604.5 / self.temp

    @property
    def dens(self) -> float:
        return self.re * self.rno

    @property
    def densi(self) -> float:
        return (1.0 + self.ri) * self.re * self.rno / (1.0 - self.ri)

    @property
    def delx(self) -> float:
        return self.box / (self.Nx - 1)

    @property
    def delk(self) -> float:
        return math.pi / (self.delx * self.N)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(0.0, self.box, self.Nx)


DEFAULT = Params()

# Backwards-compatible module-level constants.
Nx = DEFAULT.Nx
N = DEFAULT.N
c = DEFAULT.c
rmo = DEFAULT.rmo
rm = DEFAULT.rm
temp = DEFAULT.temp
rmass = DEFAULT.rmass
bbeta = DEFAULT.bbeta
rhc = DEFAULT.rhc
re = DEFAULT.re
rno = DEFAULT.rno
ri = DEFAULT.ri
dens = DEFAULT.dens
densi = DEFAULT.densi
box = DEFAULT.box
well = DEFAULT.well
barrier = DEFAULT.barrier
spacer = DEFAULT.spacer
pot = DEFAULT.pot
delx = DEFAULT.delx
delk = DEFAULT.delk
pi = math.pi
chemp = DEFAULT.chemp
x = DEFAULT.x


def _p(params: Optional[Params]) -> Params:
    return DEFAULT if params is None else params


def _kernel_backend(use_numba: bool | None = True, kernel_backend: str = "auto") -> str:
    return resolve_backend(kernel_backend, bool(use_numba))


@dataclass(slots=True)
class WignerSolveWorkspace:
    """Reusable buffers for the Wigner banded solve.

    The LAPACK buffer uses ``dgbsv`` storage directly.  Its lower view has the
    same shape as SciPy's compact band format, so the existing assembly kernels
    can fill it while LAPACK can factor the full buffer without another matrix
    allocation/copy inside ``solve_banded``.
    """

    n: int
    nx: int
    lower_upper: int
    lapack_ab: np.ndarray
    rhs: np.ndarray
    rvs_work: np.ndarray

    @classmethod
    def create(cls, n: int = N, nx: int = Nx) -> "WignerSolveWorkspace":
        lower_upper = 2 * n
        size = n * nx
        return cls(
            n=n,
            nx=nx,
            lower_upper=lower_upper,
            lapack_ab=np.zeros((1 + 3 * lower_upper, size), dtype=float, order="F"),
            rhs=np.empty((size, 1), dtype=float, order="F"),
            rvs_work=np.empty(nx, dtype=float),
        )

    @property
    def band(self) -> np.ndarray:
        return self.lapack_ab[self.lower_upper:, :]


def band_to_agb(a: np.ndarray, params: Optional[Params] = None) -> np.ndarray:
    """Convert dense square matrix to LAPACK/SciPy general band storage."""
    p = _p(params)
    m = n = p.N * p.Nx
    ml = mu = 2 * p.N
    agb = np.zeros((ml + mu + 1, n), dtype=a.dtype)
    for j in range(n):
        i0 = max(0, j - mu)
        i1 = min(m, j + ml + 1)
        rows = mu + np.arange(i0, i1) - j
        agb[rows, j] = a[i0:i1, j]
    return agb


def agb_to_band(agb: np.ndarray, params: Optional[Params] = None) -> np.ndarray:
    """Convert LAPACK/SciPy general band storage to a dense square matrix."""
    p = _p(params)
    m = n = p.N * p.Nx
    ml = mu = 2 * p.N
    a = np.zeros((m, n), dtype=agb.dtype)
    for j in range(n):
        i0 = max(0, j - mu)
        i1 = min(m, j + ml + 1)
        rows = mu + np.arange(i0, i1) - j
        a[i0:i1, j] = agb[rows, j]
    return a


def fermi_function(n: int = N, chemp_value: float = chemp, params: Optional[Params] = None) -> np.ndarray:
    p = _p(params)
    j = np.arange(1, n + 1, dtype=float)
    deno = p.rmass / (math.pi * p.bbeta * (p.rhc**2))
    rhmc = (p.rhc * p.delk) ** 2 / (8.0 * p.rmass)
    arg = p.bbeta * (chemp_value - rhmc * (2.0 * j - n - 1.0) ** 2)
    # logaddexp is stable for large positive/negative arguments.
    return deno * np.logaddexp(0.0, arg)


def fbndry(b: np.ndarray, n: int = N, nx: int = Nx, chemp_value: float = chemp,
           params: Optional[Params] = None) -> np.ndarray:
    """Fill boundary source vector in-place and return it."""
    p = _p(params)
    b.fill(0.0)
    cte = p.delk * (p.rhc**2) / (4.0 * p.delx * p.rmass)
    nh = n // 2
    f = fermi_function(n, chemp_value, p)
    j = np.arange(1, n + 1)
    ram = cte * (2 * j - n - 1)

    pos = np.arange(nh, n)       # 0-based j indices for k > 0
    neg = np.arange(0, nh)       # 0-based j indices for k < 0

    b[pos] = -3.0 * ram[pos] * f[pos]
    b[n + pos] = ram[pos] * f[pos]
    b[nx * n - n + neg] = 3.0 * ram[neg] * f[neg]
    b[nx * n - 2 * n + neg] = -ram[neg] * f[neg]
    return b


def rvscalc(rvs: np.ndarray, nx: int = Nx, bias: float = 0.0, isc: int = 0,
            params: Optional[Params] = None) -> np.ndarray:
    """Calculate external double-barrier potential in-place and return it."""
    p = _p(params)
    r1 = (p.box - p.well) / 2.0 - p.barrier
    r2 = (p.box - p.well) / 2.0
    ra = r1 - p.spacer
    ira = int(ra / p.delx) + 1
    nxh = nx // 2

    if isc == 0:
        rvs[:nx] = 0.0
    elif isc == 2:
        idx = np.arange(nxh)
        rx = idx * p.delx
        vals = 0.18 * p.pot / (1.0 + np.exp((r1 - 50.0 - rx) / 28.0))
        rvs[idx] = vals
        rvs[nx - 1 - idx] = vals

    idx = np.arange(max(ira - 1, 0), nxh)
    rx = idx * p.delx
    mask = (rx <= r2) & (rx >= r1)
    idxb = idx[mask]
    rvs[idxb] += p.pot
    rvs[nx - 1 - idxb] += p.pot

    if isc == 0:
        rvs[nx - ira:nx] -= bias
        slope = bias / (p.well + 4.0 * p.barrier)
        idx = np.arange(ira, nxh)
        rx = idx * p.delx - ra
        mask = rx >= 0.0
        idx = idx[mask]
        rx = rx[mask]
        rvs[idx] -= slope * rx
        rvs[nx - 1 - idx] += slope * (rx - p.well - 4.0 * p.barrier)

    return rvs


def integral(x_grid: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    dx = x_grid[1] - x_grid[0]
    return np.sum(y * dx, axis=axis)


def get_exchange(rho: np.ndarray, x_grid: Optional[np.ndarray] = None) -> np.ndarray:
    rho_safe = np.maximum(rho, 0.0)
    return -((3.0 / math.pi) ** (1.0 / 3.0)) * np.cbrt(rho_safe)


def get_hartree(rho: np.ndarray, x_grid: np.ndarray, eps: float = 1e-1) -> np.ndarray:
    dx = x_grid[1] - x_grid[0]
    return np.sum(rho[None, :] * dx / np.sqrt((x_grid[None, :] - x_grid[:, None]) ** 2 + eps), axis=-1)


@lru_cache(maxsize=16)
def _sin_matrix(n: int) -> np.ndarray:
    j = np.arange(1, n)[:, None]
    ip = np.arange(1, n // 2 + 1)[None, :]
    return np.sin((2.0 * math.pi / n) * np.mod(ip * j, n))


def aicalc(rvs: np.ndarray, ai: Optional[np.ndarray] = None, n: int = N, nx: int = Nx, i: int = 1,
           params: Optional[Params] = None) -> np.ndarray:
    """Vectorized non-local potential coefficient for one 1-based spatial index i."""
    i0 = i - 1
    ips = np.arange(1, n // 2 + 1)
    right = np.minimum(i0 + ips, nx - 1)
    left = np.maximum(i0 - ips, 0)
    diff = rvs[right] - rvs[left]

    out = np.zeros(n) if ai is None else ai
    out.fill(0.0)
    out[: n - 1] = (2.0 / n) * (_sin_matrix(n) @ diff)
    return out


def _potential_rvs(
    external_rvs: np.ndarray,
    rho: np.ndarray,
    exchange: bool,
    params: Params,
    rvs_work: Optional[np.ndarray] = None,
) -> np.ndarray:
    source = np.asarray(external_rvs, dtype=float)
    if not exchange:
        return source

    if rvs_work is None:
        rvs = source.copy()
    else:
        if rvs_work.shape != source.shape:
            raise ValueError(f"rvs workspace must have shape {source.shape}")
        np.copyto(rvs_work, source)
        rvs = rvs_work
    rvs += get_exchange(np.asarray(rho, dtype=float), params.x)
    return rvs


def _fill_potential_banded_inplace(
    amx: np.ndarray,
    external_rvs: np.ndarray,
    rho: np.ndarray,
    n: int = N,
    nx: int = Nx,
    exchange: bool = False,
    params: Optional[Params] = None,
    use_numba: bool | None = True,
    kernel_backend: str = "auto",
    rvs_work: Optional[np.ndarray] = None,
) -> np.ndarray:
    p = _p(params)
    rvs = _potential_rvs(external_rvs, rho, exchange, p, rvs_work)
    backend = _kernel_backend(use_numba, kernel_backend)
    if _native_fill_potential_banded(amx, rvs, _sin_matrix(n), n, nx, backend):
        return amx

    center = 2 * n
    ai = np.empty(n)
    for ix in range(nx):
        aicalc(rvs, ai, n, nx, ix + 1, p)
        base = ix * n
        for j0 in range(n - 1):
            val = ai[j0]
            if val == 0.0:
                continue
            # Original 1-based indexing converted to 0-based rows/columns.
            amx[center + (j0 + 1), base: base + n - (j0 + 1)] = -val
            amx[center - (j0 + 1), base + (j0 + 1): base + n] = val
    return amx


def potential(external_rvs: np.ndarray, rho: np.ndarray, n: int = N, nx: int = Nx,
              exchange: bool = False, params: Optional[Params] = None,
              use_numba: bool | None = True, kernel_backend: str = "auto") -> np.ndarray:
    """Return potential matrix directly in banded storage: shape (4*n+1, n*nx)."""
    amx = np.zeros((4 * n + 1, n * nx), dtype=float)
    return _fill_potential_banded_inplace(
        amx,
        external_rvs,
        rho,
        n,
        nx,
        exchange,
        params,
        use_numba,
        kernel_backend,
    )


@lru_cache(maxsize=16)
def _kinetic_cached(n: int, nx: int, irlx: int, delk_: float, delx_: float,
                    rhc_: float, rmass_: float) -> np.ndarray:
    T = np.zeros((4 * n + 1, n * nx), dtype=float)
    cte = delk_ * (rhc_**2) / (4.0 * delx_ * rmass_)
    cols = np.arange(n * nx)
    j0 = cols % n
    j1 = j0 + 1
    ram = cte * (2 * j1 - n - 1)
    nh = n // 2

    neg = j0 < nh
    T[4 * n, neg] = 0.0
    T[3 * n, neg] = 0.0
    T[2 * n, neg] = 3.0 * ram[neg]
    T[1 * n, neg] = -4.0 * ram[neg]
    T[0, neg] = ram[neg]

    pos = ~neg
    T[4 * n, pos] = -ram[pos]
    T[3 * n, pos] = 4.0 * ram[pos]
    T[2 * n, pos] = -3.0 * ram[pos]
    T[1 * n, pos] = 0.0
    T[0, pos] = 0.0

    if irlx >= 3:
        T[2 * n, :] += -2.0 * 0.6582186935

    return T


def kinetic(n: int = N, nx: int = Nx, irlx: int = 0, params: Optional[Params] = None) -> np.ndarray:
    p = _p(params)
    return _kinetic_cached(n, nx, irlx, p.delk, p.delx, p.rhc, p.rmass).copy()


def arelax(S: np.ndarray, b: np.ndarray, n: int = N, nx: int = Nx,
           params: Optional[Params] = None, use_numba: bool | None = True,
           kernel_backend: str = "auto") -> np.ndarray:
    """Relaxation/scattering band matrix. No debug printing."""
    tau = 0.5255074e3
    tcol = 0.65821869340 / tau
    backend = _kernel_backend(use_numba, kernel_backend)
    if _native_fill_scattering_banded(S, np.ascontiguousarray(b, dtype=float), n, nx, tcol, backend):
        return S

    b2 = np.asarray(b).reshape(nx, n)
    rho = b2.sum(axis=1)
    center = 2 * n

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(rho[:, None] != 0.0, tcol * b2 / rho[:, None], 0.0)

    for ix in range(nx):
        base = ix * n
        for j in range(n):
            # Add column-wise equilibrium redistribution.
            for jp in range(n):
                S[center + j - jp, base + jp] += weights[ix, j]
            S[center, base + j] -= tcol
    return S


def scattering(b: np.ndarray, n: int = N, nx: int = Nx, params: Optional[Params] = None,
               use_numba: bool | None = True, kernel_backend: str = "auto") -> np.ndarray:
    return arelax(np.zeros((4 * n + 1, n * nx), dtype=float), b, n, nx, params, use_numba, kernel_backend)


def _assemble_wigner_banded_inplace(
    out: np.ndarray,
    bo: np.ndarray,
    rvs: np.ndarray,
    rho: np.ndarray,
    n: int = N,
    nx: int = Nx,
    irlx: int = 0,
    exchange: bool = False,
    params: Optional[Params] = None,
    use_numba: bool | None = True,
    kernel_backend: str = "auto",
    rvs_work: Optional[np.ndarray] = None,
) -> np.ndarray:
    p = _p(params)
    np.copyto(out, _kinetic_cached(n, nx, irlx, p.delk, p.delx, p.rhc, p.rmass))
    _fill_potential_banded_inplace(
        out,
        rvs,
        rho,
        n,
        nx,
        exchange,
        p,
        use_numba,
        kernel_backend,
        rvs_work,
    )
    if irlx != 0:
        arelax(out, bo, n, nx, p, use_numba, kernel_backend)
    return out


def _solve_wigner_lapack(
    lapack_ab: np.ndarray,
    rhs: np.ndarray,
    lower_upper: int,
    overwrite: bool,
    workspace: Optional[WignerSolveWorkspace] = None,
) -> np.ndarray:
    size = lapack_ab.shape[1]
    rhs_arr = np.asarray(rhs, dtype=float)
    if rhs_arr.shape != (size,):
        raise ValueError(f"rhs must have shape ({size},)")

    can_overwrite_rhs = (
        overwrite
        and isinstance(rhs, np.ndarray)
        and rhs_arr is rhs
        and rhs_arr.flags.c_contiguous
        and rhs_arr.dtype == np.float64
    )
    if can_overwrite_rhs:
        rhs_2d = rhs_arr.reshape(size, 1)
    elif workspace is not None and workspace.rhs.shape == (size, 1):
        rhs_2d = workspace.rhs
        rhs_2d[:, 0] = rhs_arr
    else:
        rhs_2d = np.array(rhs_arr.reshape(size, 1), dtype=float, order="F", copy=True)

    _, _, solution, info = spla.lapack.dgbsv(
        lower_upper,
        lower_upper,
        lapack_ab,
        rhs_2d,
        overwrite_ab=True,
        overwrite_b=True,
    )
    if info < 0:
        raise ValueError(f"LAPACK dgbsv argument {-info} had an illegal value")
    if info > 0:
        raise np.linalg.LinAlgError(f"LAPACK dgbsv found a singular pivot at U({info}, {info})")

    solved = solution[:, 0]
    if overwrite:
        if can_overwrite_rhs and np.shares_memory(solved, rhs):
            return rhs
        rhs[:] = solved
        return rhs
    return solved.copy()


def wigstd(f: np.ndarray, bo: np.ndarray, rvs: np.ndarray, rho: np.ndarray,
           n: int = N, nx: int = Nx, irlx: int = 0, exchange: bool = False,
           params: Optional[Params] = None, overwrite: bool = True,
           use_numba: bool | None = True, kernel_backend: str = "auto",
           workspace: Optional[WignerSolveWorkspace] = None) -> np.ndarray:
    """
    Solve the Wigner steady-state equation.

    This path keeps T, V, and S in banded storage and calls LAPACK ``dgbsv``
    directly.  With a ``WignerSolveWorkspace`` it reuses the LAPACK band and RHS
    buffers across transient steps.
    """
    lower_upper = 2 * n
    size = n * nx
    if workspace is not None:
        if workspace.n != n or workspace.nx != nx or workspace.lower_upper != lower_upper:
            raise ValueError("workspace dimensions do not match n/nx")
        lapack_ab = workspace.lapack_ab
        band = workspace.band
        rvs_work = workspace.rvs_work
    else:
        lapack_ab = np.zeros((1 + 3 * lower_upper, size), dtype=float, order="F")
        band = lapack_ab[lower_upper:, :]
        rvs_work = None

    _assemble_wigner_banded_inplace(
        band,
        bo,
        rvs,
        rho,
        n,
        nx,
        irlx,
        exchange,
        params,
        use_numba,
        kernel_backend,
        rvs_work,
    )
    return _solve_wigner_lapack(lapack_ab, f, lower_upper, overwrite, workspace)


def tridag(a: np.ndarray, b: np.ndarray, c: np.ndarray, r: np.ndarray, nx: int) -> np.ndarray:
    """Thomas algorithm for tridiagonal systems."""
    u = np.zeros(nx, dtype=float)
    gam = np.zeros(nx, dtype=float)

    bet = b[0]
    if bet == 0.0:
        raise ZeroDivisionError("tridag failed: zero pivot at row 0")
    u[0] = r[0] / bet

    for j in range(1, nx):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]
        if bet == 0.0:
            raise ZeroDivisionError(f"tridag failed: zero pivot at row {j}")
        u[j] = (r[j] - a[j] * u[j - 1]) / bet

    for j in range(nx - 2, -1, -1):
        u[j] -= gam[j + 1] * u[j + 1]
    return u


def poissonb(b: np.ndarray, rvs: np.ndarray, nx: int = Nx, n: int = N, vo: float = 0.0,
             vb: float = 0.0, rfa: float = 650.0, amax: Optional[np.ndarray] = None,
             params: Optional[Params] = None, save_csv: bool = False) -> np.ndarray:
    p = _p(params)
    b2 = np.asarray(b).reshape(nx, n)
    charge_sum = b2.sum(axis=1)

    deno = 12.910 * 8.8542e-12 / (1e10 * 1.6e-19)
    ep2 = 1.0 / p.delx
    cc = p.delk / (2.0 * math.pi)

    rx = np.arange(nx) * p.delx
    r1 = (p.box - p.well) / 2.0 - p.barrier - p.spacer
    r4 = (p.box + p.well) / 2.0 + p.barrier + p.spacer
    inside = (rx >= r1) & (rx <= r4)

    rhs = np.empty(nx, dtype=float)
    rhs[inside] = (cc * charge_sum[inside] / deno) * p.delx / 2.0
    rhs[~inside] = -((p.dens - cc * charge_sum[~inside]) / deno) * p.delx / 2.0
    rhs[0] += ep2 * vo / 2.0
    rhs[-1] += ep2 * vb / 2.0

    a = np.full(nx, -0.5 * ep2)
    bg = np.full(nx, 1.0 * ep2)
    cvec = np.full(nx, -0.5 * ep2)

    pvec = np.empty(nx, dtype=float)
    pvec[0] = (0.5 * rvs[1] - rvs[0]) * ep2 + rhs[0]
    pvec[-1] = (-rvs[-1] + 0.5 * rvs[-2]) * ep2 + rhs[-1]
    pvec[1:-1] = (0.5 * rvs[2:] - rvs[1:-1] + 0.5 * rvs[:-2]) * ep2 + rhs[1:-1]

    rvs_new = tridag(a, bg, cvec, pvec, nx)
    if save_csv:
        np.savetxt("poisson_eqsol.csv", rvs, delimiter=",", fmt="%s")

    max_abs = float(np.max(np.abs(rvs_new)))
    if amax is not None:
        amax[0] = max_abs
    du = 1.0 if max_abs == 0.0 else math.log1p(rfa * max_abs) / (rfa * max_abs)

    rvs[:] = du * rvs_new + rvs
    rvscalc(rvs, nx, 0.0, 1, p)
    return rvs


def poisson(b: np.ndarray, xa: np.ndarray, nx: int = Nx, n: int = N, vo: float = 0.0,
            vb: float = 0.0, m: int = 1, params: Optional[Params] = None) -> np.ndarray:
    p = _p(params)
    b2 = np.asarray(b).reshape(nx, n)
    charge_sum = b2.sum(axis=1)

    deno = 12.910 * 8.8542e-12 / (1.0e10 * 1.6e-19)
    cc = p.delk / (2.0 * math.pi)
    rx = np.arange(nx) * p.delx
    r1 = (p.box - p.well) / 2.0 - p.barrier - p.spacer
    r4 = (p.box + p.well) / 2.0 + p.barrier + p.spacer
    inside = (rx >= r1) & (rx <= r4)

    rhs = np.empty(nx, dtype=float)
    rhs[inside] = ((cc * charge_sum[inside] / deno) * (p.delx**2)) / 2.0
    rhs[~inside] = -(((p.dens - cc * charge_sum[~inside]) / deno) * (p.delx**2)) / 2.0
    rhs[0] += vo / 2.0
    rhs[-1] += vb / 2.0

    a = np.full(nx, -0.5)
    bg = np.ones(nx)
    cvec = np.full(nx, -0.5)
    xa[:] = tridag(a, bg, cvec, rhs, nx)
    rvscalc(xa, nx, 0.0, 1, p)
    return xa


def curcalc(b: np.ndarray, rj: np.ndarray, n: int = N, nx: int = Nx, irj: int = 2,
            params: Optional[Params] = None, use_numba: bool | None = True,
            kernel_backend: str = "auto") -> np.ndarray:
    p = _p(params)
    B = np.asarray(b).reshape(nx, n)
    nh = n // 2

    if irj == 2:
        cofj = (5915774.594 * 1.6e12 * p.delk**2) / (8.0 * math.pi * p.rmass)
        backend = _kernel_backend(use_numba, kernel_backend)
        if _native_curcalc_current(np.ascontiguousarray(B), rj, nx, n, cofj, backend):
            return rj

        jneg = np.arange(1, nh + 1)
        wneg = (2 * jneg - n - 1)
        jpos = np.arange(nh + 1, n + 1)
        wpos = (2 * jpos - n - 1)

        vals = np.zeros(nx)
        # Original uses 1-based i=2..Nx-2, accessing rows i, i+1 and i-2, i-1 in 0-based B.
        for i0 in range(1, nx - 2):
            vals[i0] = (
                np.dot(wneg, 3.0 * B[i0 + 1, :nh] - B[i0 + 2, :nh])
                + np.dot(wpos, -B[i0 - 1, nh:] + 3.0 * B[i0, nh:])
            )
        rj[:] = cofj * vals
        rj[1] = 2.0 * rj[2] - rj[3]
        rj[0] = rj[1]
        rj[nx - 2] = 2.0 * rj[nx - 3] - rj[nx - 4]
        rj[nx - 1] = rj[nx - 2]

    elif irj == 1:
        coef = p.delk / (2.0 * math.pi)
        backend = _kernel_backend(use_numba, kernel_backend)
        if _native_curcalc_density(np.ascontiguousarray(B), rj, nx, n, coef, backend):
            return rj
        rj[:] = coef * B.sum(axis=1)
    else:
        raise ValueError("irj must be 1 for density or 2 for current")

    return rj


def comp(x: np.ndarray, xo: np.ndarray, n: int = N, nx: int = Nx, eps: float = 7.0e-10) -> tuple[int, float]:
    amx = float(np.max(np.abs(np.asarray(x[:nx]) - np.asarray(xo[:nx]))))
    return (0 if amx > eps else 1), amx


def subsrvs(rvs: np.ndarray, nx: int = Nx, params: Optional[Params] = None) -> np.ndarray:
    p = _p(params)
    r1 = (p.box - p.well) / 2.0 - p.barrier
    r2 = (p.box - p.well) / 2.0
    ra = r1 - p.barrier
    ira = int(ra / p.delx) + 1
    nxh = nx // 2
    idx = np.arange(max(ira - 1, 0), nxh)
    rx = idx * p.delx
    idx = idx[(rx <= r2) & (rx >= r1)]
    rvs[idx] -= p.pot
    rvs[nx - 1 - idx] -= p.pot
    return rvs


def initial_state(params: Optional[Params] = None, exchange: bool = True, irlx_value: int = 0):
    """Compute the default steady-state initialization."""
    p = _p(params)
    rvs = np.zeros(p.Nx)
    rho = np.zeros(p.Nx)
    f = np.zeros(p.N * p.Nx)
    rj = np.zeros(p.Nx)

    rvscalc(rvs, p.Nx, 0.0, 2, p)
    fbndry(f, p.N, p.Nx, p.chemp, p)
    wigstd(f, f, rvs, rho, p.N, p.Nx, irlx_value, exchange, p)
    curcalc(f, rj, p.N, p.Nx, 2, p)
    return rvs, rho, f, rj

# -----------------------------------------------------------------------------
# Compatibility layer for the lww_transport package API
# -----------------------------------------------------------------------------
# Supports both array-first calls, e.g. fbndry(b, n, nx), and config-first calls,
# e.g. fbndry(cfg, b).

_orig_fbndry = fbndry
_orig_rvscalc = rvscalc
_orig_kinetic = kinetic
_orig_wigstd = wigstd
_orig_poissonb = poissonb
_orig_poisson = poisson
_orig_curcalc = curcalc
_orig_comp = comp
_orig_subsrvs = subsrvs
_orig_fermi_function = fermi_function


def _is_cfg(obj) -> bool:
    return hasattr(obj, "nx") and hasattr(obj, "n")


def _cfg_params(cfg, params: Optional[Params] = None) -> Params:
    """Build a core ``Params`` object from ``LWWConfig``."""
    src = params if params is not None else getattr(cfg, "params", DEFAULT)
    vals = {}
    for field in Params.__dataclass_fields__:
        if field == "Nx":
            vals[field] = int(getattr(cfg, "nx", getattr(src, "Nx", DEFAULT.Nx)))
        elif field == "N":
            vals[field] = int(getattr(cfg, "n", getattr(src, "N", DEFAULT.N)))
        else:
            vals[field] = getattr(src, field, getattr(DEFAULT, field))
    return Params(**vals)


def _extra_pos_to_kwargs(extra, names, kwargs):
    kwargs = dict(kwargs)
    for name, value in zip(names, extra):
        kwargs.setdefault(name, value)
    return kwargs


def fbndry(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 2:
            raise TypeError("fbndry(cfg, b) requires b")
        b = args[1]
        p = _cfg_params(cfg, kwargs.pop("params", None))
        chemp_value = kwargs.pop("chemp_value", getattr(p, "chemp", chemp))
        return _orig_fbndry(b, n=int(cfg.n), nx=int(cfg.nx), chemp_value=chemp_value, params=p)
    return _orig_fbndry(*args, **kwargs)


def fermi_function(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        p = _cfg_params(cfg, kwargs.pop("params", None))
        chemp_value = kwargs.pop("chemp_value", getattr(p, "chemp", chemp))
        return _orig_fermi_function(n=int(cfg.n), chemp_value=chemp_value, params=p)
    return _orig_fermi_function(*args, **kwargs)


def rvscalc(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 2:
            raise TypeError("rvscalc(cfg, rvs, ...) requires rvs")
        rvs = args[1]
        extra = args[2:]
        kwargs = _extra_pos_to_kwargs(extra, ["bias", "isc"], kwargs)
        p = _cfg_params(cfg, kwargs.pop("params", None))
        bias = kwargs.pop("bias", 0.0)
        isc = kwargs.pop("isc", 0)
        return _orig_rvscalc(rvs, nx=int(cfg.nx), bias=bias, isc=isc, params=p)
    return _orig_rvscalc(*args, **kwargs)


def kinetic(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        p = _cfg_params(cfg, kwargs.pop("params", None))
        if len(args) > 1:
            kwargs.setdefault("irlx", args[1])
        return _orig_kinetic(n=int(cfg.n), nx=int(cfg.nx), params=p, **kwargs)
    return _orig_kinetic(*args, **kwargs)


def wigstd(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 5:
            raise TypeError("wigstd(cfg, f, bo, rvs, rho, ...) requires five positional arguments")
        f, bo, rvs, rho = args[1], args[2], args[3], args[4]
        p = _cfg_params(cfg, kwargs.pop("params", None))
        return _orig_wigstd(
            f, bo, rvs, rho,
            n=int(cfg.n),
            nx=int(cfg.nx),
            irlx=kwargs.pop("irlx", 0),
            exchange=kwargs.pop("exchange", False),
            params=p,
            overwrite=kwargs.pop("overwrite", True),
            use_numba=kwargs.pop("use_numba", getattr(cfg, "use_numba", True)),
            kernel_backend=kwargs.pop("kernel_backend", getattr(cfg, "kernel_backend", "auto")),
            workspace=kwargs.pop("workspace", None),
        )
    return _orig_wigstd(*args, **kwargs)


def poissonb(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 3:
            raise TypeError("poissonb(cfg, b, rvs, ...) requires b and rvs")
        b, rvs = args[1], args[2]
        kwargs = _extra_pos_to_kwargs(args[3:], ["vo", "vb", "rfa", "amax"], kwargs)
        p = _cfg_params(cfg, kwargs.pop("params", None))
        return _orig_poissonb(
            b, rvs,
            nx=int(cfg.nx),
            n=int(cfg.n),
            vo=kwargs.pop("vo", 0.0),
            vb=kwargs.pop("vb", 0.0),
            rfa=kwargs.pop("rfa", 650.0),
            amax=kwargs.pop("amax", None),
            params=p,
            save_csv=kwargs.pop("save_csv", False),
        )
    return _orig_poissonb(*args, **kwargs)


def poisson(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 3:
            raise TypeError("poisson(cfg, b, xa, ...) requires b and xa")
        b, xa = args[1], args[2]
        kwargs = _extra_pos_to_kwargs(args[3:], ["vo", "vb", "m"], kwargs)
        p = _cfg_params(cfg, kwargs.pop("params", None))
        return _orig_poisson(
            b, xa,
            nx=int(cfg.nx),
            n=int(cfg.n),
            vo=kwargs.pop("vo", 0.0),
            vb=kwargs.pop("vb", 0.0),
            m=kwargs.pop("m", 1),
            params=p,
        )
    return _orig_poisson(*args, **kwargs)


def curcalc(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 3:
            raise TypeError("curcalc(cfg, b, mode) or curcalc(cfg, b, rj, ...) requires at least b and mode/rj")
        b = args[1]
        third = args[2]
        if np.isscalar(third):
            rj = np.zeros(int(cfg.nx), dtype=float)
            kwargs = _extra_pos_to_kwargs(args[3:], [], kwargs)
            irj = kwargs.pop("irj", int(third))
        else:
            rj = third
            kwargs = _extra_pos_to_kwargs(args[3:], ["irj"], kwargs)
            irj = kwargs.pop("irj", 2)
        p = _cfg_params(cfg, kwargs.pop("params", None))
        return _orig_curcalc(
            b, rj,
            n=int(cfg.n),
            nx=int(cfg.nx),
            irj=irj,
            params=p,
            use_numba=kwargs.pop("use_numba", getattr(cfg, "use_numba", True)),
            kernel_backend=kwargs.pop("kernel_backend", getattr(cfg, "kernel_backend", "auto")),
        )
    return _orig_curcalc(*args, **kwargs)


def comp(*args, **kwargs) -> tuple[int, float]:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 3:
            raise TypeError("comp(cfg, x, xo, ...) requires x and xo")
        x, xo = args[1], args[2]
        kwargs = _extra_pos_to_kwargs(args[3:], ["eps"], kwargs)
        return _orig_comp(x, xo, nx=int(cfg.nx), eps=kwargs.pop("eps", 7.0e-10))
    return _orig_comp(*args, **kwargs)


def subsrvs(*args, **kwargs) -> np.ndarray:
    if args and _is_cfg(args[0]):
        cfg = args[0]
        if len(args) < 2:
            raise TypeError("subsrvs(cfg, rvs) requires rvs")
        rvs = args[1]
        p = _cfg_params(cfg, kwargs.pop("params", None))
        return _orig_subsrvs(rvs, nx=int(cfg.nx), params=p)
    return _orig_subsrvs(*args, **kwargs)
