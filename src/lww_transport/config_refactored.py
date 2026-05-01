"""Grouped configuration objects for the LWW quantum transport simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import numpy as np


@dataclass(slots=True)
class DiscretizationParams:
    """Spatial and momentum grid parameters.
    
    Parameters
    ----------
    nx : int
        Number of spatial grid points. Default 86 matches the reference RTD
        configuration.
    n : int
        Number of momentum grid points. Must be even.
        Default 72 matches the reference RTD configuration.
    """
    nx: int = 86              # Spatial grid points
    n: int = 72               # Momentum grid points


@dataclass(slots=True)
class MaterialParams:
    """Physical constants and semiconductor band structure.
    
    Defaults are for GaAs-like material at T=77K.
    
    Parameters
    ----------
    c : float
        Speed of light [nm/ps]. Default 2997.92.
    rmo : float
        Effective mass ratio. Default 0.0569.
    rm : float
        Effective mass scaling. Default 0.0667.
    temp : float
        Temperature [K]. Default 77.0 K.
    rhc : float
        ℏc product [eV·nm]. Default 1973.29.
    re : float
        Electron density scaling. Default 2.0.
    rno : float
        Nominal doping density [1/nm³]. Default 1e-6.
    ri : float
        Ionized impurity ratio. Default 0.3.
    chemp : float
        Equilibrium chemical potential [eV]. Default 0.0864 (at 77K).
    """
    c: float = 2997.924580           # Speed of light [nm/ps]
    rmo: float = 0.05685675170       # Effective mass ratio
    rm: float = 6.67e-2               # Effective mass scaling
    temp: float = 77.0                # Temperature [K]
    rhc: float = 1973.29              # ℏc [eV·nm]
    re: float = 2.0                   # Electron density scaling
    rno: float = 1.0e-6               # Nominal doping [1/nm³]
    ri: float = 0.3                   # Ionized impurity ratio
    chemp: float = 0.0863814          # Chemical potential [eV]


@dataclass(slots=True)
class GeometryParams:
    """Double-barrier quantum well (DBQW) geometry.
    
    Describes the device structure: emitter | spacer | barrier | well | barrier | spacer | collector
    
    Parameters
    ----------
    box : float
        Total device length [nm]. Default 550 nm.
    well : float
        Quantum well width [nm]. Default 50 nm.
    barrier : float
        Barrier position/height parameter [nm]. Default 30 nm.
    spacer : float
        Spacer region width [nm]. Default 30 nm.
    pot : float
        Potential barrier height [eV]. Default 0.3 eV.
    """
    box: float = 550.0                # Total device length [nm]
    well: float = 50.0                # Quantum well width [nm]
    barrier: float = 30.0             # Barrier parameter [nm]
    spacer: float = 30.0              # Spacer region [nm]
    pot: float = 0.3                  # Barrier height [eV]


@dataclass(slots=True)
class OperatingConditions:
    """Bias voltage and thermal settings.
    
    Parameters
    ----------
    bias : float
        Applied bias voltage [V]. Default 0.0 V (zero bias).
    temperature : float
        Operating temperature [K]. Default 77.0 K (liquid nitrogen).
    """
    bias: float = 0.0                 # Applied bias [V]
    temperature: float = 77.0         # Temperature [K]


@dataclass(slots=True)
class SolverParams:
    """Convergence and numerical method settings.
    
    Parameters
    ----------
    irlx : int
        Relaxation type flag. Default 4.
    relaxation_tau : float
        Relaxation time parameter [ps]. Default 525.51.
    hbar_like : float
        Planck constant scaling factor. Default 0.6582.
    convergence_eps : float
        Density convergence threshold. Default 7e-10.
        Smaller values tighten convergence and increase runtime.
    poisson_vmax_stop : float
        Potential convergence criterion [eV]. Default 0.001.
    density_stop : float
        Density convergence criterion [1/nm³]. Default 3e-9.
    exchange : bool
        Include optional density-dependent exchange correction. Default False.
    rfa : float
        Relaxation factor for Poisson solver. Default 650.0.
    """
    irlx: int = 4
    relaxation_tau: float = 525.5074
    hbar_like: float = 0.65821869340
    convergence_eps: float = 7.0e-10
    poisson_vmax_stop: float = 1.0e-3
    density_stop: float = 3.0e-9
    exchange: bool = False
    rfa: float = 650.0


@dataclass(slots=True)
class ComputeParams:
    """Performance and debugging options.
    
    Parameters
    ----------
    dense_solver : str
        Linear solver backend. Default 'numpy'.
    kernel_backend : str
        Computational kernel implementation.
        Options: 'auto', 'cpp', 'numba', 'python'.
        Default 'auto', which selects the fastest available backend.
    use_numba : bool
        Enable Numba JIT compilation for hot loops.
        Default True.
    verbose : bool
        Print diagnostic information during solve.
        Default False.
    """
    dense_solver: str = "numpy"
    kernel_backend: str = "auto"
    use_numba: bool = True
    verbose: bool = False


@dataclass(slots=True, init=False)
class LWWConfig:
    """Unified 1D LWW quantum transport simulator configuration.
    
    Parameters are grouped by discretization, material, geometry, operating
    conditions, solver controls, and compute backend.
    
    Attributes
    ----------
    discretization : DiscretizationParams
        Grid resolution (nx, n).
    material : MaterialParams
        Semiconductor properties (GaAs defaults).
    geometry : GeometryParams
        Device structure (double-barrier QW).
    operating : OperatingConditions
        Bias voltage and temperature.
    solver : SolverParams
        Numerical solver settings.
    compute : ComputeParams
        Backend and performance options.
    
    Examples
    --------
    >>> cfg = LWWConfig.standard_rtd()
    >>> cfg = LWWConfig.quick_test(nx=32, n=32)
    >>> cfg = LWWConfig(
    ...     discretization=DiscretizationParams(nx=48, n=48),
    ...     material=MaterialParams(temp=300.0),
    ...     operating=OperatingConditions(bias=0.1),
    ... )
    """

    discretization: DiscretizationParams = field(default_factory=DiscretizationParams)
    material: MaterialParams = field(default_factory=MaterialParams)
    geometry: GeometryParams = field(default_factory=GeometryParams)
    operating: OperatingConditions = field(default_factory=OperatingConditions)
    solver: SolverParams = field(default_factory=SolverParams)
    compute: ComputeParams = field(default_factory=ComputeParams)

    # Cached grid coordinates.
    x: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)

    def __init__(
        self,
        discretization: DiscretizationParams | None = None,
        material: MaterialParams | None = None,
        geometry: GeometryParams | None = None,
        operating: OperatingConditions | None = None,
        solver: SolverParams | None = None,
        compute: ComputeParams | None = None,
        **flat_overrides,
    ) -> None:
        """Create a grouped configuration with optional flat overrides.

        Flat keyword arguments are supported for compatibility, e.g.
        ``LWWConfig(nx=10, n=8, exchange=False)``.
        """
        self.discretization = discretization if discretization is not None else DiscretizationParams()
        self.material = material if material is not None else MaterialParams()
        self.geometry = geometry if geometry is not None else GeometryParams()
        self.operating = operating if operating is not None else OperatingConditions()
        self.solver = solver if solver is not None else SolverParams()
        self.compute = compute if compute is not None else ComputeParams()
        self._apply_flat_overrides(flat_overrides)
        self.__post_init__()

    def _apply_flat_overrides(self, overrides: dict[str, object]) -> None:
        if not overrides:
            return

        flat_targets = {
            "nx": (self.discretization, "nx"),
            "n": (self.discretization, "n"),
            "c": (self.material, "c"),
            "rmo": (self.material, "rmo"),
            "rm": (self.material, "rm"),
            "rhc": (self.material, "rhc"),
            "re": (self.material, "re"),
            "rno": (self.material, "rno"),
            "ri": (self.material, "ri"),
            "chemp": (self.material, "chemp"),
            "box": (self.geometry, "box"),
            "well": (self.geometry, "well"),
            "barrier": (self.geometry, "barrier"),
            "spacer": (self.geometry, "spacer"),
            "pot": (self.geometry, "pot"),
            "bias": (self.operating, "bias"),
            "irlx": (self.solver, "irlx"),
            "exchange": (self.solver, "exchange"),
            "relaxation_tau": (self.solver, "relaxation_tau"),
            "hbar_like": (self.solver, "hbar_like"),
            "convergence_eps": (self.solver, "convergence_eps"),
            "poisson_vmax_stop": (self.solver, "poisson_vmax_stop"),
            "density_stop": (self.solver, "density_stop"),
            "rfa": (self.solver, "rfa"),
            "dense_solver": (self.compute, "dense_solver"),
            "kernel_backend": (self.compute, "kernel_backend"),
            "use_numba": (self.compute, "use_numba"),
            "verbose": (self.compute, "verbose"),
        }

        for name, value in overrides.items():
            if name in {"temp", "temperature"}:
                self.operating.temperature = float(value)
                self.material.temp = float(value)
                continue
            target = flat_targets.get(name)
            if target is None:
                raise TypeError(f"unexpected LWWConfig parameter: {name}")
            obj, attr = target
            setattr(obj, attr, value)

    def __post_init__(self) -> None:
        """Validate parameters and compute cached grids."""
        self.material.temp = self.operating.temperature

        if self.discretization.n <= 1 or self.discretization.n % 2:
            raise ValueError("n must be an even integer > 1")
        if self.discretization.nx <= 3:
            raise ValueError("nx must be > 3")
        
        valid_backends = {"auto", "cpp", "numba", "python"}
        if self.compute.kernel_backend not in valid_backends:
            raise ValueError(f"kernel_backend must be one of {valid_backends}")
        
        self.x = np.linspace(0.0, self.geometry.box, self.discretization.nx)
        
        delx = self.geometry.box / (self.discretization.nx - 1)
        delk = np.pi / (delx * self.discretization.n)
        self.k = (np.arange(self.discretization.n) - self.discretization.n / 2 + 0.5) * delk

    @property
    def nx(self) -> int:
        return self.discretization.nx

    @property
    def n(self) -> int:
        return self.discretization.n

    @property
    def c(self) -> float:
        return self.material.c

    @property
    def rmo(self) -> float:
        return self.material.rmo

    @property
    def rm(self) -> float:
        return self.material.rm

    @property
    def temp(self) -> float:
        return self.operating.temperature

    @property
    def rhc(self) -> float:
        return self.material.rhc

    @property
    def re(self) -> float:
        return self.material.re

    @property
    def rno(self) -> float:
        return self.material.rno

    @property
    def ri(self) -> float:
        return self.material.ri

    @property
    def box(self) -> float:
        return self.geometry.box

    @property
    def well(self) -> float:
        return self.geometry.well

    @property
    def barrier(self) -> float:
        return self.geometry.barrier

    @property
    def spacer(self) -> float:
        return self.geometry.spacer

    @property
    def pot(self) -> float:
        return self.geometry.pot

    @property
    def irlx(self) -> int:
        return self.solver.irlx

    @property
    def chemp(self) -> float:
        return self.material.chemp

    @property
    def exchange(self) -> bool:
        return self.solver.exchange

    @property
    def relaxation_tau(self) -> float:
        return self.solver.relaxation_tau

    @property
    def hbar_like(self) -> float:
        return self.solver.hbar_like

    @property
    def convergence_eps(self) -> float:
        return self.solver.convergence_eps

    @property
    def rfa(self) -> float:
        return self.solver.rfa

    @property
    def poisson_vmax_stop(self) -> float:
        return self.solver.poisson_vmax_stop

    @property
    def density_stop(self) -> float:
        return self.solver.density_stop

    @property
    def dense_solver(self) -> str:
        return self.compute.dense_solver

    @property
    def verbose(self) -> bool:
        return self.compute.verbose

    @property
    def use_numba(self) -> bool:
        return self.compute.use_numba

    @property
    def kernel_backend(self) -> str:
        return self.compute.kernel_backend

    @property
    def rmass(self) -> float:
        """Effective mass [eV·ps²/nm²]."""
        return self.material.rm * self.material.rmo * (self.material.c**2)

    @property
    def bbeta(self) -> float:
        """Inverse thermal energy [1/eV]."""
        return 11604.5 / self.operating.temperature

    @property
    def dens(self) -> float:
        """Nominal electron density [1/nm³]."""
        return self.material.re * self.material.rno

    @property
    def densi(self) -> float:
        """Ionized impurity density [1/nm³]."""
        return (1.0 + self.material.ri) * self.dens / (1.0 - self.material.ri)

    @property
    def delx(self) -> float:
        """Spatial grid spacing [nm]."""
        return self.geometry.box / (self.discretization.nx - 1)

    @property
    def delk(self) -> float:
        """Momentum grid spacing [1/nm]."""
        return np.pi / (self.delx * self.discretization.n)

    @property
    def size(self) -> int:
        """Total Wigner phase-space dimension (nx × n)."""
        return self.discretization.nx * self.discretization.n

    def with_bias(self, bias: float) -> "LWWConfig":
        """Set the applied bias voltage."""
        self.operating.bias = bias
        self.__post_init__()
        return self

    def with_temperature(self, temperature: float) -> "LWWConfig":
        """Set operating temperature and return self."""
        self.operating.temperature = temperature
        self.material.temp = temperature
        self.__post_init__()
        return self

    def with_grid(self, nx: int, n: int) -> "LWWConfig":
        """Set grid resolution and return self."""
        self.discretization.nx = nx
        self.discretization.n = n
        self.__post_init__()
        return self

    def with_exchange(self, exchange: bool) -> "LWWConfig":
        """Set density-dependent exchange correction."""
        self.solver.exchange = exchange
        self.__post_init__()
        return self

    def with_verbose(self, verbose: bool) -> "LWWConfig":
        """Enable/disable verbose output."""
        self.compute.verbose = verbose
        self.__post_init__()
        return self

    @classmethod
    def quick_test(cls, nx: int = 16, n: int = 16) -> "LWWConfig":
        """Small-grid configuration for unit tests and smoke tests."""
        cfg = cls()
        cfg.discretization.nx = nx
        cfg.discretization.n = n
        cfg.solver.convergence_eps = 1e-6
        cfg.__post_init__()
        return cfg

    @classmethod
    def standard_rtd(cls) -> "LWWConfig":
        """Standard resonant tunneling diode at 77K.
        
        Reproduces the reference 1D RTD LWW configuration.
        - GaAs at 77K
        - Full resolution (86×72 grid)
        - Double-barrier QW geometry
        """
        return cls(
            discretization=DiscretizationParams(nx=86, n=72),
            material=MaterialParams(temp=77.0, chemp=0.0863814),
            geometry=GeometryParams(box=550.0, well=50.0, barrier=30.0, spacer=30.0, pot=0.3),
            operating=OperatingConditions(bias=0.0, temperature=77.0),
            solver=SolverParams(),
            compute=ComputeParams(),
        )

    @classmethod
    def room_temperature_rtd(cls) -> "LWWConfig":
        """RTD configuration at 300 K."""
        cfg = cls.standard_rtd()
        cfg.operating.temperature = 300.0
        cfg.material.temp = 300.0
        cfg.material.chemp = 0.0373
        cfg.__post_init__()
        return cfg

    @classmethod
    def coarse_grid_rtd(cls, nx: int = 32, n: int = 32) -> "LWWConfig":
        """Coarse-grid RTD configuration for parameter sweeps."""
        cfg = cls.standard_rtd()
        cfg.discretization.nx = nx
        cfg.discretization.n = n
        cfg.__post_init__()
        return cfg

    @classmethod
    def fine_grid_rtd(cls, nx: int = 128, n: int = 96) -> "LWWConfig":
        """High-resolution RTD configuration."""
        cfg = cls.standard_rtd()
        cfg.discretization.nx = nx
        cfg.discretization.n = n
        cfg.solver.convergence_eps = 1e-12
        cfg.__post_init__()
        return cfg

    @classmethod
    def narrow_well_rtd(cls) -> "LWWConfig":
        """RTD configuration with a 25 nm quantum well."""
        cfg = cls.standard_rtd()
        cfg.geometry.well = 25.0
        cfg.__post_init__()
        return cfg

    @classmethod
    def wide_well_rtd(cls) -> "LWWConfig":
        """RTD configuration with a 75 nm quantum well."""
        cfg = cls.standard_rtd()
        cfg.geometry.well = 75.0
        cfg.__post_init__()
        return cfg


def format_config_summary(
    cfg: LWWConfig,
    extra: Mapping[str, object] | None = None,
) -> str:
    """Return a human-readable summary of the configuration.
    
    Examples
    --------
    >>> cfg = LWWConfig.standard_rtd()
    >>> print(format_config_summary(cfg))
    """
    lines = [
        "=" * 70,
        "LWW QUANTUM TRANSPORT CONFIGURATION SUMMARY",
        "=" * 70,
    ]

    if extra:
        lines.extend(["", "[RUN]"])
        for key, value in extra.items():
            lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "[DISCRETIZATION]",
        f"  Spatial points (nx):   {cfg.discretization.nx}",
        f"  Momentum points (n):   {cfg.discretization.n}",
        f"  Total phase-space:     {cfg.size} points",
        f"  Spatial step (Δx):     {cfg.delx:.4f} nm",
        f"  Momentum step (Δk):    {cfg.delk:.6f} nm⁻¹",
        "",
        "[MATERIAL] (GaAs)",
        f"  Temperature:           {cfg.operating.temperature:.1f} K",
        f"  Chemical potential:    {cfg.material.chemp:.6f} eV",
        f"  Electron density:      {cfg.dens:.3e} nm⁻³",
        "",
        "[GEOMETRY] (Double-Barrier QW)",
        f"  Device length (box):   {cfg.geometry.box:.1f} nm",
        f"  Well width:            {cfg.geometry.well:.1f} nm",
        f"  Barrier height (pot):  {cfg.geometry.pot:.2f} eV",
        f"  Barrier position:      {cfg.geometry.barrier:.1f} nm",
        "",
        "[OPERATING CONDITIONS]",
        f"  Applied bias:          {cfg.operating.bias:.4f} V",
        f"  Temperature:           {cfg.operating.temperature:.1f} K",
        "",
        "[SOLVER]",
        f"  Exchange interactions: {cfg.solver.exchange}",
        f"  Convergence threshold: {cfg.solver.convergence_eps:.2e}",
        f"  Relaxation parameter:  {cfg.solver.relaxation_tau:.2f} ps",
        "",
        "[COMPUTE]",
        f"  Kernel backend:        {cfg.compute.kernel_backend}",
        f"  Use Numba JIT:         {cfg.compute.use_numba}",
        f"  Verbose output:        {cfg.compute.verbose}",
        "=" * 70,
    ])
    return "\n".join(lines) + "\n"


def print_config_summary(
    cfg: LWWConfig,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Print a human-readable summary of the configuration."""
    print(format_config_summary(cfg, extra), end="")


def save_config_summary(
    cfg: LWWConfig,
    output_dir: str | Path,
    filename: str = "config_summary.txt",
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Save the human-readable configuration summary and return its path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    path.write_text(format_config_summary(cfg, extra), encoding="utf-8")
    return path


__all__ = [
    "ComputeParams",
    "DiscretizationParams",
    "GeometryParams",
    "LWWConfig",
    "MaterialParams",
    "OperatingConditions",
    "SolverParams",
    "format_config_summary",
    "print_config_summary",
    "save_config_summary",
]

