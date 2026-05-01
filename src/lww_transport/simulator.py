from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .config import LWWConfig, format_config_summary, print_config_summary, save_config_summary
from .core import WignerSolveWorkspace, comp, curcalc, fbndry, poisson, poissonb, rvscalc, subsrvs, wigstd


@dataclass(slots=True)
class SimulationState:
    rvs: np.ndarray
    f: np.ndarray
    fr: np.ndarray
    fo: np.ndarray
    density_previous: np.ndarray
    current_or_density: np.ndarray

    def copy(self) -> "SimulationState":
        return SimulationState(
            rvs=self.rvs.copy(),
            f=self.f.copy(),
            fr=self.fr.copy(),
            fo=self.fo.copy(),
            density_previous=self.density_previous.copy(),
            current_or_density=self.current_or_density.copy(),
        )

@dataclass(slots=True)
class SteadyStateResult:
    state: SimulationState
    density: np.ndarray
    current: np.ndarray
    iterations: int
    converged: bool
    max_density_change: float
    max_potential_update: float


@dataclass(slots=True)
class TransientResult:
    state: SimulationState
    bias_currents: dict[float, pd.DataFrame]


class LWW1DSimulator:
    """One-dimensional LWW simulation workflow."""

    def __init__(self, config: LWWConfig | None = None):
        self.cfg = config or LWWConfig()

    def config_summary(self, extra: dict[str, object] | None = None) -> str:
        """Return a human-readable summary for the simulator configuration."""
        return format_config_summary(self.cfg, extra)

    def print_config_summary(self, extra: dict[str, object] | None = None) -> None:
        """Print a human-readable summary for the simulator configuration."""
        print_config_summary(self.cfg, extra)

    def save_config_summary(
        self,
        output_dir: str | Path,
        filename: str = "config_summary.txt",
        extra: dict[str, object] | None = None,
    ) -> Path:
        """Save the simulator configuration summary next to simulation output."""
        return save_config_summary(self.cfg, output_dir, filename=filename, extra=extra)

    def zeros_state(self) -> SimulationState:
        cfg = self.cfg
        return SimulationState(
            rvs=np.zeros(cfg.nx, dtype=float),
            f=np.zeros(cfg.size, dtype=float),
            fr=np.zeros(cfg.size, dtype=float),
            fo=np.zeros(cfg.size, dtype=float),
            density_previous=np.zeros(cfg.nx, dtype=float),
            current_or_density=np.zeros(cfg.nx, dtype=float),
        )

    def initial_zero_bias_state(self, exchange: bool | None = None) -> SimulationState:
        """Compute the zero-bias initial Wigner state."""
        cfg = self.cfg
        st = self.zeros_state()
        rho = np.zeros(cfg.nx, dtype=float)
        workspace = WignerSolveWorkspace.create(cfg.n, cfg.nx)
        rvscalc(cfg, st.rvs, 0.0, 2)
        fbndry(cfg, st.f)
        st.f[:] = wigstd(
            cfg,
            st.f,
            st.f,
            st.rvs,
            rho,
            irlx=0,
            exchange=cfg.exchange if exchange is None else exchange,
            workspace=workspace,
        )
        curcalc(cfg, st.f, st.current_or_density, irj=2)
        st.fr[:] = st.f
        st.fo[:] = st.f
        return st

    def solve_steady_state(
        self,
        bias: float = 0.0,
        dbias: float = 0.1,
        max_iterations: int = 200,
        initial_state: SimulationState | None = None,
        exchange: bool | None = None,
    ) -> SteadyStateResult:
        """Run the self-consistent steady-state Wigner-Poisson loop."""
        del dbias
        cfg = self.cfg
        exchange = cfg.exchange if exchange is None else exchange
        st = initial_state.copy() if initial_state is not None else self.initial_zero_bias_state(exchange=exchange)
        bm = np.zeros(cfg.size, dtype=float)
        fbndry(cfg, bm)
        rhs = np.empty(cfg.size, dtype=float)
        rho = np.zeros(cfg.nx, dtype=float)
        density = np.empty(cfg.nx, dtype=float)
        workspace = WignerSolveWorkspace.create(cfg.n, cfg.nx)
        vmax = [0.0]
        aamax = 1.0
        nmx = 0
        amax = float("inf")
        iterations = 0

        while True:
            poissonb(cfg, st.f, st.rvs, 0.0, -bias, cfg.rfa, vmax)
            fbndry(cfg, rhs)
            st.f[:] = wigstd(cfg, rhs, st.fo, st.rvs, rho, irlx=1, exchange=exchange, workspace=workspace)
            curcalc(cfg, st.f, density, irj=1)
            nmx, amax = comp(cfg, density, st.density_previous)
            iterations += 1
            if cfg.verbose:
                print(
                    f"steady: bias={bias}, nmx={nmx}, iteration={iterations}, "
                    f"amax={amax}, vmax={vmax[0]}"
                )
            if amax < aamax:
                aamax = amax
            if nmx == 1 or iterations > max_iterations or (amax <= cfg.density_stop and vmax[0] <= cfg.poisson_vmax_stop):
                st.current_or_density[:] = density
                break
            subsrvs(cfg, st.rvs)
            st.density_previous[:] = density

        current = np.empty(cfg.nx, dtype=float)
        curcalc(cfg, st.f, current, irj=2)
        return SteadyStateResult(
            state=st,
            density=st.current_or_density.copy(),
            current=current,
            iterations=iterations,
            converged=bool(nmx == 1 or (amax <= cfg.density_stop and vmax[0] <= cfg.poisson_vmax_stop)),
            max_density_change=float(amax),
            max_potential_update=float(vmax[0]),
        )

    def run_transient(
        self,
        state: SimulationState,
        ivn: int = 45,
        itn: int = 1000,
        dbias: float = 0.008,
        exchange: bool | None = None,
        sample_every: int = 10,
        progress_every: int | None = 50,
        output_dir: str | Path | None = None,
        transient_prefix: str = "lww_tcurl",
        state_prefix: str = "lww",
    ) -> TransientResult:
        """Run the transient bias sweep.

        When ``cfg.verbose`` is true, print flushed progress lines at the start
        and end of each bias point and every ``progress_every`` iterations.
        If ``output_dir`` is supplied, transient current trace CSV files are
        written incrementally as samples are produced, and state checkpoint
        files are refreshed after each completed bias point.
        """
        cfg = self.cfg
        exchange = cfg.exchange if exchange is None else exchange
        st = state.copy()
        bm = np.zeros(cfg.size, dtype=float)
        rho = np.zeros(cfg.nx, dtype=float)
        workspace = WignerSolveWorkspace.create(cfg.n, cfg.nx)
        rc = -4.0 * 0.6582186935
        results: dict[float, pd.DataFrame] = {}
        progress_enabled = cfg.verbose and progress_every is not None and progress_every > 0
        output = Path(output_dir) if output_dir is not None else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            self.save_config_summary(
                output,
                extra={
                    "mode": "transient",
                    "ivn": ivn,
                    "itn": itn,
                    "dbias": dbias,
                    "sample_every": sample_every,
                    "progress_every": progress_every,
                },
            )

        for iv in range(1, ivn + 1):
            bias = iv * dbias
            itsteps: list[int] = []
            currents: list[float] = []
            nnp = 0
            fbndry(cfg, bm)
            trace_path = output / f"{transient_prefix}_{float(bias):.4f}.csv" if output is not None else None
            if trace_path is not None:
                pd.DataFrame(columns=["itstep", "current"]).to_csv(trace_path, index=False)
            if cfg.verbose:
                trace_message = f", trace={trace_path}" if trace_path is not None else ""
                print(
                    f"transient: bias={bias:.6g} ({iv}/{ivn}) start, max_steps={itn}{trace_message}",
                    flush=True,
                )
            last_itstep = 0
            last_amax = float("nan")
            last_nmx = 0
            stop_reason = "max_steps"
            for itstep in range(1, itn + 1):
                last_itstep = itstep
                nnp += 1
                poisson(cfg, st.f, st.rvs, 0.0, -bias)
                st.fo[:] = st.f
                st.f[:] = rc * st.f + 2.0 * bm
                st.f[:] = wigstd(
                    cfg,
                    st.f,
                    st.fr,
                    st.rvs,
                    rho,
                    irlx=cfg.irlx,
                    exchange=exchange,
                    workspace=workspace,
                )
                st.f[:] = st.f - st.fo
                curcalc(cfg, st.f, rho, irj=1)
                nmx, amax = comp(cfg, rho, st.density_previous)
                last_nmx = nmx
                last_amax = float(amax)
                st.density_previous[:] = rho
                if progress_enabled and (itstep == 1 or itstep % progress_every == 0):
                    print(
                        f"transient: bias={bias:.6g} ({iv}/{ivn}) "
                        f"step={itstep}/{itn}, amax={amax:.3e}, nmx={nmx}, samples={len(currents)}",
                        flush=True,
                    )
                if nmx == 3:
                    stop_reason = "nmx=3"
                    break
                if amax <= 1.0e-15:
                    stop_reason = "amax<=1e-15"
                    break
                if nnp == sample_every:
                    current = curcalc(cfg, st.f, st.current_or_density, irj=2)
                    itsteps.append(itstep)
                    sampled_current = float(current[cfg.nx - 1])
                    currents.append(sampled_current)
                    if trace_path is not None:
                        pd.DataFrame(
                            {"itstep": [itstep], "current": [sampled_current]}
                        ).to_csv(trace_path, mode="a", header=False, index=False)
                    nnp = 0
            results[float(bias)] = pd.DataFrame({"itstep": itsteps, "current": currents})
            if output is not None:
                self.save_state(st, output, prefix=state_prefix)
            if cfg.verbose:
                print(
                    f"transient: bias={bias:.6g} ({iv}/{ivn}) done, "
                    f"steps={last_itstep}, samples={len(currents)}, "
                    f"last_amax={last_amax:.3e}, last_nmx={last_nmx}, reason={stop_reason}",
                    flush=True,
                )
        return TransientResult(state=st, bias_currents=results)

    def save_state(self, state: SimulationState, output_dir: str | Path, prefix: str = "lww") -> None:
        """Write state arrays in the legacy CSV format."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "config_summary.txt").exists():
            self.save_config_summary(output)
        np.savetxt(output / f"{prefix}_pypoten.csv", state.rvs, delimiter=",")
        np.savetxt(output / f"{prefix}_pydensity.csv", state.density_previous, delimiter=",")
        np.savetxt(output / f"{prefix}_pycurrent.csv", state.current_or_density, delimiter=",")
        np.savetxt(output / f"{prefix}_pywigner.csv", state.f, delimiter=",")
        np.savetxt(output / f"{prefix}_pywigner_ss.csv", state.fr, delimiter=",")

    def load_state(self, input_dir: str | Path, prefix: str = "lww") -> SimulationState:
        """Load state arrays from the legacy CSV format."""
        inp = Path(input_dir)
        st = self.zeros_state()
        st.rvs[:] = np.loadtxt(inp / f"{prefix}_pypoten.csv", delimiter=",").reshape(-1)
        st.density_previous[:] = np.loadtxt(inp / f"{prefix}_pydensity.csv", delimiter=",").reshape(-1)
        st.f[:] = np.loadtxt(inp / f"{prefix}_pywigner.csv", delimiter=",").reshape(-1)
        st.fr[:] = np.loadtxt(inp / f"{prefix}_pywigner_ss.csv", delimiter=",").reshape(-1)
        st.fo[:] = st.f
        return st

    def save_transient(self, result: TransientResult, output_dir: str | Path, prefix: str = "lww_tcurl") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for bias, df in result.bias_currents.items():
            df.to_csv(out / f"{prefix}_{float(bias):.4f}.csv", index=False)
