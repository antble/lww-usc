from pathlib import Path

from lww_transport import LWWConfig, LWW1DSimulator

# ── Output ────────────────────────────────────────────────────────────────────
output_folder = Path("lww_output_transient_v2")

# ── Discretization ────────────────────────────────────────────────────────────
nx = 86
n  = 72

# ── Material (GaAs defaults) ──────────────────────────────────────────────────
temp  = 77.0          # Temperature [K]
chemp = 0.0863814     # Chemical potential [eV]
rno   = 1.0e-6        # Nominal doping [1/nm³]
re    = 2.0           # Spin degeneracy

# ── Geometry ──────────────────────────────────────────────────────────────────
box     = 550.0       # Device length [nm]
well    = 100.0        # Quantum well width [nm]
barrier = 30.0        # Barrier width [nm]
spacer  = 30.0        # Spacer region [nm]
pot     = 0.30        # Barrier height [eV]

# ── Solver ────────────────────────────────────────────────────────────────────
exchange  = False     # Density-dependent exchange correction
irlx      = 4        # Relaxation/scattering flag (0 = off)
rfa       = 650.0    # Poisson relaxation factor

# ── Transient sweep ───────────────────────────────────────────────────────────
initial_bias             = 0.0
bias_step                = 0.008   # [V]
number_of_bias_points    = 50      # ivn
transient_steps_per_bias = 1000    # itn
max_steady_iterations    = 200
sample_every             = 10
progress_every           = 50

# ── Compute ───────────────────────────────────────────────────────────────────
kernel_backend = "cpp"             # "python", "numba", or "cpp"
verbose        = True

# ─────────────────────────────────────────────────────────────────────────────

output_folder.mkdir(parents=True, exist_ok=True)

cfg = LWWConfig(
    nx=nx,
    n=n,
    temp=temp,
    chemp=chemp,
    rno=rno,
    re=re,
    box=box,
    well=well,
    barrier=barrier,
    spacer=spacer,
    pot=pot,
    exchange=exchange,
    irlx=irlx,
    rfa=rfa,
    bias=initial_bias,
    kernel_backend=kernel_backend,
    verbose=verbose,
)

sim = LWW1DSimulator(cfg)

sim.save_config_summary(
    output_folder,
    extra={
        "mode": "transient",
        "initial_bias": initial_bias,
        "max_steady_iterations": max_steady_iterations,
        "number_of_bias_points": number_of_bias_points,
        "transient_steps_per_bias": transient_steps_per_bias,
        "bias_step": bias_step,
        "sample_every": sample_every,
        "progress_every": progress_every,
        "output": str(output_folder),
    },
)

steady = sim.solve_steady_state(
    bias=initial_bias,
    max_iterations=max_steady_iterations,
)

if verbose:
    print(f"Steady-state convergence: {steady.converged}")
    print(f"Steady-state iterations:  {steady.iterations}")
    print(f"Steady-state J(L):        {steady.current[-1]:.6e} A/m²")
    print("-" * 40)

result = sim.run_transient(
    steady.state,
    ivn=number_of_bias_points,
    itn=transient_steps_per_bias,
    dbias=bias_step,
    sample_every=sample_every,
    progress_every=progress_every,
    output_dir=output_folder,
)

sim.save_state(result.state, output_folder)
sim.save_transient(result, output_folder)

print(
    f"transient done: wrote {len(result.bias_currents)} bias traces and "
    f"state CSV files to {output_folder}"
)
