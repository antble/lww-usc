from pathlib import Path
from lww_transport import LWWConfig, LWW1DSimulator


# Edit these values before running the script.
# example_folder = Path(__file__).resolve().parent
output_folder = "lww_output_transient"
nx = 86
n = 72
initial_bias = 0.0
max_steady_iterations = 200      # Max iterations for the Wigner-Poisson loop
number_of_bias_points = 50        # ivn: Total bias points in the sweep
transient_steps_per_bias = 1000  # itn: Max time-steps per bias level
bias_step = 0.008
sample_every = 10
progress_every = 50
kernel_backend = "cpp"           # "python", "numba", or "cpp"
verbose = True

cfg = LWWConfig(
    nx=nx,
    n=n,
    bias=initial_bias,
    verbose=verbose,
    kernel_backend=kernel_backend,
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
        "output": output_folder,
    },
)

steady = sim.solve_steady_state(
    bias=initial_bias,
    max_iterations=max_steady_iterations,
)
if verbose:
    print(f"Steady-state convergence: {steady.converged}")
    print(f"Steady-state iterations: {steady.iterations}")
    # Accessing the terminal current density J(L) from the spatial profile
    print(f"Steady-state terminal current density: {steady.current[-1]:.6e} A/m^2")
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
