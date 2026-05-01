from __future__ import annotations

import argparse
from pathlib import Path

from .config import LWWConfig
from .simulator import LWW1DSimulator
from .visualization import save_rtd_geometry_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 1D LWW Wigner-Poisson simulator.")
    parser.add_argument("mode", choices=["steady", "transient", "geometry"], help="simulation mode")
    parser.add_argument("--output", default="lww_output", help="output directory")
    parser.add_argument("--nx", type=int, default=86)
    parser.add_argument("--n", type=int, default=72)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--ivn", type=int, default=45)
    parser.add_argument("--itn", type=int, default=1000)
    parser.add_argument("--dbias", type=float, default=0.008)
    parser.add_argument("--sample-every", type=int, default=10, help="transient current sampling interval in iterations")
    parser.add_argument("--progress-every", type=int, default=50, help="verbose transient progress interval in iterations")
    parser.add_argument("--no-exchange", action="store_true")
    parser.add_argument("--no-numba", action="store_true", help="legacy alias for --backend python")
    parser.add_argument(
        "--backend",
        choices=["auto", "cpp", "numba", "python"],
        default="auto",
        help="kernel backend for matrix assembly/reductions",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    kernel_backend = "python" if args.no_numba else args.backend

    cfg = LWWConfig(
        nx=args.nx,
        n=args.n,
        bias=args.bias,
        exchange=not args.no_exchange,
        verbose=args.verbose,
        use_numba=not args.no_numba,
        kernel_backend=kernel_backend,
    )
    sim = LWW1DSimulator(cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "geometry":
        run_extra = {
            "mode": "geometry",
            "output": out,
        }
        sim.print_config_summary(run_extra)
        sim.save_config_summary(out, extra=run_extra)
        image_path = save_rtd_geometry_image(cfg, out / "rtd_geometry.png")
        print(f"geometry image wrote {image_path}")
    elif args.mode == "steady":
        run_extra = {
            "mode": "steady",
            "bias": args.bias,
            "max_iterations": args.max_iterations,
            "output": out,
        }
        sim.print_config_summary(run_extra)
        sim.save_config_summary(out, extra=run_extra)
        res = sim.solve_steady_state(bias=args.bias, max_iterations=args.max_iterations)
        sim.save_state(res.state, out)
        print(
            f"steady done: converged={res.converged}, iterations={res.iterations}, "
            f"max_density_change={res.max_density_change}, max_potential_update={res.max_potential_update}"
        )
    else:
        run_extra = {
            "mode": "transient",
            "initial_bias": args.bias,
            "max_iterations": args.max_iterations,
            "ivn": args.ivn,
            "itn": args.itn,
            "dbias": args.dbias,
            "sample_every": args.sample_every,
            "progress_every": args.progress_every,
            "output": out,
        }
        sim.print_config_summary(run_extra)
        sim.save_config_summary(out, extra=run_extra)
        steady = sim.solve_steady_state(bias=args.bias, max_iterations=args.max_iterations)
        res = sim.run_transient(
            steady.state,
            ivn=args.ivn,
            itn=args.itn,
            dbias=args.dbias,
            sample_every=args.sample_every,
            progress_every=args.progress_every,
            output_dir=out,
        )
        sim.save_state(res.state, out)
        sim.save_transient(res, out)
        sim.save_config_summary(out, extra=run_extra)
        print(f"transient done: wrote {len(res.bias_currents)} bias traces to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
