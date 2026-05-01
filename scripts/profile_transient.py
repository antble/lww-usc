from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

from lww_transport import LWWConfig, LWW1DSimulator
from lww_transport.core import CPP_AVAILABLE, NUMBA_AVAILABLE, OPENMP_ENABLED, OPENMP_THREADS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a small transient LWW simulation.")
    parser.add_argument("--nx", type=int, default=43)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--ivn", type=int, default=1)
    parser.add_argument("--itn", type=int, default=2)
    parser.add_argument("--dbias", type=float, default=0.008)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--exchange", action="store_true")
    parser.add_argument("--no-numba", action="store_true")
    parser.add_argument("--backend", choices=["auto", "cpp", "numba", "python"], default="auto")
    parser.add_argument("--sort", default="cumtime", choices=["cumtime", "tottime", "calls"])
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_numba = not args.no_numba
    kernel_backend = "python" if args.no_numba else args.backend
    cfg = LWWConfig(
        nx=args.nx,
        n=args.n,
        exchange=args.exchange,
        verbose=False,
        use_numba=use_numba,
        kernel_backend=kernel_backend,
    )
    sim = LWW1DSimulator(cfg)
    state = sim.initial_zero_bias_state(exchange=args.exchange)

    prof = cProfile.Profile()
    start = time.perf_counter()
    prof.enable()
    sim.run_transient(
        state,
        ivn=args.ivn,
        itn=args.itn,
        dbias=args.dbias,
        exchange=args.exchange,
        sample_every=args.sample_every,
        progress_every=None,
    )
    prof.disable()
    elapsed = time.perf_counter() - start

    print(
        f"profile: nx={args.nx}, n={args.n}, ivn={args.ivn}, itn={args.itn}, "
        f"backend={kernel_backend}, cpp_available={CPP_AVAILABLE}, "
        f"numba_available={NUMBA_AVAILABLE}, openmp={OPENMP_ENABLED}, "
        f"openmp_threads={OPENMP_THREADS}, elapsed={elapsed:.6f}s"
    )
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).strip_dirs().sort_stats(args.sort).print_stats(args.limit)
    print(stream.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
