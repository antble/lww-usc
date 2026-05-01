from lww_transport import LWWConfig, LWW1DSimulator

cfg = LWWConfig(nx=86, n=72, exchange=True, verbose=True)
sim = LWW1DSimulator(cfg)

steady = sim.solve_steady_state(bias=0.0, max_iterations=200)
print(steady.converged, steady.iterations)
print(steady.current[-1])

sim.save_state(steady.state, "lww_output")