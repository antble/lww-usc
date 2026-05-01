import numpy as np

from lww_transport import LWWConfig, LWW1DSimulator, format_config_summary
from lww_transport.core import CPP_AVAILABLE, WignerSolveWorkspace, curcalc, fbndry, fermi_function, rvscalc, wigstd
from lww_transport.visualization import (
    geometry_potential_profile,
    rtd_geometry_regions,
    save_rtd_geometry_image,
    save_wigner_phase_space_image,
    wigner_phase_space_grids,
)


def test_boundary_and_fermi_shapes():
    cfg = LWWConfig(nx=10, n=8, exchange=False)
    fermi = fermi_function(cfg)
    b = np.zeros(cfg.size)
    fbndry(cfg, b)
    assert fermi.shape == (cfg.n,)
    assert b.shape == (cfg.size,)
    assert np.any(b != 0)


def test_external_potential_shape():
    cfg = LWWConfig(nx=10, n=8, exchange=False)
    rvs = np.zeros(cfg.nx)
    rvscalc(cfg, rvs, bias=0.0, isc=2)
    assert rvs.shape == (cfg.nx,)
    assert np.all(np.isfinite(rvs))


def test_initial_zero_bias_small_grid():
    cfg = LWWConfig(nx=10, n=8, exchange=False)
    sim = LWW1DSimulator(cfg)
    state = sim.initial_zero_bias_state(exchange=False)
    assert state.f.shape == (cfg.size,)
    assert state.rvs.shape == (cfg.nx,)
    assert np.all(np.isfinite(state.f))
    assert np.all(np.isfinite(curcalc(cfg, state.f, 1)))


def test_run_transient_verbose_progress(capsys):
    cfg = LWWConfig(nx=10, n=8, exchange=False, verbose=True)
    sim = LWW1DSimulator(cfg)
    state = sim.initial_zero_bias_state(exchange=False)

    result = sim.run_transient(
        state,
        ivn=1,
        itn=2,
        dbias=0.008,
        exchange=False,
        sample_every=1,
        progress_every=1,
    )

    captured = capsys.readouterr()
    assert "transient: bias=0.008 (1/1) start" in captured.out
    assert "step=1/2" in captured.out
    assert "transient: bias=0.008 (1/1) done" in captured.out
    assert 0.008 in result.bias_currents


def test_run_transient_incremental_output(tmp_path):
    cfg = LWWConfig(nx=10, n=8, exchange=False)
    sim = LWW1DSimulator(cfg)
    state = sim.initial_zero_bias_state(exchange=False)

    result = sim.run_transient(
        state,
        ivn=1,
        itn=2,
        dbias=0.008,
        exchange=False,
        sample_every=1,
        output_dir=tmp_path,
    )

    trace_file = tmp_path / "lww_tcurl_0.008.csv"
    assert trace_file.exists()
    trace = np.loadtxt(trace_file, delimiter=",", skiprows=1)
    assert trace.shape == (2, 2)
    assert (tmp_path / "lww_pywigner.csv").exists()
    assert (tmp_path / "config_summary.txt").exists()
    assert len(result.bias_currents[0.008]) == 2


def test_refactored_config_supports_flat_overrides():
    cfg = LWWConfig(nx=12, n=10, exchange=False, temp=300.0, kernel_backend="python")

    assert cfg.nx == 12
    assert cfg.n == 10
    assert cfg.exchange is False
    assert cfg.temp == 300.0
    assert cfg.material.temp == 300.0
    assert "LWW QUANTUM TRANSPORT CONFIGURATION SUMMARY" in format_config_summary(cfg)


def test_rtd_geometry_image_output(tmp_path):
    cfg = LWWConfig(nx=10, n=8, exchange=False)
    regions = rtd_geometry_regions(cfg)
    x, potential = geometry_potential_profile(cfg, points=100)
    image_path = save_rtd_geometry_image(cfg, tmp_path / "rtd_geometry.png")

    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert x.shape == potential.shape == (100,)
    assert any(region.kind == "barrier" for region in regions)
    assert np.isclose(potential.max(), cfg.geometry.pot)


def test_wigner_phase_space_image_output(tmp_path):
    cfg = LWWConfig(nx=10, n=8, exchange=False)
    sim = LWW1DSimulator(cfg)
    state = sim.initial_zero_bias_state(exchange=False)
    image_path = save_wigner_phase_space_image(
        state.f,
        tmp_path / "wigner_phase_space.png",
        cfg=cfg,
        title=None,
    )
    x, k = wigner_phase_space_grids(cfg)

    assert image_path.exists()
    assert image_path.stat().st_size > 0
    assert x.shape == (cfg.nx,)
    assert k.shape == (cfg.n,)

    floating_path = save_wigner_phase_space_image(
        state.f,
        tmp_path / "wigner_phase_space_floating.png",
        cfg=cfg,
        style="floating",
        x_lim=(-0.5, 0.5),
        k_lim=(-8.0, 8.0),
        z_lim=(-1.0e-4, 1.0e-3),
        title=None,
    )
    assert floating_path.exists()
    assert floating_path.stat().st_size > 0


def test_cpp_backend_matches_python_when_available():
    if not CPP_AVAILABLE:
        return

    cfg_python = LWWConfig(nx=10, n=8, exchange=False, kernel_backend="python")
    cfg_cpp = LWWConfig(nx=10, n=8, exchange=False, kernel_backend="cpp")
    sim = LWW1DSimulator(cfg_python)
    state = sim.initial_zero_bias_state(exchange=False)
    state_cpp = LWW1DSimulator(cfg_cpp).initial_zero_bias_state(exchange=False)

    np.testing.assert_allclose(state_cpp.f, state.f, rtol=1.0e-12, atol=1.0e-12)
    density_python = curcalc(cfg_python, state.f, 1)
    density_cpp = curcalc(cfg_cpp, state.f, 1)
    current_python = curcalc(cfg_python, state.f, 2)
    current_cpp = curcalc(cfg_cpp, state.f, 2)

    np.testing.assert_allclose(density_cpp, density_python, rtol=1.0e-12, atol=1.0e-18)
    np.testing.assert_allclose(current_cpp, current_python, rtol=1.0e-12, atol=1.0e-6)


def test_wigstd_workspace_matches_standalone_solve():
    cfg = LWWConfig(nx=10, n=8, exchange=False, kernel_backend="python")
    sim = LWW1DSimulator(cfg)
    state = sim.zeros_state()
    rho = np.zeros(cfg.nx)
    rvscalc(cfg, state.rvs, bias=0.0, isc=2)
    fbndry(cfg, state.f)

    standalone = wigstd(
        cfg,
        state.f.copy(),
        state.f.copy(),
        state.rvs,
        rho,
        irlx=0,
        exchange=False,
        workspace=None,
    )
    workspace = WignerSolveWorkspace.create(cfg.n, cfg.nx)
    reused = wigstd(
        cfg,
        state.f.copy(),
        state.f.copy(),
        state.rvs,
        rho,
        irlx=0,
        exchange=False,
        workspace=workspace,
    )

    np.testing.assert_allclose(reused, standalone, rtol=1.0e-12, atol=1.0e-12)
