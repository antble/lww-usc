Configuration Guide
===================

``LWWConfig`` groups simulation parameters into focused sections while
preserving flat keyword compatibility for existing scripts.

Presets
-------

.. code-block:: python

   from lww_transport import LWWConfig

   cfg = LWWConfig.standard_rtd()
   cfg = LWWConfig.quick_test(nx=32, n=32)
   cfg = LWWConfig.room_temperature_rtd()
   cfg = LWWConfig.coarse_grid_rtd(nx=48, n=48)
   cfg = LWWConfig.fine_grid_rtd(nx=128, n=96)

Grouped Parameters
------------------

.. code-block:: python

   from lww_transport import (
       ComputeParams,
       DiscretizationParams,
       GeometryParams,
       LWWConfig,
       MaterialParams,
       OperatingConditions,
       SolverParams,
   )

   cfg = LWWConfig(
       discretization=DiscretizationParams(nx=86, n=72),
       material=MaterialParams(temp=77.0, chemp=0.0863814),
       geometry=GeometryParams(box=550.0, well=50.0, pot=0.3),
       operating=OperatingConditions(bias=0.0, temperature=77.0),
       solver=SolverParams(exchange=True, convergence_eps=1e-10),
       compute=ComputeParams(kernel_backend="auto", verbose=False),
   )

Compatibility Form
------------------

Flat keyword arguments remain supported:

.. code-block:: python

   cfg = LWWConfig(nx=86, n=72, exchange=True, kernel_backend="cpp")

The grouped fields and flat properties reference the same values:

.. code-block:: python

   assert cfg.nx == cfg.discretization.nx
   assert cfg.exchange == cfg.solver.exchange

Common Adjustments
------------------

.. code-block:: python

   cfg = LWWConfig.standard_rtd()
   cfg.with_grid(64, 64)
   cfg.with_bias(0.08)
   cfg.with_temperature(300.0)
   cfg.with_exchange(False)
   cfg.with_verbose(True)

After direct edits to nested geometry or discretization fields, call
``cfg.__post_init__()`` to refresh cached grids:

.. code-block:: python

   cfg.geometry.well = 75.0
   cfg.geometry.pot = 0.45
   cfg.__post_init__()

Summary Output
--------------

.. code-block:: python

   from lww_transport import format_config_summary, save_config_summary

   print(format_config_summary(cfg))
   save_config_summary(cfg, "output")
