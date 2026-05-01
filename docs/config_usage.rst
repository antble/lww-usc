LWWConfig Integration Notes
===========================

``LWWConfig`` is the shared configuration object for the simulator, CLI,
tests, visualization helpers, and profiling script. The public import is:

.. code-block:: python

   from lww_transport import LWWConfig

Primary Consumers
-----------------

.. list-table::
   :header-rows: 1

   * - Module
     - Role
   * - ``lww_transport.simulator``
     - Allocates state arrays and controls steady and transient workflows.
   * - ``lww_transport.cli``
     - Builds configuration from command-line options.
   * - ``lww_transport.core``
     - Receives configuration through compatibility wrappers.
   * - ``lww_transport.visualization``
     - Draws RTD geometry from configuration fields.
   * - ``scripts/profile_transient.py``
     - Creates benchmark configurations.
   * - ``tests/test_core.py``
     - Uses small-grid configurations for smoke tests.

Frequently Used Fields
----------------------

.. list-table::
   :header-rows: 1

   * - Field
     - Purpose
   * - ``nx``, ``n``, ``size``
     - Grid dimensions and phase-space vector length.
   * - ``exchange``
     - Enables the density-dependent exchange correction.
   * - ``verbose``
     - Enables solver progress output.
   * - ``kernel_backend``
     - Selects ``auto``, ``cpp``, ``numba``, or ``python`` kernels.
   * - ``irlx``
     - Controls inclusion of relaxation and scattering terms.
   * - ``rfa``, ``density_stop``, ``poisson_vmax_stop``
     - Self-consistency controls.
   * - ``box``, ``well``, ``barrier``, ``spacer``, ``pot``
     - RTD geometry and barrier potential.

Construction Patterns
---------------------

.. code-block:: python

   cfg = LWWConfig.standard_rtd()
   cfg = LWWConfig.quick_test(nx=10, n=8)
   cfg = LWWConfig(nx=86, n=72, exchange=True, kernel_backend="cpp")

Grouped parameters are available through:

.. code-block:: python

   cfg.discretization
   cfg.material
   cfg.geometry
   cfg.operating
   cfg.solver
   cfg.compute

Flat properties remain available for compatibility:

.. code-block:: python

   cfg.nx
   cfg.n
   cfg.exchange
   cfg.kernel_backend
