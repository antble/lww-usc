Command Line Interface
======================

The package installs the ``lww-transport`` command.

Steady state
------------

.. code-block:: bash

   lww-transport steady --output output --nx 86 --n 72 --bias 0.0

Transient run
-------------

.. code-block:: bash

   lww-transport transient \
       --output output \
       --nx 86 \
       --n 72 \
       --ivn 45 \
       --itn 1000 \
       --dbias 0.008 \
       --sample-every 10 \
       --verbose \
       --progress-every 50 \
       --backend auto

When ``--verbose`` is set, transient runs print flushed progress lines at the
start and end of each bias point and every ``--progress-every`` iterations.
The ``lww_tcurl_<bias>.csv`` files (e.g. ``lww_tcurl_0.0080.csv``, bias
formatted to four decimal places) are written to ``--output`` during the run
every ``--sample-every`` iterations, rather than only after the full transient
finishes. State CSV checkpoint files are refreshed after each completed bias
point.

Every CLI run prints the configuration summary before solving and writes the
same text to ``config_summary.txt`` in ``--output``.

``--backend auto`` prefers the C++ extension, then Numba, then the Python
fallback. Use ``--backend cpp``, ``--backend numba``, or ``--backend python`` to
force a specific implementation. ``--no-numba`` is kept as a legacy alias for
``--backend python``. The Wigner linear solve itself uses reusable direct
LAPACK band storage; the backend option selects the assembly/reduction kernels
around that solve.

Plot device geometry
--------------------

.. code-block:: bash

   lww-transport geometry --output output --nx 86 --n 72

This draws the double-barrier RTD potential profile for the given configuration
and writes ``rtd_geometry.png`` and ``config_summary.txt`` to ``--output``.

.. tip::

   For fast development checks, reduce ``--nx`` and ``--n`` for all commands,
   and reduce ``--ivn`` and ``--itn`` for transient runs.
