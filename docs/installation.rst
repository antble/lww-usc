Installation
============

Clone the repository from GitHub:

.. code-block:: bash

   git clone https://github.com/antble/lww-usc.git
   cd lww-usc

Install the package in editable mode:

.. code-block:: bash

   pip install -e .

To install the documentation tools as well:

.. code-block:: bash

   pip install -e ".[docs]"

To install optional Numba-compiled kernels:

.. code-block:: bash

   pip install -e ".[speedups]"

The package includes a C++ extension for the hot matrix assembly and reduction
kernels, built automatically by the editable install above.

To enable OpenMP support in the C++ extension, rebuild with:

.. code-block:: bash

   LWW_TRANSPORT_OPENMP=1 pip install -e .

On macOS, OpenMP requires ``libomp``. If it is installed in a non-standard
location, set ``LWW_TRANSPORT_OPENMP_INCLUDE`` and
``LWW_TRANSPORT_OPENMP_LIB`` before rebuilding. At runtime, use
``OPENMP_NUM_THREADS`` to control the number of OpenMP threads.

Documentation
-------------

To build the HTML documentation locally:

.. code-block:: bash

   cd lww-usc
   sphinx-build -b html docs docs/_build/html

The generated site is written to ``docs/_build/html``.
