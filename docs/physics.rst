Physics Background
==================

What LWW Means
--------------

LWW stands for the lattice Weyl-Wigner formulation of quantum transport.  It is
a phase-space method: instead of propagating only a wave function or only a
classical distribution, it solves for a Wigner distribution ``f(x, k, t)`` on a
discrete lattice in position and momentum.  The formulation was developed for
open nonequilibrium quantum devices and connects Wigner transport ideas with
nonequilibrium Green-function language [BuotJensen1990]_ [Buot2009]_
[KadanoffBaym1963]_.

For the one-dimensional model implemented here, the simulation region is a
finite device between two contacts.  The phase-space grid has ``nx`` spatial
points and an even number ``n`` of momentum points.  Positive and negative
momenta are treated separately so that particles entering from each contact can
be handled with the correct upwind/downwind boundary source.

The Wigner Distribution
-----------------------

The Wigner distribution is the central unknown. It provides a phase-space
representation while retaining quantum behavior. The function itself can become
negative, so it should not be interpreted as an ordinary probability
density at each point.  Its marginals and weighted sums, however, give physical
observables such as carrier density and current density.

In this package:

* ``wigstd`` solves the discretized Wigner equation.
* ``curcalc(..., irj=1)`` evaluates carrier density from the Wigner function.
* ``curcalc(..., irj=2)`` evaluates current density.
* ``run_transient`` advances the Wigner distribution through a bias sweep and
  records the current trace.

Terms in the 1D LWW Equation
----------------------------

The implemented 1D equation contains three principal terms.

The drift or kinetic term transports the Wigner distribution across the spatial
grid.  The code uses a second-order upwind/downwind finite-difference stencil:
the sign of momentum determines which spatial direction supplies the upstream
values.  This is why the boundary source has separate left-contact and
right-contact entries.

The nonlocal potential term couples momentum points at the same spatial
location.  It is built from differences of the effective device potential at
positions around each grid point.  This is the term responsible for quantum
features such as tunneling and interference in a barrier/well structure.  In an
RTD, oscillations and sign changes in the Wigner function are expected markers
of coherent quantum behavior [JensenBuot1991Method]_ [Barraud2009]_.

The optional scattering term models relaxation toward a local equilibrium
distribution.  The original Jensen-Buot implementation used a modified
relaxation-time approximation that preserves detailed balance rather than a
plain ``-(f - f0) / tau`` decay [JensenBuot1991Method]_.  In this code path,
``irlx`` controls whether this relaxation contribution is included.

Self-Consistent Potential
-------------------------

The Wigner equation needs a device potential.  The carrier density produced by
the Wigner function also changes the electrostatic potential.  The package
therefore couples Wigner transport to a one-dimensional Poisson solve:

* solve Wigner transport for the current potential,
* compute density from the Wigner function,
* update the electrostatic potential from Poisson's equation,
* repeat until the density/potential change is small enough.

This self-consistent loop is used for steady-state bias points and as part of
the transient workflow.

What the Model Can Do
---------------------

The current implementation targets one-dimensional resonant-tunneling
structures, especially double-barrier RTDs. Within that scope it supports
study of:

* steady-state Wigner functions at a selected bias,
* density and current density profiles through the device,
* transient current traces during a bias sweep,
* tunneling and quantum-interference structure in phase space,
* negative differential resistance behavior in RTD-like geometries,
* effects of barrier height, well width, spacer length, grid resolution, and
  relaxation settings.

The RTD use case follows the original LWW demonstration literature, where
Wigner transport was used to model particle trajectories, intrinsic bistability,
and high-frequency current oscillations in resonant tunneling structures
[JensenBuot1991Method]_ [JensenBuot1991PRL]_.

The geometry plotting helper in ``lww_transport.visualization`` draws the
barrier/well layout used by the simulation for pre-run geometry checks.

Current Limitations
-------------------

The package is a deterministic one-dimensional effective-mass implementation.
It supports the original LWW/Wigner-Poisson workflow, but it is
not yet a full atomistic or multidimensional device simulator.  Boundary
conditions for open Wigner systems are a known subtle issue [Frensley1990]_
[Jiang2011]_ [Taj2006]_.  The present implementation keeps the conventional
contact-source treatment used by the reference RTD calculations, with the code
organized so more advanced contact/self-energy treatments, such as those common
in modern open-device transport models, remain possible extensions.
