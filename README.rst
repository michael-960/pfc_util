PFC simulation in python
==============================

created by michael in 2022/04

pfc_util is a python package for PFC (phase field crystal) simulations.

Required Packages
======================
* numpy
* scipy
* matplotlib
* pyfftw
* tqdm
* torusgrid
* michael960lib


Modules
========
:code:`pfc_util.pfc` - High-level main module that contains the class :code:`pfc.PFC` whose instances are PFC models capable of recording and plotting minimization history etc. Save and load models in npz format with :code:`pfc.PFC.save()` and :code:`pfc.load_pfc_model()`.

:code:`pfc_util.core.base` - Definitions of PFC free energy functional and state functions.

:code:`pfc_util.core.evolution` - PFC minimizers, including constant chemical potential & nonlocal conserved minimization, stress relaxer and others.

:code:`pfc_util.toolkit.static` - Static objects access, mostly preminimized solid/liquid profiles.

:code:`pfc_util.toolkit.routine` - Routine high-level utility, e.g. :code:`pfc_util.toolkit.routine.find_coexistent_mu()`
uses binary search to look for solid-liquid coexistence under constant chemical potential.

:code:`pfc_util.ortho_lattice_generator` - Generates rotated profiles subject to periodic boundary condition.

:code:`pfc_util.profile_prompt` - Interactive PFC Prompt (WIP).


