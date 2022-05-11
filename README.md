PFC simulation in python
========================

created by michael in 2022/04

pfc\_util is a python package for PFC (phase field crystal) simulations.

Required Packages
=================

-   numpy
-   scipy
-   matplotlib
-   pyfftw
-   tqdm
-   torusgrid
-   michael960lib

Modules
=======

`pfc_util.pfc`{.sourceCode} - High-level main module that contains the
class `pfc.PFC`{.sourceCode} whose instances are PFC models capable of
recording and plotting minimization history etc. Save and load models in
npz format with `pfc.PFC.save()`{.sourceCode} and
`pfc.load_pfc_model()`{.sourceCode}.

`pfc_util.core.base`{.sourceCode} - Definitions of PFC free energy
functional and state functions. `pfc_util.core.evolution`{.sourceCode} -
PFC minimizers, including constant chemical potential & nonlocal
conserved minimization, stress relaxer and others.

`pfc_util.toolkit.static`{.sourceCode} - Static objects access, mostly
preminimized solid/liquid profiles.
`pfc_util.toolkit.routine`{.sourceCode} - Routine high-level utility,
e.g. `pfc_util.toolkit.routine.find_coexistent_mu()`{.sourceCode} uses
binary search to look for solid-liquid coexistence under constant
chemical potential `pfc_util.ortho_lattice_generator`{.sourceCode} -
Generates rotated profiles subject to periodic boundary condition.

`pfc_util.profile_prompt`{.sourceCode} - Interactive PFC Prompt (WIP).
