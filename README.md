# PFC simulation in python

created by michael in 2022/04

pfc_util is a python package for PFC (phase field crystal) simulations.

# Dependencies

-   numpy
-   scipy
-   matplotlib
-   pyfftw
-   torusgrid
-   rich

# Modules

`pfc_util.core` - Core module; definitions of PFC free energy
functional, state functions, as well as minimizers

`pfc_util.extra` - Extensions to vanilla PFC. Currently includes `pfc6`,
which is a 6th-order generalization to the PFC functional

`pfc_util.toolkit` - Static preminimized solid/liquid profiles & utility
functions.

# Minimizers

The package provides three main means of minimizing/evolving a PFC
system:

-   **Constant chemical potential** - The grand potential
    $\Omega = F - \mu \int d\mathbf{r} \psi$ is minimized with fixed
    $\mu$
-   **Stress relaxer** - The grand potential **density** is minimized
    with respect to the density field and system size
-   **Conserved dynamics** - The free energy is minimized with fixed
    mean density
