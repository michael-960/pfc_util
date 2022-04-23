# PFC simulation in python
created by michael on 2022/04

pfc_util is a python package that for PFC (phase field crystal) simulations.

### Documentation

#### pfc_util.base - Low-level Interfaces

##### pfc_util.base.field.Field2D(Lx, Ly, Nx, Ny)
    
2D field object with pyfftw plans included.

    set_dimensions(Lx, Ly, Nx, Ny, verbose=False)

Set the dimensions. The field content will be set to zero.

    set_size(Lx, Ly, verbose=False)

Set Lx and Ly without changing the resolution. Field content is not affected.

    initialize_fft(**fftwargs)

Create pyfftw plans.

    set_psi(psi1, verbose=False)
    save(target_npz, verbose=False)
    plot(lazy_factor=1, cmap='jet', vmin=-1, vmax=1)
    yell(s)


#### pfc_util.toolkit - Tools for Editting/Analyzing PFC Fields

#### pfc_util.editor_prompt - Interactive PFC Prompt
