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

:code:`pfc_util.core.base` - Definitions of PFC free energy functional and state functions.

:code:`pfc_util.core.evolution` - PFC minimizers, including constant chemical potential & nonlocal conserved minimization and stress relaxer.

:code:`pfc_util.toolkit` - Tools for Editting/Analyzing PFC Fields.

:code:`pfc_util.profile_prompt` - Interactive PFC Prompt (WIP).

:code:`pfc_util.pfc` - High-level main module that contains the class :code:`pfc.PFC` whose instances are PFC models capable of recording and plotting minimization history etc. Save and load models in npz format with :code:`pfc.PFC.save()` and :code:`pfc.load_pfc_model()`.
