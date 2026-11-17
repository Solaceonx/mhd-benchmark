# Nyx with the comparison suite's tests (ours): pure hydrodynamics (Nyx has no MHD), 3-D (planar tests on a slab),
# no gravity, no expansion (comoving_h = 0, initial_z = 0: a = 1), constant species. build.sh: make -j USE_CUDA=TRUE
AMREX_HOME ?= ../../subprojects/amrex
TOP ?= ../..
COMP    = gcc
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA ?= FALSE
CUDA_ARCH ?= 90
PRECISION = DOUBLE
DEBUG     = FALSE
DIM      = 3
USE_CONST_SPECIES = TRUE
Bpack := ./Make.package
Blocs := .
include $(TOP)/Exec/Make.Nyx
