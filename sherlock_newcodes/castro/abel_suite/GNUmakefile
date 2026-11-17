# Castro with the comparison suite's tests (ours; sherlock_newcodes/castro/README in build.sh): 3-D MHD (Castro's MHD
# is 3-D only; planar tests run on a slab), CTU with HLLD and constrained transport, gamma-law gas, one fluid.
# build.sh: make -j USE_CUDA=TRUE CASTRO_HOME=<tree> (abel_problems.hpp copied here first).
PRECISION  = DOUBLE
PROFILE    = FALSE
DEBUG      = FALSE
DIM        = 3
COMP       = gnu
USE_MPI    = TRUE
USE_OMP    = FALSE
USE_CUDA  ?= FALSE
CUDA_ARCH ?= 90
USE_MHD    = TRUE

CASTRO_HOME ?= ../../..
EOS_DIR     := gamma_law
NETWORK_DIR := general_null
NETWORK_INPUTS = gammalaw.net

PROBLEM_DIR ?= ./
Bpack   := $(PROBLEM_DIR)/Make.package
Blocs   := $(PROBLEM_DIR)

include $(CASTRO_HOME)/Exec/Make.Castro
