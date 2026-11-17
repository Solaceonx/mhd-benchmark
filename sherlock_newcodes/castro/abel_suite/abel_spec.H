#ifndef ABEL_SPEC_H
#define ABEL_SPEC_H
// The suite's problem spec (codes/common/abel_problems.hpp), read on the host from the file named by ABEL_SPEC and
// kept in managed memory so that the initialisation kernels can read it on the GPU.
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION AMREX_GPU_HOST_DEVICE inline
#endif
#include "abel_problems.hpp"

extern AMREX_GPU_MANAGED abel::Spec* abel_spec;
void abel_read_spec ();      // problem_initialize(): reads the file, checks the grid
#endif
