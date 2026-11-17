#ifndef problem_initialize_H
#define problem_initialize_H

#include <prob_parameters.H>
#include <eos.H>
#include "abel_spec.H"

AMREX_INLINE
void problem_initialize ()
{
    abel_read_spec();
}

#endif
