#!/bin/bash
# A launcher for the fork's regression driver (tests/regression.py calls "$MPIEXEC -np N program ..."): the same
# through srun, which is how every other run of this bundle starts MPI programs on Sherlock.
n=1; [[ "${1:-}" == "-np" ]] && { n="$2"; shift 2; }
exec srun -n "$n" "$@"
