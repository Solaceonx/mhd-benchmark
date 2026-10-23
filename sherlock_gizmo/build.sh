#!/bin/bash
# Build GIZMO for one set-up of the comparison suite.
#   bash build.sh METHOD GAMMA DIMS        METHOD: mfm; GAMMA: e.g. 1.6666666666666667, 2.0, 1.4; DIMS: 2 | 3
# -> $MHD_CODES/gizmo-build/<method>_g<gamma>_<dims>d/GIZMO. The method, the adiabatic index and the dimension are
# compile-time options in GIZMO, so each combination is its own build, in its own copy of the source (GIZMO builds in
# its source tree). Needs mpicc/mpicxx, GSL and HDF5 (no FFTW: no gravity); their paths come from GSL_PREFIX and
# HDF5_PREFIX, else gsl-config and h5cc/h5dump on the PATH (the modules the caller loaded).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/gizmo"
method="${1:?usage: build.sh METHOD GAMMA DIMS}"; gamma="${2:?usage: build.sh METHOD GAMMA DIMS}"; dims="${3:?usage: build.sh METHOD GAMMA DIMS}"
tag="${method}_g${gamma}_${dims}d"; build="${GIZMO_BUILD_ROOT:-$codes/gizmo-build}/$tag"; exe="$build/GIZMO"
[[ -x "$exe" ]] && { echo "already built: $exe"; exit 0; }
[[ -f "$src/Makefile" ]] || { echo "GIZMO source not found in $src (run fetch.sh)" >&2; exit 2; }
# Only meshless finite mass. Tried on the Mac at this commit and left out: meshless finite volume
# (HYDRO_MESHLESS_FINITE_VOLUME) and the regular grid built on it (HYDRO_REGULAR_GRID): their gas velocities are NaN
# from the start and GIZMO then removes every particle (2D and 3D, 1 to 4 ranks, with and without MHD, any
# HYDRO_FIX_MESH_MOTION, gravity on or off); the grid with MHD_CONSTRAINED_GRADIENT does not compile. Pressure-SPH
# (HYDRO_PRESSURE_SPH): runs, but 2D Orszag-Tang at 64^2 gains 60 % of its total energy by t = 0.5 (50 % with the
# quintic kernel and 40 neighbours), where MFM loses about 1 % (its Dedner cleaning removes energy, as GLM does).
case "$method" in mfm) hydro=HYDRO_MESHLESS_FINITE_MASS ;; *) echo "METHOD must be mfm" >&2; exit 2 ;; esac
rm -rf "$build"; mkdir -p "$(dirname "$build")"; cp -R "$src" "$build"; rm -rf "$build/.git"
{
    echo "$hydro"
    echo "BOX_PERIODIC"; [[ "$dims" == 2 ]] && echo "BOX_SPATIAL_DIMENSION=2"
    echo "SELFGRAVITY_OFF"
    echo "MAGNETIC"; echo "MHD_CONSTRAINED_GRADIENT=1"      # GIZMO's recommended setting: Dedner cleaning plus the CG correction
    echo "EOS_GAMMA=($gamma)"
    echo "FORCE_EQUAL_TIMESTEPS"                            # one timestep for all particles, as in the other codes
    echo "INPUT_IN_DOUBLEPRECISION"; echo "OUTPUT_IN_DOUBLEPRECISION"
    echo "OUTPUT_BFIELD_DIVCLEAN_INFO"                      # output only: div B and the cleaning field in the snapshots
} > "$build/Config.sh"
gsl="${GSL_PREFIX:-$(gsl-config --prefix 2>/dev/null || true)}"
hdf5="${HDF5_PREFIX:-}"
if [[ -z "$hdf5" ]]; then for t in h5cc h5dump; do command -v "$t" >/dev/null 2>&1 && { hdf5="$(dirname "$(dirname "$(command -v "$t")")")"; break; }; done; fi
inc() { [[ -n "$1" ]] && echo "-I$1/include" || true; }
lib() { [[ -n "$1" ]] && echo "-L$1/lib -Wl,-rpath,$1/lib" || true; }
echo "== building $tag (gsl: ${gsl:-from the environment}, hdf5: ${hdf5:-from the environment})"
# SYSTYPE=abel matches none of the Makefile's machines (command-line variables win over Makefile.systype's): everything
# machine-specific comes from this command line. Make runs one compile per task of the job.
( cd "$build" && make -j "${GIZMO_MAKE_JOBS:-${SLURM_NTASKS:-8}}" SYSTYPE=abel CC=mpicc "CXX=mpicxx -std=c++17" FC=mpicc \
      "OPTIMIZE=-O3 -g -Wno-unknown-pragmas -Wno-format-security" "GSL_INCL=$(inc "$gsl")" "GSL_LIBS=$(lib "$gsl")" \
      "HDF5INCL=$(inc "$hdf5") -DH5_USE_16_API" "HDF5LIB=$(lib "$hdf5") -lhdf5 -lz" MPICHLIB= FFTW_INCL= FFTW_LIBS= ) > "$build/build.log" 2>&1 || {
    echo "BUILD FAILED: $tag"; grep -n -i "error\|undefined\|No such file" "$build/build.log" | head -n 20; tail -n 12 "$build/build.log"; exit 1; }
[[ -x "$exe" ]] || { echo "BUILD FAILED: $tag produced no executable"; tail -n 12 "$build/build.log"; exit 1; }
{ date -u; (cd "$src" && git rev-parse HEAD); cat "$build/Config.sh"; mpicxx --version | head -n 1; module list 2>&1 || true; } > "$build/BUILD_INFO.txt"
echo "built $exe"
