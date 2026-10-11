#!/bin/bash
# Build AREPO for one set-up of the comparison suite.
#   bash build.sh MESH GAMMA DIMS        MESH: static | moving; GAMMA: e.g. 1.6666666666666667, 2.0, 1.4; DIMS: 2 | 3
# -> $MHD_CODES/arepo-build/<mesh>_g<gamma>_<dims>d/Arepo. The adiabatic index and the dimension are compile-time
# options in AREPO, so each combination is its own executable. Needs mpicc, GSL, GMP and HDF5 (modules loaded by the
# caller; their include and library paths are taken from the environment, or from *_PREFIX variables below).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/arepo"
mesh="${1:?usage: build.sh MESH GAMMA DIMS}"; gamma="${2:?usage: build.sh MESH GAMMA DIMS}"; dims="${3:?usage: build.sh MESH GAMMA DIMS}"
tag="${mesh}_g${gamma}_${dims}d"; build="$codes/arepo-build/$tag"; exe="$build/Arepo"
[[ -x "$exe" ]] && { echo "already built: $exe"; exit 0; }
[[ -f "$src/Makefile" ]] || { echo "AREPO source not found in $src (run fetch.sh)" >&2; exit 2; }
rm -rf "$build"; mkdir -p "$build"
{
    [[ "$dims" == 2 ]] && echo "TWODIMS"
    echo "GAMMA=$gamma"
    echo "MHD"; echo "MHD_POWELL"; echo "MHD_POWELL_LIMIT_TIMESTEP"; echo "RIEMANN_HLLD"
    if [[ "$mesh" == static ]]; then echo "VORONOI_STATIC_MESH"
    else echo "REGULARIZE_MESH_CM_DRIFT"; echo "REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED"; echo "REGULARIZE_MESH_FACE_ANGLE"; echo "TREE_BASED_TIMESTEPS"; fi
    echo "FORCE_EQUAL_TIMESTEPS"                 # one timestep for all cells, as in the other codes
    echo "DOUBLEPRECISION=1"; echo "INPUT_IN_DOUBLEPRECISION"; echo "OUTPUT_IN_DOUBLEPRECISION"
    echo "HAVE_HDF5"; echo "ALLOWEXTRAPARAMS"
} > "$build/Config.sh"
flag() { local p="$1" kind="$2"; [[ -n "$p" ]] && { [[ "$kind" == I ]] && echo "-I$p/include" || echo "-L$p/lib -Wl,-rpath,$p/lib"; } || true; }
gsl="${GSL_PREFIX:-$(gsl-config --prefix 2>/dev/null || true)}"
hdf5="${HDF5_PREFIX:-$(command -v h5dump >/dev/null 2>&1 && dirname "$(dirname "$(command -v h5dump)")" || true)}"
gmp="${GMP_PREFIX:-}"
echo "== building $tag (gsl: ${gsl:-from the environment}, hdf5: ${hdf5:-from the environment}, gmp: ${gmp:-from the environment})"
( cd "$src" && make -j "${SLURM_CPUS_PER_TASK:-8}" CONFIG="$build/Config.sh" BUILD_DIR="$build/build" EXEC="$exe" SYSTYPE=Sherlock \
      CC=mpicc PYTHON=python3 OPTIMIZE="-std=c11 -O3 -g -Wno-format-security -Wno-unknown-pragmas -Wno-unused-function" \
      MPICH_INCL= MPICH_LIB= GSL_INCL="$(flag "$gsl" I)" GSL_LIB="$(flag "$gsl" L) -lgsl -lgslcblas" \
      GMP_LIB="$(flag "$gmp" L) -lgmp" HDF5_INCL="$(flag "$hdf5" I) $(flag "$gmp" I) -DH5_USE_16_API" HDF5_LIB="$(flag "$hdf5" L) -lhdf5 -lz" ) > "$build/build.log" 2>&1 || {
    echo "BUILD FAILED: $tag"; grep -n -i "error\|undefined\|No such file" "$build/build.log" | head -n 20; tail -n 12 "$build/build.log"; exit 1; }
[[ -x "$exe" ]] || { echo "BUILD FAILED: $tag produced no executable"; tail -n 12 "$build/build.log"; exit 1; }
{ date -u; (cd "$src" && git rev-parse HEAD); cat "$build/Config.sh"; mpicc --version | head -n 1; module list 2>&1; } > "$build/BUILD_INFO.txt"
echo "built $exe"
