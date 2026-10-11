#!/bin/bash
# Build the external codes for the comparison suite from sources fetched by tools/fetch_codes.sh.
#   bash codes/build.sh athenak             (on a GPU node: detects the GPU architecture; fp64 --
#                                            AthenaK's fp32 build does not compile at the pinned commit)
#   bash codes/build.sh athenak-cpu fp64    (Kokkos serial/OpenMP build, for local checks)
#   bash codes/build.sh ramses              (MPI, NDIM=3, SOLVER=mhd, suite condinit)
#   bash codes/build.sh athenapp            (MPI, HLLD, nghost=3, suite problem generator)
# Sources live in $MHD_CODES/{athenak,ramses,athena}. Each build writes BUILD_INFO.txt.
set -euo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes}"
what="${1:?usage: build.sh athenak|athenak-cpu|ramses|athenapp [fp64|fp32]}"
jobs="${SLURM_CPUS_PER_TASK:-8}"
info() { { date -u; printf 'bundle %s\n' "$bundle"; (cd "$1" && git rev-parse HEAD); printf '%s\n' "${@:2}"; } > "$2/BUILD_INFO.txt"; }

case "$what" in
athenak|athenak-cpu)
    prec="${2:-fp64}"; src="$codes/athenak"
    single=OFF; [[ "$prec" == fp32 ]] && single=ON
    python3 "$bundle/codes/athenak/patch_athenak.py" "$src"
    if [[ "$what" == athenak ]]; then
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
        case "$cap" in
            70) arch=VOLTA70 ;; 75) arch=TURING75 ;; 80) arch=AMPERE80 ;; 86) arch=AMPERE86 ;;
            89) arch=ADA89 ;; 90) arch=HOPPER90 ;;
            *) printf 'Unknown GPU compute capability %s; set Kokkos arch manually\n' "$cap" >&2; exit 2 ;;
        esac
        build="$src/build-$arch-$prec"
        cmake -S "$src" -B "$build" -D CMAKE_BUILD_TYPE=Release -D PROBLEM=abel_suite \
            -D Athena_SINGLE_PRECISION="$single" -D Kokkos_ENABLE_CUDA=ON -D "Kokkos_ARCH_$arch=ON" \
            -D CMAKE_CXX_COMPILER="$src/kokkos/bin/nvcc_wrapper"
    else
        arch=cpu; build="$src/build-cpu-$prec"
        # Upstream single precision has double->float brace initializers; clang rejects them
        # as errors (GCC, used under nvcc on Sherlock, only warns).
        cmake -S "$src" -B "$build" -D CMAKE_BUILD_TYPE=Release -D PROBLEM=abel_suite \
            -D Athena_SINGLE_PRECISION="$single" -D Kokkos_ENABLE_SERIAL=ON \
            -D CMAKE_CXX_FLAGS="$( [[ "$(c++ --version)" == *clang* ]] && printf -- -Wno-c++11-narrowing )"
    fi
    cmake --build "$build" -j "$jobs"
    info "$src" "$build" "athenak $prec arch=$arch" "$(nvidia-smi -L 2>/dev/null || true)"
    printf 'Built %s/src/athena\n' "$build" ;;
ramses)
    src="$codes/ramses"; patch="$src/abel-patch"
    python3 "$bundle/codes/ramses/prepare_patch.py" "$src" "$patch"
    make -C "$src/bin" clean >/dev/null 2>&1 || true
    # Fortran modules must compile in order: serial make.
    make -C "$src/bin" SOLVER=mhd NDIM=3 NPRE=8 MPI=1 PATCH="$patch" EXEC=ramses_abel
    info "$src" "$src/bin" "ramses NDIM=3 SOLVER=mhd MPI=1 PATCH=$patch" "$(mpif90 --version | head -n1)"
    printf 'Built %s/bin/ramses_abel3d\n' "$src" ;;
athenapp)
    src="$codes/athena"
    cp "$bundle/codes/athenapp/abel_suite.cpp" "$bundle/codes/common/abel_problems.hpp" "$src/src/pgen/"
    (cd "$src" && python3 configure.py --prob abel_suite -b --flux hlld --nghost 3 -mpi --cxx g++ && make clean && make -j "$jobs")
    info "$src" "$src/bin" "athena++ abel_suite hlld nghost=3 mpi"
    printf 'Built %s/bin/athena\n' "$src" ;;
*) printf 'Unknown target %s\n' "$what" >&2; exit 2 ;;
esac
