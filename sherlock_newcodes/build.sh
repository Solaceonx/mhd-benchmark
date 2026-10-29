#!/bin/bash
# Build the new codes from patched copies of the sources (fetch.sh), in a job: bash build.sh [CODE ...]
#   athenapk   AthenaPK, CUDA for the H100 (Kokkos HOPPER90), MPI, HDF5        -> athenapk/build/bin/athenaPK
#   cholla     Cholla, the two set-ups (builds/make.type.abel_plmc, _ppmc)      -> cholla/<set-up>/bin/cholla.abel_<..>.sherlock
#   pluto      gPLUTO, OpenACC for the H100 (NVHPC), one build per set-up and dimension -> pluto/<set-up>-<2|3>d/pluto
#   enzo       Enzo, MPI, double precision                                      -> enzo/build/bin/enzo (or the path printed)
#   phantom    Phantom, OpenMP, MHD, periodic, our set-up and analysis             -> phantom/tree/bin/{phantom,phantomsetup,phantomanalysis}
# Everything under $MHD_CODES/newcodes. A build is always remade from the sources; the sources are never changed.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; root="$codes/newcodes"; src="$root/src"; logs="$root/logs"
header="${SWEEP_BUNDLE:?set SWEEP_BUNDLE to the deployed sweep bundle}/codes/common/abel_problems.hpp"
jobs="${NEWCODES_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"; mkdir -p "$logs"
fresh() {   # source directory -> a clean copy at the second argument, with its .git (Parthenon and Cholla record the commit)
    rm -rf "$2" && mkdir -p "$(dirname "$2")" && cp -a "$1" "$2"
}
step() {    # name, file that proves it is done, command...
    local name="$1" done_file="$2"; shift 2; local t0=$SECONDS
    if "$@" > "$logs/$name.log" 2>&1 && [[ -e "$done_file" ]]; then echo "   $name: built in $((SECONDS - t0)) s"; return 0; fi
    echo "BUILD FAILED: $name (end of $logs/$name.log)"; grep -n -i -E " error|error:|undefined reference|No such file|cannot find|not found" "$logs/$name.log" | head -n 12 | cut -c1-240
    tail -n 10 "$logs/$name.log" | cut -c1-240; return 1
}
cuda_mods="${NEWCODES_CUDA_MODULES:-devel system git/2.45.1 gcc/12.4.0 openmpi/4.1.6 cuda/12.4.0 cmake/3.31.4 hdf5/1.14.4}"
nvhpc_mods="${NEWCODES_NVHPC_MODULES:-devel system git/2.45.1 nvhpc/24.7 python/3.12.1}"
cpu_mods="${NEWCODES_CPU_MODULES:-devel system git/2.45.1 gcc/12.4.0 openmpi/4.1.6 cmake/3.31.4 hdf5/1.14.4}"
use() { module purge >/dev/null 2>&1; module load $1 >/dev/null 2>&1 || { echo "module load $1 failed"; module load $1; }; }
prefix_of() { dirname "$(dirname "$(command -v "$1")")"; }

build_athenapk() {
    fresh "$src/athenapk" "$root/athenapk/tree" && python3 "$here/athenapk/patch_athenapk.py" "$root/athenapk/tree" "$header" \
    && NVCC_WRAPPER_DEFAULT_COMPILER=g++ cmake -S "$root/athenapk/tree" -B "$root/athenapk/build" -DCMAKE_BUILD_TYPE=Release \
         -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DCMAKE_CXX_COMPILER="$root/athenapk/tree/external/Kokkos/bin/nvcc_wrapper" \
         -DPARTHENON_DISABLE_OPENPMD=ON -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF -DHDF5_ROOT="$(prefix_of h5pcc 2>/dev/null || prefix_of h5cc)" \
    && NVCC_WRAPPER_DEFAULT_COMPILER=g++ cmake --build "$root/athenapk/build" -j "$jobs"
}
build_cholla() {    # set-up
    local t="$root/cholla/$1"
    fresh "$src/cholla" "$t" && python3 "$here/cholla/patch_cholla.py" "$t" "$header" \
    && (cd "$t" && make -j "$jobs" TYPE="abel_$1" MACHINE=sherlock CUDA_ROOT="$(prefix_of nvcc)" HDF5_ROOT="$(prefix_of h5pcc 2>/dev/null || prefix_of h5cc)" MPI_ROOT="$(prefix_of mpicxx)")
}
pluto_defs() {      # set-up, dimensions -> definitions.hpp
    local recon=LINEAR steps=RK2 divb=CONSTRAINED_TRANSPORT emf=UCT_HLL limiter="#define  LIMITER                        VANLEER_LIM"
    [[ $1 == *best ]] && { recon=PARABOLIC; steps=RK3; emf=UCT_HLLD; limiter=""; }
    [[ $1 == glm* ]] && divb=DIV_CLEANING
    cat <<DEFS
#define  PHYSICS                        MHD
#define  DIMENSIONS                     $2
#define  GEOMETRY                       CARTESIAN
#define  MAPPED_GRID                    NO
#define  BODY_FORCE                     NO
#define  COOLING                        NO
#define  RECONSTRUCTION                 $recon
#define  TIME_STEPPING                  $steps
#define  NTRACER                        0
#define  PARTICLES                      NO
#define  USER_DEF_PARAMETERS            0

// Physics dependent declarations

#define  EOS                            IDEAL
#define  ENTROPY_SWITCH                 NO
#define  DIVB_CONTROL                   $divb
#define  BACKGROUND_FIELD               NO
#define  AMBIPOLAR_DIFFUSION            NO
#define  RESISTIVITY                    NO
#define  HALL_MHD                       NO
#define  THERMAL_CONDUCTION             NO
#define  VISCOSITY                      NO
#define  ROTATING_FRAME                 NO

// User-defined parameters (labels)


// [Beg] user-defined constants (do not change this line)

$limiter
#define  ASSIGN_VECTOR_POTENTIAL        NO
$( [[ $divb == CONSTRAINED_TRANSPORT ]] && echo "#define  CT_EMF_AVERAGE                 $emf" )

// [End] user-defined constants (do not change this line)
DEFS
}
build_pluto_tree() {
    fresh "$src/gpluto" "$root/pluto/tree" && python3 "$here/pluto/patch_gpluto.py" "$root/pluto/tree"
}
build_pluto() {     # set-up, dimensions
    local w="$root/pluto/$1-$2d"
    # setup.py --auto-update reads a pluto.ini: any will do (the runner writes each run's own)
    rm -rf "$w" && mkdir -p "$w" && cp "$here/pluto/init.cpp" "$header" "$w/" && pluto_defs "$1" "$2" > "$w/definitions.hpp" \
    && cp "$root/pluto/tree/Test_Problems/MHD/Orszag_Tang/pluto_01.ini" "$w/pluto.ini" \
    && printf 'ARCH         = linux.sherlock.acc.defs\n' > "$w/makefile" \
    && (cd "$w" && PLUTO_DIR="$root/pluto/tree" python3 "$root/pluto/tree/setup.py" --auto-update --no-curses && make -j "$jobs")
}
build_enzo() {
    fresh "$src/enzo" "$root/enzo/tree" && python3 "$here/enzo/patch_enzo.py" "$root/enzo/tree" "$header" \
    && CC=mpicc CXX=mpicxx FC=mpif90 cmake -S "$root/enzo/tree" -B "$root/enzo/build" -DCMAKE_BUILD_TYPE=Release -DENZO_PRECISION=64 -DENZO_USE_MPI=ON \
    && cmake --build "$root/enzo/build" -j "$jobs" \
    && { exe="$(find "$root/enzo/build" "$root/enzo/tree/bin" -type f -name enzo -perm -u+x 2>/dev/null | head -n 1)"; [[ -n "$exe" ]] && ln -sf "$exe" "$root/enzo/enzo"; }
}

build_phantom() {
    local t="$root/phantom/tree" opts="SETUP=mhdrotor SETUPFILE=setup_abel.f90 ANALYSIS=analysis_abel.f90 SYSTEM=gfortran"
    fresh "$src/phantom" "$t" && python3 "$here/phantom/patch_phantom.py" "$t" \
    && (cd "$t" && make $opts -j "$jobs" && make setup $opts -j "$jobs" && make analysis $opts -j "$jobs")
}

want=("$@"); [[ ${#want[@]} -gt 0 ]] || want=(athenapk cholla pluto enzo phantom)
[[ -d "$src/athenapk" && -d "$src/cholla" ]] || { echo "sources not found in $src (run fetch.sh on a login node)"; exit 2; }
echo "== building under $root, $jobs at a time, $(date -u)"; bad=0
for code in "${want[@]}"; do
    case $code in
        athenapk) use "$cuda_mods"; step athenapk "$root/athenapk/build/bin/athenaPK" build_athenapk || bad=1 ;;
        cholla) use "$cuda_mods"; for s in plmc ppmc; do step "cholla-$s" "$root/cholla/$s/bin/cholla.abel_$s.sherlock" build_cholla $s || bad=1; done ;;
        pluto) use "$nvhpc_mods"; step pluto-tree "$root/pluto/tree/Config/linux.sherlock.acc.defs" build_pluto_tree || { bad=1; continue; }
               for s in ct_matched ct_best glm_matched glm_best; do for d in 2 3; do step "pluto-$s-${d}d" "$root/pluto/$s-${d}d/pluto" build_pluto $s $d || bad=1; done; done ;;
        enzo) use "$cpu_mods"; step enzo "$root/enzo/enzo" build_enzo || bad=1 ;;
        phantom) use "$cpu_mods"; step phantom "$root/phantom/tree/bin/phantomanalysis" build_phantom || bad=1 ;;
        *) echo "unknown code $code"; bad=1 ;;
    esac
done
{ date -u; hostname; for d in athenapk cholla gpluto enzo phantom; do [[ -d "$src/$d" ]] && echo "$d $(cd "$src/$d" && git rev-parse HEAD)"; done; } > "$root/BUILD_INFO.txt"
exit $bad
