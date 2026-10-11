#!/bin/bash
# Build hypre, METIS, gslib, the MFEM fork and MHD-ALE (twice: T. Abel's fork as it is, and with the suite added).
#   bash build.sh        (in a job; compilers and python loaded by the caller; sources from fetch.sh)
# Everything under $MHD_CODES/mhdale-yipihey: src/ (sources, built in place), mfem-install/, build-stock/, build-abel/,
# logs/. A step that is already done is skipped; the two MHD-ALE builds are always remade from the sources.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; root="$codes/mhdale-yipihey"; src="$root/src"; logs="$root/logs"
bundle="${SWEEP_BUNDLE:?set SWEEP_BUNDLE to the deployed sweep bundle}"
jobs="${MHDALE_BUILD_JOBS:-${SLURM_NTASKS:-4}}"; mkdir -p "$logs"
step() {     # name, file that proves it is done, command...
    local name="$1" done_file="$2"; shift 2
    if [[ -e "$done_file" ]]; then echo "   $name: already built"; return 0; fi
    local t0=$SECONDS
    if "$@" > "$logs/$name.log" 2>&1 && [[ -e "$done_file" ]]; then echo "   $name: built in $((SECONDS - t0)) s"; return 0; fi
    echo "BUILD FAILED: $name (end of $logs/$name.log)"; grep -n -i -E "error|undefined|No such file|cannot find" "$logs/$name.log" | head -n 15 | cut -c1-240
    tail -n 12 "$logs/$name.log" | cut -c1-240; return 1
}
build_hypre() { cd "$src/hypre/src" && ./configure --disable-fortran CC=mpicc CXX=mpicxx && make -j "$jobs"; }
build_metis() { cd "$src/metis-4.0.3" && make -j "$jobs" OPTFLAGS="-O2 -Wno-error=implicit-function-declaration"; }
build_gslib() { cd "$src/gslib" && make clean && make CC=mpicc -j "$jobs"; }
build_mfem() {
    cd "$src/mfem" && make config MFEM_USE_MPI=YES MFEM_USE_GSLIB=YES MFEM_DEBUG=NO MPICXX=mpicxx \
        HYPRE_DIR="$src/hypre/src/hypre" METIS_DIR="$src/metis-4.0.3" GSLIB_DIR="$src/gslib/build" PREFIX="$root/mfem-install" \
        && make -j "$jobs" && make install PREFIX="$root/mfem-install"
}
build_stock() {       # the whole tree, with the fork's own tests (make check); MHD and its two test programs
    rm -rf "$root/build-stock" && mkdir -p "$root/build-stock" && cd "$root/build-stock" \
        && (cd "$src/mhd-ale" && tar -cf - --exclude=.git .) | tar -xf - \
        && make -j "$jobs" MFEM_INSTALL_DIR="$root/mfem-install" MHD test_mean_field_geometry test_periodic_projection
}
build_abel() {
    rm -rf "$root/build-abel" && python3 "$here/prepare_patch.py" "$src/mhd-ale" "$root/build-abel" "$bundle/codes/common/abel_problems.hpp" \
        && cd "$root/build-abel" && make -j "$jobs" MFEM_INSTALL_DIR="$root/mfem-install"
}
[[ -d "$src/mfem" && -d "$src/mhd-ale" ]] || { echo "sources not found in $src (run fetch.sh on a login node)"; exit 2; }
echo "== building under $root with $(mpicxx --version | head -n 1), $jobs at a time"
step hypre "$src/hypre/src/hypre/lib/libHYPRE.a" build_hypre || exit 1
step metis "$src/metis-4.0.3/libmetis.a" build_metis || exit 1
step gslib "$src/gslib/build/lib/libgs.a" build_gslib || exit 1
step mfem "$root/mfem-install/share/mfem/config.mk" build_mfem || exit 1
rm -f "$root/build-stock/test_periodic_projection" "$root/build-abel/MHD"
step mhd-ale-fork "$root/build-stock/test_periodic_projection" build_stock || exit 1
step mhd-ale-with-the-suite "$root/build-abel/MHD" build_abel || exit 1
{ date -u; hostname; for d in hypre gslib mfem mhd-ale; do echo "$d $(cd "$src/$d" && git rev-parse HEAD)"; done; mpicxx --version | head -n 1; module list 2>&1; } > "$root/BUILD_INFO.txt"
echo "built: $root/build-stock/MHD and $root/build-abel/MHD"
