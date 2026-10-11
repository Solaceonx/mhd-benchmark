#!/bin/bash
# RAMSES with its turbulence driving: the sweep's pinned source ($MHD_CODES/ramses), hydro solver, 3D, MPI, USE_TURB=1.
# Built in its own directory, so that the sweep's bin/ is not touched:
#   bash build_ramses.sh             as released, no patch  -> $MHD_CODES/ramses/bin-turb/ramses_turb3d
#   bash build_ramses.sh fixed       with the projection of the driving corrected -> .../bin-turb-fixed/ramses_turb3d
# "fixed": find_unitk (turb/turb_force_utils.f90) takes the direction of a wave vector from the absolute values of
# its components (TURB_GS - i for the negative half) where the Helmholtz projection needs the signed ones
# (i - TURB_GS). With comp_frac = 0 a fifth of the force's power is then compressive (README.md). The three lines
# are corrected in a copy that RAMSES's own PATCH mechanism compiles in place of the original; the source tree is
# not changed.
# The driving module needs FFTW (libfftw3 and the Fortran include file fftw3.f). By default this uses the copy the
# sweep's Julia environment already holds (the FFTW_jll artifact in the Julia depot): it is there on every node, and
# loading no module cannot swap the MPI library. TURB_FFTW_DIR=/path (with include/ and lib/) names another one.
# TURB_RAMSES_MPI=0 builds without MPI (a laptop test).
set -euo pipefail
: "${MHD_CODES:?MHD_CODES is not set: source jobs/common.sh of the sweep bundle and call prepare_codes}"
: "${SWEEP_BUNDLE:?SWEEP_BUNDLE is not set}"
which="${1:-released}"; [[ "$which" == released || "$which" == fixed ]] || { echo "usage: build_ramses.sh [released|fixed]"; exit 2; }
src="${TURB_RAMSES_SRC:-$MHD_CODES/ramses}"; mpi="${TURB_RAMSES_MPI:-1}"
build="$src/bin-turb"; [[ "$which" == fixed ]] && build="$src/bin-turb-fixed"
[[ -f "$src/bin/Makefile" ]] || { echo "no RAMSES source at $src (the sweep's tools/fetch_codes.sh puts it there)"; exit 1; }
[[ -d "$src/turb" ]] || { echo "the RAMSES source at $src has no turb/ directory: not the version the sweep pinned"; exit 1; }
want="$(source "$SWEEP_BUNDLE/codes/versions.env"; echo "$RAMSES_COMMIT")"; have="$(cd "$src" && git rev-parse HEAD 2>/dev/null || echo unknown)"
[[ "$have" == "$want" || "$have" == unknown ]] || echo "NOTE: RAMSES source is at $have, the sweep pinned $want"

fftw="${TURB_FFTW_DIR:-}"
if [[ -z "$fftw" ]]; then
    depot="${MHD_JULIA_DEPOT:-$HOME/.julia-mhd}"
    while read -r f; do
        d="${f%/include/fftw3.f}"
        if compgen -G "$d/lib/libfftw3.so*" > /dev/null || compgen -G "$d/lib/libfftw3*.dylib" > /dev/null; then fftw="$d"; break; fi
    done < <(find "$depot/artifacts" -maxdepth 3 -path '*/include/fftw3.f' 2>/dev/null | sort)
fi
[[ -n "$fftw" && -f "$fftw/include/fftw3.f" ]] || { echo "no FFTW with include/fftw3.f found (looked in the Julia depot's artifacts); set TURB_FFTW_DIR"; exit 1; }
link="-lfftw3"
if [[ ! -e "$fftw/lib/libfftw3.so" && ! -e "$fftw/lib/libfftw3.dylib" ]]; then link="-l:$(basename "$(ls "$fftw"/lib/libfftw3.so.* | head -n1)")"; fi

info="$build/BUILD_INFO.txt"; stamp="projection $which, commit $have, fftw $fftw, mpi $mpi, compiler $( (mpif90 --version 2>/dev/null || gfortran --version) | head -n1)"
have_stamp="$(head -n1 "$info" 2>/dev/null || true)"       # builds made before "fixed" existed have no "projection" in theirs
if [[ -x "$build/ramses_turb3d" && ( "$have_stamp" == "$stamp" || ( "$which" == released && "$have_stamp" == "${stamp#projection released, }" ) ) ]]; then echo "RAMSES with driving ($which) is already built: $build/ramses_turb3d"; exit 0; fi
mkdir -p "$build"; cp "$src/bin/Makefile" "$build/"; [[ -f "$src/bin/clusters.mk" ]] && cp "$src/bin/clusters.mk" "$build/"
patch=""
if [[ "$which" == fixed ]]; then
    patch="$build/patch"; mkdir -p "$patch"
    sed -e 's/if (i > limit) ii = TURB_GS - i$/if (i > limit) ii = i - TURB_GS/' -e 's/if (j > limit) jj = TURB_GS - j$/if (j > limit) jj = j - TURB_GS/' \
        -e 's/if (k > limit) kk = TURB_GS - k$/if (k > limit) kk = k - TURB_GS/' "$src/turb/turb_force_utils.f90" > "$patch/turb_force_utils.f90"
    changed="$(diff "$src/turb/turb_force_utils.f90" "$patch/turb_force_utils.f90" | grep -c '^>' || true)"
    [[ "$changed" == 3 ]] || { echo "the correction changed $changed lines of turb_force_utils.f90, not 3: this is not the source it was written for"; exit 1; }
fi
echo "building RAMSES (hydro, 3D, USE_TURB=1, projection $which) in $build with FFTW from $fftw: about three minutes"
# One file at a time: this Makefile's dependencies are not complete enough for make -j.
if ! ( cd "$build" && rm -f ./*.o ./*.mod ramses_turb3d && make SOLVER=hydro NDIM=3 NPRE=8 MPI="$mpi" USE_TURB=1 FFTW_DIR="$fftw" \
        LIBS_TURB="-L$fftw/lib $link -Wl,-rpath,$fftw/lib" PATCH="$patch" EXEC=ramses_turb ) > "$build/build.log" 2>&1; then
    echo "BUILD FAILED; the end of $build/build.log:"; grep -n -i "error" "$build/build.log" | head -n 20; tail -n 15 "$build/build.log"; exit 1
fi
[[ -x "$build/ramses_turb3d" ]] || { echo "BUILD FAILED: make ended without an executable; see $build/build.log"; exit 1; }
{ echo "$stamp"; echo "built $(date -u) on $(hostname)"; echo "make SOLVER=hydro NDIM=3 NPRE=8 MPI=$mpi USE_TURB=1 FFTW_DIR=$fftw PATCH=$patch EXEC=ramses_turb"; [[ -z "$patch" ]] || diff "$src/turb/turb_force_utils.f90" "$patch/turb_force_utils.f90" || true; module list 2>&1 || true; } > "$info"
echo "built $build/ramses_turb3d"
