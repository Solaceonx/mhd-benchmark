#!/bin/bash
# Build mini-ramses-ism for the comparison suite on a Sherlock GPU node (NVIDIA HPC SDK, CUDA Fortran).
#   bash build.sh NDIM NPRE
# NDIM: 2 (the planar tests) or 3. NPRE: 8 (double precision) or 4 (single, Eric Moseley's production setting).
# One build per (NDIM, NPRE) in its own copy of the source, $MHD_CODES/miniramses-build/nd<NDIM>_p<NPRE> (or under
# MINIRAMSES_BUILD_ROOT), patched by prepare_patch.py (the sweep's initial data, exact output times); a finished build
# is kept. MINIRAMSES_CUDA_ARCH (e.g. sm_90) names the GPU when the build host has none, as on a login node.
#
# As in sherlock_ramses2/build.sh: nvfortran 25.1 stopped with an internal compiler error on cooling/cooling_module.f90
# of RAMSES II at the Makefile's -O4. An object that fails this way is built at the highest optimisation level that
# works and the build goes on; BUILD_INFO.txt lists these objects. Any other compile error ends the build.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"
src="$codes/miniramses-ism"; ndim="${1:?usage: build.sh NDIM NPRE}"; npre="${2:?usage: build.sh NDIM NPRE}"
variant="nd${ndim}_p${npre}"; build="${MINIRAMSES_BUILD_ROOT:-$codes/miniramses-build}/$variant"; exe="$build/bin/ramses${ndim}d"
[[ -x "$exe" ]] && { echo "already built: $exe"; exit 0; }
[[ -d "$src/gpu" ]] || { echo "mini-ramses-ism source not found in $src (run fetch.sh)" >&2; exit 2; }
arch="${MINIRAMSES_CUDA_ARCH:-}"
[[ -n "$arch" ]] || arch="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
rm -rf "$build"; mkdir -p "$(dirname "$build")"; cp -R "$src" "$build"
python3 "$here/prepare_patch.py" "$src" "$build/abel-patch"
opts=(COMPILER=NVHPC "CUDA_ARCH=$arch" "NDIM=$ndim" MPI=0 HYDRO=1 MHD=1 "NPRE=$npre" "PATCH=$build/abel-patch")
echo "== building $variant: make ramses ${opts[*]}"
jobs="${SLURM_CPUS_PER_TASK:-8}"; lowered=""; ok=0
mk() { ( cd "$build/bin" && make "$@" "${opts[@]}" ); }
mk -j "$jobs" ramses > "$build/build.log" 2>&1 && ok=1
for pass in 1 2 3 4 5 6 7 8; do
    [[ $ok -eq 1 ]] && break
    # the objects make gave up on ("make: *** [x.o] Error 2", or "[Makefile:365: x.o]" from a newer make)
    failed="$(sed -n -E 's/^make(\[[0-9]+\])?: \*\*\* \[([^]]*: )?([A-Za-z0-9_]+\.o)\] Error.*/\3/p' "$build/build.log" | sort -u)"
    [[ -n "$failed" ]] || break
    cp "$build/build.log" "$build/build_pass$pass.log"; progress=0
    for obj in $failed; do
        if mk "$obj" > "$build/retry_$obj.log" 2>&1; then progress=1; continue; fi        # alone, as the Makefile has it
        grep -q "Internal compiler error" "$build/retry_$obj.log" || { echo "   $obj: an error of the code, not of the compiler"; progress=0; break; }
        for level in -O3 -O2 -O1 -O0; do
            if mk "$obj" "F90=nvfortran -Mfreeform $level" >> "$build/retry_$obj.log" 2>&1; then
                lowered="$lowered $obj($level)"; progress=1
                echo "   $obj: internal compiler error at -O4, built at $level"; break
            fi
        done
    done
    [[ $progress -eq 1 ]] || break
    mk -j "$jobs" ramses > "$build/build.log" 2>&1 && ok=1
done
[[ $ok -eq 1 ]] || {
    echo "BUILD FAILED: $variant (last lines of $build/build.log)"; grep -n -i "error\|fatal\|undefined\|killed" "$build/build.log" | head -20; tail -n 15 "$build/build.log"; exit 1; }
[[ -x "$exe" ]] || { echo "BUILD FAILED: $variant produced no executable"; tail -n 15 "$build/build.log"; exit 1; }
case "$lowered" in *" gpu_"*|*" cub_"*) echo "   NOTE: a GPU kernel is among the objects built below -O4; speeds from this build are lower bounds" ;; esac
{ date -u; (cd "$src" && git rev-parse HEAD); echo "make ramses ${opts[*]}"; echo "built on $(hostname)"; echo "objects built below -O4:${lowered:- none}"
  nvfortran --version | head -n 2; nvidia-smi -L || true; } > "$build/BUILD_INFO.txt" 2>&1
echo "built $exe${lowered:+ (below -O4:$lowered)}"
