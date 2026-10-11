#!/bin/bash
# Build RAMSES II for one speed-test variant on a Sherlock GPU node (NVIDIA HPC SDK, CUDA Fortran).
#   bash build.sh NSUBGRID NPRE
# NSUBGRID: tile size of the GPU kernels (1 or 2 for MHD; hard-coded in gpu/gpu_utils.cuf, so every value is a build).
# NPRE: 8 (double precision, the code's default) or 4. The problem is 3D Orszag-Tang (INIT=OT with our patch).
# Every variant is built in its own copy of the source, $MHD_CODES/ramses2-build/<variant> (or under
# RAMSES2_BUILD_ROOT); a finished build is kept. RAMSES2_CUDA_ARCH (e.g. sm_90) names the GPU when the build host
# has none, as on a login node.
#
# nvfortran 25.1 stops with an internal compiler error on cooling/cooling_module.f90 at the Makefile's -O4 (job
# 46663995). An object that fails this way is built at the highest optimisation level that works and the build goes
# on; BUILD_INFO.txt lists these objects. Any other compile error ends the build.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"
src="$codes/ramses2"; ns="${1:?usage: build.sh NSUBGRID NPRE}"; npre="${2:?usage: build.sh NSUBGRID NPRE}"; kind=ot
variant="${kind}_ns${ns}_p${npre}"; build="${RAMSES2_BUILD_ROOT:-$codes/ramses2-build}/$variant"; exe="$build/bin/ramses_${kind}3d"
[[ -x "$exe" ]] && { echo "already built: $exe"; exit 0; }
[[ -d "$src/gpu" ]] || { echo "RAMSES II source not found in $src (see README.md)" >&2; exit 2; }
arch="${RAMSES2_CUDA_ARCH:-}"
[[ -n "$arch" ]] || arch="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' .')"
rm -rf "$build"; mkdir -p "$(dirname "$build")"; cp -R "$src" "$build"
grep -q "integer, parameter :: nsubgrid = 1" "$build/gpu/gpu_utils.cuf" || { echo "gpu_utils.cuf: nsubgrid line not found" >&2; exit 2; }
sed -i "s/integer, parameter :: nsubgrid = 1/integer, parameter :: nsubgrid = ${ns}/" "$build/gpu/gpu_utils.cuf"
opts=(COMPILER=NVHPC "CUDA_ARCH=$arch" NDIM=3 MPI=0 HYDRO=1 MHD=1 "NPRE=$npre" "EXEC=ramses_${kind}")
python3 "$here/prepare_patch.py" "$src" "$build/abel-patch"
opts+=(INIT=OT "PATCH=$build/abel-patch")
echo "== building $variant: make ${opts[*]}"
jobs="${SLURM_CPUS_PER_TASK:-8}"; lowered=""; ok=0
mk() { ( cd "$build/bin" && make "$@" "${opts[@]}" ); }
mk -j "$jobs" > "$build/build.log" 2>&1 && ok=1
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
    mk -j "$jobs" > "$build/build.log" 2>&1 && ok=1
done
[[ $ok -eq 1 ]] || {
    echo "BUILD FAILED: $variant (last lines of $build/build.log)"; grep -n -i "error\|fatal\|undefined\|killed" "$build/build.log" | head -20; tail -n 15 "$build/build.log"; exit 1; }
[[ -x "$exe" ]] || { echo "BUILD FAILED: $variant produced no executable"; tail -n 15 "$build/build.log"; exit 1; }
case "$lowered" in *" gpu_"*|*" cub_"*) echo "   NOTE: a GPU kernel is among the objects built below -O4; speeds from this build are lower bounds" ;; esac
{ date -u; (cd "$src" && git rev-parse HEAD); echo "make ${opts[*]}"; echo "built on $(hostname)"; echo "objects built below -O4:${lowered:- none}"
  nvfortran --version | head -n 2; nvidia-smi -L || true; } > "$build/BUILD_INFO.txt" 2>&1
echo "built $exe${lowered:+ (below -O4:$lowered)}"
