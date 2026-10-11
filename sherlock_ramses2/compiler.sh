# Sourced by speed.sbatch and trial_build.sh: choose the NVIDIA HPC SDK module and build the first variant with it.
#   choose_compiler_and_build   -> 0 with the module loaded and ot_ns1_p8 built; 1 if no module could build it
# Tried in turn: RAMSES2_NVHPC_MODULE if set; else the module an earlier build under this build root used; else the
# newest nvhpc modules (RAMSES2_NVHPC_TRY of them, default 3). The module that worked is written next to the builds,
# so that a requeued job runs its executables with the libraries they were built with.
ramses2_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ramses2_cuda_path=""
use_nvhpc() {
    [[ -n "$ramses2_cuda_path" ]] && PATH="${PATH#"$ramses2_cuda_path:"}"; ramses2_cuda_path=""
    module load "$1" || { echo "   $1: the module did not load"; return 1; }       # not in a pipe: it must change this shell
    command -v nvfortran >/dev/null 2>&1 || { echo "   $1: no nvfortran after loading it"; return 1; }
    if ! command -v nvcc >/dev/null 2>&1; then                     # the CUDA compiler inside the SDK
        local cuda; cuda="$(dirname "$(command -v nvfortran)")/../../cuda/bin"
        [[ -x "$cuda/nvcc" ]] && { ramses2_cuda_path="$cuda"; export PATH="$cuda:$PATH"; }
    fi
    command -v nvcc >/dev/null 2>&1 || module load "${RAMSES2_CUDA_MODULE:-cuda/12.4.0}" 2>/dev/null || true
    echo "== $1: nvfortran $(command -v nvfortran), nvcc $(command -v nvcc || echo MISSING); $(nvfortran --version 2>&1 | sed -n 2p)"
}
choose_compiler_and_build() {
    local root="${RAMSES2_BUILD_ROOT:-${MHD_CODES:?}/ramses2-build}" list nv
    if [[ -n "${RAMSES2_NVHPC_MODULE:-}" ]]; then list="$RAMSES2_NVHPC_MODULE"
    elif [[ -s "$root/nvhpc_module.txt" ]]; then list="$(cat "$root/nvhpc_module.txt")"
    else list="$(module -t avail nvhpc 2>&1 | grep -E '^nvhpc/[0-9]' | sed 's/[( ].*//' | sort -V -r | head -n "${RAMSES2_NVHPC_TRY:-3}")"; fi
    [[ -n "$list" ]] || { echo "NO NVHPC MODULE FOUND"; module spider nvhpc 2>&1 | head -n 30; return 1; }
    for nv in $list; do
        use_nvhpc "$nv" || continue
        if bash "$ramses2_here/build.sh" 1 8; then mkdir -p "$root"; echo "$nv" > "$root/nvhpc_module.txt"; nvhpc_chosen="$nv"; return 0; fi
        echo "   $nv could not build RAMSES II"
    done
    return 1
}
