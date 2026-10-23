# Sourced by miniramses.sbatch: choose the NVIDIA HPC SDK module and build the 2D code with it (as sherlock_ramses2).
#   choose_compiler_and_build   -> 0 with the module loaded and the 2D build made; 1 if no module could build it
# Tried in turn: MINIRAMSES_NVHPC_MODULE if set; else the module an earlier build under this build root used; else the
# newest nvhpc modules (MINIRAMSES_NVHPC_TRY of them, default 3). The module that worked is written next to the builds,
# so that a requeued job runs its executables with the libraries they were built with.
miniramses_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
miniramses_cuda_path=""
use_nvhpc() {
    [[ -n "$miniramses_cuda_path" ]] && PATH="${PATH#"$miniramses_cuda_path:"}"; miniramses_cuda_path=""
    module load "$1" || { echo "   $1: the module did not load"; return 1; }       # not in a pipe: it must change this shell
    command -v nvfortran >/dev/null 2>&1 || { echo "   $1: no nvfortran after loading it"; return 1; }
    if ! command -v nvcc >/dev/null 2>&1; then                     # the CUDA compiler inside the SDK
        local cuda; cuda="$(dirname "$(command -v nvfortran)")/../../cuda/bin"
        [[ -x "$cuda/nvcc" ]] && { miniramses_cuda_path="$cuda"; export PATH="$cuda:$PATH"; }
    fi
    command -v nvcc >/dev/null 2>&1 || module load "${MINIRAMSES_CUDA_MODULE:-cuda/12.4.0}" 2>/dev/null || true
    echo "== $1: nvfortran $(command -v nvfortran), nvcc $(command -v nvcc || echo MISSING); $(nvfortran --version 2>&1 | sed -n 2p)"
}
choose_compiler_and_build() {
    local root="${MINIRAMSES_BUILD_ROOT:-${MHD_CODES:?}/miniramses-build}" list nv
    if [[ -n "${MINIRAMSES_NVHPC_MODULE:-}" ]]; then list="$MINIRAMSES_NVHPC_MODULE"
    elif [[ -s "$root/nvhpc_module.txt" ]]; then list="$(cat "$root/nvhpc_module.txt")"
    else list="$(module -t avail nvhpc 2>&1 | grep -E '^nvhpc/[0-9]' | sed 's/[( ].*//' | sort -V -r | head -n "${MINIRAMSES_NVHPC_TRY:-3}")"; fi
    [[ -n "$list" ]] || { echo "NO NVHPC MODULE FOUND"; module spider nvhpc 2>&1 | head -n 30; return 1; }
    for nv in $list; do
        use_nvhpc "$nv" || continue
        if bash "$miniramses_here/build.sh" 2 "${MINIRAMSES_NPRE:-8}"; then mkdir -p "$root"; echo "$nv" > "$root/nvhpc_module.txt"; nvhpc_chosen="$nv"; return 0; fi
        echo "   $nv could not build mini-ramses-ism"
    done
    return 1
}
