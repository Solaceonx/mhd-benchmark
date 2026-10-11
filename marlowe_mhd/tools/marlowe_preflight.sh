#!/bin/bash
# Read-only Marlowe checks plus creation of this project's scratch directories.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f site/marlowe.env ]]; then
    # shellcheck disable=SC1091
    source site/marlowe.env
fi
export MHD_SITE=marlowe MHD_BUNDLE="$PWD"
source jobs/common.sh
load_site_profile

: "${MARLOWE_PROJECT:?Copy site/marlowe.env.example to site/marlowe.env and set MARLOWE_PROJECT}"
: "${MARLOWE_ACCOUNT:?Set MARLOWE_ACCOUNT in site/marlowe.env}"
command -v sbatch >/dev/null || { printf 'sbatch is unavailable. Run this on a Marlowe login node.\n' >&2; exit 2; }
command -v sacctmgr >/dev/null || { printf 'sacctmgr is unavailable.\n' >&2; exit 2; }

printf 'Marlowe project: %s\nAccount: %s\nPartition: %s\n' \
    "$MARLOWE_PROJECT" "$MARLOWE_ACCOUNT" "$MARLOWE_PARTITION"
printf '\nAccount association:\n'
sacctmgr show assoc user="$USER" account="$MARLOWE_ACCOUNT" format=User,Account,Partition,QOS
printf '\nCurrent preempt limits and inventory:\n'
sinfo -p "$MARLOWE_PARTITION" -o '%P %a %l %D %G'
sacctmgr show qos normal format=Name,MaxTRESPerAccount,MaxWall

for directory in "$MHD_RESULTS" "$MHD_CODES" "$MHD_JULIA_DEPOT"; do
    mkdir -p "$directory"
    [[ -w "$directory" ]] || { printf 'Not writable: %s\n' "$directory" >&2; exit 2; }
done
printf '\nWritable paths:\n  results %s\n  codes   %s\n  depot   %s\n' \
    "$MHD_RESULTS" "$MHD_CODES" "$MHD_JULIA_DEPOT"

load_modules "$MHD_PYTHON_MODULES"
python3 -c 'import sys, numpy, tomllib; assert sys.version_info >= (3,11); print("Python",sys.version.split()[0],"NumPy",numpy.__version__)'

if [[ -n "${JULIA_BIN:-}" ]]; then
    [[ -x "$JULIA_BIN" ]] || { printf 'JULIA_BIN is not executable: %s\n' "$JULIA_BIN" >&2; exit 2; }
    "$JULIA_BIN" --version
else
    module load "${JULIA_MODULE:-julia}"
    julia --version
fi

load_modules "$MHD_GPU_RUN_MODULES"
command -v nvcc >/dev/null || { printf 'nvcc unavailable after loading: %s\n' "$MHD_GPU_RUN_MODULES" >&2; exit 2; }
# CUDA.jl uses the system toolkit (env/LocalPreferences.toml: CUDA_Runtime_jll local = true)
# and finds it through NVHPC_ROOT, CUDA_HOME/CUDA_PATH/CUDA_ROOT or ptxas on PATH.
if [[ -z "${NVHPC_ROOT:-}${CUDA_HOME:-}${CUDA_PATH:-}${CUDA_ROOT:-}" ]] && ! command -v ptxas >/dev/null; then
    printf 'CUDA.jl cannot locate the CUDA toolkit: none of NVHPC_ROOT, CUDA_HOME, CUDA_PATH, CUDA_ROOT or ptxas is set after loading %s\n' "$MHD_GPU_RUN_MODULES" >&2
    exit 2
fi
printf '\nCUDA toolkit: %s\nHost compiler: %s\n' "$(nvcc --version | tail -n 1)" "$(g++ --version | head -n 1)"
printf 'Toolkit location for CUDA.jl: NVHPC_ROOT=%s CUDA_HOME=%s\n' "${NVHPC_ROOT:-unset}" "${CUDA_HOME:-unset}"
# AthenaK compiles through nvcc_wrapper with the default g++. Check the pair on the login node
# (no GPU needed) so an unsupported GCC/CUDA combination fails here, not in the build job.
probe="$(mktemp -d)"
printf '__global__ void k(double *x){x[threadIdx.x]*=2.0;}\nint main(){return 0;}\n' > "$probe/probe.cu"
nvcc -arch=sm_90 -ccbin g++ -c "$probe/probe.cu" -o "$probe/probe.o" || {
    printf 'nvcc could not compile for sm_90 with this g++; load a GCC version supported by this CUDA\n' >&2
    rm -rf "$probe"; exit 2; }
rm -rf "$probe"
printf 'nvcc + g++ compile for sm_90: ok\n'
command -v cmake >/dev/null || printf 'WARNING: cmake is not currently available; add its module to MHD_GPU_BUILD_MODULES before the AthenaK build.\n'

sha256sum -c SCIENTIFIC_INPUTS.sha256
printf '\nPreflight passed. Submit the CPU-only setup job next:\n  bash tools/submit.sh setup\n'

