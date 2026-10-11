#!/bin/sh
# Build the already-downloaded official source using Apple Clang and Homebrew libomp.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
athena_dir="$task_root/external/athena"
omp_prefix=${ATHENA_LIBOMP_PREFIX:-/opt/homebrew/opt/libomp}
cd "$athena_dir"
python3 configure.py --prob orszag_tang -b --flux hlld --nghost 3 \
  --cxx clang++-apple -omp --include "$omp_prefix/include" --lib_path "$omp_prefix/lib"
make -j4
./bin/athena -c
