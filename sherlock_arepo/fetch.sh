#!/bin/bash
# Get the public AREPO source at the pinned commit (run once, on a login node) and show which library modules exist.
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/arepo"
repo=https://gitlab.mpcdf.mpg.de/vrs/arepo.git; commit=351aa8111a32e2e7433866bf8367ab7eba155d39
mkdir -p "$codes"
[[ -d "$src/.git" ]] || git clone --quiet "$repo" "$src"
(cd "$src" && git checkout --quiet "$commit")      # cd, not `git -C`: the login nodes' git predates that option
echo "AREPO at $(cd "$src" && git rev-parse HEAD) in $src"
echo "== modules the build looks for (override with AREPO_MODULES=\"...\")"
for m in gsl hdf5 gmp py-h5py py-scipy; do printf '%-9s %s\n' "$m" "$(module -t avail "$m" 2>&1 | grep -i "^$m/" | tr '\n' ' ')"; done
