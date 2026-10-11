#!/bin/bash
# Get the sources MHD-ALE needs, at pinned versions (run once, on a login node): bash fetch.sh
# -> $MHD_CODES/mhdale-yipihey/src/{hypre,metis-4.0.3,gslib,mfem,mhd-ale}. Nothing is built here (build.sh, in a job).
# MHD-ALE is T. Abel's fork (github.com/yipihey/MHD-ALE: x-periodic ALE, mean-field remap); MFEM and gslib are the
# commits its README names. hypre and METIS are not pinned there (system packages): hypre 2.31.0 and MFEM's METIS 4.
set -euo pipefail
codes="${MHD_CODES:-${SCRATCH:?set MHD_CODES or SCRATCH}/mhd-codes-sweep}"; src="$codes/mhdale-yipihey/src"; mkdir -p "$src"; cd "$src"
get() {     # directory, repository, tag or commit (a full commit id is checked out after a full clone)
    local dir="$1" url="$2" ref="$3"
    if [[ ! -d "$dir/.git" ]]; then
        if [[ ${#ref} -eq 40 ]]; then git clone --quiet "$url" "$dir"; else git clone --quiet --depth 1 --branch "$ref" "$url" "$dir"; fi
    fi
    [[ ${#ref} -ne 40 ]] || (cd "$dir" && git checkout --quiet "$ref")      # cd, not `git -C`: the login nodes' git is old
    echo "$dir: $(cd "$dir" && git rev-parse HEAD) ($ref)"
}
get hypre https://github.com/hypre-space/hypre.git v2.31.0
get gslib https://github.com/Nek5000/gslib.git 375eda25b04205d1f749ab0e00a0bb3f22225b22
get mfem https://github.com/ruijie-xi/mfem-develop.git 842fac6a3b038eaee848921f096e80ceb5f895fa
get mhd-ale https://github.com/yipihey/MHD-ALE.git 2e900cd42d3b9041cd87e7e36e1334f239798438
if [[ ! -f metis-4.0.3/Makefile ]]; then
    curl -sSL -o metis-4.0.3.tar.gz https://github.com/mfem/tpls/raw/gh-pages/metis-4.0.3.tar.gz && tar -xzf metis-4.0.3.tar.gz
fi
echo "metis-4.0.3: $(sha256sum metis-4.0.3.tar.gz 2>/dev/null | cut -c1-16) (MFEM's mirror)"
