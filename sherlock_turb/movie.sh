#!/bin/bash
# Submits the movie runs of the driven-turbulence page: Tom's Euler kernel at Mach 3, 10, 30 and 50 on 256^3 with the
# midplane density saved 20 times per driving time (config_movie.toml, config_movie_highmach.toml).
#   bash -l movie.sh
# The runs go to a results directory of their own ($SCRATCH/mhd-results-turb-movie), with the driving strengths of
# the pilot, so nothing of the pilot is touched. About four minutes a run on an H100.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
main="${TURB_RESULTS:-${SCRATCH:?SCRATCH is not set}/mhd-results-turb}"
movie="${TURB_MOVIE_RESULTS:-$SCRATCH/mhd-results-turb-movie}"
[[ -s "$main/forcing.toml" ]] || { echo "no $main/forcing.toml: the pilot's driving strengths are needed"; exit 1; }
mkdir -p "$movie" logs
cp "$main/forcing.toml" "$movie/forcing.toml"
feature=(); want="${TURB_GPU_CONSTRAINT:-GPU_SKU:H100_SXM5}"; [[ "$want" == none ]] || feature=(--constraint="$want")
for config in config_movie.toml config_movie_highmach.toml; do
    name="turb-$(basename "$config" .toml | sed 's/^config_//; s/_/-/g')"
    if squeue -h -u "$USER" -n "$name" -o %i | grep -q .; then echo "$name is already in the queue; not submitted again"; continue; fi
    id="$(sbatch --parsable --job-name="$name" --time=01:00:00 --export=ALL,TURB_CONFIG="$config",TURB_RESULTS="$movie" ${feature[@]+"${feature[@]}"} gpu.sbatch)"
    echo "submitted $name as job ${id%%;*}"
done
echo "Afterwards, on the Mac: the fetch command in README.md, section Movie. To follow: squeue -u $USER; tail logs/turb-movie*.out"
