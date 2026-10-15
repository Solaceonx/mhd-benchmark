#!/bin/bash
# Run on the Mac after the comparison jobs end. Only density slices and metadata.
set -euo pipefail
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$project/turb_compare_movie"
ssh -o ServerAliveInterval=30 lyxia@login.sherlock.stanford.edu \
  'cd /scratch/users/lyxia/mhd-results-turb-compare-movie && find . -maxdepth 2 \( -name "rho_*.bin" -o -name run.toml -o -name movie_times.csv \) | tar -cf - -T -' \
  | tar -xf - -C "$project/turb_compare_movie"
printf 'Fetched comparison movie inputs. Render with:\n'
printf '.venv-movie/bin/python scripts/make_turb_movie.py --compare --mach 30 50\n'
