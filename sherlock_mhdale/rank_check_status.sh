#!/bin/bash
# What the two runs of rank_check.sh found: bash -l rank_check_status.sh
checks="${SCRATCH:?SCRATCH is not set}/mhd-results-mhdale-checks"
squeue -h -u "$USER" -n mhdale-check -o "   in the queue: %.18i %.9T %.10M %R"
for d in "$checks"/*/; do
    for log in "$d"*/data/mhdale_ale_r1/mhdale.log; do
        [[ -f "$log" ]] || { echo "$(basename "$d"): no run yet"; continue; }
        r="$(dirname "$log")/run.toml"
        echo "$(basename "$d"): $(grep -m1 '^status' "$r" 2>/dev/null || echo 'no record yet'); remaps checked $(grep -c 'Mean-field decomposition' "$log"); largest error $(grep 'Mean-field decomposition' "$log" | awk '{print $NF}' | sort -g | tail -n 1); first errors: $(grep 'Mean-field decomposition' "$log" | awk '{print $NF}' | head -n 4 | tr '\n' ' ')"
    done
done
