#!/bin/bash -l
# Builds and runs divfree_check.cpp against the MFEM build of the fork (build.sh). On Sherlock, in this directory:
#   sbatch divfree_check.sbatch      -> $SCRATCH/mhd-results-mhdale-checks/divfree/divfree_check.txt
#SBATCH --job-name=mhdale-divfree
#SBATCH --partition=normal
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
module load ${MHDALE_COMPILER_MODULES:-gcc/12.4.0 openmpi/4.1.6}
root="${MHD_CODES:-$SCRATCH/mhd-codes-sweep}/mhdale-yipihey"; out="$SCRATCH/mhd-results-mhdale-checks/divfree"; mkdir -p "$out"
cp "$SLURM_SUBMIT_DIR/divfree_check.cpp" "$out/"
printf 'include %s\ndivfree_check: divfree_check.cpp\n\t$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)\n' \
    "$root/mfem-install/share/mfem/config.mk" > "$out/Makefile"
make -C "$out" divfree_check
srun -n "$SLURM_NTASKS" "$out/divfree_check" | tee "$out/divfree_check.txt"
