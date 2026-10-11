"""The comparison matrix: which code runs which test, at which grids, with which resources.

  python3 codes/plan.py task CONFIG CODE MODE INDEX   -> "N SEED GEOMETRY RESTRICT" for one array task
  python3 codes/plan.py count CONFIG CODE MODE        -> number of array tasks
  python3 codes/plan.py sbatch CODE MODE              -> sbatch resource options for the code
  python3 codes/plan.py matrix                        -> every submit command of the full study

CODE is gpu (the GLM/CT GPU driver), athenak, ramses or athenapp. MODE is run or reference.
Reference runs use AthenaK "reference" at reference_n (planar tests as true 2D) and save
states restricted to every comparison grid; Athena++ at reference_check_n cross-checks them.
"""
import os
import sys
import tomllib
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
FULL_3D = ("orszag_tang_3d", "field_loop_3d", "decay")
GPU_ONLY = ("kh_localized_dtjump",)        # needs the GPU driver's timestep-jump protocol
EXTERNAL_VARIANTS = {"athenak": ("matched", "best"), "ramses": ("matched", "best")}


def load(path):
    config = tomllib.loads(Path(path).read_text());config.setdefault("problem", config["name"])
    return config


def planar(config):
    if config["problem"] == "blast":return config.get("dim", 3) == 2
    return config["problem"] not in FULL_3D


def run_grids(config, code):
    grids = list(config["grids"])
    if code == "ramses":   # octree: powers of two; ramses_grids can drop CPU-expensive grids
        grids = [n for n in grids if n & (n - 1) == 0 and n in config.get("ramses_grids", grids)]
    return grids


def tasks(config, code, mode):
    seeds = config.get("seeds", [42])
    if mode == "reference":
        key = "reference_n" if code == "athenak" else "reference_check_n"
        n = int(config.get(key, 0))
        if not n:return []
        restrict = ",".join(str(m) for m in config["grids"] if n % m == 0)
        return [(n, s, "plane" if planar(config) else "cube", restrict) for s in seeds]
    if code in EXTERNAL_VARIANTS and config["name"] in GPU_ONLY:return []
    grids = run_grids(config, code)
    return [(n, s, "cube", "") for s in seeds for n in grids]


def sbatch(code, mode):
    if code == "athenak":
        partition = os.environ.get("MARLOWE_PARTITION", "preempt") if os.environ.get("MHD_SITE") == "marlowe" else "gpu"
        return f"--partition={partition} --gpus=1 --cpus-per-task=4 --mem=64G"
    ranks = 64 if mode == "reference" else 32
    return f"--partition=normal --ntasks={ranks} --cpus-per-task=1 --mem-per-cpu=4G"


def matrix():
    lines = ["# Full comparison study. Review, then run the lines you want (each is one Slurm array).",
             "bash tools/submit.sh build-gpu-codes      # once per GPU model; wait for it",
             "bash tools/submit.sh build-cpu-codes      # once; wait for it"]
    for path in sorted(p for p in (BUNDLE/"configs").glob("*.toml") if not p.name.startswith(".")):
        config = load(path);name = config["name"]
        lines.append(f"bash tools/submit.sh {name}")
        for code, variants in EXTERNAL_VARIANTS.items():
            if tasks(config, code, "run"):
                lines += [f"bash tools/submit.sh {name} --code {code}:{v}" for v in variants]
        if tasks(config, "athenak", "reference"):lines.append(f"bash tools/submit.sh {name} --reference")
        if tasks(config, "athenapp", "reference"):lines.append(f"bash tools/submit.sh {name} --reference-check")
    return "\n".join(lines)


def main(argv):
    if not argv:sys.exit(__doc__)
    cmd = argv[0]
    if cmd == "task":
        config, code, mode, index = load(argv[1]), argv[2], argv[3], int(argv[4])
        n, seed, geometry, restrict = tasks(config, code, mode)[index]
        print(n, seed, geometry, restrict or "-")
    elif cmd == "count":print(len(tasks(load(argv[1]), argv[2], argv[3])))
    elif cmd == "sbatch":print(sbatch(argv[1], argv[2]))
    elif cmd == "matrix":print(matrix())
    else:sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
