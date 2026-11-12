"""The production runs of the new codes, one line per array task: python3 tasks.py CODE [VARIANT ...] > tasks file

Each line: code|variant|test|seed|grids (comma-separated)|geometry. The tests are the sweep's 24 physics tests (as
AthenaK and Athena++ ran them; decay with each of its seeds); planar ones in 2D (Cholla: an 8-cell slab), the 3D ones
as cubes. Split by grid where a code needs different resources: NEWCODES_GRIDS limits the grids, NEWCODES_DIM to
"planar" or "3d" tests.
"""
import os
import sys
from pathlib import Path

SWEEP = Path(os.environ.get("SWEEP_BUNDLE", Path(__file__).resolve().parent.parent/"sherlock_sweep"))
TESTS = ("alfven alfven_oblique blast_2d_b0 blast_2d_b50 blast_2d_b100 blast_3d_b0 blast_3d_b50 blast_3d_b100 brio_wu "
         "brio_wu_rotated current_sheet decay decay_supersonic entropy field_loop field_loop_3d field_loop_novz gem "
         "kh_localized kh_uniform orszag_tang orszag_tang_3d rotor uniform").split()
VARIANTS = {"athenapk": ["matched", "best"], "cholla": ["plmc", "ppmc"], "pluto": ["ct_matched", "ct_best", "glm_matched", "glm_best"],
            "enzo": ["dedner", "ct"], "phantom": ["sph"],
            # single precision of the GPU codes (2026-11-12); AthenaK as in the sweep: every test in 3D (planar ones on N^3)
            "athenakf32": ["matched", "best"], "athenapkf32": ["matched", "best"], "chollaf32": ["plmc", "ppmc"]}
PHANTOM_3D_MAX = 256           # 512^3 would be 134 million particles


def runs(code, variants=(), want_grids=(), dim=""):
    """(variant, test, seed, grids, geometry) of every planned run of a code; the site reads the plan from here too."""
    sys.path.insert(0, str(SWEEP/"codes"));import external
    for v in variants or VARIANTS[code]:
        for test in TESTS:
            cfg = external.load_config(SWEEP/"configs"/f"{test}.toml");flat = external.planar(cfg)
            if (dim == "planar" and not flat) or (dim == "3d" and flat):continue
            grids = [g for g in cfg["grids"] if (not want_grids or g in want_grids) and not (code == "phantom" and not flat and g > PHANTOM_3D_MAX)]
            if not grids:continue
            for seed in cfg.get("seeds", [42]):
                yield v, test, seed, grids, "plane" if flat and code != "athenakf32" else "cube"


def main(code, variants):
    want_grids = {int(g) for g in os.environ.get("NEWCODES_GRIDS", "").split(",") if g}
    for v, test, seed, grids, geometry in runs(code, variants, want_grids, os.environ.get("NEWCODES_DIM", "")):
        print(f"{code}|{v}|{test}|{seed}|{','.join(map(str, grids))}|{geometry}")


if __name__ == "__main__":
    if len(sys.argv) < 2:sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
