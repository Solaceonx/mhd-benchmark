"""Write configs/*.toml for the comprehensive sweep from the campaign configs they extend.

Physics settings (problem parameters, t_end, frames, CFL, cleaning parameters) are copied unchanged from
the campaign configs (sherlock_mhd, the corrected Kelvin-Helmholtz/current-sheet setups of sherlock_rerun,
and sherlock_gem); only the grids, methods, repeats, references and output policy change. Run from the
bundle root: python3 tools/make_sweep_configs.py SOURCE_CONFIG_DIR... (later directories win).
"""
import sys
import tomllib
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
FULL_3D = ("orszag_tang_3d", "field_loop_3d", "decay")
METHODS = ["glm", "glm_hll16", "glm_hlld", "glm_ppm", "glm_constch", "ct"]
LADDER = [32, 64, 128, 256, 512]
GRIDS = {"kh_uniform": [64, 128, 256, 512], "kh_localized": [64, 128, 256, 512], "kh_localized_dtjump": [128, 256, 512],
         "gem": [64, 128, 256, 512], "decay_supersonic": [64, 128, 256, 512],
         "throughput_ot3d": [64, 96, 128, 192, 256, 384, 512]}   # multiples of 6 use the production kernel's TB=6 tiles
# Steps per unit N measured on Sherlock (GLM, the largest completed grid), used for the step caps below.
STEPS_PER_N = {"alfven": 13.3, "alfven_oblique": 6.1, "blast_2d_b0": 4.5, "blast_2d_b100": 6.6, "blast_2d_b50": 4.9,
    "blast_3d_b0": 4.7, "blast_3d_b100": 6.4, "blast_3d_b50": 5.0, "brio_wu": 2.3, "brio_wu_rotated": 1.6,
    "current_sheet": 33.5, "decay": 20.5, "decay_supersonic": 23.9, "entropy": 21.9, "field_loop": 73.0,
    "field_loop_3d": 31.6, "field_loop_novz": 55.0, "kh_localized": 164.0, "kh_localized_dtjump": 176.0,
    "kh_uniform": 164.0, "orszag_tang": 12.0, "orszag_tang_3d": 29.5, "rotor": 5.3, "throughput_ot3d": 1.3,
    "uniform": 1.9, "gem": 41.0}
# Tests measured against exact solutions (field loops: the advected initial state at periodic returns).
EXACT = ("uniform", "entropy", "alfven", "alfven_oblique", "field_loop", "field_loop_3d")
DROP = ("slurm_time", "slurm_mem", "ramses_grids", "max_steps", "repeats", "grids", "snapshot_frames",
        "reference_n", "reference_check_n", "methods")


def planar(c):
    p = c.get("problem", c["name"])
    return c.get("dim", 3) == 2 if p == "blast" else p not in FULL_3D


def spaced(frames, count):
    """count frame indices spread evenly over 0..frames-1, first and last included."""
    if frames <= count:return list(range(frames))
    return sorted({round(i*(frames - 1)/(count - 1)) for i in range(count)})


def value(v):
    if isinstance(v, bool):return "true" if v else "false"
    if isinstance(v, str):return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, float):return repr(v)
    if isinstance(v, list):return "[" + ", ".join(value(x) for x in v) + "]"
    if isinstance(v, dict):return "{ " + ", ".join(f'"{k}" = {value(x)}' for k, x in v.items()) + " }"
    return str(v)


def sweep_config(src):
    c = tomllib.loads(src.read_text());name = c["name"];p2 = planar(c);frames = int(c["frames"])
    out = {k: v for k, v in c.items() if k not in DROP}
    grids = GRIDS.get(name, LADDER)
    out["grids"] = grids
    out["methods"] = list(METHODS);out["split_methods"] = True
    timing = name == "throughput_ot3d"
    out["repeats"] = 5
    if not timing:out["repeats_by_grid"] = {"256": 3, "384": 1, "512": 1}
    out["checkpoint_seconds"] = 900
    # Runaway guard (dt collapse): 4x the measured steps; per-attempt wall limit left to Slurm.
    out["max_steps_per_n"] = int(4*STEPS_PER_N[name]) + 10
    out["max_run_seconds"] = 172800
    out["restart_every"] = max(1, (frames - 1)//4)       # AthenaK restart files: ~4 per run
    if p2:
        # Planar states are small up to 128^2 (every output kept); six outputs at 256^2, four at 512^2.
        if "snapshot_frames" in c:out["snapshot_frames"] = c["snapshot_frames"]
        out["snapshot_frames_by_grid"] = {str(n): spaced(frames, 6 if n < 512 else 4) for n in grids if n >= 256}
    elif timing:
        out["snapshot_frames"] = []                             # timing only: no state outputs
    else:
        out["snapshot_frames"] = c.get("snapshot_frames", spaced(frames, 3))
        out["slice_frames"] = spaced(frames, 6)                  # z midplane, for figures and movies
        out["restrict_to"] = [64, 128]
        out["spectra_every"] = 1
    # Truth: exact solutions where they exist; otherwise AthenaK PPM/HLLD/RK3 at 8x the finest planar grid
    # (true 2D) or at 512^3 in 3D (the 512^3 runs get 1024^3 references in a later stage).
    exact = c.get("problem", name) in EXACT
    if timing:
        out["cpu_codes"] = False
    elif name == "kh_localized_dtjump":
        out["external_runs"] = False
    elif not exact:
        out["reference_n"] = 4096 if p2 else 512
        out["reference_check_n"] = 2048 if p2 else 256
        if name == "decay_supersonic":out["reference_variant"] = "matched"   # best stalls (dt collapse) at Mach 9
    if name == "decay_supersonic":
        # AthenaK PPM/HLLD/RK3 stalls (dt collapse) at Mach 9 at every grid tried (64-256); keep the two
        # smallest grids to document it rather than burn hours at 256-512.
        out["code_grids"] = {"athenak_best": [64, 128]}
    return name, out


def main(dirs):
    configs = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.toml")):
            if f.name.startswith(("movie_", "gate_", ".")):continue
            name, out = sweep_config(f);configs[name] = (f, out)
    dest = BUNDLE/"configs"
    for name, (src, out) in sorted(configs.items()):
        # Physics comments carry over; the campaign's job-sizing advice does not apply to the sweep.
        head = [l for l in src.read_text().splitlines() if l.startswith("#") and not any(
            w in l for w in ("RAMSES at 256", "To include it"))]
        head = [l.replace("at grids divisible by 6, so the GLM", "including grids divisible by 6, where the GLM")
                 .replace("RAMSES (octree) needs powers of two and is excluded by the plan.", "GPU codes only.") for l in head]
        lines = head + [f"# Sweep config written by tools/make_sweep_configs.py from {src.parent.parent.name}/configs/{src.name}."]
        lines += [f"{k} = {value(v)}" for k, v in out.items()]
        (dest/f"{name}.toml").write_text("\n".join(lines) + "\n")
        print("wrote", name, out["grids"])


if __name__ == "__main__":
    if len(sys.argv) < 2:sys.exit(__doc__)
    main(sys.argv[1:])
