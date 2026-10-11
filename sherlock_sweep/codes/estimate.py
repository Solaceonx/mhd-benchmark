"""Cost model for the sweep: wall time, GPU-hours or core-hours, scratch and returned bytes per array task.

Calibrated on measurements, not guesses (sources below); every figure is (central, optimistic, pessimistic).
  * Steps per unit N: medians of completed Sherlock runs at the largest grid of each test (progress/data.js,
    2026-09-30). Codes without a measurement use the ratio of the same code to GLM on the other tests.
  * H100 throughput (cell-steps per second, stepping only): Sherlock medians by grid (GLM tiled LLF 1.26e9 at
    32^3, 2.0e9 from 128^3; CT 0.43e9 -> 1.22e9; AthenaK matched 0.03e9 -> 0.33e9, best 0.016e9 -> 0.19e9).
    glm_hlld and glm_ppm were measured at 32^3 only (1.1e9, 0.48e9) and are scaled by GLM's 32 -> 128 gain.
  * CPU: RAMSES 3D on 32 Sherlock ranks, 0.08-0.09e6 cell-steps/s per core (measured). RAMSES 2D and Athena++
    were timed serially on the Mac and scaled by RAMSES's Sherlock/Mac ratio (0.17); Athena++ is assumed to
    scale twice as well (0.3), since its measured Mac speed says nothing about MPI efficiency.
"""
import math

LEVELS = ("central", "optimistic", "pessimistic")
FULL_3D = ("orszag_tang_3d", "field_loop_3d", "decay")
EXACT = ("uniform", "entropy", "alfven", "alfven_oblique", "field_loop", "field_loop_3d")

# Steps per unit N, keyed by config name, for GLM ("glm"), CT, AthenaK (and Athena++, same CFL rule) and RAMSES.
STEPS = {
    "alfven": dict(glm=13.28, ct=13.28, athenak_matched=4.06, athenak_best=4.06, ramses_matched=6.72, ramses_best=4.22),
    "alfven_oblique": dict(glm=6.09, ct=6.09, athenak_matched=1.88, athenak_best=1.88, ramses_matched=3.12, ramses_best=1.88),
    "blast_2d_b0": dict(glm=4.51, ct=4.25, athenak_matched=1.65, athenak_best=1.57, ramses_matched=2.30, ramses_best=1.44),
    "blast_2d_b50": dict(glm=4.87, ct=4.56, athenak_matched=1.64, athenak_best=1.70, ramses_matched=2.48, ramses_best=1.53),
    "blast_2d_b100": dict(glm=6.62, ct=5.89, athenak_matched=2.04, athenak_best=2.16, ramses_matched=3.38, ramses_best=2.20),
    "blast_3d_b0": dict(glm=4.70, ct=4.18, athenak_matched=1.59, athenak_best=1.64, ramses_matched=2.42, ramses_best=1.55),
    "blast_3d_b50": dict(glm=4.95, ct=4.41, athenak_matched=1.64, athenak_best=1.73, ramses_matched=2.52, ramses_best=1.61),
    "blast_3d_b100": dict(glm=6.41, ct=5.59, athenak_matched=1.96, athenak_best=2.03, ramses_matched=3.27, ramses_best=2.13),
    "brio_wu": dict(glm=2.27, ct=2.27, athenak_matched=0.79, athenak_best=0.77, ramses_matched=1.17, ramses_best=0.70),
    "brio_wu_rotated": dict(glm=1.64, ct=1.62, athenak_matched=0.47, athenak_best=0.47),
    "current_sheet": dict(glm=33.51, ct=31.77, athenak_matched=10.08, athenak_best=13.19),
    "decay": dict(glm=20.47, ct=18.39, athenak_matched=7.11, athenak_best=7.69),
    "decay_supersonic": dict(glm=23.85, ct=18.92, athenak_matched=8.56),
    "entropy": dict(glm=21.88, ct=21.62, athenak_matched=6.72, athenak_best=6.72),
    "field_loop": dict(glm=72.97, ct=72.97, athenak_matched=23.59, athenak_best=23.59),
    "field_loop_3d": dict(glm=31.56, ct=31.56, athenak_matched=11.02, athenak_best=11.02),
    "field_loop_novz": dict(glm=55.0, ct=55.0, athenak_matched=22.03, athenak_best=22.03),
    "gem": dict(athenak_best=13.67),                                   # local AthenaK 256^2 reference run
    "kh_localized": dict(glm=164.01, ct=157.16, athenak_matched=51.95, athenak_best=52.23),
    "kh_localized_dtjump": dict(glm=175.9, ct=168.21),
    "kh_uniform": dict(glm=164.01, ct=157.16, athenak_matched=51.85, athenak_best=52.23),
    "orszag_tang": dict(glm=12.0, ct=11.52, athenak_matched=4.32, athenak_best=4.36, ramses_matched=5.92),
    "orszag_tang_3d": dict(glm=29.47, ct=24.95, athenak_matched=9.44, athenak_best=10.07),
    "rotor": dict(glm=5.25, ct=4.36, athenak_matched=1.70, athenak_best=1.91),
    "throughput_ot3d": dict(glm=1.23, ct=1.23, athenak_matched=0.42, athenak_best=0.42),
    "uniform": dict(glm=1.88, ct=1.88, athenak_matched=0.62, athenak_best=0.62),
}
# Median ratios to GLM over the tests where both were measured (fallbacks).
RATIO = dict(ct=0.93, athenak_matched=0.33, athenak_best=0.34, ramses_matched=0.52, ramses_best=0.33)


def steps_per_n(test, family):
    s = STEPS.get(test, {})
    if family in s:return s[family]
    glm = s.get("glm") or s.get("athenak_best", 1.0)/RATIO["athenak_best"]
    if family == "glm":return glm
    if family.startswith("athenapp"):return steps_per_n(test, "athenak_" + family.split("_")[1])
    return glm*RATIO[family]


# H100 cell-steps per second (central) by grid; between grids the smaller grid's value is used.
GPU_RATE = {
    "glm": {32: 1.26e9, 64: 1.98e9, 128: 2.05e9},
    "glm_hll16": {32: 1.2e9, 64: 1.93e9, 128: 2.03e9},
    "glm_constch": {32: 1.2e9, 64: 1.83e9, 128: 2.02e9},
    "glm_hlld": {32: 1.1e9, 64: 1.5e9, 128: 1.6e9},
    "glm_ppm": {32: 0.48e9, 64: 0.65e9, 128: 0.72e9},
    "ct": {32: 0.43e9, 64: 0.94e9, 128: 1.22e9},
    "athenak_matched": {32: 0.026e9, 64: 0.107e9, 128: 0.223e9, 256: 0.33e9},
    "athenak_best": {32: 0.016e9, 64: 0.062e9, 128: 0.123e9, 256: 0.19e9},
}
GPU_SPREAD = (1.0, 1.2, 0.7)           # rate multipliers: central, optimistic, pessimistic
# CPU cell-steps per second per core on Sherlock, by dimension.
CPU_RATE = {("ramses", "matched", 3): 0.09e6, ("ramses", "best", 3): 0.08e6,
            ("ramses", "matched", 2): 0.47e6, ("ramses", "best", 2): 0.37e6,
            ("athenapp", "matched", 3): 1.37e6, ("athenapp", "best", 3): 0.215e6,
            ("athenapp", "matched", 2): 2.2e6, ("athenapp", "best", 2): 0.31e6}
CPU_SPREAD = (1.0, 1.5, 0.5)
CFL_OVERHEAD = 1.15                    # GPU driver: the CFL scan adds ~15% to stepping
DRIVER_FIXED = (90.0, 60.0, 180.0)     # s per GPU-driver task: Julia start, CUDA init, compiling one method
EXTERNAL_FIXED = (60.0, 45.0, 150.0)   # s per external run: launch, init, Julia start for post-processing
BYTES_PER_S = 4e8                      # scratch write/read speed for states


def rate_at(table, n):
    keys = sorted(table)
    return table[max([k for k in keys if k <= n], default=keys[0])]


def planar(config):
    p = config.get("problem", config["name"])
    return config.get("dim", 3) == 2 if p == "blast" else p not in FULL_3D


def repeats(config, n):
    return int(config.get("repeats_by_grid", {}).get(str(n), config.get("repeats", 1)))


def snapshot_frames(config, n):
    frames = int(config["frames"])
    return list(config.get("snapshot_frames_by_grid", {}).get(str(n), config.get("snapshot_frames", range(frames))))


def slice_count(config):
    sf = config.get("slice_frames", [])
    return int(config["frames"]) if sf == "all" else len(sf)


def state_bytes(n, nz):
    return n*n*nz*9*4


def outputs_cost(config, n, cells, on_gpu):
    """Seconds per run spent on outputs (repeat 1): diagnostics, spectra, snapshot writes."""
    frames = int(config["frames"]);snaps = len(snapshot_frames(config, n))
    diag = frames*(cells*4e-9 if on_gpu else cells*6e-8)            # GPU reductions vs host
    spectra = frames*(cells*3e-8 if on_gpu else cells*4e-7) if config.get("spectra_every", 0) else 0.0
    return diag + spectra + snaps*cells*36/BYTES_PER_S


def unit_cost(config, code, variant, mode, n, geometry, method=None):
    """Per array task: dict of seconds (wall), resource-seconds (GPU or core), scratch and returned bytes."""
    test = config["name"];p2 = planar(config)
    nz = 1 if geometry == "plane" else n;cells = n*n*nz
    reps = 1 if mode == "reference" else repeats(config, n)
    frames = int(config["frames"])
    out = {}
    for i, level in enumerate(LEVELS):
        if code == "gpu":
            fam = "ct" if method == "ct" else "glm"
            steps = steps_per_n(test, fam)*n
            rate = rate_at(GPU_RATE[method], n)*GPU_SPREAD[i]
            run = steps*cells/rate*CFL_OVERHEAD
            wall = DRIVER_FIXED[i] + cells*5e-7 + reps*run + outputs_cost(config, n, cells, True)
            wall += reps*(run/900.0)*cells*36/BYTES_PER_S                  # checkpoints every 15 min
            res = wall
        elif code == "athenak":
            var = "best" if variant in ("best", "reference") else "matched"
            steps = steps_per_n(test, f"athenak_{var}")*n
            rate = rate_at(GPU_RATE[f"athenak_{var}"], max(n, round(cells**(1/3))))*GPU_SPREAD[i]
            run = steps*cells/rate
            convert = frames*cells*9*8/BYTES_PER_S*3                        # read native, write ours, read back
            wall = EXTERNAL_FIXED[i] + reps*run + convert + outputs_cost(config, n, cells, True)
            res = wall
        else:
            var = "best" if variant in ("best", "reference") else "matched"
            dim = 2 if geometry == "plane" else 3
            steps = steps_per_n(test, f"{code}_{var}" if code == "ramses" else f"athenapp_{var}")*n
            ranks = cpu_ranks(n, dim, mode)
            eff = 1.0 if ranks <= 16 else 0.85
            rate = CPU_RATE[(code, var, dim)]*CPU_SPREAD[i]*ranks*eff
            run = steps*cells/rate
            convert = frames*cells*9*8/BYTES_PER_S*4 + frames*cells*2e-6        # native read is slow (Python)
            wall = EXTERNAL_FIXED[i] + reps*run + convert + outputs_cost(config, n, cells, False)
            res = wall*ranks
        out[level] = dict(wall=wall, resource=res)
    # Planar tests save one z plane whatever the evolved geometry.
    out.update(storage(config, code, mode, n, 1 if p2 else n, reps, method))
    return out


def storage(config, code, mode, n, nz, reps, method):
    """Bytes kept on scratch after the run, bytes returned to the Mac (files up to the pack cap)."""
    cap = 50*2**20;frames = int(config["frames"])
    full = state_bytes(n, nz);files = []
    if mode == "reference":
        grids = [m for m in config["grids"] if n % m == 0 and m < n]
        files += [state_bytes(m, 1 if nz == 1 else m) for m in grids]*frames
    else:
        snaps = len(snapshot_frames(config, n))
        files += [full]*snaps
        if code == "gpu" and method == "ct":files += [n*n*nz*4*3]*snaps     # CT faces
        if nz > 1:
            files += [state_bytes(n, 1)]*slice_count(config)
            files += [state_bytes(m, m) for m in config.get("restrict_to", []) if m < n]*snaps
        files += [full]*(reps - 1) if code != "gpu" else []                  # external timing repeats keep finals
        if config.get("problem", config["name"]) in EXACT and code == "gpu":files += [full]*snaps   # ref_###.bin
    scratch = sum(files) + 2e6*reps
    returned = sum(f for f in files if f <= cap) + 1e6*reps
    return dict(scratch_bytes=scratch, returned_bytes=returned)


def cpu_ranks(n, dim, mode="run"):
    """MPI ranks for RAMSES/Athena++ (blocks of >= 16^3 or 64^2 cells per rank)."""
    if dim == 2:return {32: 1, 64: 2, 128: 4, 256: 8, 512: 16, 1024: 32, 2048: 64, 4096: 128}.get(n, 16)
    return {32: 4, 64: 8, 128: 32, 256: 64, 384: 96, 512: 128, 1024: 256}.get(n, 64)
