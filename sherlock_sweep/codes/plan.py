"""The sweep matrix: every run, and how runs are packed into Slurm jobs.

  python3 codes/plan.py runs [--json FILE]     every run (one output directory) with its cost estimate
  python3 codes/plan.py units [--json FILE]    every Slurm job of the sweep: packed runs, resources, estimates
  python3 codes/plan.py task CONFIG CODE MODE INDEX [VARIANT]   (legacy) one run of a config, by index

A *run* is one output directory: one test, code, variant, mode, grid and seed (for the GPU driver also a method
group: all six methods together below 256^3, one method per run from 256^3 up, where single runs take hours).
A *unit* is one Slurm job. Runs of one family (GPU driver, AthenaK, CPU codes with the same rank count) are
packed, cheapest first, into jobs of about PACK_SECONDS; runs longer than LONG_SECONDS get a job of their own.
tools/sweep.py writes each unit's run list to sweep_state/tasks/ and the job scripts run that list.
Planar (z-invariant) tests run as N^3 cubes in the GPU driver and AthenaK (like-for-like GPU timing) and as
true 2D grids in RAMSES (NDIM=2 build) and Athena++; references run true 2D.
"""
import json
import math
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimate  # noqa: E402

BUNDLE = Path(__file__).resolve().parents[1]
FULL_3D = estimate.FULL_3D
GPU_ONLY = ("kh_localized_dtjump",)        # needs the GPU driver's timestep-jump protocol
EXTERNAL_VARIANTS = {"athenak": ("matched", "best"), "ramses": ("matched", "best"), "athenapp": ("matched", "best")}
CPU_CODES = ("ramses", "athenapp")
PACK_SECONDS = 2*3600       # target length of a job of packed small runs (central estimate)
LONG_SECONDS = 3600         # runs longer than this get a job of their own
SPLIT_FROM = 256            # GPU driver: one method per run from this grid up
PER_RUN_GPU = 15.0          # GPU driver: per-run set-up inside a job (initial state, warm-up); start-up is per job


def load(path):
    config = tomllib.loads(Path(path).read_text());config.setdefault("problem", config["name"])
    return config


def config_paths():
    return sorted(p for p in (BUNDLE/"configs").glob("*.toml") if not p.name.startswith(("gate_", ".")))


def planar(config):
    return estimate.planar(config)


def run_grids(config, code, variant=None):
    grids = list(config["grids"])
    limits = config.get("code_grids", {})
    key = f"{code}_{variant}" if variant else code
    if key in limits:grids = [n for n in grids if n in limits[key]]
    elif code in limits:grids = [n for n in grids if n in limits[code]]
    if code == "ramses":grids = [n for n in grids if n & (n - 1) == 0]     # octree: powers of two
    return grids


def geometry(config, code, n):
    if not planar(config):return "cube"
    if code in CPU_CODES:return "plane"
    if code == "athenak" and n >= int(config.get("athenak_plane_from", 10**9)):return "plane"
    return "cube"


def reference_variant(config, code):
    return config.get("reference_variant", "reference") if code == "athenak" else "reference"


def config_runs(config, code, mode, variant=None):
    """(n, seed, geometry, extra) for every run of one config and code; extra = method list (GPU driver),
    restriction grids (references) or ''."""
    seeds = config.get("seeds", [42])
    if mode == "reference":
        key = "reference_n" if code == "athenak" else "reference_check_n"
        n = int(config.get(key, 0))
        if not n:return []
        restrict = ",".join(str(m) for m in config["grids"] if n % m == 0 and m < n)
        return [(n, s, "plane" if planar(config) else "cube", restrict) for s in seeds]
    if code != "gpu":
        if config["name"] in GPU_ONLY or not config.get("external_runs", True):return []
        if code in CPU_CODES and not config.get("cpu_codes", True):return []
    grids = run_grids(config, code, variant)
    if code == "gpu":
        methods = config.get("methods", ["glm", "ct"])
        out = []
        for s in seeds:
            for n in grids:
                if n < SPLIT_FROM:out.append((n, s, "cube", ",".join(methods)))
                else:out += [(n, s, "cube", m) for m in methods]
        return out
    return [(n, s, geometry(config, code, n), "") for s in seeds for n in grids]


# Legacy per-config task list (the resume gate and smoke run use the driver's own index decoding).
def tasks(config, code, mode, variant=None):
    return config_runs(config, code, mode, variant)


def run_dir_name(r):
    if r["code"] == "gpu":
        tag = "all" if "," in r["extra"] else r["extra"]
        return f"{r['test']}-gpu-N{r['n']}-s{r['seed']}-{tag}"
    return f"{r['test']}-{r['code']}-{r['variant']}-{r['mode']}-N{r['n']}-s{r['seed']}"


def run_cost(config, code, variant, mode, n, geom, extra):
    """Cost dict of one run (levels -> wall, resource) plus storage; GPU-driver start-up counted per job."""
    if code == "gpu":
        parts = [estimate.unit_cost(config, "gpu", "", mode, n, geom, m) for m in extra.split(",")]
        out = {}
        for i, lv in enumerate(estimate.LEVELS):
            w = sum(p[lv]["wall"] - estimate.DRIVER_FIXED[i] for p in parts) + PER_RUN_GPU
            out[lv] = dict(wall=w, resource=w)
        out["scratch_bytes"] = sum(p["scratch_bytes"] for p in parts);out["returned_bytes"] = sum(p["returned_bytes"] for p in parts)
        return out
    return estimate.unit_cost(config, code, variant, mode, n, geom)


def all_runs(settings=None):
    """Every run of the sweep, after the start-up filters (only, skip_cpu, cpu_max_n3d)."""
    settings = settings or {}
    only = settings.get("only") or [];cap3d = int(settings.get("cpu_max_n3d") or 10**9)
    out = []
    for path in config_paths():
        config = load(path);name = config["name"]
        if only and name not in only:continue
        combos = [("gpu", "", "run")] + [(c, v, "run") for c, vs in EXTERNAL_VARIANTS.items() for v in vs]
        combos += [("athenak", reference_variant(config, "athenak"), "reference"), ("athenapp", "reference", "reference")]
        for code, variant, mode in combos:
            if code in CPU_CODES and settings.get("skip_cpu"):continue
            for n, seed, geom, extra in config_runs(config, code, mode, variant or None):
                if code in CPU_CODES and geom == "cube" and n > cap3d:continue
                c = run_cost(config, code, variant, mode, n, geom, extra)
                dim = 2 if geom == "plane" else 3
                family = "gpu" if code == "gpu" else "athenak" if code == "athenak" else "cpu"
                label = "gpu" if code == "gpu" else f"{code}:{variant}"
                r = dict(test=name, code=code, variant=variant, mode=mode, n=n, seed=seed, geometry=geom, extra=extra,
                         family=family, gpu=family != "cpu", planar=planar(config),
                         ranks=estimate.cpu_ranks(n, dim, mode) if family == "cpu" else 0,
                         wall=c["central"]["wall"], wall_pessimistic=c["pessimistic"]["wall"],
                         resource={lv: c[lv]["resource"] for lv in estimate.LEVELS},
                         scratch_bytes=c["scratch_bytes"], returned_bytes=c["returned_bytes"])
                r["key"] = f"{name}|{label}|{mode}|N{n}|s{seed}" + (f"|{extra}" if code == "gpu" and "," not in extra else "")
                r["dir"] = run_dir_name(r)
                out.append(r)
    return out


def job_resources(runs):
    """sbatch options for a job running these runs one after another (partition and GPU feature added by sweep.py)."""
    family = runs[0]["family"]
    fixed = estimate.DRIVER_FIXED[2] if family == "gpu" else 0.0
    seconds = sum(r["wall_pessimistic"] for r in runs) + fixed
    limit = max(1800, min(48*3600, int(1.5*seconds) + 1200))     # a TIMEOUT is resubmitted with 1.5x and resumes
    if limit > 2*3600:limit = int(math.ceil(limit/3600))*3600
    hh, rest = divmod(limit, 3600);time = f"{hh:02d}:{rest//60:02d}:00"
    if family in ("gpu", "athenak"):
        size = max(r["n"] if r["geometry"] == "cube" else round((r["n"]*r["n"])**(1/3)) for r in runs)
        mem = "16G" if size <= 128 else "32G" if size <= 256 else "64G" if size <= 384 else "96G"
        return ["--gpus=1", "--cpus-per-task=4", f"--mem={mem}", f"--time={time}"]
    ranks = max(r["ranks"] for r in runs)
    # RAMSES reads the whole initial state on every rank (codes/ramses/condinit.f90): N^3 x 48 bytes each in 3D
    # (6.4 GB at 512^3), on top of ~4 GB for the code; fewer ranks per node keep a node under ~96 GB.
    ic = max((r["n"]**3*48 if r["code"] == "ramses" and r["geometry"] == "cube" else 0) for r in runs)
    mem = 4 + math.ceil(1.2*ic/2**30)
    per_node = 16 if mem <= 4 else min(16, max(4, 96//mem))
    opts = [f"--ntasks={ranks}", "--cpus-per-task=1", f"--mem-per-cpu={mem}G", f"--time={time}"]
    if ranks >= 32 or per_node < 16:
        opts.insert(1, f"--ntasks-per-node={min(per_node, ranks)}")   # also >= 64 GB on the node that converts output
    return opts


def pack(runs):
    """Greedy packing, cheapest first: jobs of about PACK_SECONDS; long runs alone."""
    jobs, cur, t = [], [], 0.0
    for r in sorted(runs, key=lambda r: (r["wall"], r["key"])):
        if r["wall"] > LONG_SECONDS:jobs.append([r]);continue
        if cur and t + r["wall"] > PACK_SECONDS:jobs.append(cur);cur, t = [], 0.0
        cur.append(r);t += r["wall"]
    if cur:jobs.append(cur)
    return jobs


def make_unit(key, runs):
    family = runs[0]["family"]
    fixed = [estimate.DRIVER_FIXED[i] if family == "gpu" else 0.0 for i in range(3)]
    res = {lv: sum(r["resource"][lv] for r in runs) + fixed[i]*(1 if family != "cpu" else 0)
           for i, lv in enumerate(estimate.LEVELS)}
    tests = sorted({r["test"] for r in runs})
    return dict(key=key, family=family, gpu=family != "cpu", runs=runs, tasks=1, indices=[0], sbatch=job_resources(runs),
                wall_central=sum(r["wall"] for r in runs) + fixed[0], resource=res,
                scratch_bytes=sum(r["scratch_bytes"] for r in runs), returned_bytes=sum(r["returned_bytes"] for r in runs),
                tests=tests, test=tests[0] if len(tests) == 1 else f"{len(tests)} tests",
                n=max(r["n"] for r in runs), runs_count=len(runs))


def units(settings=None, pilots=None):
    """Every Slurm job. `pilots` = {name: [run keys]}: those runs form their own units (the gates) and are
    left out of the packs. Deterministic for a given bundle and settings."""
    runs = all_runs(settings);by_key = {r["key"]: r for r in runs}
    out, taken = [], set()
    for name, keys in (pilots or {}).items():
        rs = [by_key[k] for k in keys if k in by_key]
        if rs:out.append(make_unit(f"pilot:{name}", rs));taken |= {r["key"] for r in rs}
    groups = {}
    for r in runs:
        if r["key"] in taken:continue
        cls = r["family"] if r["family"] != "cpu" else f"cpu-{'2d' if r['geometry'] == 'plane' else '3d'}-{r['ranks']}r"
        groups.setdefault(cls, []).append(r)
    for cls in sorted(groups):
        for k, job in enumerate(pack(groups[cls])):
            key = job[0]["key"] if len(job) == 1 else f"pack:{cls}:{k:03d}"
            out.append(make_unit(key, job))
    return out


def safe(key):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)


def task_lines(unit):
    """The run list a job executes: RUNDIR TEST CODE VARIANT MODE N SEED GEOMETRY EXTRA (extra '-' if empty)."""
    return [" ".join(str(x) for x in (r["dir"], r["test"], r["code"], r["variant"] or "-", r["mode"], r["n"], r["seed"],
                                      r["geometry"], r["extra"] or "-")) for r in unit["runs"]]


def main(argv):
    if not argv:sys.exit(__doc__)
    cmd = argv[0]
    if cmd == "task":
        config, code, mode, index = load(argv[1]), argv[2], argv[3], int(argv[4])
        variant = argv[5] if len(argv) > 5 else None
        n, seed, geom, extra = config_runs(config, code, mode, variant)[index]
        print(n, seed, geom, extra or "-")
    elif cmd in ("runs", "units"):
        items = all_runs() if cmd == "runs" else units()
        if "--json" in argv:Path(argv[argv.index("--json") + 1]).write_text(json.dumps(items, indent=1))
        else:
            for u in items:
                print(u["key"], f"{u['wall']/3600 if cmd == 'runs' else u['wall_central']/3600:.2f} h",
                      "" if cmd == "runs" else f"{u['runs_count']} runs " + " ".join(u["sbatch"]))
    else:sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
