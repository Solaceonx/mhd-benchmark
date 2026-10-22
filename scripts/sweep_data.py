"""Read the results of the comprehensive sweep (sherlock_sweep) on this machine.

The results live on Sherlock under $SCRATCH/mhd-results-sweep, one directory per run:

    <run>/data/input.toml                 the test's config
    <run>/data/<method>_r<repeat>/        one *part*: run.toml, diagnostics.csv, spectra.csv (3D tests),
                                          dense_diagnostics.csv (timestep-jump runs), state_###.bin (kept frames),
                                          slice_###.bin (3D tests: the z midplane), ref_N<m>_###.bin (references)

The files this module's WANTED table names are copied, with the same paths, to sweep_results/ here by
scripts/sweep_fetch.py; everything below reads from there. Which runs exist is taken from the plan in the bundle
(sherlock_sweep/codes/plan.py) and from the newest status export under progress/sweep_status/ (scripts/sweep_status.py).

Binary files are little-endian Float32 in Fortran order: (n, n, 9) for planar tests and slices, fields
rho, mx, my, mz, E, Bx, By, Bz, psi (psi is zero for codes without divergence cleaning).
"""
import functools
import json
import os
import re
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/"sherlock_sweep"
SITE = Path(os.environ.get("MHD_SITE", ROOT/"progress"))                        # override: tests of this tooling
RESULTS = Path(os.environ.get("MHD_SWEEP_RESULTS", ROOT/"sweep_results"))     # override: tests of this tooling
REMOTE = "/scratch/users/lyxia/mhd-results-sweep"
sys.path.insert(0, str(BUNDLE/"codes"))
sys.path.insert(0, str(ROOT/"scripts"))
import plan  # noqa: E402
import sweep_status  # noqa: E402

GPU_METHODS = ("glm", "glm_hll16", "glm_hlld", "glm_ppm", "glm_constch", "ct")
EXTERNAL = ("athenak_matched", "athenak_best", "ramses_matched", "ramses_best", "athenapp_matched", "athenapp_best")
REFERENCES = ("athenak_reference", "athenapp_reference")
METHODS = GPU_METHODS + EXTERNAL
# AREPO (sherlock_arepo/): run after the sweep, outside its plan and its status export, on the planar tests at seed
# 42. Its runs are known here by the labels below (the names of its part directories); their state is read from the
# fetched run.toml files. They are not in METHODS, so the pages built from the plan do not change.
AREPO = ("arepo_moving", "arepo_static")
AREPO_TESTS = ("entropy", "alfven", "alfven_oblique", "uniform", "field_loop", "field_loop_novz", "brio_wu", "brio_wu_rotated",
               "orszag_tang", "rotor", "current_sheet", "blast_2d_b0", "blast_2d_b50", "blast_2d_b100", "kh_uniform", "kh_localized", "gem")
# The fixed kernels of the sweep-fixes branch (sherlock_fixes/README.md, "The full rerun"): every sweep test at every
# grid and seed, one directory per kernel, fetched into fix_results/ (git-ignored). Each is set beside the original it
# fixes (FIXES); like AREPO they are outside the plan, so a page shows them only when it asks (labels(fixed=True)).
FIXES = {"glm_fix_hll16": "glm_hll16", "glm_fct16": "glm_hll16", "glm_fix_hlld": "glm_hlld", "glm_fix_ppm": "glm_ppm",
         "ct_fix": "ct"}
FIXED = tuple(FIXES)
RERUN = Path(os.environ.get("MHD_RERUN_RESULTS", ROOT/"fix_results"/"mhd-results-rerun"))
FIELDS = ("rho", "mx", "my", "mz", "E", "Bx", "By", "Bz", "psi")

# The binary files the paper figures read, per test: frames of the full state for every method (STATE: {grid: frames},
# "end" = the last output), the same for the z-midplane slices of 3D tests (SLICE), and the grids at which the
# references' block averages are read (REF: the frames are those of STATE). The initial frame is taken from one
# method only (FIRST), since every code starts from the same data. Text files (run.toml, the CSVs) are read for
# every part of every test.
WANTED = {
    "kh_uniform": dict(state={64: ["end"], 128: ["end"], 256: ["end"], 512: ["end"]}, first=(64, 128, 256, 512),
                       ref=(64, 128, 256, 512)),
    "kh_localized": dict(state={64: ["end"], 128: [12, "end"], 256: [12, "end"], 512: ["end"]}, first=(64, 128, 256, 512),
                         ref=(64, 128, 256, 512)),
    "kh_localized_dtjump": dict(state={128: [30, 31, "end"], 256: ["end"], 512: ["end"]}),
    "field_loop": dict(state={32: ["end"], 64: [2, 10, "end"], 128: ["end"], 256: ["end"]}, first=(32, 64, 128, 256)),
    "field_loop_novz": dict(state={32: ["end"], 64: [2, 10, "end"], 128: ["end"], 256: ["end"]}, first=(32, 64, 128, 256)),
    "field_loop_3d": dict(slice={32: [0, 2, 4, 6, 8, 10], 64: [0, 2, 4, 6, 8, 10], 128: [0, 2, 4, 6, 8, 10], 256: [0, 10]}),
    "current_sheet": dict(state={256: [4, 8, 12, 16, "end"]}, first=(256,), ref=(256,)),
    "alfven": dict(state={32: ["end"], 64: ["end"], 128: ["end"]}, first=(32, 64, 128)),
    "alfven_oblique": dict(state={32: ["end"], 64: ["end"], 128: ["end"]}, first=(32, 64, 128)),
    "brio_wu": dict(state={256: ["end"]}, ref=(256,)),
    "brio_wu_rotated": dict(state={128: ["end"], 256: ["end"]}, first=(256,), ref=(128, 256)),
    "blast_2d_b50": dict(state={256: ["end"]}, ref=(256,)),
    "blast_2d_b100": dict(state={256: ["end"]}, ref=(256,)),
    "blast_3d_b50": dict(slice={256: ["end"]}),
    "blast_3d_b100": dict(slice={256: ["end"]}),
    "decay_supersonic": dict(slice={256: ["end"]}),
    "gem": dict(state={256: ["end"]}, ref=(256,)),
}
FIRST_FROM = "ct"         # the initial frame is read from this method's run


def read_csv(path):
    """{column: float array} of a diagnostics/spectra CSV (true/false -> 1/0); None if there are no rows."""
    lines = Path(path).read_text().splitlines()
    if len(lines) < 2:return None
    names = lines[0].split(",");conv = {"true": 1.0, "false": 0.0}
    rows = []
    for line in lines[1:]:
        cells = line.split(",")
        if len(cells) != len(names):continue          # a torn last line of a run that was stopped
        rows.append([conv[c] if c in conv else float(c) for c in cells])
    if not rows:return None
    a = np.array(rows, dtype=float)
    return {name: a[:, i] for i, name in enumerate(names)}


class Sweep:
    """The plan, the status and the fetched results."""

    def __init__(self, results=RESULTS):
        self.results = Path(results)
        self.runs = plan.all_runs()
        self.configs = {p.stem: plan.load(p) for p in plan.config_paths()}
        self.by = {}                                   # (test, label, n, seed) -> run of the plan
        for r in self.runs:
            for label in (r["extra"].split(",") if r["code"] == "gpu" else [sweep_status.label_of(r)]):
                self.by[(r["test"], label, r["n"], r["seed"])] = r
        caps = {name: int(c.get("max_steps_per_n", 0)) for name, c in self.configs.items()}
        self.status = sweep_status.load(SITE, self.runs, caps)
        self.rows = {(x["test"], x["label"], x["n"], x["seed"]): x for x in (self.status or {}).get("rows", [])}
        self.export = None
        if self.status and self.status.get("source") == "export":
            data = json.loads((SITE/"sweep_status"/self.status["file"]).read_text())
            self.export = {r["key"]: {p["part"]: p for p in r.get("parts", [])} for r in data["runs"]}
        self.arepo = {}                                # (test, label, n, 42) -> status row, from the fetched records
        for test in AREPO_TESTS:
            c = self.configs.get(test)
            if c is None:continue
            for label in AREPO:
                mesh = label.split("_")[1]
                for n in c["grids"]:
                    key = (test, label, int(n), 42);d = f"{test}-arepo-{mesh}-run-N{n}-s42"
                    self.by[key] = dict(test=test, code="arepo", variant=mesh, mode="run", n=int(n), seed=42, extra="", dir=d, key=d)
                    m = self.meta(*key);row = dict(test=test, n=int(n), seed=42, label=label, frac=None, state="open")
                    if m is not None:
                        row["state"] = "done" if m.get("status") == "complete" else "stopped"
                        if row["state"] == "stopped":
                            reached = float(m.get("arepo_stopped_at_time", m.get("time_reached", 0.0)));t_end = float(c["t_end"])
                            row["frac"] = min(1.0, reached/t_end) if t_end else None;row["reached"] = reached
                            row["note"] = m.get("arepo_stopped", "")
                    self.arepo[key] = row;self.rows[key] = row
        self.fixed = {}                                # (test, label, n, seed) -> status row, from the fetched records
        for d in sorted(RERUN.glob("*-N*-s*-*")) if RERUN.is_dir() else ():
            m = re.fullmatch(r"(.+)-N(\d+)-s(\d+)-(\w+)", d.name)
            if not m or m[4] not in FIXES or m[1] not in self.configs:continue
            test, n, seed, label = m[1], int(m[2]), int(m[3]), m[4];key = (test, label, n, seed)
            # dir is absolute: self.results/dir is then the rerun's directory
            self.by[key] = dict(test=test, code="fixed", variant=label, mode="run", n=n, seed=seed, extra=label, dir=str(d), key=d.name)
            meta = self.meta(*key);row = dict(test=test, n=n, seed=seed, label=label, frac=None, state="open")
            if meta is not None:
                row["state"] = "done" if meta.get("status") == "complete" else "stopped"
                if row["state"] == "stopped":
                    t_end = float(self.configs[test]["t_end"]);row["reached"] = float(meta.get("time_reached", 0.0))
                    row["frac"] = min(1.0, row["reached"]/t_end) if t_end else None;row["note"] = meta.get("error", "").split("\n")[0]
            self.fixed[key] = row;self.rows[key] = row

    # ---- the plan ----
    def frames(self, test):
        return int(self.configs[test]["frames"])

    def times(self, test):
        c = self.configs[test];return np.linspace(0.0, float(c["t_end"]), int(c["frames"]))

    def kept_frames(self, test, n):
        """Frames whose full state a repeat-1 run keeps."""
        c = self.configs[test]
        return list(c.get("snapshot_frames_by_grid", {}).get(str(n), c.get("snapshot_frames", range(int(c["frames"])))))

    def slice_frames(self, test):
        c = self.configs[test];sf = c.get("slice_frames", [])
        return list(range(int(c["frames"]))) if sf == "all" else list(sf)

    def grids(self, test, label):
        return sorted({n for (t, l, n, s) in self.by if t == test and l == label})

    def seeds(self, test):
        return list(self.configs[test].get("seeds", [42]))

    def labels(self, test, references=False, fixed=False):
        """Methods planned for a test, in display order; fixed: with the rerun's fixed kernels (after the originals)."""
        have = {l for (t, l, n, s) in self.by if t == test}
        return [m for m in GPU_METHODS + (FIXED if fixed else ()) + EXTERNAL + (REFERENCES if references else ()) if m in have]

    def reference_n(self, test, label):
        g = self.grids(test, label);return g[0] if g else None

    # ---- paths ----
    def part(self, test, label, n, seed=42, rep=1):
        """Relative path of a part's directory, or None when the plan has no such run."""
        r = self.by.get((test, label, n, seed))
        return None if r is None else f'{r["dir"]}/data/{label}_r{rep}'

    def path(self, test, label, n, name, seed=42, rep=1):
        p = self.part(test, label, n, seed, rep)
        return None if p is None else self.results/p/name

    def has(self, test, label, n, name, seed=42):
        p = self.path(test, label, n, name, seed)
        return p is not None and p.is_file() and p.stat().st_size > 0

    # ---- status ----
    def state_of(self, test, label, n, seed=42):
        """done / stopped / redo / open / skipped / unknown, or None when not planned."""
        row = self.rows.get((test, label, n, seed))
        return row["state"] if row else None

    def recorded(self, test, label, n, seed=42, rep=1):
        """The export's record of a part (status, reached, frames, steps, ...), or None."""
        r = self.by.get((test, label, n, seed))
        if r is not None and r["code"] in ("arepo", "fixed"):          # not in the export: from the fetched record
            m = self.meta(test, label, n, seed, rep);row = self.rows.get((test, label, n, seed), {})
            return None if m is None else dict(part=f"{label}_r{rep}", status=m.get("status"), frames=len(m.get("frames", [])),
                                               reached=row.get("reached", m.get("time_reached")), steps=m.get("steps"))
        if r is None or self.export is None:return None
        return self.export.get(r["key"], {}).get(f"{label}_r{rep}")

    # ---- data ----
    @functools.lru_cache(maxsize=None)
    def meta(self, test, label, n, seed=42, rep=1):
        p = self.path(test, label, n, "run.toml", seed, rep)
        return tomllib.loads(p.read_text()) if p is not None and p.is_file() else None

    @functools.lru_cache(maxsize=None)
    def diag(self, test, label, n, seed=42, name="diagnostics.csv"):
        p = self.path(test, label, n, name, seed)
        return read_csv(p) if p is not None and p.is_file() and p.stat().st_size > 0 else None

    @functools.lru_cache(maxsize=None)
    def finite_rows(self, test, label, n, seed=42):
        """How many leading outputs of a part are finite (its energies are numbers and its state is flagged finite), or
        None without diagnostics. A code that keeps running after its state turned to NaN is recorded as complete."""
        d = self.diag(test, label, n, seed)
        if d is None:return None
        ok = np.ones(len(d["time"]), dtype=bool)
        for c in ("mass", "energy", "kinetic", "magnetic", "thermal"):
            if c in d:ok &= np.isfinite(d[c])
        if "finite" in d:ok &= d["finite"] != 0
        return len(ok) if ok.all() else int(np.argmin(ok))

    def went_bad(self, test, label, n, seed=42):
        """Time of the last finite output of a part whose later outputs are not finite; None when all are finite."""
        k = self.finite_rows(test, label, n, seed)
        if k is None:return None
        t = self.diag(test, label, n, seed)["time"]
        return None if k == len(t) else float(t[k - 1]) if k else 0.0

    def nonfinite_parts(self):
        """{(test, label, n, seed): time of the last finite output} for every fetched repeat-1 part that has outputs
        which are not finite. Such a run can be recorded as complete: the code kept stepping on NaN."""
        out = {}
        for key in self.by:
            t = self.went_bad(*key)
            if t is not None:out[key] = t
        return out

    def dense(self, test, label, n, seed=42):
        return self.diag(test, label, n, seed, "dense_diagnostics.csv")

    @functools.lru_cache(maxsize=None)
    def spectra(self, test, label, n, seed=42):
        """{frame: {column: array over k}} or None."""
        d = self.diag(test, label, n, seed, "spectra.csv")
        if d is None:return None
        out = {}
        for f in np.unique(d["frame"]):
            m = d["frame"] == f;out[int(f)] = {k: v[m] for k, v in d.items()}
        return out

    def frame_file(self, test, label, n, frame, seed=42, kind="state"):
        """Path of a kept binary frame; frame may be "end". kind: state, slice."""
        f = self.frames(test) - 1 if frame == "end" else int(frame)
        return self.path(test, label, n, f"{kind}_{f:03d}.bin", seed)

    def field(self, test, label, n, frame, seed=42, kind="state"):
        """(n, n, 9) array indexed [ix, iy, field], or None when the file is not here."""
        p = self.frame_file(test, label, n, frame, seed, kind)
        if p is None or not p.is_file() or p.stat().st_size != n*n*9*4:return None
        u = np.fromfile(p, dtype="<f4").reshape((n, n, 9), order="F")
        return u if np.isfinite(u).all() else None           # a state with NaN or Inf is not a result

    def ref_field(self, test, label, m, frame, seed=42):
        """A reference run's block average at grid m: (m, m, 9), or None."""
        big = self.reference_n(test, label)
        if big is None:return None
        f = self.frames(test) - 1 if frame == "end" else int(frame)
        p = self.path(test, label, big, f"ref_N{m}_{f:03d}.bin", seed)
        if p is None or not p.is_file():return None
        size = p.stat().st_size
        if size == m*m*9*4:return np.fromfile(p, dtype="<f4").reshape((m, m, 9), order="F")
        if size == m**3*9*4:return np.fromfile(p, dtype="<f4").reshape((m, m, m, 9), order="F")
        return None

    # ---- what to copy from Sherlock ----
    def wanted(self):
        """[(relative path, bytes or None, known_absent)] of every file the figures and the site read.

        known_absent: the status export shows the file cannot exist (the part did not get that far). Parts the
        export does not list (still running or waiting when it was written) are asked for anyway."""
        out, seen = [], set()

        def add(path, size=None, absent=False):
            if path not in seen:seen.add(path);out.append((path, size, absent))
        for r in self.runs:
            labels = r["extra"].split(",") if r["code"] == "gpu" else [sweep_status.label_of(r)]
            test, n, seed = r["test"], r["n"], r["seed"];base = f'{r["dir"]}/data';c = self.configs[test]
            parts = (self.export or {}).get(r["key"]) if self.export is not None else None
            add(f"{base}/input.toml")
            repeats = int(c.get("repeats_by_grid", {}).get(str(n), c.get("repeats", 1))) if r["mode"] == "run" else 1
            spec = WANTED.get(test, {});last = int(c["frames"]) - 1;planar = plan.planar(c)
            for label in labels:
                for rep in range(1, repeats + 1):
                    rec = None if parts is None else parts.get(f"{label}_r{rep}")
                    gone = parts is not None and bool(parts) and rec is None and rep > 1 and f"{label}_r1" in parts \
                        and parts[f"{label}_r1"].get("status") == "failed"       # no timing repeats after a failed repeat 1
                    add(f"{base}/{label}_r{rep}/run.toml", None, gone)
                    add(f"{base}/{label}_r{rep}/diagnostics.csv", None, gone)
                    if rep == 1 and not planar and int(c.get("spectra_every", 0)) > 0:
                        add(f"{base}/{label}_r1/spectra.csv")
                    if rep == 1 and r["code"] == "gpu" and c.get("dt_jump"):
                        add(f"{base}/{label}_r1/dense_diagnostics.csv")
                rec = None if parts is None else parts.get(f"{label}_r1")
                got = None if rec is None else int(rec.get("frames") or 0)       # frames 0 .. got-1 were written
                frame_no = lambda f: last if f == "end" else int(f)
                if r["mode"] == "reference":
                    frames = sorted({frame_no(f) for fs in spec.get("state", {}).values() for f in fs} | ({0} if spec.get("first") else set()))
                    for m in spec.get("ref", ()):
                        if n % m or m >= n:continue
                        for f in frames:
                            add(f"{base}/{label}_r1/ref_N{m}_{f:03d}.bin", m*m*9*4, got is not None and f >= got)
                    continue
                kept = set(self.kept_frames(test, n))
                frames = [frame_no(f) for f in spec.get("state", {}).get(n, [])]
                if n in spec.get("first", ()) and label == (FIRST_FROM if FIRST_FROM in self.labels(test) else labels[0]):frames.append(0)
                for f in sorted(set(frames)):
                    if planar and f in kept:add(f"{base}/{label}_r1/state_{f:03d}.bin", n*n*9*4, got is not None and f >= got)
                sliced = set(self.slice_frames(test))
                for f in sorted({frame_no(f) for f in spec.get("slice", {}).get(n, [])}):
                    if not planar and f in sliced:add(f"{base}/{label}_r1/slice_{f:03d}.bin", n*n*9*4, got is not None and f >= got)
        return out
