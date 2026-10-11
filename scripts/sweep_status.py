"""Which runs of the comprehensive sweep exist: the status the progress site shows on its Sweep page.

Sources, kept under progress/sweep_status/ (the newest by its time stamp is used):

  export_*.json        written on Sherlock by the export command in sherlock_sweep/SWEEP_RUN.md and copied here:
                       every run and every part (method, repeat) with its recorded status. Exact.
                       An optional export_*.since.json records what is known to have changed after that export
                       ({"base": the export's time, "skipped": [run keys], "remarks": [...]}); it applies only to the
                       export whose time it names, so the next export supersedes it.
  summary_*.txt        the text printed by `tools/sweep.py runs` and `status`: per method and grid, how many repeat-1
                       parts completed, and the failed ones by test with how far they got. Exact for every method/grid
                       group it accounts for completely; for the others it says how many runs are unfinished but not
                       which. An optional summary_*.notes.json names what is known from elsewhere (see NOTES below).

A *part* is one method of one run, repeat 1 (the repeat that carries the data; later repeats are timing only).
States: done; stopped (the solver stopped, or stalled into the step cap: a result); redo (failed for a reason that is
not the solver's, to be run again); open (no result yet: running, queued, or left unfinished by its job); unknown
(done or open: the summary does not say which); skipped (taken out of the sweep).
"""
import json
import re
from pathlib import Path

GPU_METHODS = ("glm", "glm_hll16", "glm_hlld", "glm_ppm", "glm_constch", "ct")
# The GPU driver stops on these by design; anything else that fails there is not a solver result.
SOLVER_STOPS = ("Nonphysical/nonfinite state", "Invalid state at output", "Invalid timestep", "Step limit reached",
                "Out-of-plane variation exceeds tolerance", "Uniform state preservation failed",
                "Initial face divergence too large", "Per-run wall limit reached")
STATES = ("done", "stopped", "redo", "open", "unknown", "skipped")


def label_of(run):
    """Column label of an external-code run of the plan: athenak_matched, ramses_best, athenapp_reference, ..."""
    return f'{run["code"]}_{"reference" if run["mode"] == "reference" else run["variant"]}'


def planned_parts(runs):
    """{(label, n): [(test, seed), ...]} for every part of the plan, in plan order."""
    out = {}
    for r in runs:
        labels = r["extra"].split(",") if r["code"] == "gpu" else [label_of(r)]
        for label in labels:out.setdefault((label, r["n"]), []).append((r["test"], r["seed"]))
    return out


def lane_line(text, lane):
    m = re.search(rf"^{lane}: (\d+)/(\d+) jobs submitted; (.*?); (\d+) (?:GPU-h|core-h) used of ~(\d+) estimated", text, re.M)
    if not m:return None
    states = dict((k, int(v)) for k, v in (x.rsplit(" ", 1) for x in m.group(3).split(", ")))
    return dict(submitted=int(m.group(1)), total=int(m.group(2)), states=states, used=int(m.group(4)), estimate=int(m.group(5)))


def from_summary(text, planned, notes):
    """Status from the text of `sweep.py runs` (+ `status`). notes (NOTES): what the text cannot say.

    NOTES keys: time; redo = {label: [tests]} failed parts that are not solver results; open = [[label, n, test, why],
    ...] unfinished runs known by name; override = [[label, n, test, state, why], ...] a part whose record says
    complete but which is known to be something else (a damaged run); timing = [text, ...] runs whose data repeat is
    done but a timing repeat failed; stuck = [text, ...]; remarks = [text, ...]."""
    groups = {}
    for m in re.finditer(r"^[ \t]+(\S+)[ \t]+N=[ \t]*(\d+)[ \t]+ok[ \t]+(\d+)[ \t]+failed[ \t]+(\d+)[ \t]*(.*)$", text, re.M):
        fails = [(t, int(p)/100) for t, p in re.findall(r"(\S+?)\((\d+)%\)", m.group(5))]
        if len(fails) != int(m.group(4)):raise ValueError(f"cannot read the failures of {m.group(1)} N={m.group(2)}")
        groups[(m.group(1), int(m.group(2)))] = (int(m.group(3)), fails)
    redo = notes.get("redo", {});known_open = {(x[0], int(x[1]), x[2]): (x[3] if len(x) > 3 else "") for x in notes.get("open", [])}
    override = {(x[0], int(x[1]), x[2]): (x[3], x[4] if len(x) > 4 else "") for x in notes.get("override", [])}
    rows, ambiguous = [], []
    for (label, n), members in planned.items():
        ok, fails = groups.get((label, n), (0, []))
        left = list(members)
        for test, frac in fails:                       # which seed failed is not in the text: taken in seed order
            hit = next(((t, s) for t, s in left if t == test), None)
            if hit is None:raise ValueError(f"{label} N={n}: failed test {test} is not in the plan")
            left.remove(hit)
            solver = label in GPU_METHODS or test not in redo.get(label, [])
            rows.append(dict(test=test, n=n, seed=hit[1], label=label, state="stopped" if solver else "redo", frac=frac))
        unfinished = len(left) - ok
        if unfinished < 0:raise ValueError(f"{label} N={n}: more complete parts ({ok}) than planned ({len(left)})")
        named = [(t, s) for t, s in left if (label, n, t) in known_open]
        if unfinished == 0:state_of = lambda t, s: "done"
        elif len(named) == unfinished:state_of = lambda t, s, named=named: "open" if (t, s) in named else "done"
        else:
            state_of = lambda t, s: "unknown"
            ambiguous.append(dict(label=label, n=n, planned=len(members), complete=ok, unfinished=unfinished, failed=len(fails)))
        for t, s in left:
            row = dict(test=t, n=n, seed=s, label=label, state=state_of(t, s), frac=None)
            if row["state"] == "open" and known_open.get((label, n, t)):row["note"] = known_open[(label, n, t)]
            if (label, n, t) in override and row["state"] in ("done", "unknown"):row["state"], row["note"] = override[(label, n, t)]
            rows.append(row)
    fam = {}
    for m in re.finditer(r"^(GPU driver|AthenaK|RAMSES/Athena\+\+|All)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.M):
        fam[m.group(1)] = dict(zip(("complete", "running", "failed", "not started", "total"), map(int, m.groups()[1:])))
    return dict(source="summary", time=notes.get("time", ""), rows=rows, ambiguous=ambiguous, families=fam,
                lanes=dict(gpu=lane_line(text, "GPU"), cpu=lane_line(text, "CPU")), timing=notes.get("timing", []),
                stuck=notes.get("stuck", []), remarks=notes.get("remarks", []), skipped=[])


def part_state(label, part, cap):
    """State of a recorded repeat-1 part from the export (status, error, steps), see the module docstring."""
    if part.get("status") == "complete":return "done"
    if part.get("status") != "failed":return "open"
    err = part.get("error") or ""
    if label in GPU_METHODS:return "stopped" if err.startswith(SOLVER_STOPS) else "redo"
    # External codes: the code ended normally (exit code 0) without reaching every output: it ran into the step
    # cap, a stall. Athena++ could also end normally and fail the old time match, with far fewer steps than the cap.
    if err.startswith("exit code 0;"):
        return "redo" if label.startswith("athenapp") and cap and (part.get("steps") or 0) < cap else "stopped"
    return "redo"


def from_export(data, runs, caps):
    """Status from the Sherlock export. runs: the plan's runs; caps: {test: max_steps_per_n}."""
    rows, timing, stuck = [], [], [];by_key = {r["key"]: r for r in data["runs"]};skipped = set(data.get("skipped", []))
    waiting = set(data.get("waiting", []))
    for r in runs:
        rec = by_key.get(r["key"]);labels = r["extra"].split(",") if r["code"] == "gpu" else [label_of(r)]
        unit = (data.get("units") or {}).get((rec or {}).get("unit"), {})
        parts = {p["part"]: p for p in (rec or {}).get("parts", [])}
        for label in labels:
            p = parts.get(f"{label}_r1");row = dict(test=r["test"], n=r["n"], seed=r["seed"], label=label, frac=None)
            if r["key"] in skipped:row["state"] = "skipped"
            elif p is None:
                row["state"] = "open"
                if unit.get("live") == "RUNNING":row["note"] = "its job is running"
                elif unit.get("live") == "PENDING":row["note"] = "its job is waiting in the queue"
                elif (rec or {}).get("unit") in waiting:row["note"] = "waiting to be submitted again"
                elif unit.get("state") in (None, "NOT SUBMITTED"):row["note"] = "not submitted yet"
                else:
                    row["note"] = f'its job ended ({unit.get("state", "?").lower()}) without finishing it'
                    if r["dir"] not in stuck:stuck.append(r["dir"])
            else:
                row["state"] = part_state(label, p, caps.get(r["test"], 0)*r["n"])
                if row["state"] in ("stopped", "redo"):
                    t_end = p.get("t_end") or 0;row["frac"] = min(1.0, (p.get("reached") or 0)/t_end) if t_end else None
                    row["note"] = (p.get("error") or "").splitlines()[0][:120] if p.get("error") else ""
                later = [q for name, q in parts.items() if name.startswith(label + "_r") and name != f"{label}_r1"]
                if row["state"] == "done" and any(q.get("status") == "failed" for q in later):
                    timing.append(f'{r["dir"]}: a timing repeat of {label} failed')
            rows.append(row)
    lanes = {k: lane_line("\n".join(data.get("lanes", [])), name) for k, name in (("gpu", "GPU"), ("cpu", "CPU"))}
    return dict(source="export", time=data.get("time", ""), rows=rows, ambiguous=[], families={}, lanes=lanes, timing=timing,
                stuck=stuck, remarks=data.get("remarks", []), skipped=sorted(skipped))


def mark_nonfinite(status, nonfinite, t_end):
    """Runs recorded as complete whose outputs stop being finite are solver stops, not results to the end.

    nonfinite: {(test, label, n, seed): time of the last finite output}, from the fetched diagnostics (scripts/sweep_data.py);
    t_end: {test: end time}. The record on Sherlock cannot say this: the code kept stepping on NaN and exited normally."""
    hit = []
    for row in status["rows"]:
        t = nonfinite.get((row["test"], row["label"], row["n"], row["seed"]))
        if t is None or row["state"] != "done":continue
        row["state"] = "stopped";row["frac"] = min(1.0, t/t_end[row["test"]]) if t_end.get(row["test"]) else None
        row["note"] = f"state not finite after t = {t:g}; the code ran on to the end and the run is recorded as complete"
        hit.append(row)
    if hit:
        nice = lambda l: l.replace("athenapp", "Athena++").replace("athenak", "AthenaK").replace("ramses", "RAMSES").replace("_", " ")
        names = sorted({(nice(r["label"]), r["test"]) for r in hit})
        status.setdefault("remarks", []).append(
            f"{len(hit)} runs recorded as complete are counted as stopped: their state is not finite after an early output "
            f"({'; '.join(f'{l} on {t}' for l, t in names)}). Found in the fetched diagnostics, not in the record on Sherlock.")
    return status


def load(site, runs, caps):
    """The newest status under site/sweep_status, or None."""
    d = Path(site)/"sweep_status";best = None
    for f in sorted(x for x in d.glob("export_*.json") if not x.name.endswith(".since.json")) if d.is_dir() else []:
        data = json.loads(f.read_text());since = f.with_name(f.stem + ".since.json")
        if since.is_file():
            extra = json.loads(since.read_text())
            if extra.get("base") == data.get("time"):
                data["skipped"] = sorted(set(data.get("skipped", [])) | set(extra.get("skipped", [])));data["remarks"] = extra.get("remarks", [])
        if best is None or data.get("time", "") >= best[0]:best = (data.get("time", ""), "export", f, data)
    for f in sorted(d.glob("summary_*.txt")) if d.is_dir() else []:
        note = f.with_suffix(".notes.json");notes = json.loads(note.read_text()) if note.is_file() else {}
        if best is None or notes.get("time", "") > best[0]:best = (notes.get("time", ""), "summary", f, notes)
    if best is None:return None
    _, kind, f, payload = best
    st = from_export(payload, runs, caps) if kind == "export" else from_summary(f.read_text(), planned_parts(runs), payload)
    st["file"] = f.name
    return st
