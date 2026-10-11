"""Build the "Sweep" sheet of the progress site (progress/sweep-plan.html): which runs exist, and the plan and its cost.

The plan comes from the sweep bundle itself (sherlock_sweep/codes/plan.py: which runs what; codes/estimate.py: the
calibrated cost model), so the page and the jobs cannot disagree. Which runs exist comes from the newest status under
progress/sweep_status/ (scripts/sweep_status.py: an exact export from Sherlock, or the text of `sweep.py runs`).
Stages 2-4 are not in the bundle yet; their rough costs are computed below from the same rates and the stated physics.
Run: python3 scripts/build_sweep_plan.py
Also adds the page to progress/tests.json and to the navigation of the existing pages (no full site rebuild).
"""
from datetime import datetime
import html
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
BUNDLE = ROOT/"sherlock_sweep"
sys.path.insert(0, str(BUNDLE/"codes"))
sys.path.insert(0, str(ROOT/"scripts"))
import estimate  # noqa: E402
import plan  # noqa: E402
import sweep_status  # noqa: E402

PAGE = dict(id="sweep-plan", page="sweep-plan.html", kind="sweep", number="07", title="Comprehensive sweep",
            nav="Sweep", status="Running", document="../sherlock_sweep/SWEEP_RUN.md",
            description="Every test at every grid with every code.", chips=[])
# One column of the status grid per method (the plan sheet groups the five GLM kernels).
METHODS = [("glm", "GLM", "GLM original: step_tiled!, Float32, LLF"), ("glm_hll16", "GLM f16", "GLM production kernel: step_plm!, f16, HLL"),
           ("glm_hlld", "GLM HLLD", "GLM step_plm!, Float32, HLLD"), ("glm_ppm", "GLM PPM", "GLM step_plm!, f16, PPM, HLL"),
           ("glm_constch", "GLM cₕ", "GLM f16 / HLL with a constant cleaning speed"), ("ct", "CT", "CT prototype"),
           ("athenak_matched", "AthenaK m", "AthenaK matched: PLM / HLLE / RK2"), ("athenak_best", "AthenaK b", "AthenaK best: PPM / HLLD / RK3"),
           ("ramses_matched", "RAMSES m", "RAMSES matched: HLL"), ("ramses_best", "RAMSES b", "RAMSES best: HLLD"),
           ("athenapp_matched", "Athena++ m", "Athena++ matched: PLM / HLLE / VL2"), ("athenapp_best", "Athena++ b", "Athena++ best: PPM / HLLD / RK3"),
           ("athenak_reference", "Reference", "AthenaK reference (truth)"), ("athenapp_reference", "Check", "Athena++ check of the reference")]
STATE_TEXT = dict(done="finished", stopped="stopped by the solver", redo="failed, to rerun", open="not finished",
                  unknown="finished or not: only the count is known", skipped="taken out of the sweep")
COLUMNS = [("gpu", "glm", "GLM ×5", "the five GLM kernels (T. Abel)"), ("gpu", "ct", "CT", "CT prototype (T. Abel)"),
           ("athenak", "matched", "AthenaK m", "PLM / HLLE / RK2"), ("athenak", "best", "AthenaK b", "PPM / HLLD / RK3"),
           ("ramses", "matched", "RAMSES m", "HLL"), ("ramses", "best", "RAMSES b", "HLLD"),
           ("athenapp", "matched", "Athena++ m", "PLM / HLLE / VL2"), ("athenapp", "best", "Athena++ b", "PPM / HLLD / RK3"),
           ("athenak", "reference", "Reference", "AthenaK truth"), ("athenapp", "reference", "Check", "Athena++ truth check")]
LANE = {True: "GPU-h", False: "core-h"}
e = html.escape


def fmt(x, digits=0):
    if x >= 100:return f"{x:,.0f}"
    return f"{x:,.{digits}f}" if digits else f"{x:,.0f}"


def load_units():
    """Runs (one output directory each), the Slurm jobs they are packed into, and the configs."""
    spec = importlib.util.spec_from_file_location("sweep_tool", BUNDLE/"tools"/"sweep.py")
    tool = importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    runs = plan.all_runs();jobs = plan.units(None, tool.PILOTS)
    configs = {p.stem: plan.load(p) for p in (BUNDLE/"configs").glob("*.toml") if not p.name.startswith("gate_")}
    return runs, jobs, configs


def column_of(u):
    return f"{u['code']}:{'reference' if u['mode'] == 'reference' else u['variant']}"


def matrix(units, configs):
    """test -> column -> dict(grids, cost, geometry), from the runs."""
    cells = {}
    for u in units:
        if u["code"] == "gpu":
            # Split the GPU-driver runs into GLM kernels and CT for the sheet.
            for col, keep in (("gpu:glm", lambda m: m != "ct"), ("gpu:ct", lambda m: m == "ct")):
                ms = [m for m in u["extra"].split(",") if keep(m)]
                if not ms:continue
                share = sum(estimate.unit_cost(configs[u["test"]], "gpu", "", "run", u["n"], "cube", m)["central"]["resource"]
                            - estimate.DRIVER_FIXED[0] + plan.PER_RUN_GPU for m in ms)
                c = cells.setdefault(u["test"], {}).setdefault(col, dict(grids=[], cost=0.0, geometry=set(), gpu=True))
                c["grids"].append(u["n"]);c["cost"] += share;c["geometry"].add(u["geometry"])
            continue
        c = cells.setdefault(u["test"], {}).setdefault(column_of(u), dict(grids=[], cost=0.0, geometry=set(), gpu=u["gpu"]))
        c["grids"].append(u["n"]);c["cost"] += u["resource"]["central"];c["geometry"].add(u["geometry"])
    return cells


def later_stages(units, configs):
    """Rough costs of the stages that still need code (same calibrated rates; see the notes on the page)."""
    by = lambda pred, lane: sum(u["resource"]["central"] for u in units if pred(u) and u["gpu"] == lane)/3600
    resist = lambda u: u["test"] in ("gem", "current_sheet")
    s2 = dict(gpu=1.1*by(resist, True), cpu=1.1*by(resist, False))
    # Pakmor et al. (2026): driven Mach 0.2, isothermal, to t = 100, grids 32-256 (GPU codes also 512).
    # GLM steps: t * N * sum_d(|v_d| + c_f) / C ~ 100 N (3 x 1.6) / 0.25 = 1920 N; per-direction CFL codes ~533 N.
    estimate.STEPS["pakmor_dynamo"] = dict(glm=1920, ct=1920, athenak_matched=533, athenak_best=533,
                                            ramses_matched=800, ramses_best=533)
    cfg = dict(name="pakmor_dynamo", problem="pakmor_dynamo", frames=101, repeats=1, spectra_every=1, grids=[32, 64, 128, 256, 512])
    gpu3 = cpu3 = 0.0
    for n in cfg["grids"]:
        for m in ("glm", "glm_hll16", "glm_hlld", "glm_ppm", "glm_constch", "ct"):
            gpu3 += estimate.unit_cost(cfg, "gpu", "", "run", n, "cube", m)["central"]["resource"]
        for v in ("matched", "best"):
            gpu3 += estimate.unit_cost(cfg, "athenak", v, "run", n, "cube")["central"]["resource"]
            if n <= 256:
                for code in ("ramses", "athenapp"):
                    cpu3 += estimate.unit_cost(cfg, code, v, "run", n, "cube")["central"]["resource"]
    # Kritsuk start: AthenaK drives to Ms ~ 9 for 3 t_d = 0.15 at 1024^3 on 8 GPUs; the decay then reruns
    # decay_supersonic from that state (Stage 1's decay_supersonic cost again).
    k_steps = 8.56/0.2*0.15*1024;cells = 1024**3;multi = 8*0.19e9*0.8
    kritsuk_start = k_steps*cells/multi*8/3600
    decay = by(lambda u: u["test"] == "decay_supersonic", True), by(lambda u: u["test"] == "decay_supersonic", False)
    s3 = dict(gpu=gpu3/3600 + kritsuk_start + decay[0], cpu=cpu3/3600 + decay[1], pakmor_gpu=gpu3/3600, pakmor_cpu=cpu3/3600,
              kritsuk_start=kritsuk_start)
    # 1024^3 AthenaK PPM/HLLD/RK3 references for the 3D tests (8 GPUs, 80% scaling); decays per seed.
    s4 = 0.0
    for name, c in configs.items():
        if plan.planar(c) or name == "throughput_ot3d":continue
        var = "athenak_matched" if name == "decay_supersonic" else "athenak_best"
        steps = estimate.steps_per_n(name, var)*1024;seeds = len(c.get("seeds", [42]))
        s4 += seeds*steps*cells/multi*8/3600
    return dict(stage2=s2, stage3=s3, stage4=dict(gpu=s4, cpu=0.0))


def project_size():
    try:
        out = subprocess.run(["du", "-sk", str(ROOT)], capture_output=True, text=True, timeout=300).stdout.split()[0]
        return int(out)*1024
    except Exception:
        return None


def bars(rows, unit, tone):
    top = max((v for _, v in rows), default=1) or 1
    out = []
    for label, v in rows:
        out.append(f'<tr><td>{label}</td><td class="bar-cell"><span class="bar {tone}" style="width:{max(0.5, 100*v/top):.1f}%"></span></td>'
                   f'<td>{fmt(v)} {unit}</td></tr>')
    return '<table class="metrics-table bar-table"><tbody>' + "".join(out) + "</tbody></table>"


def when(stamp):
    """2026-10-01T16:45Z -> 2026-10-01 16:45 UTC."""
    return stamp.replace("T", " ").replace("Z", " UTC") if stamp else "unknown time"


def combine(rows):
    """One mark for the seeds of a test at one grid and method: the state to show, how far a stopped run got, a tip."""
    states = [r["state"] for r in rows]
    state = next((x for x in ("redo", "stopped", "open", "unknown", "skipped") if x in states), "done")
    fracs = [r["frac"] for r in rows if r["state"] == "stopped" and r["frac"] is not None]
    seeds = len(rows)
    if state == "done":tip = "finished" + (f" (all {seeds} seeds)" if seeds > 1 else "")
    elif state == "stopped":
        how = ", ".join(f"{f:.0%}" for f in sorted(fracs))
        tip = f"stopped by the solver at {how} of the run" if seeds == 1 else \
            f"{states.count('stopped')} of {seeds} seeds stopped by the solver (at {how} of the run), {states.count('done')} finished"
    elif state == "redo":
        note = next((r.get("note") for r in rows if r["state"] == "redo" and r.get("note")), "")
        tip = "to be rerun: " + (note or "it failed for a reason that is not the solver's")
    elif state == "open":
        note = next((r.get("note") for r in rows if r["state"] == "open" and r.get("note")), "")
        tip = "no result yet" + (f": {note}" if note else "")
    elif state == "unknown":tip = "finished or not: the summary gives only a count for this method and grid"
    else:tip = "taken out of the sweep"
    return state, (min(fracs) if fracs else 0.0), tip


def tally(status):
    """Per method: planned, finished, stopped, redo, unfinished (the counts are exact even where the cells are not)."""
    out = {m: dict(planned=0, finished=0, stopped=0, redo=0, unfinished=0, skipped=0) for m, _, _ in METHODS}
    for r in status["rows"]:
        t = out[r["label"]];t["planned"] += 1
        key = {"done": "finished", "stopped": "stopped", "redo": "redo", "open": "unfinished", "skipped": "skipped"}.get(r["state"])
        if key:t[key] += 1
    for a in status["ambiguous"]:
        out[a["label"]]["finished"] += a["complete"];out[a["label"]]["unfinished"] += a["unfinished"]
    return out


def arepo_card():
    """AREPO ran after the sweep, outside its plan: a card with what its fetched records say (empty when none are here)."""
    import sweep_data
    rows = sweep_data.Sweep().arepo
    if not any(r["state"] != "open" for r in rows.values()):return ""
    parts = []
    for label, name in (("arepo_moving", "moving mesh"), ("arepo_static", "static mesh")):
        mine = [(k, r) for k, r in rows.items() if k[1] == label]
        done = sum(1 for _, r in mine if r["state"] == "done");stopped = sorted((k[0], k[2], r.get("reached") or 0.0) for k, r in mine if r["state"] == "stopped")
        text = f"<strong>{e(name)}</strong>: {done} of {len(mine)} runs reached the end"
        if stopped:text += "; AREPO ended " + ", ".join(f"{e(t.replace('_', ' '))} {n}² at t = {x:.3g}" for t, n, x in stopped) + " itself, its timestep collapsing"
        parts.append(f"<li>{text}.</li>")
    return ('<div class="card"><h3>Run after the sweep: AREPO</h3><p>The public AREPO on the 17 planar tests at every grid, with Powell\'s scheme and the HLLD solver, '
            'on its moving mesh and on a static Cartesian mesh (CPUs; not in the plan or the tables above).</p>'
            f'<ul class="checks plain">{"".join(parts)}</ul><p class="small">Figures: <a href="paper-figures.html#arepo">the AREPO section of the paper figures</a>. '
            'A static-mesh run that reached the end is not thereby right: the field loop is unstable there at every grid.</p></div>')


def status_html(status, configs, families, family_of):
    """The status tiles, grid and tables."""
    by = {}                                              # (test, label) -> n -> rows
    for r in status["rows"]:by.setdefault((r["test"], r["label"]), {}).setdefault(r["n"], []).append(r)
    amb = {(a["label"], a["n"]): a for a in status["ambiguous"]}
    ladder = [32, 64, 96, 128, 192, 256, 384, 512]
    rows = []
    for family in families:
        tests = sorted({t for t, _ in by if family_of.get(t) == family})
        if not tests:continue
        rows.append(f'<tr class="family-row"><th colspan="{len(METHODS) + 2}">{e(family)}</th></tr>')
        for t in tests:
            c = configs[t];p2 = plan.planar(c);tds = [];fin = tot = 0
            for label, _, _ in METHODS:
                cell = by.get((t, label))
                if not cell:
                    tds.append('<td class="not-configured">—</td>');continue
                cube = p2 and not label.startswith(("ramses", "athenapp")) and not label.endswith("reference")
                geo = "²" if p2 and not cube else "³"
                marks, tips = [], []
                grids = sorted(cell) if label.endswith("reference") else [n for n in ladder if n in c["grids"] or n in cell]
                for n in grids:
                    if n not in cell:
                        marks.append('<i class="g none"></i>');continue
                    state, frac, tip = combine(cell[n])
                    if state == "unknown":
                        a = amb[(label, n)];tip += f" ({a['complete']} of {a['complete'] + a['unfinished']} such runs are finished)"
                    tot += sum(x["state"] != "skipped" for x in cell[n]);fin += sum(x["state"] == "done" for x in cell[n])
                    full = f"{n}{geo}: {tip}"
                    marks.append(f'<i class="g s-{state}"' + (f' style="--f:{frac:.2f}"' if state == "stopped" else "") + f' title="{e(full)}"></i>')
                    tips.append(full)
                if label.endswith("reference"):
                    tds.append(f'<td><span class="ladder" title="{e(" · ".join(tips))}"><span class="grid-tag">{grids[0]}{geo}</span>{"".join(marks)}</span></td>')
                else:
                    tds.append(f'<td><span class="ladder" title="{e(" · ".join(tips))}">{"".join(marks)}</span></td>')
            unsure = any(x["state"] == "unknown" for (tt, _), cell in by.items() if tt == t for xs in cell.values() for x in xs)
            rows.append(f'<tr><td><code>{e(t)}</code> <span class="divb-flag">{"2D" if p2 else "3D"}</span></td>{"".join(tds)}'
                        f'<td class="num">{"≥ " if unsure else ""}{fin}/{tot}</td></tr>')
    head = "".join(f'<th title="{e(d)}">{e(l)}</th>' for _, l, d in METHODS)
    grid = (f'<div class="matrix-scroll"><table class="suite-matrix sweep-matrix sweep-status"><thead><tr><th>Test</th>{head}'
            f'<th class="num" title="finished of planned, over all methods and grids">Finished</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    # ---- counts by method, with a part-to-whole bar (the numbers beside it are the table view)
    t = tally(status);tot = {k: sum(v[k] for v in t.values()) for k in ("planned", "finished", "stopped", "redo", "unfinished", "skipped")}
    seg = lambda v: "".join(f'<i class="seg-{k}" style="flex:{v[k]}" title="{v[k]} {STATE_TEXT[s]}"></i>'
                            for k, s in (("finished", "done"), ("stopped", "stopped"), ("redo", "redo"), ("unfinished", "open"), ("skipped", "skipped")) if v[k])
    out_col = lambda v: f'<td class="num">{v["skipped"] or "·"}</td>' if tot["skipped"] else ""
    trs = "".join(f'<tr><td title="{e(d)}">{e(l)}</td><td class="num">{v["planned"]}</td><td class="num">{v["finished"]}</td>'
                  f'<td class="num">{v["stopped"] or "·"}</td><td class="num">{v["redo"] or "·"}</td><td class="num">{v["unfinished"] or "·"}</td>{out_col(v)}'
                  f'<td class="bar-cell"><span class="stack">{seg(v)}</span></td></tr>' for (m, l, d), v in ((x, t[x[0]]) for x in METHODS))
    counts = (f'<table class="metrics-table sweep-text sweep-counts"><thead><tr><th>Method</th><th class="num">Planned</th><th class="num">Finished</th>'
              f'<th class="num">Stopped</th><th class="num">To rerun</th><th class="num">Not finished</th>{"<th class=num>Taken out</th>" if tot["skipped"] else ""}<th></th></tr></thead><tbody>{trs}'
              f'<tr class="total"><td><strong>All</strong></td><td class="num">{tot["planned"]:,}</td><td class="num">{tot["finished"]:,}</td><td class="num">{tot["stopped"]}</td>'
              f'<td class="num">{tot["redo"]}</td><td class="num">{tot["unfinished"]}</td>{out_col(tot)}<td class="bar-cell"><span class="stack">{seg(tot)}</span></td></tr></tbody></table>')

    # ---- what stopped, by method and grid
    stops = {}
    for r in status["rows"]:
        if r["state"] == "stopped":stops.setdefault(r["label"], {}).setdefault(r["n"], []).append((r["test"], r["frac"]))
    grids = sorted({n for v in stops.values() for n in v})
    def items(xs):
        acc = {}
        for name, f in sorted(xs):acc.setdefault(name, []).append(f)
        return ", ".join(f'{e(name)} <span class="at">{"/".join("?" if f is None else f"{f:.0%}" for f in fs)}</span>' for name, fs in acc.items())
    srows = "".join(f'<tr><td title="{e(d)}">{e(l)}</td>' + "".join(f'<td>{items(stops[m].get(n, [])) or "·"}</td>' for n in grids) + "</tr>"
                    for m, l, d in METHODS if m in stops)
    stopped = (f'<table class="metrics-table sweep-text sweep-stops"><thead><tr><th>Method</th>' + "".join(f"<th>{n}</th>" for n in grids) +
               f'</tr></thead><tbody>{srows}</tbody></table>') if stops else "<p>No run was stopped by its solver.</p>"

    # ---- what is open
    notes = []
    redo = sorted({(r["label"], r["test"], r["n"]) for r in status["rows"] if r["state"] == "redo"})
    if redo:
        acc = {}
        for label, test, n in redo:acc.setdefault((label, test), []).append(n)
        name = {m: l for m, l, _ in METHODS}
        notes.append("<strong>To rerun</strong> (failed for a reason that is not the solver's): " +
                     "; ".join(f"{e(test)}, {e(name[label])}, at {', '.join(map(str, ns))}" for (label, test), ns in sorted(acc.items())) + ".")
    groups = {}
    for r in status["rows"]:
        if r["state"] == "open":
            g = groups.setdefault(r.get("note") or "no result yet", {});g[(r["test"], r["label"], r["n"])] = g.get((r["test"], r["label"], r["n"]), 0) + 1
    name = {m: l for m, l, _ in METHODS}
    for note, items in sorted(groups.items(), key=lambda x: (sum(x[1].values()), x[0])):
        listed = sorted(items)
        notes.append(f"<strong>Not finished</strong> ({sum(items.values())}; {e(note)}): " +
                     "; ".join(f"{e(t)}, {e(name[l])}, {n}" + (f" ({items[(t, l, n)]} seeds)" if items[(t, l, n)] > 1 else "") for t, l, n in listed[:16]) +
                     (f" and {len(listed) - 16} more" if len(listed) > 16 else "") + ".")
    if status["ambiguous"]:
        name = {m: l for m, l, _ in METHODS}
        notes.append("<strong>Counted but not identified</strong> (hatched cells): " + "; ".join(
            f"{e(name[a['label']])} at {a['n']}: {a['unfinished']} of {a['planned'] - a['failed']} not finished" for a in status["ambiguous"]) +
            ". The summary this page is built from gives these counts but not which runs they are; an export from Sherlock settles it "
            '(<a href="../sherlock_sweep/SWEEP_RUN.md">SWEEP_RUN.md</a>, “Status for the site”).')
    for label, key in (("Timing repeats to rerun", "timing"), ("Left unfinished by their job", "stuck")):
        if status.get(key):notes.append(f"<strong>{label}</strong>: " + "; ".join(e(x) for x in status[key]) + ".")
    gone = sorted({(r["test"], r["label"], r["n"], r["seed"]) for r in status["rows"] if r["state"] == "skipped"})
    if gone:notes.append(f"<strong>Taken out of the sweep</strong> ({len(gone)}): " +
                         "; ".join(f"{e(t)}, {e(name[l])}, {n}, seed {sd}" for t, l, n, sd in gone) + ".")
    notes += [e(x) for x in status.get("remarks", [])]
    return dict(grid=grid, counts=counts, stopped=stopped, notes="".join(f"<li>{x}</li>" for x in notes), tally=t, total=tot)


def build():
    units, jobs, configs = load_units()        # units: the runs (per-run tables); jobs: the Slurm jobs
    gpu = [u for u in units if u["gpu"]];cpu = [u for u in units if not u["gpu"]]
    gjobs = [j for j in jobs if j["gpu"]];cjobs = [j for j in jobs if not j["gpu"]]
    h = lambda us, lv="central": sum(u["resource"][lv] for u in us)/3600
    tot = dict(gpu={lv: h(gjobs, lv) for lv in estimate.LEVELS}, cpu={lv: h(cjobs, lv) for lv in estimate.LEVELS},
               runs=len(units), jobs=len(jobs), gpu_jobs=len(gjobs), cpu_jobs=len(cjobs),
               single=sum(j["runs_count"] == 1 for j in jobs), scratch=sum(u["scratch_bytes"] for u in units),
               returned=sum(u["returned_bytes"] for u in units))
    longest = {lane: max(u["wall_central"] for u in us)/3600 for lane, us in (("gpu", gjobs), ("cpu", cjobs))}
    longest_unit = {lane: max(us, key=lambda u: u["wall_central"])["key"] for lane, us in (("gpu", gjobs), ("cpu", cjobs))}
    stages = later_stages(units, configs)
    cells = matrix(units, configs)
    fam = json.loads((SITE/"gpu_tests.json").read_text())
    families = fam["families"];family_of = {k: v["family"] for k, v in fam["tests"].items()}
    size = project_size()
    status = sweep_status.load(SITE, units, {name: c.get("max_steps_per_n", 0) for name, c in configs.items()})
    if status and (ROOT/"sweep_results").is_dir():
        # With the diagnostics fetched (scripts/sweep_fetch.py), runs whose state turned to NaN while the code ran on are found.
        import sweep_data
        bad = sweep_data.Sweep().nonfinite_parts()
        sweep_status.mark_nonfinite(status, bad, {name: float(c["t_end"]) for name, c in configs.items()})
    sv = status_html(status, configs, families, family_of) if status else None
    data = dict(built=datetime.now().astimezone().isoformat(timespec="minutes"), totals=tot, longest=longest,
                status=dict(time=status["time"], source=status["source"], file=status["file"], by_method=sv["tally"], total=sv["total"],
                            ambiguous=status["ambiguous"], lanes=status["lanes"]) if status else None,
                longest_unit=longest_unit, stages=stages, project_bytes=size,
                jobs=[dict(key=j["key"], family=j["family"], runs=[r["key"] for r in j["runs"]], sbatch=j["sbatch"],
                           wall_central=j["wall_central"], resource=j["resource"]) for j in jobs],
                runs=[{k: u[k] for k in ("key", "dir", "test", "code", "variant", "mode", "n", "seed", "geometry", "extra",
                                        "wall", "resource", "scratch_bytes", "returned_bytes")} for u in units])
    (SITE/"sweep_plan.json").write_text(json.dumps(data, indent=1))

    # ---- matrix rows
    ladder = [32, 64, 96, 128, 192, 256, 384, 512]
    rows = []
    for family in families:
        tests = sorted(t for t in cells if family_of.get(t) == family)
        if not tests:continue
        rows.append(f'<tr class="family-row"><th colspan="{len(COLUMNS) + 3}">{e(family)}</th></tr>')
        for t in tests:
            c = configs[t];p2 = plan.planar(c)
            tds = []
            for code, variant, label, _ in COLUMNS:
                cell = cells[t].get(f"{code}:{variant}" if code != "gpu" else f"gpu:{variant}")
                if not cell:
                    tds.append('<td class="not-configured">—</td>');continue
                gs = sorted(set(cell["grids"]))
                if variant == "reference":
                    geo = "²" if p2 else "³"
                    tds.append(f'<td><span class="grid-tag" title="{fmt(cell["cost"]/3600, 1)} {LANE[cell["gpu"]]}">{gs[0]}{geo}</span></td>')
                    continue
                marks = "".join(f'<i class="g{" on" if n in gs else ""}" title="{n}{"²" if p2 and code in ("ramses", "athenapp") else "³"}"></i>'
                                for n in ladder if n in c["grids"] or n in gs)
                tip = f'{", ".join(map(str, gs))} · {fmt(cell["cost"]/3600, 1)} {LANE[cell["gpu"]]}'
                tds.append(f'<td><span class="ladder" title="{e(tip)}">{marks}</span></td>')
            g = sum(v["cost"] for v in cells[t].values() if v["gpu"])/3600
            k = sum(v["cost"] for v in cells[t].values() if not v["gpu"])/3600
            geo = "2D" if p2 else "3D"
            rows.append(f'<tr><td><code>{e(t)}</code> <span class="divb-flag">{geo}</span></td>'
                        f'<td class="grid-range">{min(c["grids"])}–{max(c["grids"])}</td>{"".join(tds)}'
                        f'<td class="num">{fmt(g, 1)}</td><td class="num">{fmt(k)}</td></tr>')
    head = "".join(f'<th title="{e(d)}">{e(l)}</th>' for _, _, l, d in COLUMNS)
    matrix_html = (f'<div class="matrix-scroll"><table class="suite-matrix sweep-matrix"><thead><tr><th>Test</th><th>Grids</th>{head}'
                   f'<th class="num">GPU-h</th><th class="num">core-h</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    # ---- breakdowns
    def by_code(us, lane):
        acc = {}
        for u in us:
            label = {"gpu": "GLM kernels + CT", "athenak": "AthenaK", "ramses": "RAMSES", "athenapp": "Athena++"}[u["code"]]
            label += {"matched": " matched", "best": " best"}.get(u["variant"], "") if u["mode"] == "run" and u["code"] != "gpu" else ""
            label += " (reference)" if u["mode"] == "reference" else ""
            acc[label] = acc.get(label, 0) + u["resource"]["central"]/3600
        return sorted(acc.items(), key=lambda x: -x[1])

    def by_grid(us):
        acc = {}
        for u in us:
            key = f'{u["n"]}{"²" if u["geometry"] == "plane" else "³"}'
            acc[key] = acc.get(key, 0) + u["resource"]["central"]/3600
        order = sorted(acc, key=lambda k: (int(k[:-1]), k[-1]))
        return [(k, acc[k]) for k in order]

    gb = lambda b: b/2**30
    s2, s3, s4 = stages["stage2"], stages["stage3"], stages["stage4"]
    all_gpu = tot["gpu"]["central"] + s2["gpu"] + s3["gpu"] + s4["gpu"];all_cpu = tot["cpu"]["central"] + s2["cpu"] + s3["cpu"]

    def days(total, conc, longest_h):
        return max(total/conc, longest_h)/24

    scen = [("Busy week", 15, 512), ("Typical", 30, 1536), ("Quiet cluster", 60, 3072)]
    scen_rows = "".join(
        f'<tr><td>{name}</td><td>{g} GPUs · {c:,} cores</td><td class="num">{days(tot["gpu"]["central"], g, longest["gpu"]):.1f} d</td>'
        f'<td class="num">{days(tot["cpu"]["central"], c, longest["cpu"]):.1f} d</td>'
        f'<td class="num">{max(days(tot["gpu"]["central"], g, longest["gpu"]), days(tot["cpu"]["central"], c, longest["cpu"])) + 0.2:.1f} d</td></tr>'
        for name, g, c in scen)
    used = size/2**30 if size else None
    tests_n = len(cells)
    badge, stage1, status_block, plan_open = "Stage 1 ready to submit", "Bundle ready", "", " open"
    if status:
        tt = sv["total"];left = tt["unfinished"] + tt["redo"];pct = tt["finished"]/max(tt["planned"] - tt["skipped"], 1)
        badge = "Stage 1 complete" if left == 0 else f"Stage 1 running · {pct:.0%} finished"
        stage1 = "Complete" if left == 0 else f"Running: {tt['finished']:,} of {tt['planned']:,} finished"
        plan_open = ""
        lanes = status["lanes"];g, c = lanes.get("gpu"), lanes.get("cpu")
        hours = (f'<div class="tile"><span class="control-label">Used so far</span><strong>{g["used"]:,}</strong><span>GPU-hours · {c["used"]/1000:.1f}k core-hours</span>'
                 f'<small>of ~{g["estimate"]:,} and ~{c["estimate"]/1000:.0f}k estimated · jobs submitted: {g["submitted"]}/{g["total"]} GPU, {c["submitted"]}/{c["total"]} CPU</small></div>') if g and c else ""
        exact = status["source"] == "export"
        source = ("an export of every run's records from Sherlock" if exact else "the run summary printed on Sherlock (<code>tools/sweep.py runs</code>)")
        hidden = sum(a["unfinished"] for a in status["ambiguous"])
        status_block = f'''
<section class="sweep-tiles">
<div class="tile"><span class="control-label">Finished</span><strong>{tt["finished"]:,}</strong><span>of {tt["planned"]:,} method runs</span>
<small>{pct:.0%}{f" of the {tt['planned'] - tt['skipped']:,} still planned · {tt['skipped']} taken out" if tt["skipped"] else " · one method on one test, grid and seed"}</small></div>
<div class="tile"><span class="control-label">Stopped by the solver</span><strong>{tt["stopped"]}</strong><span>results, not errors</span>
<small>a nonphysical state, or a stall into the step cap</small></div>
<div class="tile"><span class="control-label">To rerun</span><strong>{tt["redo"]}</strong><span>failed outside the solver</span>
<small>{"none" if not tt["redo"] else "listed under Still open"}</small></div>
<div class="tile"><span class="control-label">Not finished</span><strong>{tt["unfinished"]}</strong><span>running, queued or left by their job</span>
<small>{f"{hidden} of them counted but not identified" if hidden else "each one identified"}</small></div>
{hours}
</section>

<details class="suite-status" open><summary><span class="control-label">Which runs exist · {when(status["time"])}</span><span class="legend">
<i class="g s-done"></i> finished <i class="g s-stopped" style="--f:.5"></i> stopped by the solver, filled to how far it got
<i class="g s-redo"></i> failed, to rerun <i class="g s-open"></i> not finished{' <i class="g s-unknown"></i> finished or not: only the count is known' if status["ambiguous"] else ""}
{' <i class="g s-skipped"></i> taken out' if tt["skipped"] else ""} <i class="g none"></i> not planned</span></summary>
{sv["grid"]}
<p class="sweep-note">One mark per grid (32, 64, 128, 256, 512 a side; the throughput test also 96, 192, 384), one column per method; hover a mark for the detail.
Tests with several seeds show one mark per grid: it is a stop mark if any seed stopped. Status as of {when(status["time"])}, from {source}
(<code>progress/sweep_status/{e(status["file"])}</code>){"" if exact else "; where that summary gives a count without naming the runs, the cells are hatched"}.
A stopped run is a result: the driver ends a run at the first cell with a nonphysical state (the blast tests use the solvers' own floor), and every run has a step cap.</p>
</details>

<div class="sweep-grid">
<div class="card"><h3>By method</h3>{sv["counts"]}
<p class="small">Bars: <i class="g s-done"></i> finished <i class="g s-stopped" style="--f:1"></i> stopped <i class="g s-redo"></i> to rerun <i class="g s-open"></i> not finished{' <i class="g s-skipped"></i> taken out' if tt["skipped"] else ""}.
Only the original GLM kernel and CT run nearly everything; they are also the most diffusive, so finishing is not the same as being accurate.</p></div>
<div class="card"><h3>What the solvers stopped on</h3>{sv["stopped"]}
<p class="small">Test and the fraction of the run it reached, by grid; several seeds are separated by a slash.</p></div>
</div>
<div class="card"><h3>Still open</h3><ul class="checks plain">{sv["notes"] or "<li>Nothing: every planned run has a result.</li>"}</ul></div>
{arepo_card()}

<h2 class="section-title">The plan and its cost</h2>
'''
    page_main = f'''
<div class="page-heading"><div><div class="eyebrow"><span>07 · Comprehensive sweep</span><span class="badge">{badge}</span></div>
<h1>Every test, every grid, every code</h1>
<p>{tests_n} tests from 32 to 512 cells a side, run with the five GLM kernels and the CT prototype of T. Abel, AthenaK, RAMSES and
Athena++, each in a setup matched to the GPU kernels and in its best setup, against exact solutions or AthenaK references
at 4096² and 512³. It reproduces the comparisons of Tomida et al. (2026), Kritsuk et al. (2011), Pakmor et al. (2026),
the GEM challenge and Gardiner &amp; Stone (2005, 2008), then extends each to all codes and grids.</p></div>
<a class="method-link" href="../sherlock_sweep/SWEEP_RUN.md">How to run it <span>↗</span></a></div>
<div class="chips"><span class="chip">{tot["runs"]:,} runs in {tot["jobs"]} Slurm jobs</span><span class="chip">6 GPU-driver methods</span>
<span class="chip">3 external codes × 2 setups</span><span class="chip">32–512 per side</span><span class="chip">Sherlock owners · H100 SXM5</span>
<span class="chip">checkpoint / resume in every code</span></div>
{status_block}
<section class="sweep-tiles">
<div class="tile"><span class="control-label">GPU · stage 1</span><strong>{fmt(tot["gpu"]["central"])}</strong><span>GPU-hours</span>
<small>{fmt(tot["gpu"]["optimistic"])}–{fmt(tot["gpu"]["pessimistic"])} · {tot["gpu_jobs"]} jobs</small></div>
<div class="tile"><span class="control-label">CPU · stage 1</span><strong>{fmt(tot["cpu"]["central"]/1000, 0)}k</strong><span>core-hours</span>
<small>{fmt(tot["cpu"]["optimistic"]/1000)}k–{fmt(tot["cpu"]["pessimistic"]/1000)}k · {tot["cpu_jobs"]} jobs</small></div>
<div class="tile"><span class="control-label">Sherlock scratch</span><strong>{tot["scratch"]/1e12:.1f}</strong><span>TB kept</span>
<small>of the 100 TB quota; purged after 90 days</small></div>
<div class="tile"><span class="control-label">To this Mac</span><strong>{gb(tot["returned"]):.0f}</strong><span>GB, uncompressed</span>
<small>summaries alone ~{gb(tot["runs"]*1.5e6):.0f} GB{f" · project now {used:.0f} GB" if used else ""}</small></div>
<div class="tile"><span class="control-label">Wall clock · stage 1</span><strong>{days(tot["gpu"]["central"], 30, longest["gpu"]) + 0.2:.0f}–{max(days(tot["cpu"]["central"], 1536, longest["cpu"]), days(tot["cpu"]["central"], 512, longest["cpu"])) + 0.2:.0f}</strong><span>days</span>
<small>GPU work under a day; the 512³ CPU runs set the end</small></div>
</section>

<details class="suite-status"{plan_open}><summary><span class="control-label">The planned grid, with hours</span><span class="legend"><i class="g on"></i> runs
<i class="g"></i> not run (octree needs powers of two, or not useful) · hover for grids and hours · <span class="divb-flag">2D</span> planar test:
GLM, CT and AthenaK evolve it as an N³ cube (like-for-like GPU timing), RAMSES and Athena++ as a true N² grid</span></summary>
{matrix_html}
<p class="sweep-note">Jobs: the {tot["runs"]:,} runs (one output directory each: a test, code, setup, grid and seed; the GPU driver's six
methods share one below 256³) are packed, cheapest first, into {tot["jobs"]} Slurm jobs of about 2 h each; {tot["single"]} runs longer than
an hour have a job of their own. Timing repeats: 5 up to 128, 3 at 256, 1 at 512 (the throughput benchmark keeps 5). Outputs: diagnostics at every
output; power spectra (velocity with solenoidal/compressive split, √ρv, ρ<sup>1/3</sup>v, B, ρ) at every output of every 3D run;
full states at selected outputs; 3D z-midplane slices and 64³/128³ block averages for the large grids.</p>
</details>

<div class="sweep-grid">
<div class="card"><h3>GPU-hours by code</h3>{bars(by_code(gpu, True), "GPU-h", "gpu")}</div>
<div class="card"><h3>CPU core-hours by code</h3>{bars(by_code(cpu, False), "core-h", "cpu")}</div>
<div class="card"><h3>GPU-hours by grid</h3>{bars(by_grid(gpu), "GPU-h", "gpu")}
<p class="small">Planar tests cost as much as 3D ones on the GPU codes, which only evolve cubes: the Kelvin–Helmholtz runs at 512 are the
largest GPU items. Thin-slab support in the GLM kernels would remove most of this.</p></div>
<div class="card"><h3>CPU core-hours by grid</h3>{bars(by_grid(cpu), "core-h", "cpu")}
<p class="small">3D RAMSES at 512³ is 80% of the CPU total (20–33 h on 128 cores per run). <code>--cpu-max-n3d 256</code> drops it and
cuts the CPU total to about {fmt(sum(u["resource"]["central"] for u in cpu if plan.planar(configs[u["test"]]) or u["n"] <= 256)/3600/1000)}k core-hours.</p></div>
</div>

<div class="sweep-grid">
<div class="card"><h3>How long it takes</h3>
<table class="metrics-table sweep-text"><thead><tr><th>Cluster</th><th>Concurrent</th><th class="num">GPU work</th><th class="num">CPU work</th><th class="num">Stage 1</th></tr></thead>
<tbody>{scen_rows}</tbody></table>
<p class="small">Days = total hours ÷ concurrency, but never less than the longest single task (GPU: {longest["gpu"]:.1f} h,
<code>{e(longest_unit["gpu"])}</code>; CPU: {longest["cpu"]:.0f} h, <code>{e(longest_unit["cpu"])}</code>), plus ~5 h of gates.
The per-user submit limit (~100 queued jobs) caps concurrency at about 60 GPU and 30 CPU jobs. Owner-node preemption costs
little: every code resumes from its last checkpoint, restart file or output.</p></div>
<div class="card"><h3>Before the sweep: gates</h3>
<ul class="checks"><li>Environment and initial conditions (setup)</li><li>Every config and method at 32³ on the H100 (smoke)</li>
<li>Checkpoint/resume matches an uninterrupted run bit for bit, 2D and 3D, all six methods (resume check)</li>
<li>AthenaK (CUDA), RAMSES 2D and 3D, Athena++ HLLE and HLLD build</li><li>Pilot runs of each external code in 2D and 3D, through post-processing</li></ul>
<p class="small">Tested on this Mac before shipping: RAMSES, AthenaK and Athena++ killed mid-run and resumed reproduce an uninterrupted run
exactly; the GPU-side diagnostics match the original host ones to 1e-9; the coordinator was driven through a simulated Slurm
(waves under the cap, time-outs resubmitted with 1.5× time, a failed gate stops everything).</p></div>
</div>

<div class="card"><h3>Stages</h3>
<table class="metrics-table sweep-text"><thead><tr><th>Stage</th><th>What</th><th>Status</th><th class="num">GPU-h</th><th class="num">core-h</th></tr></thead><tbody>
<tr><td>1 · Standard sweep</td><td>All {tests_n} tests, all grids, all codes; 2D references at 4096², 3D at 512³</td><td>{stage1}</td>
<td class="num">{fmt(tot["gpu"]["central"])}</td><td class="num">{fmt(tot["cpu"]["central"])}</td></tr>
<tr><td>2 · Explicit resistivity</td><td>Resistive GEM (η = 0.005) and current sheet in every code (GLM: split η∇²B; CT: resistive EMF; AthenaK, RAMSES, Athena++ built in)</td>
<td>To build</td><td class="num">{fmt(s2["gpu"])}</td><td class="num">{fmt(s2["cpu"])}</td></tr>
<tr><td>3 · Turbulence driving</td><td>Pakmor et al. dynamo (Mach 0.2 to t = 100, 32³–256³, GPU codes to 512³): {fmt(s3["pakmor_gpu"])} GPU-h, {fmt(s3["pakmor_cpu"])} core-h.
Kritsuk et al. driven start at 1024³ ({fmt(s3["kritsuk_start"])} GPU-h), then the Mach 9 decay from it</td><td>To build</td><td class="num">{fmt(s3["gpu"])}</td><td class="num">{fmt(s3["cpu"])}</td></tr>
<tr><td>4 · 1024³ references</td><td>AthenaK on 8 GPUs for every 3D test, so the 512³ runs have a truth</td><td>To build</td><td class="num">{fmt(s4["gpu"])}</td><td class="num">—</td></tr>
<tr><td><strong>All stages</strong></td><td></td><td></td><td class="num"><strong>{fmt(all_gpu)}</strong></td><td class="num"><strong>{fmt(all_cpu)}</strong></td></tr>
</tbody></table>
<p class="small">Stages 2–4 use the same measured rates; their step counts come from the stated physics (Pakmor: 1,920·N GLM steps at
Mach 0.2 to t = 100) and are rougher than stage 1's, which are measured.</p></div>

<div class="card"><h3>Papers reproduced, and what the sweep adds</h3>
<table class="metrics-table sweep-text"><thead><tr><th>Paper</th><th>Their runs</th><th>Here</th></tr></thead><tbody>
<tr><td>Tomida et al. 2026 (GLM leakage, Δt jump)</td><td>KH at h = 1/128, 1/256; their code</td><td>KH uniform and localized, Δt jump, 64²–512², six GPU methods and six external setups</td></tr>
<tr><td>Kritsuk et al. 2011 (Mach 9 decay)</td><td>256³, 512³ (1024³ for two codes); 9 codes</td><td>64³–512³, all codes; spectra every output; driven 1024³ start in stage 3</td></tr>
<tr><td>Pakmor et al. 2026 (small-scale dynamo)</td><td>32³–256³, identical driving</td><td>Stage 3: the same driving table in every code, GPU codes to 512³</td></tr>
<tr><td>Birn et al. 2001 (GEM)</td><td>MHD, Hall, PIC codes</td><td>Ideal MHD 64²–512² with a 4096² reference; resistive in stage 2</td></tr>
<tr><td>Gardiner &amp; Stone 2005, 2008</td><td>Field loops and current sheet, 128²–256² (and 3D)</td><td>32–512, 2D and 3D loops, all codes</td></tr>
</tbody></table></div>

<div class="card"><h3>Assumptions</h3><ul class="checks">
<li>Step counts per grid are Sherlock medians from the completed campaign; throughputs are measured H100 medians by grid.</li>
<li>CPU rates: RAMSES 3D measured on Sherlock (0.08–0.09 M cell-steps/s per core); RAMSES 2D and Athena++ timed on this Mac and scaled
by RAMSES's Sherlock/Mac ratio. The Athena++ figures are the least certain (the range is 0.5–1.5×).</li>
<li>AthenaK PPM/HLLD/RK3 stalls at Mach 9 (Δt collapse); it runs only at 64³ and 128³ there. Every run has a step cap of 4× its expected steps.
The sweep found that RAMSES matched (HLL) stalls there too: at 64³ and 128³ it reaches 10–20% of the run before the cap, so its 256³ and 512³
runs would cost far more than planned to record the same stall.</li>
<li>Returned data counts files up to 50 MB (larger 3D states stay on scratch, listed per run); each run also has a small summary archive.</li>
</ul></div>
'''
    return page_main, data


def nav_html(tests, current):
    return "".join(f'<a href="{e(t["page"])}"' + (' aria-current="page"' if t["id"] == current else "") +
                   f'><span>{t["number"]}</span>{e(t["nav"])}</a>' for t in tests)


def render(main_html, tests):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="The comprehensive GLM/CT/AthenaK/RAMSES/Athena++ sweep: grid, cost and schedule.">
<title>Sweep · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="sweep-plan"><a class="skip-link" href="#main">Skip to the sweep</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> SWEEP · SHERLOCK</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav_html(tests, "sweep-plan")}</nav>
<main id="main">{main_html}</main>
<footer><span>The plan is computed from sherlock_sweep/codes (plan.py, estimate.py); which runs exist, from progress/sweep_status/ (scripts/sweep_status.py).</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main():
    tests = json.loads((SITE/"tests.json").read_text())
    main_html, data = build()
    page = dict(PAGE, status=("Complete" if data["status"]["total"]["unfinished"] + data["status"]["total"]["redo"] == 0 else "Running")
                if data["status"] else "Ready to submit")
    if not any(t["id"] == PAGE["id"] for t in tests):tests.append(page)
    tests = [page if t["id"] == PAGE["id"] else t for t in tests]
    (SITE/"tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=2) + "\n")
    (SITE/PAGE["page"]).write_text(render(main_html, tests))
    # Add the page to the navigation of the pages already built (build_progress_site.py does the same on a rebuild).
    link = f'<a href="{PAGE["page"]}"><span>{PAGE["number"]}</span>{PAGE["nav"]}</a>'
    for t in tests:
        path = SITE/t["page"]
        if t["id"] == PAGE["id"] or not path.is_file():continue
        text = path.read_text()
        if PAGE["page"] not in text and '</nav>' in text:text = text.replace("</nav>", link + "</nav>", 1)
        # the link's label follows PAGE["nav"]
        text = re.sub(rf'<a href="{re.escape(PAGE["page"])}"><span>{PAGE["number"]}</span>[^<]*</a>', link, text)
        path.write_text(text)
    tt = data["totals"]
    print(f"Sweep plan: {tt['runs']} runs in {tt['jobs']} jobs; GPU {tt['gpu']['central']:.0f} GPU-h, CPU {tt['cpu']['central']:.0f} core-h;"
          f" page {SITE/PAGE['page']}")


if __name__ == "__main__":
    main()
