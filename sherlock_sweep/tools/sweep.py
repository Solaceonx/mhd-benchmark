#!/usr/bin/env python3
"""Run the comprehensive sweep on Sherlock: gates first, then every unit of codes/plan.py, within the QOS cap.

  python3 tools/sweep.py start [options]   submit the gates and the coordinator job (run through tools/start_sweep.sh)
  python3 tools/sweep.py tick              one coordinator step (jobs/20_sweep_coordinator.sbatch runs it every 5 min)
  python3 tools/sweep.py status            jobs: progress, hours used, what is queued
  python3 tools/sweep.py runs [--json F]   runs: complete / running / failed / not started, failures grouped by error;
                                           --json writes every run and part with its recorded status (for the progress site)
  python3 tools/sweep.py plan [--json F]   the units that would run with the saved (or default) options, and their cost
  python3 tools/sweep.py regate            after a fix: resubmit failed/cancelled gates (keeps completed ones)
  python3 tools/sweep.py retry [KEY...] [--lane cpu|gpu] [--clean-error TEXT]
                                           hand jobs back to the coordinator (the named ones; the FAILED/CANCELLED ones
                                           of a lane); --clean-error instead deletes the runs that have a part which
                                           failed with TEXT, in any job, so that they start over (a failed run is otherwise kept)
  python3 tools/sweep.py audit             check every run's records and files and every job that ran twice; lists the
                                           runs to redo. Changes nothing (writes sweep_state/audit.json).
  python3 tools/sweep.py redo [RUN...] [--audit]     delete these run directories (--audit: the ones audit listed) and
                                           hand their jobs back
  python3 tools/sweep.py repack [RUN...] [--audit]   pack runs again whose archives are missing from returns/
  python3 tools/sweep.py dedupe [--dry-run]          cancel duplicate jobs: two jobs of one unit in the queue at once
  python3 tools/sweep.py skip [KEY...] [--undo]      take long runs out of the sweep (their jobs are cancelled and never
                                           submitted again); --undo puts them back; without keys, lists them

A unit is one Slurm job running a list of runs (codes/plan.py packs small runs into jobs of about 2 h; long runs
get a job of their own). Its run list is written to sweep_state/tasks/ when it is submitted. Sherlock counts every
queued job against a per-user submit limit (QOSMaxSubmitJobPerUserLimit, ~100), so units are submitted in waves:
cheapest first, GPU and CPU work in separate lanes.

One job per unit at a time: nothing is submitted for a unit that still has a job in the queue. Every job is
submitted with --requeue, so Slurm itself requeues a preempted job (same job ID); the coordinator leaves it alone.
A job that has left the queue in TIMEOUT or OUT_OF_MEMORY (or PREEMPTED, NODE_FAIL, BOOT_FAIL without having been
requeued) is resubmitted, up to --max-attempts, with 1.5x time or 2x memory; it resumes in the same run
directories. FAILED jobs are not retried: they are results (a solver that stops) or bugs to look at. When every job
has ended, a job that left runs unfinished is submitted once more.
State lives in sweep_state/ (settings, gate and submission ledgers, requeue.jsonl, status.txt).
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE/"codes"))
import plan  # noqa: E402

STATE = BUNDLE/"sweep_state"
# States after which a job is resubmitted, but only once it has left the queue: every job is submitted with
# --requeue, so Slurm itself requeues a preempted (or node-failed) job under the same ID, and it stays in the queue.
# Resubmitting such a job as well would run the same run directories twice at once.
RETRY = ("TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "OUT_OF_MEMORY")
REQUEUED_BY_SLURM = ("NODE_FAIL", "PREEMPTED", "BOOT_FAIL")
SETTLE_SECONDS = 600        # a job in one of those states is resubmitted only this long after it ended, if not requeued
# Queue states of a job that is not executing (waiting to start, or requeued and waiting).
WAITING = ("PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SPECIAL_EXIT", "CONFIGURING")
ACTIVE = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING", "REQUEUE_HOLD",
          "REQUEUE_FED", "SIGNALING", "STAGE_OUT", "STOPPED")
DEFAULTS = dict(gpu_partition="gpu,owners", cpu_partition="normal,owners", gpu_feature="GPU_SKU:H100_SXM5",
                cap=90, cpu_cap=30, max_attempts=3, skip_cpu=False, cpu_max_n3d=512, only=[])
# Pilot units run behind the builds and the smoke test, one small job per family; their runs are real sweep runs,
# left out of the packs. They exercise a packed GPU-driver job, AthenaK in 2D (as a cube) and 3D, RAMSES NDIM=2 and 3,
# and both Athena++ builds (HLLE matched, HLLD best), each through post-processing and packing.
PILOTS = {"gpu": ["current_sheet|gpu|run|N32|s42", "orszag_tang_3d|gpu|run|N32|s42"],
          "athenak": ["current_sheet|athenak:matched|run|N64|s42", "orszag_tang_3d|athenak:best|run|N32|s42"],
          "cpu": ["current_sheet|ramses:matched|run|N64|s42", "orszag_tang_3d|ramses:best|run|N32|s42",
                  "current_sheet|athenapp:matched|run|N64|s42", "orszag_tang_3d|athenapp:best|run|N32|s42"]}
PILOT_BUILD = {"gpu": None, "athenak": "build-gpu", "cpu": "build-cpu"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def save_json(path, data):
    tmp = path.with_suffix(".tmp");tmp.write_text(json.dumps(data, indent=1));tmp.replace(path)


# The environment every job is submitted with (module versions, code and result directories), whichever shell runs
# this tool: jobs use --export=ALL, so a submission from a bare login shell would otherwise fall back to defaults.
ENV_KEYS = ("JULIA_MODULE", "MHD_PYTHON_MODULES", "MHD_GPU_BUILD_MODULES", "MHD_GPU_RUN_MODULES", "MHD_CPU_BUILD_MODULES",
            "MHD_CODES", "MHD_RESULTS", "MHD_PACK_MAX_FILE_MB", "JULIA_BIN", "MHD_JULIA_DEPOT")


def default_env():
    """Mirrors tools/sweep_env.sh."""
    scratch = os.environ.get("SCRATCH", "")
    return dict(JULIA_MODULE="julia/1.12.6", MHD_PYTHON_MODULES="python/3.12.1 py-numpy/2.2.6_py312",
                MHD_GPU_BUILD_MODULES="devel cmake/3.31.4 cuda/12.4.0 gcc/12.4.0",
                MHD_GPU_RUN_MODULES="devel cuda/12.4.0 gcc/12.4.0", MHD_CPU_BUILD_MODULES="gcc/12.4.0 openmpi/4.1.6",
                MHD_CODES=f"{scratch}/mhd-codes-sweep", MHD_RESULTS=f"{scratch}/mhd-results-sweep", MHD_PACK_MAX_FILE_MB="50")


def job_env(settings):
    return {**default_env(), **(settings or {}).get("env", {})}


def run(cmd, check=True, env=None):
    p = subprocess.run(cmd, cwd=BUNDLE, text=True, capture_output=True, env=env)
    if check and p.returncode:raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}): {p.stdout}{p.stderr}")
    return p


def selected_units(settings, skipped=None):
    """The sweep's jobs for these settings (filters apply to runs before packing, so this is deterministic). Runs
    taken out with `skip` (sweep_state/skip.json) are removed after packing, so every other job keeps its key and its
    run list; `skipped`, a list, receives them."""
    out = plan.units(settings, PILOTS)
    skip = set(load_json(STATE/"skip.json", []))
    if skip:
        kept = []
        for u in out:
            runs = [r for r in u["runs"] if r["key"] not in skip]
            if skipped is not None:skipped += [r for r in u["runs"] if r["key"] in skip]
            if runs:kept.append(u if len(runs) == len(u["runs"]) else dict(u, runs=runs, runs_count=len(runs)))
        out = kept
    # Cheapest first within each lane, so every test reports early and the large grids come last.
    out.sort(key=lambda u: (not u["gpu"], u["wall_central"], u["key"]))
    return out


# ------------------------------------------------------------------------------ Slurm
def submit_environment(settings):
    """The environment sbatch runs in: this process's minus the Slurm allocation it may run inside, plus the sweep's
    job environment. The coordinator is itself a job: with --export=ALL its SLURM_* variables would reach the jobs it
    submits, and srun refuses a CPU job whose inherited SLURM_MEM_PER_NODE meets its own SLURM_MEM_PER_CPU."""
    env = {k: v for k, v in os.environ.items()
           if not ((k.startswith("SLURM_") and k != "SLURM_CONF") or k.startswith(("SBATCH_", "SLURMD_")))}
    return {**env, **job_env(settings)}


def sbatch(options, script, export, settings):
    cmd = ["sbatch", "--parsable", f"--chdir={BUNDLE}", *options, f"--export={export}", script]
    out = run(cmd, env=submit_environment(settings)).stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", out):raise RuntimeError(f"unexpected sbatch output {out!r}")
    return out


def lane_options(gpu, settings):
    if gpu:return [f"--partition={settings['gpu_partition']}", f"--constraint={settings['gpu_feature']}", "--requeue"]
    return [f"--partition={settings['cpu_partition']}", "--requeue"]


def task_file(u):
    """Write (or rewrite, identically) the unit's run list; the job reads it."""
    d = STATE/"tasks";d.mkdir(parents=True, exist_ok=True)
    path = d/f"{plan.safe(u['key'])}.runs"
    path.write_text("\n".join(plan.task_lines(u)) + "\n")
    return path


def submit_unit(u, indices, settings, adjust=(), extra=()):
    script = "jobs/test.sbatch" if u["family"] == "gpu" else "jobs/external.sbatch"
    env = f"ALL,MHD_TASK_FILE={task_file(u)}"
    opts = list(u["sbatch"])
    for kind in adjust:
        opts = scale_option(opts, kind)
    name = "sweep-" + plan.safe(u["key"].replace("|", "-"))[:60]
    # A one-element array, so logs and accounting look the same for every job (logs/NAME-JOB_0.out).
    return sbatch([f"--job-name={name}", f"--array={','.join(map(str, indices))}", *lane_options(u["gpu"], settings),
                   *opts, *extra], script, env, settings), opts


def scale_option(opts, kind):
    """1.5x time (TIMEOUT) or 2x memory (OUT_OF_MEMORY) for a resubmission."""
    out = []
    for o in opts:
        if kind == "time" and o.startswith("--time="):
            h, m, s = map(int, o.split("=")[1].split(":"));sec = min(48*3600, int(1.5*(3600*h + 60*m + s)))
            o = f"--time={sec//3600:02d}:{sec%3600//60:02d}:00"
        if kind == "mem" and re.match(r"--mem(-per-cpu)?=\d+G$", o):
            key, val = o.split("=");o = f"{key}={min(int(val[:-1])*2, 256 if key == '--mem' else 16)}G"
        out.append(o)
    return out


def sacct_records(job_ids):
    """{(job_id, index): [record, ...]}: every accounting record of each array task, oldest first. A job that Slurm
    requeues (after a preemption or a node failure) keeps its ID and gets a new record, so a task can have several:
    the last is its current state. Pending ranges are expanded; index None is a job that is not an array."""
    records = {}
    ids = sorted(set(job_ids))
    for i in range(0, len(ids), 200):
        p = run(["sacct", "-n", "-P", "-X", "--duplicates", "--starttime=now-60days",
                 "--format=JobID,State,ElapsedRaw,NCPUS,Submit,Start,End", "-j", ",".join(ids[i:i + 200])], check=False)
        if p.returncode:raise RuntimeError("sacct failed: " + p.stderr)
        for line in p.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 7:continue
            jid = parts[0]
            rec = dict(state=parts[1].split()[0] if parts[1].strip() else "UNKNOWN", elapsed=int(parts[2] or 0),
                       ncpus=int(parts[3] or 0), submit=parts[4], start=parts[5], end=parts[6])
            m = re.fullmatch(r"(\d+)_(\d+)", jid)
            if m:records.setdefault((m.group(1), int(m.group(2))), []).append(rec);continue
            m = re.fullmatch(r"(\d+)_\[([^\]]+)\]", jid)
            if m:
                for part in m.group(2).split("%")[0].split(","):
                    a, _, b = part.partition("-")
                    for k in range(int(a), int(b or a) + 1):records.setdefault((m.group(1), k), []).append(dict(rec, elapsed=0))
                continue
            if re.fullmatch(r"\d+", jid):records.setdefault((jid, None), []).append(rec)
    stamp = lambda t: t if re.match(r"\d{4}-\d\d-\d\d", t or "") else "9999"     # Unknown/None: not yet, so last
    for recs in records.values():recs.sort(key=lambda r: (stamp(r["submit"]), stamp(r["start"])))
    return records


def latest_states(records):
    """{(job_id, index): (state, elapsed_s, ncpus, elapsed_s over every incarnation)} of each task's current record."""
    return {k: (r[-1]["state"], r[-1]["elapsed"], r[-1]["ncpus"], sum(x["elapsed"] for x in r)) for k, r in records.items()}


def sacct_states(job_ids):
    return latest_states(sacct_records(job_ids))


def clock(text):
    """Epoch seconds of a sacct time (local time, no zone), or None for Unknown/None."""
    try:return datetime.datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):return None


def queue(reasons=None):
    """[(base job id, state)] for every task of this user in the queue (pending, running, completing).
    reasons: a dict that receives {job id: why it waits} (Slurm's reason, e.g. Priority, or "... held")."""
    user = os.environ.get("USER", "")
    # %F: the array's base job id (what sbatch --parsable returned); %A would give each element its own id.
    # -r lists every pending array element separately, so each line is one task against the submit limit.
    p = run(["squeue", "-u", user, "-h", "-r", "-o", "%F %T %r"], check=False)
    if p.returncode:raise RuntimeError("squeue failed: " + p.stderr)
    out = []
    for f in (l.split(None, 2) for l in p.stdout.splitlines()):
        if not f:continue
        out.append((f[0], f[1] if len(f) > 1 else "UNKNOWN"))
        if reasons is not None and len(f) > 2:reasons[f[0]] = f[2].strip()
    return out


def held_jobs(entries, q, reasons):
    """Lines for sweep jobs that Slurm holds (e.g. requeued in a held state after a launch failure): they wait until
    released, and their unit waits with them."""
    key = {e["job"]: e["key"] for e in entries}
    return [f"HELD by Slurm: {key[j]} job {j} ({reasons.get(j) or st}). Release it with: scontrol release {j}"
            for j, st in dict(q).items() if j in key and (st == "REQUEUE_HOLD" or "held" in reasons.get(j, "").lower())]


def live_units(entries, q):
    """{unit key: "RUNNING" | "PENDING"} for units with a job in the queue (RUNNING if any of its jobs executes)."""
    state = {}
    for job, st in q:state[job] = "RUNNING" if st not in WAITING or state.get(job) == "RUNNING" else "PENDING"
    live = {}
    for e in entries:
        if e["job"] in state:live[e["key"]] = "RUNNING" if "RUNNING" in (state[e["job"]], live.get(e["key"])) else "PENDING"
    return live


# ------------------------------------------------------------------------------ gates
def gate_specs(units_by_key):
    """(label, script or pilot unit key, gpu, dependencies, extra sbatch options) in submission order."""
    specs = [("setup", "jobs/00_setup.sbatch", False, [], ["--time=01:00:00"]),
             ("build-gpu", "jobs/10_build_gpu_codes.sbatch", True, [], ["--time=02:00:00"]),
             ("build-cpu", "jobs/11_build_cpu_codes.sbatch", False, [], ["--time=02:00:00"]),
             ("smoke", "jobs/01_smoke.sbatch", True, ["setup"], ["--time=03:00:00", "--mem=32G"]),
             ("resume-check", "jobs/05_resume_check.sbatch", True, ["setup"], [])]
    for name in PILOTS:
        key = f"pilot:{name}"
        if key in units_by_key:
            deps = ["setup", "smoke"] + ([PILOT_BUILD[name]] if PILOT_BUILD[name] else [])
            specs.append((key, key, None, deps, []))
    return specs


def cmd_start(args):
    STATE.mkdir(exist_ok=True)
    settings = dict(DEFAULTS)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:settings[k] = v
    if (STATE/"gates.json").exists() and not args.dry_run:
        sys.exit("sweep_state/gates.json exists: the sweep was started already. Use `status`, or restart the "
                 "coordinator with `sbatch jobs/20_sweep_coordinator.sbatch` (see SWEEP_RUN.md).")
    units = selected_units(settings);by_key = {u["key"]: u for u in units}
    report(units, settings)
    if args.dry_run:
        print("\nGates:");[print(f"  {g[0]} (after {', '.join(g[3]) or 'nothing'})") for g in gate_specs(by_key)]
        return
    if not shutil.which("sbatch"):sys.exit("sbatch not found: run on a Sherlock login node")
    # Record the job environment (tools/start_sweep.sh sourced tools/sweep_env.sh) for every later submission.
    settings["env"] = {**default_env(), **{k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)}}
    save_json(STATE/"settings.json", settings)
    gates = {}
    for spec in gate_specs(by_key):
        job = submit_gate(spec, [gates[d] for d in spec[3]], settings, by_key)
        gates[spec[0]] = job;save_json(STATE/"gates.json", gates)
        print(f"gate {spec[0]}: job {job}", flush=True)
    coord = sbatch([], "jobs/20_sweep_coordinator.sbatch", "ALL", settings)
    (STATE/"coordinator.txt").write_text(f"{coord} {now()}\n")
    print(f"coordinator: job {coord}. It submits the sweep once every gate has completed.")


def submit_gate(spec, dep_ids, settings, by_key, attempt=1):
    label, target, gpu, deps, extra = spec
    opts = ([f"--dependency=afterok:{':'.join(dep_ids)}"] if dep_ids else []) + extra
    if target in by_key:                        # pilot unit: also a sweep unit, recorded in the ledger
        u = by_key[target];job, used = submit_unit(u, u["indices"], settings, extra=opts)
        record(dict(key=u["key"], job=job, indices=u["indices"], attempt=attempt, options=used, time=now(), pilot=True, deps=dep_ids))
        return job
    return sbatch([*lane_options(gpu, settings), *opts], target, "ALL", settings)


def gate_states(gates, states):
    """COMPLETED when every task of the gate's job completed; else its failure state, or an active state."""
    out = {}
    for label, job in gates.items():
        s = [v[0] for (j, _), v in states.items() if j == job]
        out[label] = "COMPLETED" if s and all(x == "COMPLETED" for x in s) else \
            next((x for x in s if x not in ACTIVE and x != "COMPLETED"), None) or (s[0] if s else "UNKNOWN")
    return out


def cmd_regate(args):
    """Resubmit the gates that failed or were cancelled (e.g. pilots cancelled behind a failed smoke test), after a fix.
    Completed gates are kept; the new jobs depend only on gates not yet completed. Then restart the coordinator."""
    settings = load_json(STATE/"settings.json", None);gates = load_json(STATE/"gates.json", None)
    if not settings or not gates:sys.exit("No sweep here (sweep_state/settings.json, gates.json); use tools/start_sweep.sh")
    by_key = {u["key"]: u for u in selected_units(settings)}
    state = gate_states(gates, sacct_states(list(gates.values())))
    attempts = {}
    for e in ledger():attempts[e["key"]] = max(attempts.get(e["key"], 0), e["attempt"])
    redo = [spec for spec in gate_specs(by_key) if state.get(spec[0]) not in ACTIVE + ("COMPLETED", "UNKNOWN")]
    if not redo:print("No failed or cancelled gates: " + ", ".join(f"{k}={v}" for k, v in state.items()));return
    new = dict(gates)
    for spec in redo:
        deps = [new[d] for d in spec[3] if d in {r[0] for r in redo} or state.get(d) != "COMPLETED"]
        job = submit_gate(spec, deps, settings, by_key, attempt=attempts.get(spec[1], 0) + 1)
        new[spec[0]] = job;save_json(STATE/"gates.json", new)
        print(f"gate {spec[0]}: was {state[spec[0]]}, resubmitted as job {job}" + (f" (after {', '.join(deps)})" if deps else ""), flush=True)
    print("Now restart the coordinator: bash tools/restart_coordinator.sh")


def record(entry):
    entry.setdefault("ts", time.time())
    with (STATE/"ledger.jsonl").open("a") as f:f.write(json.dumps(entry) + "\n")


def ledger():
    path = STATE/"ledger.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.is_file() else []


def entry_ts(e):
    """When a ledger entry was written (epoch seconds); entries from before patch 7 carry only the UTC text."""
    if "ts" in e:return e["ts"]
    return datetime.datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()


def hand_back(keys):
    """Ask the coordinator to submit these jobs again. Append-only (requeue.jsonl): a request cannot be lost to a
    coordinator step running at the same moment. A job is waiting until a newer ledger entry exists for it."""
    with (STATE/"requeue.jsonl").open("a") as f:
        for k in keys:f.write(json.dumps(dict(key=k, ts=time.time(), time=now())) + "\n")


def handed_back(entries):
    """Jobs waiting to be resubmitted, oldest request first."""
    legacy = STATE/"requeue.json"               # the list kept before patch 7: everything in it is still waiting
    if legacy.is_file():
        hand_back(load_json(legacy, []))
        try:legacy.rename(STATE/"requeue.json.migrated")
        except OSError:pass
    asked = {}
    path = STATE/"requeue.jsonl"
    for line in (path.read_text().splitlines() if path.is_file() else []):
        if line.strip():r = json.loads(line);asked[r["key"]] = r["ts"]
    last = {}
    for e in entries:last[e["key"]] = max(last.get(e["key"], 0.0), entry_ts(e))
    return [k for k in sorted(asked, key=asked.get) if asked[k] > last.get(k, 0.0)]


# ------------------------------------------------------------------------------ coordinator step
def cmd_tick(args):
    STATE.mkdir(exist_ok=True)
    with (STATE/"lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return tick()


def tick():
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));gates = load_json(STATE/"gates.json", {})
    entries = ledger();units = selected_units(settings);by_key = {u["key"]: u for u in units}
    records = sacct_records(list(gates.values()) + [e["job"] for e in entries]);states = latest_states(records)
    lines = [f"Sweep status {now()}"]
    # Gates: every task of every gate job must complete.
    gate_state = gate_states(gates, states)
    failed = {k: v for k, v in gate_state.items() if v not in ACTIVE + ("COMPLETED", "UNKNOWN")}
    lines.append("Gates: " + ", ".join(f"{k}={v}" for k, v in gate_state.items()))
    if failed:
        lines.append("GATE FAILED: " + ", ".join(f"{k} ({v})" for k, v in failed.items()) +
                     ". Nothing more is submitted. Read logs/, fix, then `python3 tools/sweep.py regate` and "
                     "`bash tools/restart_coordinator.sh` (SWEEP_RUN.md).")
        write_status(lines);return 3
    if not gates or any(v != "COMPLETED" for v in gate_state.values()):
        lines.append("Waiting for the gates.");write_status(lines);return 0
    # Latest attempt per (unit, index).
    latest = {}
    for e in entries:
        for i in e["indices"]:latest[(e["key"], i)] = e
    task_state = {}
    for (key, i), e in latest.items():
        task_state[(key, i)] = states.get((e["job"], i), (states.get((e["job"], None), ("UNKNOWN", 0, 0, 0))[0], 0, 0, 0))[0]
    reasons = {};q = queue(reasons);live = live_units(entries, q)   # units with a job in the queue: nothing is submitted for them
    lines += held_jobs(entries, q, reasons)
    ours = {e["job"]: by_key.get(e["key"], {}).get("gpu", True) for e in entries}
    in_queue = len(q);cpu_in_queue = sum(1 for j, _ in q if j in ours and not ours[j])
    # Lanes: CPU work may hold at most cpu_cap slots, and GPU work leaves those slots free while CPU units remain.
    done_keys = {e["key"] for e in entries}
    cpu_left = any(not u["gpu"] and u["key"] not in done_keys for u in units)
    reserve = settings["cpu_cap"] if cpu_left else 0
    submitted_now = 0

    def fits(gpu, n):
        if in_queue + n > settings["cap"]:return False
        if gpu:return in_queue - cpu_in_queue + n <= settings["cap"] - reserve
        return cpu_in_queue + n <= settings["cpu_cap"]

    def submit(u, idx, attempt, after=None, adjust=(), options=None):
        """The only place a sweep job is submitted: never for a unit that has a job in the queue."""
        nonlocal in_queue, cpu_in_queue, submitted_now
        if u["key"] in live:return None
        job, used = submit_unit(dict(u, sbatch=options or u["sbatch"]), idx, settings, adjust=adjust)
        e = dict(key=u["key"], job=job, indices=idx, attempt=attempt, options=used, time=now())
        if after:e["after"] = after
        record(e);live[u["key"]] = "PENDING";done_keys.add(u["key"])
        for i in idx:latest[(u["key"], i)] = e;task_state[(u["key"], i)] = "PENDING"
        in_queue += len(idx);cpu_in_queue += 0 if u["gpu"] else len(idx);submitted_now += len(idx)
        return job
    scale = lambda st: ["time"] if st == "TIMEOUT" else ["mem"] if st == "OUT_OF_MEMORY" else []
    # 0) jobs handed back by `retry`/`redo` (after a fix), within the caps, oldest request first; each waits until its
    #    unit has no job left in the queue
    waiting = [k for k in handed_back(entries) if k in by_key]
    for key in list(waiting):
        u = by_key[key];i0 = u["indices"][0]
        if key in live or not fits(u["gpu"], u["tasks"]):continue
        prev = latest.get((key, i0))
        job = submit(u, u["indices"], (prev["attempt"] if prev else 0) + 1, after="retry",
                     adjust=scale(task_state.get((key, i0))), options=prev.get("options") if prev else None)
        if job:waiting.remove(key);lines.append(f"resubmitted {key} (retry) as job {job}")
    if waiting:
        lines.append(f"Waiting to be resubmitted: {len(waiting)} jobs ({sum(k in live for k in waiting)} still have a job in the queue)")
    # Brake: when most of a lane's finished jobs failed within 15 minutes, something is systematically wrong (a
    # missing module, a bad setting); stop submitting in that lane instead of failing every job of it.
    paused = {}
    for lane in (True, False):
        fin = [k for k, st in task_state.items() if k[0] in by_key and by_key[k[0]]["gpu"] == lane
               and st not in ACTIVE + ("UNKNOWN", "CANCELLED") and k[0] not in waiting]   # handed back: not failures
        fast = [k for k in fin if task_state[k] == "FAILED" and states.get((latest[k]["job"], k[1]), ("", 0, 0, 0))[1] < 900]
        if len(fast) >= 6 and len(fast) >= 0.5*len(fin):
            paused[lane] = True
            lines.append(f"{'GPU' if lane else 'CPU'} LANE PAUSED: {len(fast)} of its {len(fin)} finished jobs failed within "
                         "15 minutes. Nothing new is submitted in this lane. Read a failed job's log (logs/), fix, then "
                         f"`python3 tools/sweep.py retry --lane {'gpu' if lane else 'cpu'}` (SWEEP_RUN.md).")
    # 1) resubmit interrupted jobs that have left the queue (they resume). A preempted job is normally requeued by
    #    Slurm itself and stays in the queue; only one that ended a while ago and is not in the queue is resubmitted.
    retry = {}
    for (key, i), st in task_state.items():
        e = latest[(key, i)]
        if st not in RETRY or key not in by_key or key in live or e["attempt"] >= settings["max_attempts"]:continue
        if st in REQUEUED_BY_SLURM:
            ended = clock(records[(e["job"], i)][-1]["end"]) if (e["job"], i) in records else None
            if ended is not None and time.time() - ended < SETTLE_SECONDS:continue
        retry.setdefault((key, st), []).append(i)
    for (key, st), idx in sorted(retry.items()):
        u = by_key[key]
        if paused.get(u["gpu"]) or not fits(u["gpu"], len(idx)):continue
        prev = latest[(key, idx[0])]
        job = submit(u, idx, prev["attempt"] + 1, after=st, adjust=scale(st), options=prev.get("options"))
        if job:lines.append(f"resubmitted {key} tasks {idx} after {st} as job {job}")
    # 2) new units, in order, per lane
    blocked = {True: paused.get(True, False), False: paused.get(False, False)}
    for u in units:
        if u["key"] in done_keys or blocked[u["gpu"]]:continue
        if not fits(u["gpu"], u["tasks"]):
            blocked[u["gpu"]] = True;continue
        submit(u, u["indices"], 1)
    remaining = [u["key"] for u in units if u["key"] not in done_keys] + waiting
    active = [k for k, st in task_state.items()
              if st in ACTIVE + ("UNKNOWN",) or st in RETRY and latest[k]["attempt"] < settings["max_attempts"]]
    ended = not remaining and not active and not submitted_now
    closing = []
    if ended:
        # 3) every job has ended. A job can end with runs it never finished (it died while writing a result; a run was
        #    in use by another job): such a job is submitted once more, and its runs reported if still unfinished then.
        results = Path(job_env(settings)["MHD_RESULTS"]);tried = {e["key"] for e in entries if e.get("after") == "unfinished runs"}
        for u in units:
            unfinished = [r["dir"] for r in u["runs"] if run_outcome(r, results)[0] in ("running", "not started")]
            e = latest.get((u["key"], u["indices"][0]))
            if not unfinished or not e:continue
            if u["key"] in tried or paused.get(u["gpu"]):
                closing.append(f"UNFINISHED in {u['key']}: {', '.join(unfinished[:4])}{' ...' if len(unfinished) > 4 else ''}")
                continue
            ended = False                                   # resubmitted now, or next step when the queue has room
            if not fits(u["gpu"], u["tasks"]):continue
            job = submit(u, u["indices"], e["attempt"] + 1, after="unfinished runs", options=e.get("options"))
            if job:lines.append(f"resubmitted {u['key']} ({len(unfinished)} unfinished runs) as job {job}")
    lines += summary(units, ledger(), sacct_states([e["job"] for e in ledger()]), in_queue, submitted_now)
    if ended:
        if len(closing) > 20:closing = closing[:20] + [f"... and {len(closing) - 20} more jobs with unfinished runs (python3 tools/sweep.py audit)"]
        write_status(lines + closing + ["ALL UNITS FINISHED. Check the results: python3 tools/sweep.py audit"]);return 10
    write_status(lines)
    return 0


def summary(units, entries, states, in_queue, submitted_now):
    latest = {}
    for e in entries:
        for i in e["indices"]:latest[(e["key"], i)] = e
    counts = {True: {}, False: {}};used = {True: 0.0, False: 0.0}
    gpu_of = {u["key"]: u["gpu"] for u in units};seen = set()
    # A job whose unit is no longer in the plan (skipped): its lane from its own options; its hours still count.
    lane = lambda e: gpu_of[e["key"]] if e["key"] in gpu_of else any(o.startswith("--gpus") for o in e.get("options", []))
    for e in entries:
        for i in e["indices"]:
            if (e["job"], i) in seen:continue             # a job kept by `dedupe` has two entries
            seen.add((e["job"], i))
            total, nc = states.get((e["job"], i), ("UNKNOWN", 0, 0, 0))[3], states.get((e["job"], i), ("UNKNOWN", 0, 0, 0))[2]
            used[lane(e)] += total*(1 if lane(e) else max(nc, 1))/3600
    for (key, i), e in latest.items():
        if key not in gpu_of:continue                     # skipped: not counted
        st = states.get((e["job"], i), ("UNKNOWN", 0, 0, 0))[0];g = gpu_of[key]
        counts[g][st] = counts[g].get(st, 0) + 1
    total = {True: sum(u["tasks"] for u in units if u["gpu"]), False: sum(u["tasks"] for u in units if not u["gpu"])}
    out = []
    for g, lane in ((True, "GPU"), (False, "CPU")):
        c = counts[g];sub = sum(c.values())
        est = sum(u["resource"]["central"] for u in units if u["gpu"] == g)/3600
        out.append(f"{lane}: {sub}/{total[g]} jobs submitted; " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) +
                   f"; {used[g]:.0f} {'GPU-h' if g else 'core-h'} used of ~{est:.0f} estimated")
    out.append(f"Queue: {in_queue} jobs (this step submitted {submitted_now})")
    return out


def write_status(lines):
    text = "\n".join(lines) + "\n";(STATE/"status.txt").write_text(text);print(text, end="", flush=True)


def cmd_status(args):
    if not (STATE/"gates.json").is_file():sys.exit("The sweep has not been started (no sweep_state/gates.json).")
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));units = selected_units(settings);entries = ledger()
    gates = load_json(STATE/"gates.json", {})
    states = sacct_states(list(gates.values()) + [e["job"] for e in entries])
    print("Gates: " + ", ".join(f"{k}={states.get((j, None), [None])[0] or ','.join(sorted({v[0] for (jj, _), v in states.items() if jj == j}))}"
                                for k, j in gates.items()))
    for line in summary(units, entries, states, len(queue()), 0):print(line)
    waiting = handed_back(entries)
    if waiting:print(f"Waiting to be resubmitted: {len(waiting)} jobs: {', '.join(waiting[:6])}{' ...' if len(waiting) > 6 else ''}")
    bad = sorted({(e["key"], i, states[(e["job"], i)][0]) for e in entries for i in e["indices"]
                  if states.get((e["job"], i), ("",))[0] in ("FAILED", "CANCELLED") + RETRY})
    for key, i, st in bad[:40]:print(f"  {st:14s} {key} task {i}")
    if (STATE/"status.txt").is_file():print("\nLast coordinator step:\n" + (STATE/"status.txt").read_text())


def report(units, settings):
    gpu = [u for u in units if u["gpu"]];cpu = [u for u in units if not u["gpu"]]
    h = lambda us, lv: sum(u["resource"][lv] for u in us)/3600
    runs = sum(u["runs_count"] for u in units)
    print(f"{runs} runs packed into {len(units)} Slurm jobs ({len(gpu)} GPU, {len(cpu)} CPU; "
          f"{sum(u['runs_count'] == 1 for u in units)} long runs have a job of their own)")
    print(f"GPU: {h(gpu, 'central'):.0f} GPU-h (optimistic {h(gpu, 'optimistic'):.0f}, pessimistic {h(gpu, 'pessimistic'):.0f})")
    print(f"CPU: {h(cpu, 'central'):.0f} core-h (optimistic {h(cpu, 'optimistic'):.0f}, pessimistic {h(cpu, 'pessimistic'):.0f})")
    print(f"Scratch kept: {sum(u['scratch_bytes'] for u in units)/1e12:.1f} TB; returned to the Mac: "
          f"{sum(u['returned_bytes'] for u in units)/2**30:.0f} GB (uncompressed)")
    print(f"Settings: {json.dumps(settings)}")


ARCHIVES = (".tar.gz", ".tar.gz.sha256", ".summary.tar.gz", ".summary.tar.gz.sha256")


def read_toml(path):
    """A TOML file as a dict; {} when it is missing or cut short."""
    try:return tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):return {}


def remove_run(r, results):
    """Delete one run's directory and its packed archives, so that it starts over."""
    shutil.rmtree(results/r["dir"], ignore_errors=True)
    for suffix in ARCHIVES:
        (BUNDLE/"returns"/(r["dir"] + suffix)).unlink(missing_ok=True)


def wanted_repeats(r):
    return 1 if r["mode"] == "reference" else plan.estimate.repeats(plan.load(BUNDLE/"configs"/f"{r['test']}.toml"), r["n"])


def run_final(r, results):
    """Whether every part of a run has a final record (complete or failed). No job works in such a run: jobs skip
    finished parts."""
    d = results/r["dir"]/"data"
    if r["code"] == "gpu":return read_toml(d/"suite.toml").get("status") in ("complete", "failed")
    done = [t for t in d.glob("*_r*/run.toml") if read_toml(t).get("status") in ("complete", "failed")]
    return len(done) >= wanted_repeats(r)


def unit_view(settings):
    """(units by key, latest ledger entry per unit, its Slurm state, units with a job in the queue)."""
    by_key = {u["key"]: u for u in selected_units(settings)}
    entries = ledger();states = sacct_states([e["job"] for e in entries]);live = live_units(entries, queue())
    latest = {e["key"]: e for e in entries}
    state = {k: states.get((e["job"], e["indices"][0]), ("UNKNOWN", 0, 0, 0))[0] for k, e in latest.items()}
    return by_key, latest, state, live


def clear_runs(picks, live, results):
    """Delete the run directories of `picks` ([(unit key, run)]) so that those runs start over. While a unit has a
    running job, a run of it is deleted only if every part has a final record (the job is past it or will skip it);
    otherwise that job may be working in it, and it is left alone. Returns (removed, left alone, units to hand back,
    units whose waiting job will run the removed runs itself)."""
    removed, left, back, queued = [], [], [], []
    if not results.is_dir():sys.exit(f"results directory {results} not found (MHD_RESULTS of this sweep): nothing deleted")
    for key, r in picks:
        if live.get(key) == "RUNNING" and not run_final(r, results):left.append(r["dir"]);continue
        remove_run(r, results);removed.append(r["dir"])
        target = queued if live.get(key) == "PENDING" else back
        if key not in target:target.append(key)
    return removed, left, back, queued


def report_cleared(removed, left, back, queued):
    short = lambda xs: ", ".join(xs[:8]) + (" ..." if len(xs) > 8 else "")
    hand_back(back)
    if removed:print(f"{len(removed)} run directories removed: {short(removed)}")
    if back:print(f"{len(back)} jobs handed back to the coordinator ({len(handed_back(ledger()))} waiting in all).")
    if queued:print(f"{len(queued)} jobs are waiting in the queue and will run their removed runs when they start.")
    if left:print(f"{len(left)} runs left alone, because their job is running and may be working in them: {short(left)}. "
                  "Repeat this command when that job has ended.")
    if back:print("The coordinator resubmits a handed-back job, within the caps, once its unit has no job left in the queue. "
                  "If the coordinator is not running: bash tools/restart_coordinator.sh")


def cmd_retry(args):
    """Hand jobs back to the coordinator: the jobs named (whatever their state) and, with --lane, the lane's
    FAILED/CANCELLED jobs. With --clean-error TEXT, instead: every run of the jobs named or of the lane with a part
    (a method or a repeat) that failed with TEXT in its error is deleted whole, so that it starts over (a run recorded
    as failed is otherwise kept and skipped), and its job is handed back unless a waiting job will run it anyway."""
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));by_key, latest, state, live = unit_view(settings)
    for k in args.keys:
        if k not in by_key:sys.exit(f"unknown job {k!r} (see `python3 tools/sweep.py status`)")
    lane = [k for k in sorted(latest) if k in by_key and by_key[k]["gpu"] == (args.lane == "gpu") and k not in args.keys] if args.lane else []
    if not args.clean_error:
        keys = list(args.keys) + [k for k in lane if state.get(k) in ("FAILED", "CANCELLED") and not k.startswith("pilot:")]
        if not keys:print("No FAILED or CANCELLED jobs selected.");return
        report_cleared([], [], keys, [])
        return
    results = Path(job_env(settings)["MHD_RESULTS"]);picks = []
    for k in list(args.keys) + lane:
        for r in by_key[k]["runs"]:
            metas = [read_toml(t) for t in (results/r["dir"]).glob("data/*_r*/run.toml")]
            if any(m.get("status") == "failed" and args.clean_error in m.get("error", "") for m in metas):picks.append((k, r))
    if not picks:print(f"No run failed with {args.clean_error!r} in the selected jobs.");return
    report_cleared(*clear_runs(picks, live, results))


def run_outcome(r, results):
    """complete / failed / running / not started, from the run's own records (suite.toml for the GPU driver,
    one run.toml per repeat for the external codes)."""
    d = results/r["dir"]/"data"
    if not d.is_dir():return "not started", ""
    if r["code"] == "gpu":
        st = read_toml(d/"suite.toml").get("status", "running")
        if st == "failed":
            errs = [read_toml(t).get("error", "") for t in d.glob("*_r*/run.toml")]
            return "failed", next((e for e in errs if e), "")
        return ("complete" if st == "complete" else "running"), ""
    metas = [read_toml(t) for t in d.glob("*_r*/run.toml")]
    bad = [m for m in metas if m.get("status") == "failed"]
    if bad:return "failed", bad[0].get("error", "")
    if len(metas) >= wanted_repeats(r) and all(m.get("status") == "complete" for m in metas):return "complete", ""
    return "running", ""


def export_status(settings, units, skipped, results):
    """Every run and every part (method, repeat) with what its record says, each job's state, and the hours used:
    what the progress site shows (scripts/sweep_status.py on the Mac reads it)."""
    entries = ledger();q = queue();live = live_units(entries, q)
    states = sacct_states([e["job"] for e in entries]);latest = {e["key"]: e for e in entries}
    out = dict(time=now(), source="export", results=str(results), lanes=summary(units, entries, states, len(q), 0),
               waiting=handed_back(entries), skipped=[r["key"] for r in skipped], units={}, runs=[])
    for u in units:
        e = latest.get(u["key"])
        out["units"][u["key"]] = dict(job=e["job"] if e else None, attempt=e["attempt"] if e else 0, live=live.get(u["key"]),
                                      state=states.get((e["job"], e["indices"][0]), ("UNKNOWN",))[0] if e else "NOT SUBMITTED")
        for r in u["runs"]:
            parts = []
            for t in sorted((results/r["dir"]/"data").glob("*_r*/run.toml")):
                m = read_toml(t)
                parts.append(dict(part=t.parent.name, status=m.get("status"), reached=m.get("time_reached"), t_end=m.get("t_end_requested"),
                                  steps=m.get("steps"), frames=len(m.get("frames", [])), valid=m.get("timing_valid"),
                                  rate=m.get("million_cell_steps_per_second"), error=(m.get("error") or "")[:300]))
            out["runs"].append(dict(key=r["key"], dir=r["dir"], unit=u["key"], outcome=run_outcome(r, results)[0], parts=parts))
    return out


def cmd_runs(args):
    """Progress by run (1,265 of them), and failures grouped by error."""
    skipped = [];settings = load_json(STATE/"settings.json", dict(DEFAULTS));units = selected_units(settings, skipped)
    results = Path(job_env(settings)["MHD_RESULTS"])
    if args.json:
        data = export_status(settings, units, skipped, results);Path(args.json).write_text(json.dumps(data))
        print(f"{len(data['runs'])} runs with {sum(len(r['parts']) for r in data['runs'])} recorded parts written to {args.json} ({data['time']})")
        return
    counts, errors = {}, {}
    names = {"gpu": "GPU driver", "athenak": "AthenaK", "cpu": "RAMSES/Athena++"}
    for u in units:
        for r in u["runs"]:
            o, e = run_outcome(r, results)
            counts.setdefault(names[r["family"]], {}).setdefault(o, 0);counts[names[r["family"]]][o] += 1
            if o == "failed":
                lines = [l for l in e.splitlines() if l.strip()]
                head = re.sub(r"\[[^\]]*\]?", "[...]", lines[0])[:60] if lines else "?"
                tail = re.sub(r"srun: error: \S+: tasks? [\d,-]+:", "srun: error:", lines[-1])[-80:] if len(lines) > 1 else ""
                tail = "" if "@ Base" in tail or "cpu_second" in tail else tail      # backtrace / timing footers say nothing
                errors.setdefault((names[r["family"]], head + (" ... " + tail if tail else "")), []).append(r["dir"])
    cols = ("complete", "running", "failed", "not started")
    print(f"{'':18s}" + "".join(f"{c:>13s}" for c in cols) + f"{'total':>9s}")
    tot = {c: 0 for c in cols}
    for fam, c in counts.items():
        print(f"{fam:18s}" + "".join(f"{c.get(k, 0):13d}" for k in cols) + f"{sum(c.values()):9d}")
        for k in cols:tot[k] += c.get(k, 0)
    print(f"{'All':18s}" + "".join(f"{tot[k]:13d}" for k in cols) + f"{sum(tot.values()):9d}")
    if skipped:print(f"(plus {len(skipped)} runs taken out with `skip`: {', '.join(r['dir'] for r in skipped[:6])}{' ...' if len(skipped) > 6 else ''})")
    if errors:
        print("\nFailed runs by error:")
        for (fam, err), dirs in sorted(errors.items(), key=lambda x: -len(x[1])):
            print(f"  {len(dirs):4d} x {fam}: {err}\n         e.g. {', '.join(dirs[:2])}")
    # By code/method and grid (repeat 1): finished, failed, and each failure as test(fraction of the run reached).
    ok, bad = {}, {}
    for t in results.glob("*/data/*_r1/run.toml"):
        m = tomllib.loads(t.read_text())
        if m.get("status") == "running" or "method" not in m or t.parts[-4].startswith(("smoke-", "resume-check-")):continue
        k = (m["method"], m["N"])
        if m["status"] == "complete":ok[k] = ok.get(k, 0) + 1
        else:bad.setdefault(k, []).append(f"{m['test']}({m.get('time_reached', 0)/(m.get('t_end_requested') or 1):.0%})")
    print("\nBy method and grid (failures: test and how far it got):")
    for meth in sorted({k[0] for k in list(ok) + list(bad)}):
        for n in sorted({k[1] for k in list(ok) + list(bad) if k[0] == meth}):
            b = sorted(bad.get((meth, n), []))
            print(f"  {meth:18s} N={n:5d}  ok {ok.get((meth, n), 0):3d}  failed {len(b):3d}  {' '.join(b)}")


# ------------------------------------------------------------------------------ audit
# A GPU-driver run stops on these by design (results); anything else that fails is looked at.
SOLVER_STOPS = ("Nonphysical/nonfinite state", "Invalid state at output", "Invalid timestep", "Step limit reached",
                "Out-of-plane variation exceeds tolerance", "Uniform state preservation failed",
                "Initial face divergence too large", "Per-run wall limit reached")
# Failures of the files, not of a solver: what a run shows when another job worked in its directory.
FILE_ERRORS = ("could not be opened", "No such file or directory", "FileNotFoundError", "Corrupt Fortran record",
               "does not cover the uniform level", "Duplicate output at frame", "Stale file handle", "Input/output error",
               "IOError", "SystemError", "ENOENT", "EOFError")


UNMEASURED = "converted but not measured (post-processing did not finish)"


# An external code that was killed from outside (preemption, requeue, cancel, time limit), as codes/external.py tells it
# (KILL_CODES, KILL_TEXT, OOM_TEXT there). Before patch 10 the driver recorded such a run as failed.
KILL_CODES = (129, 130, 137, 143, -1, -2, -9, -15)
KILL_TEXT = re.compile(r"srun: error: \S+: tasks? [\d,\-]+: (Killed|Terminated)|srun: Force Terminated|srun: Job step aborted|"
                       r"CANCELLED AT|DUE TO (PREEMPTION|TIME LIMIT|NODE FAILURE)")
OOM_TEXT = re.compile(r"oom[-_ ]kill|out of memory", re.I)


def failure_kind(error, gpu):
    """files (redo) / solver (a result) / killed (not a result: to be continued) / other, from a failed part's recorded error."""
    text = (error.splitlines() or [""])[0] if gpu else error       # GPU driver: the message, not its backtrace
    if any(x in text for x in FILE_ERRORS):return "files"
    if gpu and text.startswith(SOLVER_STOPS):return "solver"
    m = None if gpu else re.match(r"exit code (-?\d+);", error)
    if m and not OOM_TEXT.search(error) and (int(m.group(1)) in KILL_CODES or (int(m.group(1)) != 0 and KILL_TEXT.search(error))):return "killed"
    return "other"


def csv_column(path, name):
    """One column of a CSV as text, or None when the file is damaged (a short or torn line, no such column)."""
    try:lines = path.read_text(errors="replace").splitlines()
    except OSError:return None
    if not lines:return None
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]
    if name not in header or any(len(r) != len(header) for r in rows):return None
    return [r[header.index(name)] for r in rows]


def check_part(mdir, meta, config, gpu):
    """Problems in the records and files of one finished part of a run (a <method>_r<repeat> directory): outputs
    listed twice or missing, CSV rows that do not match the outputs, files missing or of the wrong size. Empty when
    sound. These are what two jobs working in one directory leave behind."""
    out = [];frames = meta.get("frames", []);idx = [f.get("frame") for f in frames]
    if idx != sorted(set(idx)):out.append("run.toml lists outputs twice or out of order")
    elif config.get("frames"):
        want = list(range(int(config["frames"])))
        if not gpu and meta.get("repeat", 1) != 1:want = want[-1:]      # external codes: timing repeats keep the last output
        if idx != want:out.append(f"run.toml lists {len(idx)} of {len(want)} outputs")
    # Files the manifest names: present, with the size their shape implies (Float32).
    n = int(meta.get("N", 0));full = 4
    for x in meta.get("saved_shape", []):full *= int(x)
    sliced = bool(meta.get("slice"))                                    # midplane-only movie runs: sizes not checked
    for f in frames:
        sizes = {f[k]: None if sliced else full for k in ("file", "final_file", "reference_file") if isinstance(f.get(k), str)}
        if isinstance(f.get("slice"), str):sizes[f["slice"]] = 36*n*n
        for m, name in (f.get("restricted") or {}).items():
            sizes[name] = None if sliced else 36*int(m)**2*(1 if meta.get("planar") else int(m))
        for name, size in sizes.items():
            try:got = (mdir/name).stat().st_size
            except OSError:out.append(f"{name} is missing");continue
            if size is not None and got != size:out.append(f"{name} has {got} bytes, not {size}")
    # diagnostics.csv: one row per measured output, in order.
    if gpu:measured = idx
    else:
        measured = [f.get("frame") for f in frames if "file" in f or "final_file" in f or f.get("measured_then_deleted")]
        if not meta.get("postprocessed"):
            if measured and not sliced:out.append(UNMEASURED)     # finished by running the job again, nothing to delete
            measured = None
    if measured:
        col = csv_column(mdir/"diagnostics.csv", "frame")
        if col is None:out.append("diagnostics.csv is missing or has a damaged line")
        elif col != [str(x) for x in measured]:out.append(f"diagnostics.csv has rows for outputs {compact(col)}, not {compact(measured)}")
    # spectra.csv: the same shells for every output that has spectra, each output once, in order.
    spectra = [f.get("frame") for f in frames if f.get("spectra") is True]
    if spectra:
        col = csv_column(mdir/"spectra.csv", "frame")
        if col is None:out.append("spectra.csv is missing or has a damaged line")
        else:
            groups = []
            for x in col:
                if groups and groups[-1][0] == x:groups[-1][1] += 1
                else:groups.append([x, 1])
            if [g[0] for g in groups] != [str(x) for x in spectra]:
                out.append(f"spectra.csv has outputs {compact([g[0] for g in groups])}, not {compact(spectra)}")
            elif len({g[1] for g in groups}) > 1:out.append("spectra.csv: outputs have different numbers of shells")
    dense = mdir/"dense_diagnostics.csv"
    if gpu and dense.is_file() and dense.stat().st_size:
        col = csv_column(dense, "step")
        try:steps = [int(x) for x in col]
        except (TypeError, ValueError):steps = None
        if steps is None or steps != sorted(set(steps)):out.append("dense_diagnostics.csv has steps twice, out of order or a damaged line")
    return out


def compact(xs, limit=12):
    xs = [str(x) for x in xs]
    return "[" + ",".join(xs if len(xs) <= limit else xs[:limit//2] + ["..."] + xs[-limit//2:]) + f"] ({len(xs)})"


def audit_run(r, results):
    """(outcome, problems of the finished parts, [(part, kind, first line of the error)] of the failed parts)."""
    d = results/r["dir"]/"data"
    if not d.is_dir():return "not started", [], []
    outcome = run_outcome(r, results)[0];config = read_toml(d/"input.toml");problems, failures = [], []
    for t in sorted(d.glob("*_r*/run.toml")):
        meta = read_toml(t);part = t.parent.name;gpu = r["code"] == "gpu"
        if not meta:problems.append(f"{part}: run.toml is cut short");continue
        if meta.get("status") == "complete":
            try:found = check_part(t.parent, meta, config, gpu)
            except Exception as e:found = [f"could not be checked ({e!r})"]
            problems += [f"{part}: {x}" for x in found]
        elif meta.get("status") == "failed":
            err = meta.get("error", "");lines = [l.strip() for l in err.splitlines() if l.strip()] or ["?"];kind = failure_kind(err, gpu)
            line = next((l for l in lines if any(x in l for x in FILE_ERRORS)), lines[0]) if kind == "files" else lines[0] if gpu else lines[-1]
            failures.append((part, kind, line[:110]))
    return outcome, problems, failures


def last_words(r, results):
    """Why an unfinished run stopped, as far as its own files say: its initial data never completed, or the last line
    its driver logged (external codes; the GPU driver logs to the job's file in logs/)."""
    d = results/r["dir"]
    if (d/"data").is_dir() and any((d/"data").iterdir()) and not (d/"data"/"input.toml").is_file():
        return "stopped while writing its initial data (starts over when its job runs again)"
    try:lines = [l.strip() for l in (d/"driver.log").read_text(errors="replace").splitlines() if l.strip()]
    except OSError:return ""
    return lines[-1][:150] if lines else ""


def ran(records, job, index):
    """[(start, end)] in epoch seconds of every incarnation of a job that started; one still running ends now."""
    out = []
    for rec in records.get((job, index), []):
        a = clock(rec["start"])
        if a is None:continue
        b = clock(rec["end"]) or time.time()
        if b > a:out.append((a, b))
    return out


def hhmm(seconds):
    return f"{int(seconds)//3600}:{int(seconds)%3600//60:02d}"


def cmd_audit(args):
    """Check the sweep: jobs of one unit that ran at the same time, duplicate jobs still in the queue, and every run's
    records and files. Lists what to redo; changes nothing (writes sweep_state/audit.json for `redo --audit`)."""
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));units = selected_units(settings);by_key = {u["key"]: u for u in units}
    results = Path(job_env(settings)["MHD_RESULTS"]);entries = ledger()
    reasons = {};records = sacct_records([e["job"] for e in entries]);q = queue(reasons);live = live_units(entries, q)
    waiting = set(handed_back(entries));submitted = {e["key"] for e in entries}
    print(f"Audit {now()}: {sum(u['runs_count'] for u in units)} runs in {len(units)} jobs, results in {results}")
    if not results.is_dir():sys.exit(f"results directory {results} not found (MHD_RESULTS of this sweep)")
    # 1) Jobs. Every submission of a unit, and whether two of them ever ran at the same time.
    jobs_of = {}
    for e in entries:
        if (e["job"], e["indices"][0]) not in jobs_of.setdefault(e["key"], []):jobs_of[e["key"]].append((e["job"], e["indices"][0]))
    requeued = sum(1 for k, recs in records.items() if len(recs) > 1)
    again = [e for e in entries if e.get("after") in REQUEUED_BY_SLURM]
    together = []
    for key, js in sorted(jobs_of.items()):
        for a in range(len(js)):
            for b in range(a + 1, len(js)):
                sec = sum(max(0.0, min(y, w) - max(x, v)) for x, y in ran(records, *js[a]) for v, w in ran(records, *js[b]))
                if sec > 0:together.append((key, js[a][0], js[b][0], sec))
    print(f"\nJobs: {sum(len(v) for v in jobs_of.values())} submitted for {len(jobs_of)} units; Slurm requeued {requeued} of them "
          f"itself (preemption). The coordinator resubmitted {len(again)} after a preemption or node failure.")
    hit = sorted({k for k, *_ in together})
    if together:
        print(f"Two jobs of one unit ran at the same time in {len(hit)} units (their runs are checked below):")
        for key, a, b, sec in together:print(f"  {key}: jobs {a} and {b} together for {hhmm(sec)} h")
    else:print("No two jobs of one unit ever ran at the same time.")
    state = {}
    for job, st in q:state.setdefault(job, []).append(st)
    dup = {k: [j for j, _ in js if j in state] for k, js in jobs_of.items()}
    dup = {k: v for k, v in dup.items() if len(v) > 1}
    if dup:
        print(f"DUPLICATE JOBS IN THE QUEUE for {len(dup)} units: run `python3 tools/sweep.py dedupe` now.")
        for k, v in sorted(dup.items()):print(f"  {k}: {', '.join(f'{j} ({state[j][0]})' for j in v)}")
    # 2) Runs. Records and files of every finished part; failures by kind; runs nobody is working on.
    counts = {};redo, suspect, stalled, repack, notes, unmeasured = [], [], {}, [], [], [];why = {}
    latest = {e["key"]: e for e in entries};states = latest_states(records)
    kinds = {"solver": 0, "files": 0, "killed": 0, "other": 0};done = 0;n_runs = sum(u["runs_count"] for u in units);killed = []
    for line in held_jobs(entries, q, reasons):print(line)
    for u in units:
        for r in u["runs"]:
            done += 1
            if done % 50 == 0 or done == n_runs:print(f"\r  checking runs: {done}/{n_runs}", end="\n" if done == n_runs else "", file=sys.stderr, flush=True)
            outcome, problems, failures = audit_run(r, results)
            counts[outcome] = counts.get(outcome, 0) + 1
            where = " [two jobs ran together here]" if u["key"] in hit else ""
            if problems and all(x.endswith(UNMEASURED) for x in problems):
                problems = []                     # measured after the last repeat: normal while its job is at work
                if u["key"] not in live:
                    unmeasured.append(r["dir"])
                    if u["key"] in submitted and u["key"] not in waiting:stalled.setdefault(u["key"], []).append(r["dir"] + " (to measure)")
            if problems:
                redo.append(r["dir"])
                notes.append(f"  DAMAGED {r['dir']}{where}: {problems[0]}" + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
                if args.verbose:notes += [f"      {x}" for x in problems[1:]]
            for part, kind, line in failures:kinds[kind] += 1
            bad = [f for f in failures if f[1] == "files"]
            if bad and r["dir"] not in redo:
                redo.append(r["dir"]);notes.append(f"  FAILED ON FILES {r['dir']}{where}: {bad[0][0]}: {bad[0][2]}")
            elif u["key"] in hit and any(f[1] == "other" for f in failures) and r["dir"] not in redo:
                f = next(f for f in failures if f[1] == "other");suspect.append(r["dir"])
                notes.append(f"  (not redone) failed where two jobs ran together, error not recognised: {r['dir']}: {f[0]}: {f[2]}")
            if any(f[1] == "killed" for f in failures) and r["dir"] not in redo:
                killed.append(r["dir"])           # the code was killed, not stopped by its solver: its job continues it
                if u["key"] not in live and u["key"] not in waiting:stalled.setdefault(u["key"], []).append(r["dir"] + " (killed while running)")
            if outcome in ("running", "not started") and u["key"] in submitted and u["key"] not in live and u["key"] not in waiting:
                stalled.setdefault(u["key"], []).append(r["dir"])
                if outcome == "running" and last_words(r, results):why[r["dir"]] = last_words(r, results)
            if outcome in ("complete", "failed") and r["dir"] not in redo and u["key"] not in live \
                    and not all((BUNDLE/"returns"/(r["dir"] + x)).is_file() for x in ARCHIVES):
                repack.append(r["dir"])           # (a job still in the queue packs its runs itself)
    total = sum(counts.values())
    print(f"\nRuns: " + ", ".join(f"{counts.get(k, 0)} {k}" for k in ("complete", "failed", "running", "not started")) + f" (of {total})")
    print(f"Failed parts: {kinds['solver']} GPU-driver solver stops, {kinds['files']} file errors, {kinds['killed']} killed from outside, "
          f"{kinds['other']} other (external-code failures are 'other'; `python3 tools/sweep.py runs` groups them by error)")
    if killed:print(f"Recorded as failed, but the code had been killed (preemption, requeue, cancel or time limit); their jobs continue "
                    f"them from the last output ({len(killed)}): {', '.join(killed[:8])}{' ...' if len(killed) > 8 else ''}")
    for line in notes:print(line)
    if unmeasured:print(f"Converted but not measured, and their job has ended ({len(unmeasured)}; running the job again "
                        f"finishes them): {', '.join(unmeasured[:6])}{' ...' if len(unmeasured) > 6 else ''}")
    if stalled:
        print(f"Jobs that ended and left runs unfinished ({len(stalled)} jobs, {sum(len(v) for v in stalled.values())} runs):")
        for k, v in sorted(stalled.items()):
            e = latest[k];st = states.get((e["job"], e["indices"][0]), ("UNKNOWN",))[0]
            print(f"  {k} (job {e['job']} {st}, attempt {e['attempt']}): {', '.join(v[:4])}{' ...' if len(v) > 4 else ''}")
            for x in v[:4]:
                if x in why:print(f"      {x}: {why[x]}")
    if repack:print(f"Not packed for download ({len(repack)}): {', '.join(repack[:6])}{' ...' if len(repack) > 6 else ''}")
    save_json(STATE/"audit.json", dict(time=now(), redo=redo, suspect=suspect, stalled=sorted(stalled), repack=repack, unmeasured=unmeasured, killed=killed,
                                       together=[dict(key=k, jobs=[a, b], seconds=sec) for k, a, b, sec in together]))
    print(f"\nTo redo: {len(redo)} runs (damaged, or failed on file errors); {len(stalled)} jobs to hand back; {len(repack)} runs to pack.")
    if redo or stalled:print("  python3 tools/sweep.py redo --audit      deletes those runs and hands their jobs (and the ended ones) back")
    if repack:print("  python3 tools/sweep.py repack --audit    packs the unpacked runs (on this node)")
    if not (redo or stalled or repack or dup):print("Nothing to do.")


def cmd_redo(args):
    """Delete run directories so that their runs start over, and hand their jobs back to the coordinator: the runs
    named, and with --audit the runs the last audit listed (plus the jobs it found ended with unfinished runs)."""
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));by_key, latest, state, live = unit_view(settings)
    results = Path(job_env(settings)["MHD_RESULTS"]);unit_of = {r["dir"]: (k, r) for k, u in by_key.items() for r in u["runs"]}
    names = [Path(x).name for x in args.runs];stalled = []
    if args.audit:
        a = load_json(STATE/"audit.json", None)
        if a is None:sys.exit("No audit yet: python3 tools/sweep.py audit")
        names += [x for x in a["redo"] if x not in names];stalled = [k for k in a["stalled"] if k in by_key and k not in live]
    unknown = [x for x in names if x not in unit_of]
    if unknown:sys.exit(f"not runs of this sweep: {', '.join(unknown[:5])} (names as in `python3 tools/sweep.py audit`)")
    if not names and not stalled:print("Nothing to redo.");return
    if len(names) > 40 and not args.yes:
        sys.exit(f"{len(names)} runs would be deleted and run again. Look at the list first (python3 tools/sweep.py audit); "
                 "to go ahead, add --yes")
    removed, left, back, queued = clear_runs([unit_of[x] for x in names], live, results)
    report_cleared(removed, left, back + [k for k in stalled if k not in back], queued)


def cmd_repack(args):
    """Pack runs for download again (returns/NAME.tar.gz and NAME.summary.tar.gz): the runs named, and with --audit
    the finished runs the last audit found without archives."""
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));results = Path(job_env(settings)["MHD_RESULTS"])
    names = [Path(x).name for x in args.runs]
    if args.audit:names += [x for x in load_json(STATE/"audit.json", {}).get("repack", []) if x not in names]
    if not names:print("Nothing to pack.");return
    env = {**os.environ, **job_env(settings)};done = 0
    for x in names:
        if not (results/x).is_dir():print(f"  {x}: no such run directory");continue
        p = run(["bash", str(BUNDLE/"tools/pack_results.sh"), str(results/x)], check=False, env=env)
        ok = p.returncode == 0 and all((BUNDLE/"returns"/(x + y)).is_file() for y in ARCHIVES);done += ok
        if not ok:print(f"  {x}: not packed: {(p.stdout + p.stderr).strip()[-300:]}")
    print(f"{done} of {len(names)} runs packed.")


def cmd_dedupe(args):
    """Cancel duplicate jobs: two jobs of one unit in the queue at once. (Before patch 7 the coordinator resubmitted
    a preempted job although Slurm had requeued it itself; the two then ran the same runs.) Of each such group the
    job that is running is kept (the earliest started if several run), otherwise the newest; the others are cancelled."""
    entries = ledger();state = {}
    for job, st in queue():state.setdefault(job, []).append(st)
    jobs_of, entry_of = {}, {}
    for e in entries:
        entry_of[e["job"]] = e
        if e["job"] in state and e["job"] not in jobs_of.setdefault(e["key"], []):jobs_of[e["key"]].append(e["job"])
    dup = {k: js for k, js in jobs_of.items() if len(js) > 1}
    latest = {e["key"]: e for e in entries}
    # A unit whose newest job has ended while an earlier one is still in the queue (the requeued original outlived the
    # resubmission): the earlier job is the unit's current job.
    for key, js in sorted(jobs_of.items()):
        if len(js) == 1 and latest[key]["job"] != js[0]:
            print(f"{key}: job {js[0]} ({'/'.join(sorted(set(state[js[0]])))}) is the unit's job again; its later job {latest[key]['job']} has ended")
            if not args.dry_run:
                e = entry_of[js[0]];record(dict(key=key, job=js[0], indices=e["indices"], attempt=latest[key]["attempt"],
                                               options=e.get("options", []), time=now(), after="dedupe"))
    if not dup:print("No duplicate jobs in the queue.");return
    records = sacct_records([j for js in dup.values() for j in js])
    for key, js in sorted(dup.items()):
        running = [j for j in js if any(s not in WAITING for s in state[j])]
        started = lambda j: min((a for a, _ in ran(records, j, entry_of[j]["indices"][0])[-1:]), default=float("inf"))
        keep = min(running, key=started) if running else js[-1]
        for j in js:
            if j == keep:continue
            if not args.dry_run:
                p = run(["scancel", j], check=False)
                if p.returncode:print(f"  scancel {j} failed: {p.stderr.strip()}");continue
            print(f"{key}: {'would cancel' if args.dry_run else 'cancelled'} job {j} ({'/'.join(sorted(set(state[j])))}); "
                  f"keeping job {keep} ({'/'.join(sorted(set(state[keep])))})"
                  + ("; both were running: `audit` checks the runs they touched" if j in running else ""))
        if latest[key]["job"] != keep and not args.dry_run:
            # The kept job becomes the unit's current job again for the coordinator.
            e = entry_of[keep];record(dict(key=key, job=keep, indices=e["indices"], attempt=latest[key]["attempt"],
                                           options=e.get("options", []), time=now(), after="dedupe"))


def cmd_skip(args):
    """Take runs out of the sweep (`skip KEY ...`), put them back (`skip --undo KEY ...`) or list them (`skip`).
    A skipped run is not submitted or resubmitted, its job in the queue is cancelled, and it is left out of the
    counts; what it has written stays on scratch. Only runs with a job of their own (the long runs) can be skipped:
    their key is the job key that `status` prints."""
    settings = load_json(STATE/"settings.json", dict(DEFAULTS));skip = load_json(STATE/"skip.json", [])
    if not args.keys:
        print(f"{len(skip)} runs are skipped" + (":" if skip else "."))
        for k in skip:print(f"  {k}")
        return
    every = {u["key"]: u for u in plan.units(settings, PILOTS)}
    for k in args.keys:
        if k not in every or len(every[k]["runs"]) != 1 or every[k]["runs"][0]["key"] != k:
            sys.exit(f"{k!r} is not a run with a job of its own (keys as in `python3 tools/sweep.py plan --json F`; "
                     "a run packed with others cannot be skipped)")
    if args.undo:
        save_json(STATE/"skip.json", [k for k in skip if k not in args.keys])
        print(f"{len([k for k in args.keys if k in skip])} runs put back; the coordinator submits them when their turn comes.");return
    save_json(STATE/"skip.json", skip + [k for k in args.keys if k not in skip])
    state = {}
    for job, st in queue():state.setdefault(job, []).append(st)
    for e in ledger():
        if e["key"] in args.keys and e["job"] in state:
            p = run(["scancel", e["job"]], check=False);was = "/".join(sorted(set(state.pop(e["job"]))))
            print(f"  {e['key']}: " + (f"scancel {e['job']} failed: {p.stderr.strip()}" if p.returncode else f"cancelled job {e['job']} ({was})"))
    hours = sum(every[k]["resource"]["central"] for k in args.keys)/3600
    print(f"{len(args.keys)} runs taken out of the sweep (estimated {hours:.0f} {'GPU-h' if every[args.keys[0]]['gpu'] else 'core-h'} by the plan's "
          "model). They are not submitted again; `skip --undo KEY ...` puts them back.")


def cmd_plan(args):
    settings = load_json(STATE/"settings.json", dict(DEFAULTS))
    units = selected_units(settings);report(units, settings)
    if args.json:Path(args.json).write_text(json.dumps(units, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start");s.add_argument("--dry-run", action="store_true")
    s.add_argument("--gpu-partition");s.add_argument("--cpu-partition");s.add_argument("--gpu-feature")
    s.add_argument("--cap", type=int, help="queued-task ceiling (Sherlock's limit is ~100 per user)")
    s.add_argument("--cpu-cap", type=int, help="most CPU tasks queued at once, so GPU work is never starved")
    s.add_argument("--max-attempts", type=int);s.add_argument("--skip-cpu", action="store_true", default=None)
    s.add_argument("--cpu-max-n3d", type=int, help="drop 3D RAMSES/Athena++ runs above this grid (default 512)")
    s.add_argument("--only", nargs="*", help="run only these tests")
    sub.add_parser("tick");sub.add_parser("status")
    q = sub.add_parser("plan");q.add_argument("--json")
    r = sub.add_parser("retry");r.add_argument("keys", nargs="*")
    r.add_argument("--lane", choices=("cpu", "gpu"), help="the FAILED/CANCELLED jobs of this lane (with --clean-error: every job of it)")
    r.add_argument("--clean-error", help="delete the run directories that have a part which failed with this text in its error")
    sub.add_parser("regate")
    sub.add_parser("runs").add_argument("--json", help="write every run and part with its recorded status to this file (for the progress site)")
    a = sub.add_parser("audit");a.add_argument("--verbose", action="store_true", help="every problem of a damaged run")
    for name in ("redo", "repack"):
        d = sub.add_parser(name);d.add_argument("runs", nargs="*", help="run directory names");d.add_argument("--audit", action="store_true")
        if name == "redo":d.add_argument("--yes", action="store_true", help="go ahead with more than 40 runs")
    sub.add_parser("dedupe").add_argument("--dry-run", action="store_true")
    k = sub.add_parser("skip");k.add_argument("keys", nargs="*", help="job keys of runs that have a job of their own")
    k.add_argument("--undo", action="store_true", help="put these runs back")
    args = p.parse_args()
    if args.cmd == "start":cmd_start(args)
    elif args.cmd == "tick":sys.exit(cmd_tick(args))
    else:{"status": cmd_status, "plan": cmd_plan, "retry": cmd_retry, "regate": cmd_regate, "runs": cmd_runs,
          "audit": cmd_audit, "redo": cmd_redo, "repack": cmd_repack, "dedupe": cmd_dedupe, "skip": cmd_skip}[args.cmd](args)


if __name__ == "__main__":
    main()
