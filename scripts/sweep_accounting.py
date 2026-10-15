"""Summarize allocation CPU-hours from sacct, independently of decayed fairshare usage.

    python3 scripts/sweep_accounting.py

Inputs are downloaded by fetch_sweep_accounting.sh. Allocation records only:
no batch/extern/solver steps, separate array tasks, separate requeue attempts.
CPUTimeRAW is allocated CPU-seconds, not measured busy CPU time.
"""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "progress/sweep_status/accounting"
OUTPUT = ROOT / "progress/media/accounting"


def number(value):
    return int(value.strip() or 0)


def gpu_count(tres):
    entries = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    if "gres/gpu" in entries:
        return int(entries["gres/gpu"])
    return sum(int(value) for key, value in entries.items() if key.startswith("gres/gpu:"))


def summarize(source=SOURCE):
    if not (source / "accounting.psv").is_file():
        return None
    query = json.loads((source / "query.json").read_text())
    ledger = [json.loads(line) for line in (source / "ledger.jsonl").read_text().splitlines() if line.strip()]
    sweep_jobs = {str(entry["job"]): entry for entry in ledger}
    unique = {}
    with (source / "accounting.psv").open(newline="") as stream:
        for raw in csv.DictReader(stream, delimiter="|"):
            row = {key.strip(): (value or "").strip() for key, value in raw.items() if key}
            job = row["JobID"]
            if not re.fullmatch(r"\d+(?:_\d+)?", job):
                continue  # Steps and unexpanded array ranges are never allocations to sum.
            cpus = number(row["AllocCPUS"])
            elapsed = number(row["ElapsedRaw"])
            seconds = number(row["CPUTimeRAW"])
            if seconds != cpus*elapsed:
                raise ValueError(f"{job}: CPUTimeRAW {seconds} differs from AllocCPUS × ElapsedRaw {cpus*elapsed}")
            entry = sweep_jobs.get(job.split("_")[0])
            gpus = gpu_count(row["AllocTRES"])
            sweep_gpu = bool(entry) and any(opt.startswith("--gpus") for opt in entry.get("options", []))
            result = dict(job_id=job, raw_id=row["JobIDRaw"], name=row["JobName"], account=row["Account"],
                          partition=row["Partition"], state=row["State"], cpus=cpus, gpus=gpus,
                          elapsed_seconds=elapsed, allocated_cpu_seconds=seconds, core_hours=seconds/3600,
                          gpu_hours=gpus*elapsed/3600, submit=row["Submit"], start=row["Start"], end=row["End"],
                          sweep=entry is not None, sweep_lane=("gpu" if sweep_gpu else "cpu") if entry else "outside sweep ledger",
                          unit=entry["key"] if entry else "")
            # Repeated exports of one incarnation count once; different starts/submissions preserve requeues.
            key = (result["raw_id"], result["submit"], result["start"])
            if key not in unique or seconds > unique[key]["allocated_cpu_seconds"]:
                unique[key] = result
    allocations = sorted(unique.values(), key=lambda row: (row["start"], row["job_id"]))
    active = [row for row in allocations if row["elapsed_seconds"] > 0 and row["cpus"] > 0]
    scopes = {}
    groups = {"all_user_jobs": active,
              "cpu_only_jobs": [row for row in active if row["gpus"] == 0],
              "gpu_jobs_host_cpus": [row for row in active if row["gpus"] > 0],
              "sweep_cpu_lane": [row for row in active if row["sweep_lane"] == "cpu"],
              "sweep_gpu_lane": [row for row in active if row["sweep_lane"] == "gpu"],
              "outside_sweep_ledger": [row for row in active if not row["sweep"]]}
    for name, group in groups.items():
        scopes[name] = dict(allocation_attempts=len(group), core_hours=sum(row["core_hours"] for row in group),
                            gpu_hours=sum(row["gpu_hours"] for row in group),
                            cpu_counts=sorted({row["cpus"] for row in group}))
    missing = sorted(job for job in sweep_jobs if not any(row["job_id"].split("_")[0] == job for row in allocations))
    return dict(query=query, scopes=scopes, allocations=allocations, missing_ledger_jobs=missing,
                accounting_basis="Allocated CPUs × elapsed seconds / 3600; CPUTimeRAW cross-checked; no job steps",
                duplicate_basis="One record per JobIDRaw, Submit, Start; distinct requeue incarnations retained",
                fairshare_is_not_lifetime_cpu_hours=True)


def write(report, output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    rows = report["allocations"]
    with (output / "allocations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["job_id"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    report = summarize()
    if report is None:
        raise SystemExit("Fetch accounting first: bash scripts/fetch_sweep_accounting.sh")
    write(report)
    print("Slurm allocation accounting as of", report["query"]["fetched_at"])
    for scope, values in report["scopes"].items():
        print(f"{scope}: {values['core_hours']:,.2f} allocated core-hours, {values['gpu_hours']:,.2f} GPU-hours; "
              f"{values['allocation_attempts']} attempts; CPUs/job: {values['cpu_counts']}")
    print("Sweep ledger job IDs absent from this query:", len(report["missing_ledger_jobs"]))
