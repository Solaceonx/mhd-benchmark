"""Plan/submit missing GLM and RAMSES dense movies; keep the original results.

    bash -l compare_movies.sh                 # plan only
    bash -l compare_movies.sh --submit        # Mach 30, 50 at 256³, seed 1
    bash -l compare_movies.sh --submit --mach 3 10 30 50

RAMSES array runs one 64-core task at a time. Euler/HLL dense runs already exist
in mhd-results-turb-movie and are reused by the renderer; they are not repeated.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

HERE = Path(__file__).resolve().parent


def complete(path, count=201):
    if not (path/"run.toml").is_file():
        return False
    meta = tomllib.loads((path/"run.toml").read_text())
    return (meta.get("status") == "complete" and meta.get("movie_frames") == count
            and all((path/f"rho_{i:04d}.bin").is_file() and (path/f"rho_{i:04d}.bin").stat().st_size == 256*256*4 for i in range(count)))


def configuration(machs):
    high = machs[0] >= 30
    if any((mach >= 30) != high for mach in machs):
        raise ValueError("Low and high Mach numbers require different gamma configurations")
    base = tomllib.loads((HERE/("config_movie_highmach.toml" if high else "config_movie.toml")).read_text())
    base["machs"] = machs; base["gpu"]["methods"] = ["glm"]
    base["gpu"]["check_methods"] = []; base["gpu"]["check_grids"] = []
    return base


def queued(name):
    return subprocess.check_output(["squeue", "-h", "-u", os.environ["USER"], "-n", name, "-o", "%i"], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--mach", nargs="+", type=int, choices=[3, 10, 30, 50], default=[30, 50])
    parser.add_argument("--results", type=Path, default=Path(os.environ.get("TURB_COMPARE_RESULTS", str(Path(os.environ.get("SCRATCH", "/scratch/users/lyxia"))/"mhd-results-turb-compare-movie"))))
    parser.add_argument("--pilot", type=Path, default=Path(os.environ.get("TURB_PILOT_RESULTS", str(Path(os.environ.get("SCRATCH", "/scratch/users/lyxia"))/"mhd-results-turb"))))
    args = parser.parse_args(argv)
    machs = sorted(set(args.mach))
    forcing_path = args.pilot/"forcing.toml"
    if not forcing_path.is_file():
        parser.error(f"Pilot forcing calibration not found: {forcing_path}")
    forcing = tomllib.loads(forcing_path.read_text())
    if any(f"mach_{mach}" not in forcing for mach in machs):
        parser.error("Missing requested Mach calibration in pilot forcing.toml")
    gpu = [mach for mach in machs if not complete(args.results/f"gpu-glm-M{mach}-N256-s1")]
    cpu = [dict(variant=variant, mach=mach, n=256) for mach in machs for variant in ("matched", "best")
           if not complete(args.results/f"ramsesfix-{variant}-M{mach}-N256-s1")]
    print(f"Results: {args.results}\nGPU: {len(gpu)} missing GLM/LLF runs; 1 GPU + 4 host cores per job")
    print(f"CPU: {len(cpu)} missing RAMSES runs; 64 cores per task, array concurrency 1")
    print("Each run: 256³, seed 1, 10T, 201 density snapshots. Original Euler/HLL movies are reused.")
    print("Allow ~250 GB temporary scratch for the one active RAMSES run; dense movie output is ~53 MB/run.")
    print("Mach 30/50 RAMSES pilot evolution took ~320 core-hours in total; dense native output adds I/O overhead.")
    if not args.submit:
        print("Plan only; no jobs submitted. Add --submit to launch."); return
    # Import the same TOML writer and setup used by the pilot only when preparing a real submission.
    import ramses_turb as R
    executable = Path(os.environ.get("MHD_CODES", str(Path(os.environ.get("SCRATCH", "/scratch/users/lyxia"))/"mhd-codes-sweep")))/"ramses/bin-turb-fixed/ramses_turb3d"
    if cpu and not os.access(executable, os.X_OK):
        parser.error(f"Corrected RAMSES executable is missing: {executable}")
    args.results.mkdir(parents=True, exist_ok=True); (HERE/"logs").mkdir(exist_ok=True)
    destination = args.results/"forcing.toml"
    if destination.is_file() and destination.read_bytes() != forcing_path.read_bytes():
        parser.error("Existing movie forcing differs from the pilot; choose a fresh --results directory")
    if not destination.is_file():shutil.copyfile(forcing_path, destination)
    for high in (False, True):
        selected = [mach for mach in gpu if (mach >= 30) == high]
        if not selected:continue
        name = "turb-cmp-glm-"+("high" if high else "low")
        if queued(name):
            print(name, "already queued; kept"); continue
        config = HERE/f"config_compare_{'high' if high else 'low'}.toml"
        config.write_text(R.X.toml_dump(configuration(selected)))
        constraint = os.environ.get("TURB_GPU_CONSTRAINT", "GPU_SKU:H100_SXM5")
        options = [] if constraint == "none" else [f"--constraint={constraint}"]
        command = ["sbatch", "--parsable", f"--chdir={HERE}", f"--job-name={name}", "--time=02:00:00",
                   f"--export=ALL,TURB_CONFIG={config.name},TURB_RESULTS={args.results}", *options, str(HERE/"gpu.sbatch")]
        print("submitted", name, subprocess.check_output(command, text=True).strip())
    if cpu:
        name = "turb-cmp-ramses"
        if queued(name):
            print(name, "already queued; kept"); return
        tasks = HERE/"compare_cpu_tasks.json"
        for task in cpu:
            config = HERE/f"config_compare_M{task['mach']}.toml"
            config.write_text(R.X.toml_dump(configuration([task["mach"]])))
            task.update(config=str(config), results=str(args.results), exe=str(executable))
        tasks.write_text(json.dumps(cpu, indent=2)+"\n")
        command = ["sbatch", "--parsable", f"--chdir={HERE}", f"--job-name={name}",
                   f"--array=0-{len(cpu)-1}%1", f"--export=ALL,TURB_COMPARE_TASKS={tasks}", str(HERE/"compare_cpu.sbatch")]
        print("submitted", name, subprocess.check_output(command, text=True).strip())


if __name__ == "__main__":main()
