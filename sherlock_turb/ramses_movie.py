"""Dense RAMSES density movies in a separate results tree, without repeating analysis.

Uses the validated hydro setup and corrected forcing executable from the pilot.
RAMSES writes native outputs at 20/T. Only midplane density is exported; native
outputs are removed after successful conversion. Allow ~250 GB temporary scratch
per 256³ run. Completed density snapshots are kept across interrupted conversion.
"""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tomllib

import numpy as np
import ramses_turb as R


def density_slice(outdir, n):
    """Read the conservative density midplane directly from native per-rank records.

    Same oct indexing as the pilot reader; avoids allocating nine full volumes.
    Density may be nonpositive: preserve it and let the renderer floor for display.
    """
    suffix = outdir.name.split("_")[-1]
    info = (outdir/f"info_{suffix}.txt").read_text()
    ncpu = int(re.search(r"^ncpu\s*=\s*(\d+)", info, re.M)[1])
    level = int(round(np.log2(n)))
    result = np.full((n, n), np.nan, dtype=np.float32)
    for rank in range(1, ncpu+1):
        amr = R.X.FortranRecords(outdir/f"amr_{suffix}.out{rank:05d}")
        hydro = R.X.FortranRecords(outdir/f"hydro_{suffix}.out{rank:05d}")
        amr.read("<i4"); ndim = amr.read("<i4")[0]; amr.read("<i4")
        levels = amr.read("<i4")[0]; amr.read("<i4"); boundaries = amr.read("<i4")[0]
        amr.read("<i4"); boxlen = amr.read("<f8")[0]
        if ndim != 3:
            raise ValueError("Movie reader requires a 3D hydro output")
        amr.skip(11); amr.read("<i4"); amr.read("<i4")
        numbl = amr.read("<i4").reshape(levels, ncpu); amr.read("<i4")
        if boundaries:
            raise ValueError("Movie reader requires a periodic grid without boundary grids")
        amr.skip(1); amr.skip(1); amr.skip(1); amr.skip(3)
        hydro.read("<i4"); nvar = hydro.read("<i4")[0]
        hydro.read("<i4"); hydro.read("<i4"); hydro.read("<i4"); hydro.read("<f8")
        if nvar != 5:
            raise ValueError(f"Expected five conservative hydro variables, found {nvar}")
        for ilevel in range(levels):
            for owner in range(ncpu):
                cells = numbl[ilevel, owner]
                hydro.read("<i4"); hydro.read("<i4")
                if cells == 0:
                    continue
                amr.skip(3)
                centers = np.stack([amr.read("<f8") for _ in range(ndim)])
                amr.skip(1+2*ndim+3*2**ndim)
                for child in range(2**ndim):
                    rho = hydro.read("<f8")
                    hydro.skip(nvar-1)
                    if owner != rank-1 or ilevel+1 != level:
                        continue
                    dx = boxlen/2**level
                    offset = np.array([(child >> axis) & 1 for axis in range(ndim)])
                    ijk = np.floor((centers.T+(offset-.5)*dx)/dx).astype(int)
                    select = ijk[:, 2] == n//2
                    result[ijk[select, 0], ijk[select, 1]] = rho[select]
    if not np.isfinite(result).all():
        raise ValueError(f"{outdir}: density midplane is incomplete or nonfinite")
    return result


def run(args):
    config = tomllib.loads(args.config.read_text())
    per = int(config["movie_per_T"])
    config["frames_per_T"] = per
    n = args.n; mach = args.mach
    forcing_file = args.results/"forcing.toml"
    forcing = tomllib.loads(forcing_file.read_text())[f"mach_{R.mkey(mach)}"]
    seed = int(config["ramses"]["seed"])
    directory = args.results/f"ramsesfix-{args.variant}-M{R.mkey(mach)}-N{n}-s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    meta_file = directory/"run.toml"
    count = round(float(config["t_end_T"])*per)+1
    expected = [directory/f"rho_{i:04d}.bin" for i in range(count)]
    previous = tomllib.loads(meta_file.read_text()) if meta_file.is_file() else {}
    if previous.get("status") == "complete" and all(path.is_file() and path.stat().st_size == n*n*4 for path in expected):
        print("kept", directory.name); return 0
    state, outputs, perf = R.solve(config, directory, args.variant, mach, n, forcing["a_rms"], seed,
                                   float(config["t_end_T"]), args.exe, args.mpi)
    if state == "killed":
        print("Interrupted: rerun to resume from the newest native output", flush=True)
        return R.X.KILLED
    settings = R.VARIANTS[args.variant]
    meta = dict(code="ramses", method=args.variant, N=n, mach_target=mach, seed=seed,
        solver=f"RAMSES hydro / {settings['riemann'].upper()}, slope_type={settings['slope_type']}",
        gamma=R.gamma_of(config), a_rms=forcing["a_rms"], comp_frac=config["comp_frac"],
        spectrum=config["spectrum"], forcing_projection="corrected signed wave numbers",
        forcing_sha256=hashlib.sha256(forcing_file.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        movie_per_T=per, movie_frames=0, t_end_T=config["t_end_T"], turb_T=1/(2*mach),
        status="converting", device=f"CPU {os.uname().nodename} x {R.X.mpi_ranks(args.mpi)} ranks",
        precision="fp64 evolution; Float32 density midplanes", perf=perf,
        density_floor_for_display=1e-6, output_only=True,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    meta_file.write_text(R.X.toml_dump(meta))
    time_rows = {}
    times_file = directory/"movie_times.csv"
    if times_file.is_file():
        for line in times_file.read_text().splitlines()[1:]:
            time_rows[int(line.split(",")[0])] = line
    for index, output in sorted(outputs.items()):
        rho = density_slice(output, n)
        temporary = expected[index].with_suffix(".partial")
        rho.ravel(order="F").tofile(temporary); temporary.replace(expected[index])
        time_rows[index] = f"{index},{R.X.ramses_time(output):.17g},{int(np.count_nonzero(rho<=0))},{float(rho.min()):.9g}"
        times_file.write_text("frame,time,nonpositive_density_cells,rho_min\n"+"\n".join(time_rows[key] for key in sorted(time_rows))+"\n")
        print(f"density {index:04d}/{count-1}: {directory.name}", flush=True)
    present = [path for path in expected if path.is_file() and path.stat().st_size == n*n*4]
    meta["movie_frames"] = len(present)
    meta["status"] = "complete" if state == "complete" and len(present) == count else "failed"
    if meta["status"] == "failed":
        meta["reason"] = perf.get("reason", f"Expected {count} density frames; found {len(present)}")
    meta_file.write_text(R.X.toml_dump(meta))
    if meta["status"] == "complete":
        for output in directory.glob("output_*"):
            shutil.rmtree(output)
    print(meta["status"], directory.name, len(present), "density frames", flush=True)
    return 0 if meta["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("variant", choices=R.VARIANTS)
    parser.add_argument("mach", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--mpi", default="")
    sys.exit(run(parser.parse_args()))
