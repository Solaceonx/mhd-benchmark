"""Compare saved CPU GLM/CT runs with Athena++ using norms, spectra and energies."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
from compare_athena_reference import ROOT, load_athena, load_local, restrict, plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from mhd_spectra import (divergence, energies, error_norms, face_divergence,
                         mean_square, mode_grid, physical_fields, spectrum)

COLORS = {"glm": "#167d9a", "ct": "#ce6d27", "athena": "#222222", "check": "#888888"}
TIMES = (0.0, 0.1, 0.25, 0.5)


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_faces(root, frame, n):
    arrays = []
    for component in ("bx", "by"):
        path = root / f"ct_{component}face_{frame:03d}.bin"
        if not path.exists():
            raise ValueError(f"Missing native face output: {path}; rerun with updated driver")
        arrays.append(np.fromfile(path, dtype="<f4").reshape((n, n), order="F").T.astype(float))
    return arrays


def draw(ax, power, n, label, color, style="-"):
    # Only complete annuli: bins with upper edge <= axial Nyquist.
    k = np.arange(1, n // 2)
    positive = power[k] > 0
    ax.loglog(k[positive] + 0.5, power[k][positive], style, color=color, label=label, lw=1.6)
    ax.set(xlabel=r"Shell midpoint $kL$ (cycles per box)", ylabel="Shell-integrated power")
    ax.set_xticks([2**j for j in range(1, 12) if 2**j <= n//2])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.2, which="both")


def plot_one(out, n, nref, ncheck, cache, history):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for ax, field, title in zip(axes, ("density_fluctuation", "kinetic_energy", "magnetic_energy"),
                                ("Density fluctuations", "Kinetic energy", "Magnetic energy")):
        for name in ("athena", "check", "glm", "ct"):
            if (name, field) in cache:
                label = {"athena": f"Athena++ {nref}² → {n}²", "check": f"Athena++ {ncheck}² → {n}²"}.get(name, name.upper())
                draw(ax, cache[name, field], n, label, COLORS[name], "--" if name == "check" else "-")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    qualifier = " · equal-resolution reference" if n == nref else ""
    fig.suptitle(f"{n}² Orszag–Tang · t = 0.5 · matched cell averages{qualifier}")
    fig.savefig(out / "power_spectra.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, field in zip(axes, ("density", "magnetic")):
        for name in ("glm", "ct", "check"):
            if (name, "error_" + field) in cache:
                label = f"{nref}² − {ncheck}² reference difference" if name == "check" else name.upper() + " − reference"
                draw(ax, cache[name, "error_" + field], n, label, COLORS[name], "--" if name == "check" else "-")
        ax.set_title(field.capitalize() + " error spectrum"); ax.legend(fontsize=8)
    fig.suptitle(f"{n}² · t = 0.5 · Fourier transform of the field difference (phase sensitive){qualifier}", fontsize=12)
    fig.savefig(out / "error_spectra.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for ax, op, title in zip(axes, ("spectral", "centered", "native_face"),
                            ("Common Fourier derivative of cell B", "Common centered difference of cell B", "CT native face divergence")):
        for name in ("athena", "glm", "ct"):
            key = (name, "divergence_" + op)
            if key in cache:
                draw(ax, cache[key], n, name.upper(), COLORS[name])
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(f"{n}² · t = 0.5 · operators differ: only the right panel measures CT's native constraint")
    fig.savefig(out / "divergence_spectra.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, field in zip(axes.flat, ("kinetic", "magnetic", "thermal", "total")):
        for name in ("glm", "ct", "athena"):
            rows = [r for r in history if r["N"] == n and r["method"] == name and r["representation"] == "comparison_grid"]
            values = [r[field] for r in rows]
            if field == "total":
                values = [(v-values[0])/values[0] for v in values]
            ax.plot([r["time"] for r in rows], values,
                    "o--" if name == "athena" else "-", color=COLORS[name],
                    label=f"Athena++ {nref}² → {n}²" if name == "athena" else name.upper(), ms=4)
        ax.set(title=field.capitalize(), xlabel="time", ylabel="Volume-averaged energy density")
        if field == "total":
            ax.set(title="Total energy drift from each run's initial value", ylabel=r"$(E(t)-E(0))/E(0)$")
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle(f"{n}² · resolved energies on the comparison grid\nThermal remainder after restriction includes unresolved kinetic/magnetic energy", fontsize=12)
    fig.savefig(out / "energy_history.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locals", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "scripts/out/athena_1024")
    parser.add_argument("--check-reference", type=Path, default=ROOT / "scripts/out/athena_512")
    parser.add_argument("--out", type=Path, default=ROOT / "scripts/out/spectral_study/analysis")
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    refmeta = json.loads((args.reference / "run_manifest.json").read_text())
    checkmeta = json.loads((args.check_reference / "run_manifest.json").read_text())
    if not refmeta.get('completed') or not checkmeta.get('completed'):
        raise ValueError('Both reference runs must be complete before analysis')
    nref, ncheck = refmeta["N"], checkmeta["N"]
    all_spectra, all_norms, all_energy, timings, checks, summaries, divstats = [], [], [], [], [], [], []
    reference = {t: load_athena(args.reference, t) for t in TIMES}
    check_reference = {t: load_athena(args.check_reference, t) for t in TIMES}
    for name, states, nf in (("athena", reference, nref), ("athena_check", check_reference, ncheck)):
        for t, state in states.items():
            all_energy.append(dict(N=nf, method=name, time=t, representation="native_grid", **energies(state)))

    def record(q, factor, n, t, name, field, ref_n=0):
        power = spectrum(q, factor)
        expected = factor * mean_square(q)
        residual = abs(power.sum() - expected) / max(expected, 1e-300)
        assert residual < 2e-12, (name, field, residual)
        checks.append(residual)
        _, _, bins = mode_grid(n)
        counts = np.bincount(bins.ravel())
        for k, value in enumerate(power):
            all_spectra.append(dict(N=n, time=t, method=name, diagnostic=field, reference_N=ref_n,
                                    k_lower=k, k_upper=k + 1, mode_count=int(counts[k]),
                                    complete_annulus=(k + 1 <= n / 2), power=float(value)))
        return power

    for root in args.locals:
        root = root.resolve()
        meta = tomllib.loads((root / "metadata.toml").read_text()); n = meta["N"]
        if n > nref or nref % n:
            raise ValueError(f"Reference {nref} must be at least N={n} and divisible by it")
        runout = out / f"N{n}"; runout.mkdir(exist_ok=True)
        diag = {m: np.genfromtxt(root / f"{m}_diagnostics.csv", delimiter=",", names=True) for m in ("glm", "ct")}
        for m in ("glm", "ct"):
            timings.append(dict(N=n, method=m, step_seconds=meta[f"{m}_step_seconds"],
                                combined_wall_seconds_excluding_compile=meta["wall_seconds_excluding_compile"],
                                steps=meta["steps"], frames=meta["frames"], threads=meta["threads"],
                                million_cell_steps_per_second=n*n*meta["steps"]/meta[f"{m}_step_seconds"]/1e6))
            for d in diag[m]:
                all_energy.append(dict(N=n, method=m, time=float(d["time"]), representation="comparison_grid",
                                       kinetic=float(d["kinetic"]), magnetic=float(d["magnetic"]),
                                       thermal=float(d["energy"]-d["kinetic"]-d["magnetic"]), total=float(d["energy"])))
        cache = {}
        for t in TIMES:
            ref = restrict(reference[t], n)
            states = {m: load_local(root, m, t, n) for m in ("glm", "ct")}
            states["athena"] = ref
            check = restrict(check_reference[t], n) if n <= ncheck and ncheck % n == 0 else None
            if check is not None:
                states["check"] = check
            all_energy.append(dict(N=n, method="athena", time=t, representation="comparison_grid", **energies(ref)))
            for m, state in states.items():
                assert all(np.isfinite(q).all() for q in state.values())
                thermal = state["energy"] - np.sum(state["momentum"]**2, axis=-1)/(2*state["density"]) - np.sum(state["magnetic"]**2, axis=-1)/2
                assert state["density"].min() > 0 and thermal.min() > 0
                for field, (q, factor) in physical_fields(state).items():
                    power = record(q, factor, n, t, m, field)
                    if t == TIMES[-1]: cache[m, field] = power
                for op in ("spectral", "centered"):
                    div = divergence(state["magnetic"], op)
                    brms = np.sqrt(mean_square(state["magnetic"]))
                    divstats.append(dict(N=n, time=t, method=m, operator=op,
                        rms=float(np.sqrt(mean_square(div))), maximum=float(abs(div).max()),
                        h_rms_over_Brms=float(np.sqrt(mean_square(div))/(n*brms))))
                    power = record(div, 1, n, t, m, "divergence_"+op)
                    if t == TIMES[-1]: cache[m, "divergence_"+op] = power
                if m in ("glm", "ct"):
                    for refname, refstate, refn in (("athena", ref, nref), ("check", check, ncheck)):
                        if refstate is None: continue
                        for field in ("density", "magnetic", "momentum", "energy"):
                            norms = error_norms(state[field], refstate[field])
                            all_norms.append(dict(N=n, time=t, method=m, field=field, reference_N=refn, **norms))
                            if field in ("density", "magnetic"):
                                power = record(state[field]-refstate[field], 1, n, t, m, "error_"+field, refn)
                                assert np.isclose(power.sum(), norms["absolute_L2"]**2, rtol=2e-12, atol=1e-28)
                                if t == TIMES[-1] and refname == "athena":
                                    cache[m, "error_"+field] = power
                                    # Bands include every Fourier mode; DC is explicitly separate.
                                    bands = (("DC", 0, 1), ("large_1_to_4", 1, 4), ("intermediate_4_to_16", 4, 16), ("small_16_and_up", 16, len(power)))
                                    summaries.append(dict(N=n, method=m, field=field, reference_N=nref,
                                        relative_L1=norms["relative_L1"], relative_L2=norms["relative_L2"], relative_Linf=norms["relative_Linf"],
                                        error_power_fraction={label: float(power[lo:hi].sum()/power.sum()) for label,lo,hi in bands}))
                if m == "ct":
                    frame = int(diag[m]["frame"][np.argmin(abs(diag[m]["time"]-t))])
                    bx, by = load_faces(root, frame, n)
                    centered = np.stack(((bx+np.roll(bx,-1,axis=1))/2, (by+np.roll(by,-1,axis=0))/2),axis=-1)
                    assert np.allclose(centered, state["magnetic"][...,:2], rtol=1e-6, atol=1e-8)
                    div = face_divergence(bx, by)
                    expected_max = diag[m]["div_face_max"][np.argmin(abs(diag[m]["time"]-t))]
                    assert np.isclose(abs(div).max(), expected_max, rtol=1e-12, atol=1e-20)
                    divstats.append(dict(N=n, time=t, method=m, operator="native_face",
                        rms=float(np.sqrt(mean_square(div))), maximum=float(abs(div).max()),
                        h_rms_over_Brms=float(np.sqrt(mean_square(div))/(n*np.sqrt(mean_square(state["magnetic"]))))))
                    power = record(div, 1, n, t, m, "divergence_native_face")
                    if t == TIMES[-1]: cache[m, "divergence_native_face"] = power
            if check is not None:
                for field in ("density", "magnetic"):
                    all_norms.append(dict(N=n, time=t, method="reference_difference", field=field,
                        reference_N=nref, **error_norms(check[field], ref[field])))
                    power = record(ref[field]-check[field], 1, n, t, "reference_difference", "error_"+field, nref)
                    if t == TIMES[-1]: cache["check", "error_"+field] = power
        plot_one(runout, n, nref, ncheck, cache, all_energy)
        print(f"Analyzed {n}²: Parseval, L2/error-spectrum identity, face divergence, positivity passed", flush=True)

    write_csv(out / "spectra.csv", all_spectra)
    write_csv(out / "norms.csv", all_norms)
    write_csv(out / "energies.csv", all_energy)
    write_csv(out / "timings.csv", timings)
    write_csv(out / "divergence.csv", divstats)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    validation = dict(max_relative_parseval_residual=max(checks), spectra_checked=len(checks),
        reference=str(args.reference.resolve()), check_reference=str(args.check_reference.resolve()),
        local_runs=[str(p.resolve()) for p in args.locals], comparison_times=TIMES,
        same_resolution_reference_N=[r["N"] for r in timings if r["method"]=="glm" and r["N"]==nref],
        limitations=[f"{nref}² is provisional, not exact; differences are not rigorous error bounds",
                    "Cell-centered divergence is not CT native face divergence",
                    "Magnetic/kinetic energy changes include ideal energy exchange, not only dissipation",
                    "Restricted thermal remainder includes unresolved kinetic and magnetic energy",
                    "Single timing samples, shared timestep, different compile/output timing scopes; some analysis overlapped the 512² run"],
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("mhd_spectra.py"), Path(__file__).with_name("mhd_2d_cpu.jl"))})
    (out / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, field in zip(axes[:2], ("density", "magnetic")):
        for m in ("glm", "ct"):
            rows = sorted([r for r in summaries if r["method"]==m and r["field"]==field],key=lambda r:r["N"])
            for norm, style in (("L1", "-o"), ("L2", "--s"), ("Linf", ":^")):
                ax.loglog([r["N"] for r in rows], [100*r["relative_"+norm] for r in rows],style,color=COLORS[m],label=m.upper()+" "+norm)
        ax.set(title=field.capitalize()+" error at t=0.5",xlabel="Grid N",ylabel="Relative error (%)")
        ax.grid(alpha=.2,which="both"); ax.legend(fontsize=8)
    for m in ("glm","ct"):
        rows=sorted([r for r in timings if r["method"]==m],key=lambda r:r["N"])
        axes[2].loglog([r["N"] for r in rows],[r["step_seconds"] for r in rows],"-o",color=COLORS[m],label=m.upper())
    axes[2].set(title="CPU stepping cost · one sample",xlabel="Grid N",ylabel="Step time (seconds)")
    axes[2].grid(alpha=.2,which="both"); axes[2].legend()
    for ax in axes:
        ax.set_xticks(sorted(set(r["N"] for r in timings)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(f"Against Athena++ {nref}² · reference averaged to each grid\nFinite-reference differences, not proven convergence rates",fontsize=12)
    fig.savefig(out/"resolution_comparison.png",dpi=160);plt.close(fig)
    print("Saved diagnostics:", out)


if __name__ == "__main__":
    main()
