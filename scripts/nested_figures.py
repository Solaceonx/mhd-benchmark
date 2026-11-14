"""Figures of Tom Abel's nested-grid code (NestedMHD) on Sherlock's H100s: a nest against a uniform grid at its finest
resolution, driven isothermal Mach-10 MHD (sherlock_nested/compare.jl, movie.jl).
    python3 scripts/nested_figures.py
Reads nested_results/ (copied from Sherlock; sherlock_nested/README.md) and writes progress/media/nested/: the figures,
the movies (from movie.jl's slices, if fetched) and nested.json, the numbers the page quotes. A few seconds; the
movies about a minute each."""
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_style as ps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"nested_results"
CMP = DATA/"compare"
OUT = ROOT/"progress"/"media"/"nested"
LO, HI = 0.375, 0.625                              # the central cube = the nests' finest level (central 1/4 of the box)
EDGES = dict(lnrho=np.linspace(-12, 6, 181), vx=np.linspace(-60, 60, 241), v=np.linspace(0, 80, 161), logb=np.linspace(-4, 1.5, 111))
SIZES = [180, 240, 160, 110]
# Colour = the code (Tom's kernels green, the uniform reference grey); the two nests differ by line style.
CASES = {"nest64": dict(label="nest 3 × 64³ (finest 256³)", color=ps.RUN, ls=(0, (5, 2)), lw=1.8),
         "uni256": dict(label="uniform 256³", color=ps.REFERENCE, ls="-", lw=2),
         "nest128": dict(label="nest 3 × 128³ (finest 512³)", color=ps.RUN, ls="-", lw=2.2)}
LIM = dict(rho=(-6, 3.5), v=(0, 30), b=(-2.2, 0.8))
DEN, MAG, SPEED = ps.field_map("density"), ps.field_map("magnetic"), "cmr.torch"


def tex(s):
    return s.replace("³", "$^3$").replace("×", "$\\times$")


# ---------------------------------------------------------------- records
def snap(case, kind, t):
    a = np.fromfile(CMP/case/f"snap_{kind}_{t:.3f}.bin", np.float32)
    n = round((a.size // 7) ** (1 / 3))
    return a.reshape(7, n, n, n)                   # [field, z, y, x]: ρ, vx, vy, vz, Bx, By, Bz


def snap_times(case):
    return sorted(float(f.name[7:12]) for f in (CMP/case).glob("snap_c_*.bin"))


def midplane(s):
    m = s.shape[1] // 2
    sl = 0.5 * (s[:, m - 1] + s[:, m])             # the two layers either side of z = 1/2
    return np.log(sl[0]), np.sqrt((sl[1:4] ** 2).sum(0)), np.log10(np.sqrt((sl[4:7] ** 2).sum(0)))


def hists(case):
    a = np.fromfile(CMP/case/"hist.bin", np.float64).reshape(-1, 3 * sum(SIZES) + 1)
    out, i = {}, 0
    for where in ("central", "box", "rootres"):
        for name, n in zip(EDGES, SIZES):
            out[where, name] = a[:, i:i + n]; i += n
    out["t"] = a[:, -1]
    return out


def logs():
    """Per case, from the job logs: t, steps per level, M_m, M_v, wall seconds at every output (every 0.01)."""
    runs, cur = {}, None
    for f in sorted((DATA/"logs").glob("nested-compare-*.out")):
        for line in f.read_text().splitlines():
            m = re.match(r"== (\w+): (.*)", line)
            if m and m.group(1) in CASES or (m and m.group(1).startswith("uni")):
                cur = runs.setdefault(m.group(1), dict(t=[], steps=[], mm=[], mv=[], wall=[], header=m.group(2), end=""))
                continue
            m = re.match(r"\s+t ([\d.]+)\s+steps \[([\d, ]+)\]\s+M_m ([\d.]+)\s+M_v ([\d.]+)\s+(\d+) s", line)
            if m and cur is not None:
                cur["t"].append(float(m.group(1))); cur["steps"].append([int(x) for x in m.group(2).split(",")])
                cur["mm"].append(float(m.group(3))); cur["mv"].append(float(m.group(4))); cur["wall"].append(float(m.group(5)))
            elif cur is not None and ("finished:" in line or "CANCELLED" in line):
                cur["end"] = line.strip()
    return {k: {q: (np.array(v) if isinstance(v, list) else v) for q, v in r.items()} for k, r in runs.items()}


# ---------------------------------------------------------------- pictures
def image(ax, img, cmap, lim, ext=(0, 1, 0, 1)):
    ps.frame(ax)
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def composite(case, t, q):
    r, c = midplane(snap(case, "r", t))[q], midplane(snap(case, "c", t))[q]
    F = 4 * r.shape[0]; o = 3 * F // 8
    img = np.kron(r, np.ones((4, 4), np.float32)); img[o:o + c.shape[0], o:o + c.shape[0]] = c
    return img


def boxes(ax, labels=True):
    for lo, hi, lab in ((0.25, 0.75, "level 2"), (LO, HI, "level 3")):
        ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec="white", lw=.9, ls=(0, (4, 3))))
        if labels:
            ax.text(hi - .008, lo + .008, lab, color="white", fontsize=9, ha="right", va="bottom", bbox=dict(fc="black", alpha=.45, lw=0, pad=1.5))


def fig_overview(case, t, name):
    n = snap(case, "r", t).shape[1]
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 8.6), layout="constrained")
    for row, (q, cmap, lim, lab) in enumerate(((0, DEN, LIM["rho"], r"$\ln\rho$"), (2, MAG, LIM["b"], r"$\log_{10}|B|$"))):
        im = image(axs[row, 0], composite(case, t, q), cmap, lim); boxes(axs[row, 0])
        image(axs[row, 1], midplane(snap(case, "c", t))[q], cmap, lim, (LO, HI, LO, HI))
        fig.colorbar(im, ax=axs[row], shrink=.85, label=lab)
    axs[0, 0].set_title(f"whole box: level 1 ({n}$^3$) with level 3 pasted in")
    axs[0, 1].set_title(f"central cube: level 3 ({4 * n}$^3$-equivalent)")
    fig.suptitle(f"Nest 3 $\\times$ {n}$^3$, driven Mach 10, mid-plane $z = 1/2$, $t = {t:g}$", fontsize=13)
    fig.savefig(OUT/f"{name}.png", dpi=150); plt.close(fig)


def fig_slices(t):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.6), layout="constrained")
    cols = [("r", 0, DEN, LIM["rho"], r"$\ln\rho$", "whole box at 64$^3$"), ("c", 0, DEN, LIM["rho"], r"$\ln\rho$", "central cube, 256$^3$ cells"),
            ("c", 1, SPEED, LIM["v"], r"$|v|/c_s$", "central cube, 256$^3$ cells"), ("c", 2, MAG, LIM["b"], r"$\log_{10}|B|$", "central cube, 256$^3$ cells")]
    for j, (kind, q, cmap, lim, lab, title) in enumerate(cols):
        ext = (0, 1, 0, 1) if kind == "r" else (LO, HI, LO, HI)
        for i, case in enumerate(("nest64", "uni256")):
            im = image(axs[i, j], midplane(snap(case, kind, t))[q], cmap, lim, ext)
            if kind == "r":
                axs[i, j].add_patch(Rectangle((LO, LO), HI - LO, HI - LO, fill=False, ec="white", lw=.9, ls=(0, (4, 3))))
        axs[0, j].set_title(title)
        fig.colorbar(im, ax=axs[:, j], orientation="horizontal", shrink=.85, label=lab)
    axs[0, 0].set_ylabel("nest 3 $\\times$ 64$^3$", fontsize=13); axs[1, 0].set_ylabel("uniform 256$^3$", fontsize=13)
    fig.suptitle(f"Same driving, same seed, $t = {t:g}$ (the uniform run's last output), mid-plane $z = 1/2$; "
                 "the uniform run's whole box is drawn as 4$^3$ block averages", fontsize=12)
    fig.savefig(OUT/"nested-slices.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- statistics
def pdf(h, name, sel=slice(None)):
    c = h["central", name][sel].sum(0); e = EDGES[name]
    return 0.5 * (e[1:] + e[:-1]), c / c.sum() / np.diff(e)


def rms(h, where, name):
    c = h[where, name]; e = EDGES[name]; x = 0.5 * (e[1:] + e[:-1])
    x = 10 ** x if name == "logb" else x
    return np.sqrt((c * x ** 2).sum(1) / c.sum(1))


def lowest(h, where):
    return np.array([EDGES["lnrho"][np.nonzero(r)[0][0]] for r in h[where, "lnrho"]])


def legend(fig, cases, extra=()):
    hs = [Line2D([], [], color=CASES[c]["color"], ls=CASES[c]["ls"], lw=CASES[c]["lw"]) for c in cases] + [x[0] for x in extra]
    fig.legend(hs, [tex(CASES[c]["label"]) for c in cases] + [x[1] for x in extra], loc="outside lower center", ncol=len(hs))


def plot(ax, case, x, y, **kw):
    s = CASES[case]; return ax.plot(x, y, color=s["color"], ls=s["ls"], lw=s["lw"], **kw)


def fig_stats(H, tu):
    fig = plt.figure(figsize=(15, 8.6), layout="constrained")
    m = fig.subplot_mosaic([["A", "B", "C"], ["D", "E", "F"]])
    for ax, name, xl in zip((m["A"], m["B"], m["C"]), ("lnrho", "v", "logb"), (r"$\ln\rho$", r"$|v|/c_s$", r"$\log_{10}|B|$")):
        for case in CASES:
            sel = H[case]["t"] <= tu + 1e-9 if case == "nest64" else slice(None)
            x, y = pdf(H[case], name, sel); plot(ax, case, x, np.where(y > 0, y, np.nan))
        ax.set_yscale("log"); ax.set_xlabel(xl); ax.set_ylabel("PDF over the central cube"); ax.set_ylim(1e-6, None)
    m["A"].set_xlim(-10, 5); m["B"].set_xlim(0, 30); m["C"].set_xlim(-3.5, 1.5)
    for ax, where, name, yl in ((m["D"], "rootres", "v", r"rms $|v|/c_s$, whole box"), (m["E"], "rootres", "logb", r"rms $|B|$, whole box"),
                                (m["F"], "central", "logb", r"rms $|B|$, central cube")):
        for case in CASES:
            plot(ax, case, H[case]["t"], rms(H[case], where, name))
        ax.set_xlabel("$t$"); ax.set_ylabel(yl); ax.set_xlim(0.2, 1)
    m["D"].axhline(10, color=ps.MUTED, lw=.8, ls=":")
    legend(fig, CASES)
    fig.suptitle(f"Statistics: the PDFs average every output from $t$ = 0.2 to {tu:g} (nest 3 $\\times$ 64$^3$ and uniform 256$^3$) "
                 "and to 0.5 (nest 3 $\\times$ 128$^3$); whole box at the nest's root resolution", fontsize=12)
    fig.savefig(OUT/"nested-stats.png", dpi=140); plt.close(fig)


def fig_cost(H, L):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
    ax = axs[0]
    for case in CASES:
        r = L[case]; plot(ax, case, r["t"], r["wall"] / 60)
        ax.annotate(f"{r['wall'][-1] / 60:.0f} min", (r["t"][-1], r["wall"][-1] / 60), xytext=(4, 0), textcoords="offset points", va="center", fontsize=10)
    ax.set_yscale("log"); ax.set_xlabel("$t$"); ax.set_ylabel("wall clock [min], one H100"); ax.set_xlim(0.18, 1.1)
    ax = axs[1]
    for case in CASES:
        r = L[case]; st = r["steps"][:, -1]
        plot(ax, case, 0.5 * (r["t"][1:] + r["t"][:-1]), np.diff(r["t"]) / np.diff(st))
    ax.set_yscale("log"); ax.set_xlabel("$t$"); ax.set_ylabel(r"mean time step of the finest grid"); ax.set_xlim(0.2, 1)
    ax = axs[2]
    ax.axhspan(-13, -12, color=ps.PANEL, lw=0)
    ax.text(0.98, -12.5, r"below the histograms ($\rho < 6\times10^{-6}$)", ha="right", va="center", color=ps.MUTED, fontsize=9, transform=ax.get_yaxis_transform())
    for case in CASES:
        plot(ax, case, H[case]["t"], lowest(H[case], "central"))
    ax.plot(H["uni256"]["t"], lowest(H["uni256"], "box"), color=ps.REFERENCE, lw=1, ls=":")
    ax.set_ylim(-13, -2.5); ax.set_xlabel("$t$"); ax.set_ylabel(r"lowest $\ln\rho$, central cube"); ax.set_xlim(0.2, 1)
    legend(fig, CASES, [(Line2D([], [], color=ps.REFERENCE, lw=1, ls=":"), "uniform 256$^3$, whole box (right)")])
    fig.suptitle("Cost: one H100 each; outputs every 0.01", fontsize=12)
    fig.savefig(OUT/"nested-cost.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- spectra
def shells(f, window):
    """Shell-summed power, ½ Σ|f̂|² over the components of f (c, n, n, n); for a window, the mean is removed first and the power is
    divided by the window's mean square. Wave numbers in units of the box's fundamental (cube of side L/4: k = 4, 8, ...)."""
    n = f.shape[-1]
    if window:
        w1 = np.hanning(n + 2)[1:-1]; w = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
        f = (f - f.mean(axis=(1, 2, 3), keepdims=True)) * w
    F = np.fft.fftn(f, axes=(1, 2, 3)) / n ** 3
    P = 0.5 * (np.abs(F) ** 2).sum(0) / ((w ** 2).mean() if window else 1)
    k = np.fft.fftfreq(n) * n
    kk = np.rint(np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)).astype(int)
    E = np.bincount(kk.ravel(), P.ravel(), minlength=n)[:n // 2 + 1]
    return E


def spectra(case, kind, times):
    """Time-averaged spectra of v, B and ρ; kind "r" the whole box (periodic), "c" the central cube (Hann window)."""
    acc = None
    for t in times:
        s = snap(case, kind, t).astype(np.float64)
        e = [shells(s[1:4], kind == "c"), shells(s[4:7], kind == "c"), shells(s[0:1], kind == "c")]
        acc = e if acc is None else [a + b for a, b in zip(acc, e)]
    fac = 4 if kind == "c" else 1
    k = fac * np.arange(len(acc[0]))
    return k, [a / len(times) / fac for a in acc]          # E(k) per unit k in box units


def variance_per_lnk(k, E):
    """k E(k) / σ²: the share of the variance per unit ln k (= 4π k³ P(k) / σ² for the 3-D power P); the k = 0 mean is left out,
    so its area over ln k is 1."""
    dk = k[1] - k[0]
    return k * E / (E[1:].sum() * dk)


def fig_spectra(tu):
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    tsel = {c: [t for t in snap_times(c) if (c != "nest64" or t <= tu + 1e-9)] for c in CASES}
    out = {}
    for row, kind in enumerate(("r", "c")):
        for case in CASES:
            k, (ev, eb, er) = spectra(case, kind, tsel[case]); out[case, kind] = (k, ev, eb, er)
            sel = k >= (1 if kind == "r" else 4)
            for ax, E in zip(axs[row], (ev, eb, er)):
                plot(ax, case, k[sel], variance_per_lnk(k, E)[sel])
        for ax in axs[row]:
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$k$ (box units)")
            if kind == "c":
                ax.axvspan(3, 12, color=ps.PANEL, lw=0); ax.set_xlim(3, None)
        axs[row, 0].set_ylabel(r"$k\,E(k)/\sigma^2 = 4\pi k^3 P(k)/\sigma^2$" + (", whole box" if kind == "r" else ", central cube"))
    axs[1, 0].text(0.02, 0.04, "shaded: the window\nsets these", transform=axs[1, 0].transAxes, fontsize=9, color=ps.MUTED)
    for j, q in enumerate(("velocity", "magnetic field", "density")):
        axs[0, j].set_title(q)
    legend(fig, CASES)
    fig.suptitle(f"Normalized power spectra: the share of each field's variance per unit ln $k$ (area under a curve over ln $k$ = 1). "
                 f"Top: the whole periodic box at the nest's root resolution\n(uniform: 4$^3$ block averages). Bottom: the central cube at the "
                 f"finest resolution, Hann-windowed. Snapshots every 0.05 from $t$ = 0.2 to {tu:g} (nest 3 $\\times$ 64$^3$, uniform) "
                 "and to 0.5 (nest 3 $\\times$ 128$^3$)", fontsize=11.5)
    fig.savefig(OUT/"nested-spectra.png", dpi=140); plt.close(fig)
    return out


# ---------------------------------------------------------------- movies (movie.jl's slices)
def movies():
    made = {}
    for case, d in (("nest64", DATA/"movie"/"nest64"), ("nest128", DATA/"movie"/"nest128"), ("deep192", DATA/"big"/"deep192")):
        big = case == "deep192";record = d/("run.toml" if big else "movie.toml")
        if not record.is_file() or not (d/"slices.bin").is_file():
            continue
        mp4 = d/f"{case}_movie.mp4"
        if not mp4.is_file() or mp4.stat().st_mtime < (d/"slices.bin").stat().st_mtime:
            subprocess.run([sys.executable, str(ROOT/"sherlock_nested"/"plot_movie.py"), str(d)], check=True)
        shutil.copyfile(mp4, OUT/f"{case}-movie.mp4")
        # the poster as a JPEG: a PNG of a frame is about 1 MB
        subprocess.run([ps.ffmpeg_path(), "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", str(mp4), "-frames:v", "1",
                        "-update", "1", "-q:v", "3", str(OUT/f"{case}-movie.jpg")], check=True)
        meta = tomllib.load(open(record, "rb"))
        made[case] = dict(frames=meta["frames"], t_last=meta["t"] if big else meta["t_last"], dt=0.002 if big else meta["dt_frame"],
                          finished=meta["finished"], wall=meta["wall"], steps=meta["steps"], mb=round(mp4.stat().st_size / 1e6, 1))
    return made


# ---------------------------------------------------------------- the big run (bigrun.jl): root 128³ + 3 levels of 192³, capped
BIG = DATA/"big"/"deep192"
LEVEL_STYLE = [dict(color=ps.REFERENCE, ls=(0, (1, 1.5)), lw=1.6), dict(color=ps.RUN, ls=(0, (5, 2)), lw=1.6),
               dict(color=ps.RUN, ls="-", lw=1.6), dict(color=ps.RUN, ls="-", lw=2.8)]


def big_series():
    rows = [l.split(",") for l in (BIG/"series.csv").read_text().splitlines()];head = rows[0]
    a = np.array(rows[1:], float);return {h: a[:, i] for i, h in enumerate(head)}


def fig_big(L):
    """The big run: where each level's variance lives, its time steps beside the uncapped 3 × 128³ nest, the mass the cap adds."""
    meta = tomllib.load(open(BIG/"run.toml", "rb"));z = np.load(BIG/"spectra.npz");ser = big_series();nl = meta["levels"]
    sel = z["times"] >= 0.2 - 1e-9
    names = [f"level {l + 1}: {int(round(meta['root'] * 2 ** l))}$^3$-equivalent over {['the whole box', '3/4', '3/8', '3/16'][l]}" for l in range(nl)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    share = {}
    for j, key in enumerate(("E_v", "E_B", "E_rho")):
        ax = axs[0, j]
        for l in range(nl):
            E = np.nanmean(z[key][l][sel], axis=0);n = int(z["n"][l]);E = E[:n // 2 + 1]
            k = np.arange(n // 2 + 1) / z["frac"][l];dk = k[1] - k[0]           # box units; E per shell -> per unit k
            y = k * (E / dk) / (E[1:].sum());ok = (k >= (1 if l == 0 else 3 / z["frac"][l]))
            ax.plot(k[ok], y[ok], **LEVEL_STYLE[l])
            band = (k >= 32) & (k <= 96)
            share[key, l] = float(E[band].sum() / E[1:].sum()) if band.sum() > 2 and k[-1] >= 96 else None
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("$k$ (box units)")
        ax.set_title(("velocity", "magnetic field", "density")[j])
    axs[0, 0].set_ylabel(r"$k\,E(k)/\sigma^2$ of each level's own patch")
    ax = axs[1, 0]
    for l in range(nl):
        st = ser[f"steps_L{l + 1}"];ax.plot(0.5 * (ser["t"][1:] + ser["t"][:-1]), np.diff(ser["t"]) / np.diff(st), **LEVEL_STYLE[l])
    r = L["nest128"];ax.plot(0.5 * (r["t"][1:] + r["t"][:-1]), np.diff(r["t"]) / np.diff(r["steps"][:, -1]), color=ps.FAMILY["ct"], lw=1.4)
    ax.set_yscale("log");ax.set_xlabel("$t$");ax.set_ylabel("mean time step between outputs");ax.set_xlim(0, 0.5)
    ax = axs[1, 1]
    ax.plot(ser["t"], 1e5 * (ser["mass"] - meta["mass0"]) / meta["mass0"], color=ps.INK, lw=1.8)
    ax.set_xlabel("$t$");ax.set_ylabel(r"mass added by the cap and floor [$10^{-5}$ of the total]");ax.set_xlim(0, 0.5)
    ax = axs[1, 2]
    ax.plot(ser["t"], ser["M_m"], color=ps.INK, lw=1.8);ax.plot(ser["t"], ser["M_v"], color=ps.MUTED, lw=1.2, ls="--")
    ax.axhline(10, color=ps.MUTED, lw=.8, ls=":");ax.set_xlabel("$t$");ax.set_ylabel("rms Mach number (mass-weighted; dashed: volume)");ax.set_xlim(0, 0.5)
    hs = [Line2D([], [], **st) for st in LEVEL_STYLE[:nl]] + [Line2D([], [], color=ps.FAMILY["ct"], lw=1.4)]
    fig.legend(hs, names + ["finest level of the 3 $\\times$ 128$^3$ nest without the cap (time steps)"], loc="outside lower center", ncol=3)
    fig.suptitle(f"The big run: root {meta['root']}$^3$ + {nl - 1} levels of {meta['n']}$^3$, Alfvén speed capped at {meta['va_max']:g} $c_s$. "
                 "Top: share of each field's variance per ln $k$, by level ($t$ = 0.2–0.5). Bottom: time steps, mass added, Mach number", fontsize=12)
    fig.savefig(OUT/"nested-big.png", dpi=140);plt.close(fig)
    dts = {l + 1: np.diff(ser["t"]) / np.diff(ser[f"steps_L{l + 1}"]) for l in range(nl)}
    return dict(root=meta["root"], n=meta["n"], levels=nl, wall=meta["wall"], steps=meta["steps"], va_max=meta["va_max"], t=meta["t"],
                segments=meta["segments"], mass_added=meta["mass_added_fraction"], divb=max(meta["divb"]), precision=meta["precision"],
                dt_min={l: float(v.min()) for l, v in dts.items()}, dt_median={l: float(np.median(v)) for l, v in dts.items()},
                mach=float(ser["M_m"][ser["t"] >= 0.2].mean()),
                share={f"{key[2:]}_L{l + 1}": v for (key, l), v in share.items() if v is not None},
                cell_updates=float(sum(s * (meta["root"] if i == 0 else meta["n"]) ** 3 for i, s in enumerate(meta["steps"]))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.apply_movie_style(11)
    H = {c: hists(c) for c in CASES}; L = logs(); tu = float(H["uni256"]["t"][-1])
    fig_overview("nest128", snap_times("nest128")[-1], "nested-overview")
    fig_overview("nest64", 1.0, "nested-overview-64")
    fig_slices(tu)
    fig_stats(H, tu)
    fig_cost(H, L)
    S = fig_spectra(tu)
    # numbers the page quotes, all from the records above
    run = {c: tomllib.load(open(CMP/c/"run.toml", "rb")) for c in ("nest64", "nest128")}
    gpu = tomllib.load(open(DATA/"gpu_check"/"gpu_check.toml", "rb"))
    same = H["nest64"]["t"] <= tu + 1e-9
    num = dict(
        t_uniform=tu,
        runs={c: dict(header=L[c]["header"], end=L[c]["end"], t=float(L[c]["t"][-1]) if len(L[c]["t"]) else None,
                      wall=float(L[c]["wall"][-1]) if len(L[c]["wall"]) else None, steps=L[c]["steps"][-1].tolist() if len(L[c]["steps"]) else None)
              for c in L},
        cell_updates={c: run[c]["cell_updates"] for c in run}, divb={c: max(run[c]["divb"]) for c in run},
        level_wall={c: run[c]["level_wall"] for c in run},
        mach_box={c: float(rms(H[c], "rootres", "v")[(H[c]["t"] <= tu + 1e-9) if c == "nest64" else slice(None)].mean()) for c in CASES},
        brms_box_at_tu={c: float(rms(H[c], "rootres", "logb")[np.argmin(abs(H[c]["t"] - tu))]) for c in ("nest64", "uni256")},
        brms_box_mean={c: float(rms(H[c], "rootres", "logb")[(H[c]["t"] <= tu + 1e-9) if c == "nest64" else slice(None)].mean()) for c in ("nest64", "uni256")},
        brms_central_range={c: [float(x) for x in (rms(H[c], "central", "logb")[same if c == "nest64" else slice(None)].min(),
                                                   rms(H[c], "central", "logb")[same if c == "nest64" else slice(None)].max())] for c in ("nest64", "uni256")},
        lowest_box_uni=float(lowest(H["uni256"], "box").min()), lowest_central={c: float(lowest(H[c], "central").min()) for c in CASES},
        wall_02_to_tu={c: float(L[c]["wall"][np.argmin(abs(L[c]["t"] - tu))] - L[c]["wall"][0]) for c in ("nest64", "uni256")},
        steps_to_tu={c: L[c]["steps"][np.argmin(abs(L[c]["t"] - tu))].tolist() for c in ("nest64", "uni256")},
        dt_finest={c: dict(min=float((np.diff(L[c]["t"]) / np.diff(L[c]["steps"][:, -1])).min()),
                           median=float(np.median(np.diff(L[c]["t"]) / np.diff(L[c]["steps"][:, -1])))) for c in CASES},
        gpu=dict(agreement=[dict(T=a["T"], mode=a["mode"], riemann=a["riemann"], diff=max(a["max_rel_diff_by_level"]),
                                 same_steps=a["steps_cpu"] == a["steps_gpu"]) for a in gpu["agreement"]],
                 speed=[dict(setup=s["setup"], T=s["T"], mode=s["mode"], rate=s["cell_updates_per_s"]) for s in gpu["speed"]]),
        movies=movies(),
        big=fig_big(L) if (BIG/"spectra.npz").is_file() else None,
    )
    # share of each field's variance at small scales: central cube k = 32–96 (finest resolution), whole box k = 8–24
    def share(case, kind, lo, hi):
        k, *E = S[case, kind]; dk = k[1] - k[0]; sel = (k >= lo) & (k <= hi)
        return {q: float(e[sel].sum() / e[1:].sum()) for q, e in zip(("v", "B", "rho"), E)}
    num["variance_share_central"] = {c: share(c, "c", 32, 96) for c in CASES}
    num["variance_share_box"] = {c: share(c, "r", 8, 24) for c in CASES}
    num.pop("spectra_ratio_central", None)
    (OUT/"nested.json").write_text(json.dumps(num, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in num.items() if k not in ("gpu", "runs")}, indent=1, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
