"""First-look figures of the driven-turbulence pilot (sherlock_turb/README.md).

  python3 scripts/turb_figures.py          -> progress/media/turb/*.png

Reads turb_results/ (the small files of every run, fetched from $SCRATCH/mhd-results-turb): run.toml, series.csv,
spectra.csv, pdf.csv and the final midplane slice. Time averages are over time/T >= FROM, as in summarize.py.
Colours and line styles follow the other pages: hue is the code, line style the set-up within a code. The two
set-ups that run the same scheme in two codes (Tom's Euler kernel, RAMSES HLL) share one dash pattern at different
offsets, so that where their curves coincide both colours show. RAMSES as released (faulty projection of the driving)
is the paler purple.
"""
import csv
import sys
import tomllib
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as pc  # noqa: E402
from plot_style import apply_plot_style  # noqa: E402

from plot_style import field_map  # noqa: E402

IMAGE_MAP = field_map("density")         # the notebook's map for density images (scripts/plot_style.py: FIELD_MAPS)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"turb_results"
OUT = ROOT/"progress"/"media"/"turb"
FROM = 2.0                                   # averages start here, in driving times
GREEN, PURPLE = pc.FAMILY["glm"], pc.FAMILY["ramses"]
PALE = "#8f86c9"       # RAMSES as released. OKLab distance to the purple 23 (22 protan, 23 deutan), to the green 21 (15, 12); contrast on white 3.3
DASH = (4.5, 4.5)
KINDS = {   # kind: colour, line style, marker, label, short label
    "gpu-hydro": (GREEN, (0, DASH), "s", "Tom's Euler kernel: HLL, minmod", "Euler kernel"),
    "gpu-glm": (GREEN, (0, (1, 1)), "x", "Tom's MHD kernel, zero field: LLF", "MHD kernel"),
    "ramsesfix-matched": (PURPLE, (4.5, DASH), "s", "RAMSES HLL, minmod; driving corrected", "RAMSES HLL"),
    "ramsesfix-best": (PURPLE, "-", "o", "RAMSES HLLC, MonCen; driving corrected", "RAMSES HLLC"),
    "ramses-matched": (PALE, (4.5, DASH), "s", "RAMSES HLL as released", "RAMSES HLL, released"),
    "ramses-best": (PALE, "-", "o", "RAMSES HLLC as released", "RAMSES HLLC, released"),
}
LIKE = ("gpu-hydro", "gpu-glm", "ramsesfix-matched", "ramsesfix-best")          # the runs that share one driving
S_LO, S_STEP, S_BINS = -16.0, 0.1, 280                                          # the bins of pdf.csv (measure.jl)


class Run:
    def __init__(self, d):
        self.dir = d;self.name = d.name;self.meta = tomllib.loads((d/"run.toml").read_text())
        self.kind = d.name.split("-M")[0];self.mach = self.meta["mach_target"];self.n = self.meta["N"];self.seed = self.meta["seed"]
        self.T = self.meta["turb_T"]
        rows = list(csv.DictReader((d/"series.csv").open()))
        self.series = {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "finite"}
        self.window = self.series["time_over_T"] >= FROM - 1e-9

    def mean(self, column):
        return float(self.series[column][self.window].mean())

    def power(self):
        """Energy injection rate: slope of the total energy against time over the window."""
        return float(np.polyfit(self.series["time"][self.window], self.series["energy"][self.window], 1)[0])

    def re_eff(self):
        v = float((self.series["mach_mass"]*self.series["cs"])[self.window].mean())
        return v*(self.mean("enstrophy_rho") + 4/3*self.mean("dilatation_rho"))/self.power()

    def spectra(self):
        """{column: time-averaged spectrum over the window}, with k."""
        a = np.genfromtxt(self.dir/"spectra.csv", delimiter=",", names=True)
        keep = a["time"]/self.T >= FROM - 1e-9;nk = self.n//2;out = {"k": a["k"][:nk]}
        for name in a.dtype.names[3:]:out[name] = a[name][keep].reshape(-1, nk).mean(axis=0)
        return out

    def pdf(self):
        """Time-averaged volume-weighted distribution of s = ln(rho/<rho>): (bin centres, probability density)."""
        a = np.genfromtxt(self.dir/"pdf.csv", delimiter=",", names=True)
        keep = a["time"]/self.T >= FROM - 1e-9;frames = len(np.unique(a["frame"][keep]))
        p = np.zeros(S_BINS);np.add.at(p, np.rint((a["s_low"][keep] - S_LO)/S_STEP).astype(int), a["volume_fraction"][keep])
        return S_LO + (np.arange(S_BINS) + .5)*S_STEP, p/frames/S_STEP

    def slice(self):
        f = sorted(self.dir.glob("slice_*.bin"))[-1]
        return np.fromfile(f, dtype="<f4").reshape((self.n, self.n, 9), order="F")[..., 0]


def load():
    runs = [Run(d) for d in sorted(DATA.iterdir()) if (d/"run.toml").is_file() and (d/"series.csv").is_file()]
    return [r for r in runs if r.meta.get("status") == "complete" and r.kind in KINDS]


def pick(runs, kind, mach, n):
    return [r for r in runs if r.kind == kind and r.mach == mach and r.n == n]


def kw(kind, **more):
    c, ls, _, _, _ = KINDS[kind];out = dict(color=c, linestyle=ls, lw=1.5);out.update(more);return out


def legend(fig, kinds, y=1.0, ncol=2, markers=False):
    h = [Line2D([], [], label=KINDS[k][3], marker=KINDS[k][2] if markers else "", ms=4.5, **kw(k)) for k in kinds]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.5, y), ncol=ncol, handlelength=3.6, columnspacing=1.6)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True);path = OUT/f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white");plt.close(fig);print(f"wrote {path.relative_to(ROOT)}")


def cases(runs):
    return [(m, n) for m in sorted({r.mach for r in runs}) for n in sorted({r.n for r in runs})]


# ---------------------------------------------------------------- figures
def mach_time(runs):
    """Rms Mach number against time: every run reaches a steady state within two driving times."""
    machs = sorted({r.mach for r in runs});grids = sorted({r.n for r in runs})
    fig, axes = plt.subplots(len(machs), len(grids), figsize=(7.6, 5.0), sharex=True, sharey="row", squeeze=False)
    for i, m in enumerate(machs):
        for j, n in enumerate(grids):
            ax = axes[i, j];pc.style_axes(ax)
            ax.axvspan(0, FROM, color=pc.PANEL, lw=0)
            for kind in ("ramses-matched", "ramses-best", "ramsesfix-matched", "ramsesfix-best", "gpu-glm", "gpu-hydro"):
                for r in pick(runs, kind, m, n):
                    ax.plot(r.series["time_over_T"], r.series["mach_vol"], **kw(kind, lw=1.2))
            ax.set_xlim(0, 10);ax.set_ylim(0, 1.25*m);ax.set_title(f"target Mach {m}, ${n}^3$")
            if i == 0 and j == 0:ax.text(FROM/2, .07*1.25*m, "not\naveraged", ha="center", va="bottom", color=pc.MUTED, fontsize=7.5)
            if j == 0:ax.set_ylabel("rms Mach number")
            if i == len(machs) - 1:ax.set_xlabel("time / driving time $T$")
    legend(fig, list(KINDS), y=.98, ncol=2);fig.tight_layout();save(fig, "turb-mach-time")


def projection_gap(runs):
    """What the correction of RAMSES's projection does, relative to Tom's Euler kernel (mean of its two seeds)."""
    cs = cases(runs);fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    order = ("ramses-matched", "ramses-best", "ramsesfix-matched", "ramsesfix-best", "gpu-glm", "gpu-hydro")
    for ax, (what, title, lim) in zip(axes, ((lambda r: r.mean("mach_vol"), "rms Mach number", (-6, 3)), (lambda r: r.power(), "power taken up from the driving", (-22, 9)))):
        pc.style_axes(ax);ax.axhline(0, color=pc.MUTED, lw=.8)
        for x, (m, n) in enumerate(cs):
            ref = np.mean([what(r) for r in pick(runs, "gpu-hydro", m, n)])
            for o, kind in enumerate(order):
                for s, r in enumerate(pick(runs, kind, m, n)):
                    ax.plot(x + (o - 2.5)*.13 + (s - .5)*.05*(len(pick(runs, kind, m, n)) > 1), 100*(what(r)/ref - 1), ls="", marker=KINDS[kind][2], ms=5.5,
                            color=KINDS[kind][0], mfc="white" if KINDS[kind][2] != "x" else KINDS[kind][0], mew=1.4)
        ax.set_xticks(range(len(cs)));ax.set_xticklabels([f"Mach {m}\n${n}^3$" for m, n in cs]);ax.set_xlim(-.6, len(cs) - .4);ax.set_ylim(*lim)
        ax.set_title(title);ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f} %" if v else "0"))
    axes[0].set_ylabel("relative to Tom's Euler kernel")
    h = [Line2D([], [], ls="", marker=KINDS[k][2], ms=5.5, color=KINDS[k][0], mfc="white" if KINDS[k][2] != "x" else KINDS[k][0], mew=1.4, label=KINDS[k][3]) for k in order]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.5, .98), ncol=2, columnspacing=1.6);fig.tight_layout();save(fig, "turb-projection-gap")


def mean_spectrum(runs, kind, m, n, column):
    sp = [r.spectra() for r in pick(runs, kind, m, n)]
    return (sp[0]["k"], np.mean([s[column] for s in sp], axis=0)) if sp else (None, None)


def spectra(runs):
    """Velocity spectra, compensated by k^2 (flat for a field of shocks), time-averaged."""
    machs = sorted({r.mach for r in runs});grids = sorted({r.n for r in runs})
    fig, axes = plt.subplots(len(machs), len(grids), figsize=(7.6, 5.0), sharey="row", squeeze=False)
    for i, m in enumerate(machs):
        for j, n in enumerate(grids):
            ax = axes[i, j];pc.style_axes(ax)
            for kind in ("ramsesfix-best", "gpu-glm", "ramsesfix-matched", "gpu-hydro"):
                k, e = mean_spectrum(runs, kind, m, n, "v")
                if k is not None:ax.loglog(k, k**2*e, **kw(kind))
            ax.set_xlim(1, n/2);ax.set_title(f"target Mach {m}, ${n}^3$")
            lo, hi = ax.get_ylim();ax.set_ylim(max(lo, hi*2e-4), hi)
            if j == 0:ax.set_ylabel("$k^2 E_v(k)$  [$c_s^2$]")
            if i == len(machs) - 1:ax.set_xlabel("wavenumber $k$ (box = 1)")
    legend(fig, LIKE, y=.98, ncol=2);fig.tight_layout();save(fig, "turb-spectra")


def compressive_share(runs):
    """Share of the velocity power that is compressive, scale by scale: where the released projection shows."""
    machs = sorted({r.mach for r in runs});n = max(r.n for r in runs)
    fig, axes = plt.subplots(1, len(machs), figsize=(7.6, 3.0), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], machs):
        pc.style_axes(ax)
        for kind in ("ramses-matched", "ramses-best", "ramsesfix-best", "gpu-glm", "ramsesfix-matched", "gpu-hydro"):
            k, c = mean_spectrum(runs, kind, m, n, "v_compressive");_, e = mean_spectrum(runs, kind, m, n, "v")
            if k is not None:ax.semilogx(k, c/e, **kw(kind))
        ax.set_xlim(1, n/2);ax.set_ylim(0, 1);ax.set_title(f"target Mach {m}, ${n}^3$");ax.set_xlabel("wavenumber $k$ (box = 1)")
        ax.axvspan(1, 3, color=pc.PANEL, lw=0);ax.text(1.75, .96, "driven", ha="center", va="top", color=pc.MUTED, fontsize=7.5)
    axes[0, 0].set_ylabel("compressive share of $E_v(k)$")
    legend(fig, list(KINDS), y=.98, ncol=2);fig.tight_layout();save(fig, "turb-compressive-share")


def density_pdf(runs):
    """Distribution of ln(density) by volume, time-averaged, on the finer grid."""
    machs = sorted({r.mach for r in runs});n = max(r.n for r in runs)
    fig, axes = plt.subplots(1, len(machs), figsize=(7.6, 3.0), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], machs):
        pc.style_axes(ax);lo, hi = 0.0, 0.0
        for kind in ("ramses-matched", "ramses-best", "ramsesfix-best", "gpu-glm", "ramsesfix-matched", "gpu-hydro"):
            rs = pick(runs, kind, m, n)
            if not rs:continue
            s = rs[0].pdf()[0];p = np.mean([r.pdf()[1] for r in rs], axis=0);ok = p > 0
            ax.semilogy(s[ok], p[ok], **kw(kind));lo = min(lo, s[p > 1e-5].min());hi = max(hi, s[p > 1e-5].max())
        ax.set_xlim(lo - .3, hi + .3);ax.set_ylim(1e-5, 1);ax.set_title(f"target Mach {m}, ${n}^3$");ax.set_xlabel(r"$s = \ln(\rho/\langle\rho\rangle)$")
    axes[0, 0].set_ylabel("probability density, by volume")
    legend(fig, list(KINDS), y=.98, ncol=2);fig.tight_layout();save(fig, "turb-density-pdf")


def dissipation(runs):
    """Effective Reynolds number against resolution: the same scheme in two codes dissipates alike."""
    machs = sorted({r.mach for r in runs});grids = sorted({r.n for r in runs})
    fig, axes = plt.subplots(1, len(machs), figsize=(7.6, 3.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], machs):
        pc.style_axes(ax)
        for kind in LIKE:
            re = [[r.re_eff() for r in pick(runs, kind, m, n)] for n in grids];c = KINDS[kind][0]
            ax.loglog(grids, [np.mean(v) for v in re], marker=KINDS[kind][2], ms=5, mfc="white" if KINDS[kind][2] != "x" else c, mew=1.3, **kw(kind))
            for n, v in zip(grids, re):       # the single seeds, where there are two
                if len(v) > 1:ax.plot([n]*len(v), v, ls="", marker="_", ms=9, mew=1.1, color=c)
        ax.set_xticks(grids);ax.set_xticklabels([f"${n}^3$" for n in grids]);ax.minorticks_off();ax.set_xlim(grids[0]/1.15, grids[-1]*1.15)
        ticks = [1000, 1500, 2000, 3000, 4000, 6000];ax.set_yticks(ticks);ax.set_yticklabels([f"{t:,}".replace(",", " ") for t in ticks]);ax.set_ylim(900, 7500)
        ax.set_title(f"target Mach {m}");ax.set_xlabel("grid")
    axes[0, 0].set_ylabel("effective Reynolds number")
    legend(fig, LIKE, y=.98, ncol=2, markers=True);fig.tight_layout();save(fig, "turb-dissipation")


def slices(runs):
    """Density in the midplane at the end of each run on the finest grid: one row per Mach number."""
    machs = sorted({r.mach for r in runs});n = max(r.n for r in runs)
    fig, axes = plt.subplots(len(machs), len(LIKE), figsize=(7.6, 1.95*len(machs) + .25), squeeze=False)
    for i, m in enumerate(machs):
        chosen = [(pick(runs, kind, m, n) or [None])[0] for kind in LIKE]
        fields = [None if r is None else np.log10(r.slice().astype(float)) for r in chosen]
        lo, hi = np.percentile(np.concatenate([f.ravel() for f in fields if f is not None]), [0.5, 99.9]);im = None
        for j, (kind, f) in enumerate(zip(LIKE, fields)):
            ax = axes[i, j];pc.image_axes(ax)
            if f is None:pc.blank(ax, "no run")
            else:im = pc.show(ax, np.clip(f, lo, hi), cmap=IMAGE_MAP, vmin=lo, vmax=hi)
            if i == 0:ax.set_title(KINDS[kind][4], fontsize=8.5)
            if j == 0:ax.set_ylabel(f"target Mach {m}")
        cb = fig.colorbar(im, ax=axes[i].tolist(), fraction=.022, pad=.015);cb.set_label(r"$\log_{10}\,\rho$");cb.outline.set_linewidth(.6)
    save(fig, "turb-slices")


def main():
    apply_plot_style()
    matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "axes.labelcolor": pc.INK, "axes.titlecolor": pc.INK,
                                "text.color": pc.INK, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 100,
                                "savefig.dpi": 200, "legend.frameon": False})
    runs = load();print(f"{len(runs)} complete runs in {DATA.relative_to(ROOT)}")
    for figure in (mach_time, projection_gap, spectra, compressive_share, density_pdf, dissipation, slices):figure(runs)


if __name__ == "__main__":
    main()
