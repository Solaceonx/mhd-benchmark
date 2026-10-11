"""Kritsuk et al. (2011, ApJ 737, 13), "Comparing numerical methods for isothermal magnetized supersonic turbulence":
their Figs. 1-5 and Table 1 with our codes, from the Mach 9 decay test (decay_supersonic).

Theirs: nine codes start from one saturated, driven Mach 9 state (a 1000^3 snapshot) and decay for 0.2 box crossing
times at 256^3 and 512^3. Ours: every code starts from the same smooth random velocity field at rms Mach 9 with a
uniform field B0 = 0.3 (the same beta0 = 22), gamma = 1.01, and decays for the same time; the driven start is stage 3
of the sweep. So the curves are not comparable in value with the paper's, only in kind.

Spectra come from spectra.csv (src/spectra.jl): shell sums E(k) over round(|k|) = k, k = 1 .. N/2, with
sum_k E = <f^2>/2 over those shells. The paper's P(f, k) sums to <f^2>, so P = 2 E.

A run whose state stops being finite (Athena++ PPM/HLLD/RK3 here: NaN from the third output on, while the code runs
to the end and is recorded as complete) is cut at its last finite output and marked like a solver stop.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_common import ALL, INK, MUTED, REFERENCE_INK, color, handles, label, line_kw, plain, style_axes

PAPER = "kritsuk"
TEST = "decay_supersonic"
REF = "athenak_reference"
START = ("Every code starts from the same smooth random velocity field at rms Mach 9 with a uniform field (β₀ = 22), not from the paper's "
         "driven, saturated state, so the field is first amplified here; values are not comparable with the paper's, the kind of "
         "measurement is. γ = 1.01 instead of isothermal.")


def moments(F, m, n, seed=42):
    """Per finite output: time, <u^2>/2, <B^2>/2, <rho u^2>/2 and the small-scale sums, or None."""
    sw = F.sw;sp = sw.spectra(TEST, m, n, seed);d = sw.diag(TEST, m, n, seed)
    if sp is None or d is None:return None
    good = sw.finite_rows(TEST, m, n, seed);row = {int(f): i for i, f in enumerate(d["frame"])}
    frames = [f for f in sorted(set(sp) & set(row)) if row[f] < good]
    if not frames:return None
    out = {k: [] for k in ("t", "ek", "em", "ekrho", "omega", "j2")}
    for f in frames:
        s = sp[f];k2 = (2*np.pi*s["k"])**2;i = row[f]
        out["t"].append(d["time"][i]);out["ek"].append(s["v"].sum());out["em"].append(d["magnetic"][i]);out["ekrho"].append(d["kinetic"][i])
        out["omega"].append(2*(k2*(s["v_solenoidal"] + 4/3*s["v_compressive"])).sum());out["j2"].append(2*(k2*s["b"]).sum())
    return {k: np.array(v) for k, v in out.items()}


def spectrum(F, m, n, frame, seed=42):
    """The spectra of one output, or None when it is not here or the state was no longer finite."""
    sp = F.sw.spectra(TEST, m, n, seed)
    if sp is None or frame not in sp:return None
    good = F.sw.finite_rows(TEST, m, n, seed)
    return sp[frame] if good is None or frame < good else None


def draw_methods(F, ax, n, value, drawn, methods=ALL, seed=42, reference=True, positive=False):
    """One line per method of value(moments) against time; the 512^3 reference in ink when n is smaller."""
    clean = (lambda y: np.where(y > 0, y, np.nan)) if positive else (lambda y: y)
    if reference and n < F.sw.reference_n(TEST, REF):
        mo = moments(F, REF, F.sw.reference_n(TEST, REF), seed)
        if mo is not None:ax.plot(mo["t"], clean(value(mo)), color=REFERENCE_INK, lw=2.2, alpha=.5, solid_capstyle="round")
    for m in methods:
        mo = moments(F, m, n, seed)
        if mo is None:
            F.miss(TEST, m, n, seed);continue
        y = clean(value(mo));ax.plot(mo["t"], y, **line_kw(m));drawn.add(m)
        if F.stopped(TEST, m, n, seed):
            ax.plot(mo["t"][-1], y[-1], marker="x", color=color(m), ms=6, mew=1.5, clip_on=False);F.miss(TEST, m, n, seed)


def legend(fig, F, n, y, drawn, extra=()):
    """Only the methods that have a curve; the others are listed with the figure."""
    hs = handles([m for m in ALL if m in drawn])
    if n < F.sw.reference_n(TEST, REF):
        hs.append(Line2D([], [], color=REFERENCE_INK, lw=2.2, alpha=.5, label=f"AthenaK PLM/HLLE at {F.sw.reference_n(TEST, REF)}$^3$ (reference)"))
    hs.append(Line2D([], [], color=MUTED, marker="x", ms=6, mew=1.5, lw=0, label="run ended here (solver stop or NaN)"))
    hs += list(extra)
    fig.legend(handles=hs, loc="lower center", ncol=4, bbox_to_anchor=(.5, y), handlelength=3.2, columnspacing=1.4)


def fig1(F, n):
    """Fig. 1: mean specific kinetic energy, magnetic energy, sonic and Alfvenic rms Mach numbers against time."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), sharex=True, gridspec_kw=dict(wspace=.22, hspace=.1));drawn = set()
    panels = [(r"$\log_{10}\,\langle u^2\rangle/2$", lambda mo: np.log10(mo["ek"])), (r"$\log_{10}\,\langle B^2\rangle/2$", lambda mo: np.log10(mo["em"])),
              (r"$M_s = \langle u^2\rangle^{1/2}/c_s$", lambda mo: np.sqrt(2*mo["ek"])),
              (r"$M_A = (2\langle\rho u^2\rangle/\langle B^2\rangle)^{1/2}$", lambda mo: np.sqrt(2*mo["ekrho"]/mo["em"]))]
    for ax, (name, value) in zip(axes.flat, panels):
        style_axes(ax);draw_methods(F, ax, n, value, drawn);ax.set_ylabel(name);ax.set_xlim(0, .2)
    axes[1, 1].set_yscale("log");axes[1, 1].set_yticks([2, 5, 10, 20, 50]);axes[1, 1].set_yticklabels(["2", "5", "10", "20", "50"]);axes[1, 1].minorticks_off()
    for ax in axes[1]:ax.set_xlabel("t")
    legend(fig, F, n, .9, drawn)
    return fig


def fig2(F, n):
    """Fig. 2: 2 Omega + 4/3 Delta and J^2 against time (spectral sums)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw=dict(wspace=.22));drawn = set()
    for ax, (name, key) in zip(axes, ((r"$2\Omega + \frac{4}{3}\Delta$", "omega"), (r"$J^2 = \langle|\nabla\times\mathbf{B}|^2\rangle$", "j2"))):
        style_axes(ax);draw_methods(F, ax, n, lambda mo, key=key: mo[key], drawn, positive=True)
        ax.set_ylabel(name);ax.set_xlabel("t");ax.set_xlim(0, .2);ax.set_yscale("log")
    legend(fig, F, n, .93, drawn)
    return fig


def fig3(F, n, frames=(1, 3, 10)):
    """Fig. 3: velocity spectra compensated by k^(5/3) and magnetic spectra at three times."""
    times = F.sw.times(TEST);drawn = set()
    fig, axes = plt.subplots(len(frames), 2, figsize=(10.4, 3.25*len(frames)), sharex=True, gridspec_kw=dict(wspace=.2, hspace=.1))
    big = F.sw.reference_n(TEST, REF)
    for i, f in enumerate(frames):
        for j, (key, comp, name) in enumerate((("v", 5/3, r"$\log_{10}\,k^{5/3}P(u,k)$"), ("b", 0.0, r"$\log_{10}\,P(B,k)$"))):
            ax = axes[i, j];style_axes(ax)
            if n < big:
                s = spectrum(F, REF, big, f)
                if s is not None:ax.plot(np.log10(s["k"]), np.log10(2*s[key]*s["k"]**comp), color=REFERENCE_INK, lw=2.2, alpha=.5)
            for m in ALL:
                s = spectrum(F, m, n, f)
                if s is None:
                    if i == len(frames) - 1 and j == 0:F.miss(TEST, m, n)
                    continue
                ax.plot(np.log10(s["k"]), np.log10(np.maximum(2*s[key]*s["k"]**comp, 1e-300)), **line_kw(m, lw=1.2));drawn.add(m)
            ax.set_ylabel(name);ax.text(.03, .06, f"t = {times[f]:.2f}", transform=ax.transAxes, color=INK)
            ax.set_xlim(0, np.log10(n/2))
            ys = np.concatenate([l.get_ydata() for l in ax.lines]) if ax.lines else np.array([0.0, 1.0])
            ys = ys[np.isfinite(ys)]
            if ys.size:ax.set_ylim(max(ys.min(), ys.max() - 6.5) - .2, ys.max() + .3)
    for ax in axes[-1]:ax.set_xlabel(r"$\log_{10}\,k/k_{\min}$")
    legend(fig, F, n, .89, drawn)
    return fig


def band(k, ratio, nyquist, tolerance=.25):
    """Fraction of the Nyquist wavenumber up to which |ratio - 1| stays within the tolerance (from k = 2 on)."""
    bad = np.where((np.abs(ratio - 1) > tolerance) & (k >= 2))[0]
    return float(k[bad[0]])/nyquist if bad.size else 1.0


def bandwidths(F, m, n, frame, seed=42):
    """(velocity, magnetic) effective bandwidth of a run against the reference at one output, or (None, None)."""
    big = F.sw.reference_n(TEST, REF);s = spectrum(F, m, n, frame, seed);ref = spectrum(F, REF, big, frame, seed)
    if s is None or ref is None:return None, None
    k = s["k"];return band(k, s["v"]/ref["v"][:len(k)], n/2), band(k, s["b"]/ref["b"][:len(k)], n/2)


def fig4(F, n=256, frames=(1, 10)):
    """Fig. 4: spectra divided by the reference's; the 25 % tolerance gives the effective bandwidth."""
    times = F.sw.times(TEST);big = F.sw.reference_n(TEST, REF);drawn = set()
    fig, axes = plt.subplots(len(frames), 2, figsize=(10.4, 3.3*len(frames)), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw=dict(wspace=.08, hspace=.12))
    for i, f in enumerate(frames):
        ref = spectrum(F, REF, big, f)
        for j, (key, name) in enumerate((("v", r"$P(u,k)/P_{\rm ref}(u,k)$"), ("b", r"$P(B,k)/P_{\rm ref}(B,k)$"))):
            ax = axes[i, j];style_axes(ax)
            ax.axhline(1, color=REFERENCE_INK, lw=1.0);ax.axhline(.75, color=MUTED, lw=.8, ls=(0, (1, 2)));ax.axhline(1.25, color=MUTED, lw=.8, ls=(0, (1, 2)))
            if j == 0:ax.set_ylabel(name)
            else:ax.text(1.02, .5, name, transform=ax.transAxes, rotation=90, va="center", fontsize=9)
            ax.text(.03, .06, f"t = {times[f]:.2f}", transform=ax.transAxes, color=INK)
            if ref is None:
                ax.text(.5, .5, F.why(TEST, REF, big).replace("\n", " "), transform=ax.transAxes, ha="center", color=MUTED);continue
            for m in ALL:
                s = spectrum(F, m, n, f)
                if s is None:
                    if j == 0:F.miss(TEST, m, n)
                    continue
                k = s["k"];ax.plot(np.log10(k), s[key]/ref[key][:len(k)], **line_kw(m, lw=1.2));drawn.add(m)
            ax.set_xlim(0, np.log10(n/2));ax.set_ylim(0, 1.45)
    if spectrum(F, REF, big, frames[0]) is None:F.miss(TEST, REF, big)
    for ax in axes[-1]:ax.set_xlabel(r"$\log_{10}\,k/k_{\min}$")
    legend(fig, F, big, .9 if len(frames) > 1 else .93, drawn, extra=[Line2D([], [], color=MUTED, lw=.8, ls=(0, (1, 2)), label="±25 % tolerance")])
    return fig


def fig5(F, n, frames=(1, 10)):
    """Fig. 5: ratio of dilatational to solenoidal velocity power against wavenumber."""
    times = F.sw.times(TEST);big = F.sw.reference_n(TEST, REF);drawn = set()
    fig, axes = plt.subplots(1, len(frames), figsize=(10.4, 3.7), sharey=True, gridspec_kw=dict(wspace=.08))
    for ax, f in zip(axes, frames):
        style_axes(ax)
        if n < big:
            s = spectrum(F, REF, big, f)
            if s is not None:ax.plot(np.log10(s["k"]), np.log10(s["v_compressive"]/s["v_solenoidal"]), color=REFERENCE_INK, lw=2.2, alpha=.5)
        for m in ALL:
            s = spectrum(F, m, n, f)
            if s is None:
                if f == frames[-1]:F.miss(TEST, m, n)
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ax.plot(np.log10(s["k"]), np.log10(s["v_compressive"]/s["v_solenoidal"]), **line_kw(m, lw=1.2));drawn.add(m)
        ax.set_xlabel(r"$\log_{10}\,k/k_{\min}$");ax.set_xlim(0, np.log10(n/2));ax.text(.03, .06, f"t = {times[f]:.2f}", transform=ax.transAxes, color=INK)
    axes[0].set_ylabel(r"$\log_{10}\,\chi(k)$,  $\chi = P(u_d,k)/P(u_s,k)$")
    legend(fig, F, n, .93, drawn)
    return fig


COLUMNS = [("method", "Method"), ("ek512", "E_K / ref, 512³"), ("em512", "E_M / ref, 512³"), ("ek", "E_K / ref, 256³"), ("em", "E_M / ref, 256³"),
           ("omega", "2Ω + 4/3Δ / ref, 256³, t = 0.02"), ("j2", "J² / ref, 256³, t = 0.02"), ("ub1", "u bandwidth, t = 0.02"),
           ("ub", "u bandwidth, t = 0.2"), ("bb1", "B bandwidth, t = 0.02"), ("bb", "B bandwidth, t = 0.2"), ("chi10", "χ at k ≈ 10, 512³"),
           ("chi", "χ̄ (k > 100), 512³"), ("note", "")]


def table(F, n=256, big_n=512, seed=42):
    """Table 1: the paper's selected numbers for our runs, against the 512^3 reference of the same seed."""
    sw = F.sw;big = sw.reference_n(TEST, REF);ref = moments(F, REF, big, seed);last = sw.frames(TEST) - 1;rows = []
    for m in ALL:
        row = {k: None for k, _ in COLUMNS};row["method"] = plain(label(m));notes = []
        for grid, ek, em in ((big_n, "ek512", "em512"), (n, "ek", "em")):
            mo = moments(F, m, grid, seed)
            if mo is None or len(mo["t"]) <= last:
                notes.append(f"{grid}³: {F.why(TEST, m, grid, seed).replace(chr(10), ' ')}")
            elif ref is not None:row[ek] = float(mo["ek"][-1]/ref["ek"][-1]);row[em] = float(mo["em"][-1]/ref["em"][-1])
            if grid == n and mo is not None and ref is not None and len(mo["t"]) > 1:
                row["omega"] = float(mo["omega"][1]/ref["omega"][1]);row["j2"] = float(mo["j2"][1]/ref["j2"][1])
        row["ub1"], row["bb1"] = bandwidths(F, m, n, 1, seed);row["ub"], row["bb"] = bandwidths(F, m, n, last, seed)
        s = spectrum(F, m, big_n, last, seed)
        if s is not None:
            sel = s["k"] > 100;row["chi"] = float(s["v_compressive"][sel].sum()/s["v_solenoidal"][sel].sum())
            mid = (s["k"] >= 8) & (s["k"] <= 12);row["chi10"] = float(np.mean(s["v_compressive"][mid]/s["v_solenoidal"][mid]))
        row["note"] = "; ".join(notes);rows.append(row)
    # What the paper prints (its Table 1 and sec. 5.5: 512^3 against its own 1024^3 reference; bandwidths at 256^3, t = 0.02):
    # the one code we share, and the spread of its nine codes.
    rows.append(dict(method="In the paper: RAMSES", ek512=1.069, em512=.87, ek=None, em=None, omega="1.07 (512³)", j2="1.18 (512³)", ub1=.24, ub=None,
                     bb1=.09, bb=None, chi10=None, chi=.33, note="driven start; reference: PPML at 1024³"))
    rows.append(dict(method="In the paper: its nine codes", ek512="1.000 – 1.077", em512="0.70 – 0.94", ek=None, em=None, omega="0.76 – 1.93 (512³)",
                     j2="0.79 – 1.46 (512³)", ub1="0.15 – 0.28", ub=None, bb1="0.07 – 0.20", bb=None, chi10="≈ 0.33", chi="0.27 – 0.86",
                     note="driven start; reference: PPML at 1024³"))
    F.notes = []
    return rows, ref is not None


def figures():
    def one(ident, fig_no, maker, title, caption, differs, n):
        def run(F):
            fig = maker(F, n)
            F.save(fig, ident, paper=PAPER, paper_fig=fig_no, kind="analogue", test=TEST, title=title.format(n=n), caption=caption, differs=differs)
        return run

    def bandwidth(F):
        F.save(fig4(F), "kritsuk-4", paper=PAPER, paper_fig="Fig. 4", kind="analogue", test=TEST,
               title="Mach 9 decay: spectra of the 256³ runs divided by the 512³ reference",
               caption=("Velocity (left) and magnetic (right) power spectra of each 256³ run divided by the 512³ reference's at the same time, "
                        "at the first output (t = 0.02, the paper's time) and at the end. Dotted lines: the paper's ±25 % tolerance; the "
                        "wavenumber where a curve leaves it, as a fraction of the Nyquist wavenumber, is the effective bandwidth (table below). "
                        "At t = 0.02 our smooth start has only just formed its first shocks, so t = 0.2 is the fairer time to read."),
               differs=START + " The paper's reference is its least diffusive code at 1024³ filtered to 256³; ours is AthenaK PLM/HLLE at 512³, "
                               "unfiltered, which is itself a second-order scheme.")

    def numbers(F):
        rows, has_ref = table(F)
        F.entries.append(dict(id="kritsuk-table-1", paper=PAPER, paper_fig="Table 1", kind="analogue", test=TEST, file=None, missing=[],
                              title="Mach 9 decay: selected numbers against the 512³ reference (seed 42)",
                              caption=("E_K, E_M: mean specific kinetic and magnetic energy at t = 0.2 over the reference's, for the 512³ runs "
                                       "(code against code at one grid) and the 256³ runs. 2Ω + 4/3Δ and J²: at t = 0.02 over the reference's. "
                                       "Bandwidth: fraction of the Nyquist wavenumber up to which a 256³ spectrum stays within 25 % of the "
                                       "reference's. χ: dilatational over solenoidal velocity power at t = 0.2, around k = 10 and summed above "
                                       "k = 100. The reference is AthenaK PLM/HLLE at 512³, so that row's 512³ entries are 1 by construction."),
                              differs=START + " The paper compares 512³ runs with a 1024³ reference; we have no grid above 512³ yet (stage 4).",
                              table=dict(columns=[c for _, c in COLUMNS], keys=[k for k, _ in COLUMNS], rows=rows, reference=has_ref)))
        print("wrote kritsuk-table-1 (table)", flush=True)

    out = {}
    for n in (512, 256):
        out[f"kritsuk-1-{n}"] = one(f"kritsuk-1-{n}", "Fig. 1", fig1, "Mach 9 decay at {n}³: energies and rms Mach numbers against time",
                                    "Mean specific kinetic energy, mean magnetic energy, sonic and Alfvénic rms Mach numbers, with the paper's "
                                    "definitions. ⟨u²⟩ is the sum of the velocity spectrum over shells up to N/2 (it leaves out the mean flow, "
                                    "about 1 %).", START, n)
        out[f"kritsuk-2-{n}"] = one(f"kritsuk-2-{n}", "Fig. 2", fig2, "Mach 9 decay at {n}³: small-scale kinetic and magnetic measures against time",
                                    "The paper's proxies for the dissipation rates: 2Ω + 4/3Δ (enstrophy and dilatation) and the mean squared "
                                    "current J², in box units.",
                                    START + " Computed as (2πk)²-weighted sums of the shell spectra up to k = N/2, so modes in the corners of the "
                                            "Fourier cube are left out (a few per cent); the paper does not state its normalisation.", n)
        out[f"kritsuk-3-{n}"] = one(f"kritsuk-3-{n}", "Fig. 3", fig3, "Mach 9 decay at {n}³: velocity and magnetic power spectra",
                                    "Velocity spectra compensated by k^(5/3) (left) and magnetic spectra (right) at t = 0.02, 0.06 and 0.2, "
                                    "the paper's three snapshots.", START, n)
        out[f"kritsuk-5-{n}"] = one(f"kritsuk-5-{n}", "Fig. 5", fig5, "Mach 9 decay at {n}³: dilatational over solenoidal velocity power",
                                    "χ(k) at the first and the last output.", START, n)
        if n == 512:out["kritsuk-4"] = bandwidth;out["kritsuk-table-1"] = numbers
    order = ["kritsuk-1-512", "kritsuk-1-256", "kritsuk-2-512", "kritsuk-2-256", "kritsuk-3-512", "kritsuk-3-256", "kritsuk-4", "kritsuk-5-512",
             "kritsuk-5-256", "kritsuk-table-1"]
    return {k: out[k] for k in order}
