"""The Mach 9 decay test (decay_supersonic) for our paper, in the manner of Kritsuk et al. (2011, ApJ 737, 13; K11), for
every code that ran it: their Table 1 and their Figs. 1, 2 and 4 extended to all set-ups, with the time windows, seeds,
costs and failures K11 did not have.

    python3 scripts/manuscript_decay.py          -> progress/media/analysis/decay_k11.json, manuscript/tables/decay.tex
    (the figures decay_rates and decay_bandwidth are drawn by scripts/manuscript_figures.py, which imports this file)

What is measured, per set-up, grid and seed (outputs every 0.02 to t = 0.2, as K11's ten snapshots):
  E_K = <u^2>/2 (specific, from the velocity spectrum), E_M = <B^2>/2, M_s = sqrt(2 E_K)/c_s, M_A = sqrt(<rho u^2>/<B^2>);
  K11's small-scale measures 2 Omega + 4/3 Delta = <|curl u|^2> + 4/3 <(div u)^2> and J^2 = <|curl B|^2> (sums of k^2 times the
  spectra): with the kinetic energy decaying alike in every code (it does, below), the first ranks the effective Reynolds
  numbers, the second the effective magnetic Reynolds numbers (K11 sec. 5.2);
  the effective spectral bandwidth (K11 sec. 5.4): at 256^3, the fraction of the Nyquist wave number up to which the
  spectrum stays within 25 % of the reference's, for u and B; here from the spectra averaged over the outputs t = 0.1-0.2
  (a window, not one snapshot) as well as at K11's t = 0.02;
  chi = P(u_d)/P(u_s) averaged over k > 100 at 512^3 (K11 sec. 5.5);
  the cost to t = 0.2: solver seconds on one H100, or on the CPU ranks counted as 128-core nodes (as work_precision.py).
Reference: as K11, the least diffusive set-up that finished at 512^3, chosen here by the geometric mean of its two
small-scale measures (time-averaged); E_K and E_M are given relative to it at t = 0.2.
Differences from K11 kept in view: a smooth random start (rms Mach 9, B0 = 0.3 along z, beta0 = 22) rather than a
driven, saturated state, so the field is first amplified; gamma = 1.01 rather than isothermal; 256^3 and 512^3 with
our own reference at 512^3 rather than their 1024^3.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as pc  # noqa: E402
import sweep_data as sd  # noqa: E402

TEST = "decay_supersonic"
SETUPS = tuple(sd.METHODS) + tuple(sd.NEWCODES) + tuple(sd.RAMSES2)
MAIN = tuple(m for m in SETUPS if m not in sd.RAMSES2_F32)          # the figures: double precision (and Abel et al.'s kernels)
SEEDS = (42, 43)
WINDOW = (0.1, 0.2)                                                   # spectra averaged over these outputs
TOL = 0.25
OUT_JSON = sd.SITE/"media"/"analysis"/"decay_k11.json"
OUT_TEX = sd.ROOT/"manuscript"/"tables"/"decay.tex"
CORES = 128


def series(sw, m, n, seed=42):
    """Per finite output: t, E_K, E_M, kinetic <rho u^2>/2, 2Ω+4/3Δ, J^2 and the spectra; None without a record."""
    sp = sw.spectra(TEST, m, n, seed);d = sw.diag(TEST, m, n, seed)
    if sp is None or d is None:return None
    good = sw.finite_rows(TEST, m, n, seed);row = {int(f): i for i, f in enumerate(d["frame"])}
    frames = [f for f in sorted(set(sp) & set(row)) if row[f] < good]
    if not frames:return None
    out = {k: [] for k in ("frame", "t", "ek", "em", "kin", "omega", "j2")};spec = []
    for f in frames:
        s = sp[f];i = row[f];k2 = (2*np.pi*s["k"])**2
        out["frame"].append(f);out["t"].append(float(d["time"][i]));out["ek"].append(float(s["v"].sum()))
        out["em"].append(float(d["magnetic"][i]));out["kin"].append(float(d["kinetic"][i]))
        out["omega"].append(float(2*(k2*(s["v_solenoidal"] + 4/3*s["v_compressive"])).sum()));out["j2"].append(float(2*(k2*s["b"]).sum()))
        spec.append(s)
    r = {k: np.array(v) for k, v in out.items()};r["spectra"] = spec
    return r


def window_spectrum(se, key, lo=WINDOW[0], hi=WINDOW[1]):
    sel = [s for t, s in zip(se["t"], se["spectra"]) if lo - 1e-9 <= t <= hi + 1e-9]
    if len(sel) < round((hi - lo)/0.02) + 1:return None, None           # the whole window, or nothing
    return sel[0]["k"], np.mean([s[key] for s in sel], axis=0)


def bandwidth(k, ratio, nyquist, tol=TOL):
    """K11's effective spectral bandwidth: the fraction of the Nyquist wave number up to which |ratio - 1| <= tol (k >= 2)."""
    bad = np.where((np.abs(ratio - 1) > tol) & (k >= 2) & (k <= nyquist))[0]
    return float(k[bad[0]])/nyquist if bad.size else 1.0


def cost(rows, m, n, seed=42):
    """(hours on one H100, or on 128-core nodes; resource text) of a finished run, else (None, text)."""
    r = rows.get((m, n, seed))
    if not r or r["status"] != "complete" or not r.get("solver_seconds"):return None, (r or {}).get("resource")
    res = r.get("resource") or "";s = float(r["solver_seconds"])
    if "CPU" in res:
        ranks = int(res.split()[0]);return s*ranks/CORES/3600, res
    gpus = int(res.split()[0]) if res[:1].isdigit() else 1
    return s*gpus/3600, res


def analyse(sw=None):
    sw = sw or sd.Sweep()
    rows = {(r["method"], r["N"], r.get("seed", 42)): r for r in json.loads((sd.SITE/"media"/"analysis"/TEST/"data.json").read_text())["runs"]}
    S = {(m, n, s): series(sw, m, n, s) for m in SETUPS for n in (256, 512) for s in SEEDS}
    done = lambda m, n, s=42: S[m, n, s] is not None and len(S[m, n, s]["t"]) == 11
    # the small-scale measures, time-averaged over t = 0.02-0.2 at 512^3, and the reference: the largest geometric mean
    mean_small = {m: (float(S[m, 512, 42]["omega"][1:].mean()), float(S[m, 512, 42]["j2"][1:].mean())) for m in MAIN if done(m, 512)}
    ref = max(mean_small, key=lambda m: np.sqrt(mean_small[m][0]*mean_small[m][1]))
    R = S[ref, 512, 42];ro, rj = mean_small[ref]
    kr, rv = window_spectrum(R, "v");_, rb = window_spectrum(R, "b")
    _, rvs = window_spectrum(R, "v_solenoidal");_, rvd = window_spectrum(R, "v_compressive");hi = kr > 100
    out = dict(test=TEST, reference=ref, reference_label=pc.label(ref), window=list(WINDOW), tolerance=TOL, setups={})
    for m in SETUPS:
        e = dict(label=pc.label(m), family=pc.STYLE[m][0], status={}, reached={})
        for n in (256, 512):
            for s in SEEDS:
                se = S[m, n, s];key = f"{n}_s{s}"
                e["status"][key] = "complete" if done(m, n, s) else ("stopped" if se is not None or (m, n, s) in rows else "not run")
                e["reached"][key] = float(se["t"][-1]) if se is not None else 0.0
        for n in (256, 512):
            se = S[m, n, 42]
            if se is None:continue
            e[f"t_{n}"] = se["t"].tolist();e[f"ek_{n}"] = se["ek"].tolist();e[f"em_{n}"] = se["em"].tolist()
            e[f"omega_{n}"] = se["omega"].tolist();e[f"j2_{n}"] = se["j2"].tolist()
            e[f"ms_{n}"] = np.sqrt(2*se["ek"]).tolist();e[f"ma_{n}"] = np.sqrt(se["kin"]/np.maximum(se["em"], 1e-300)).tolist()
        if done(m, 512):
            se = S[m, 512, 42]
            e["ek_ratio"] = float(se["ek"][-1]/R["ek"][-1]);e["em_ratio"] = float(se["em"][-1]/R["em"][-1])
            e["re_rel"] = float(se["omega"][1:].mean()/ro);e["rm_rel"] = float(se["j2"][1:].mean()/rj)
            e["re_rel_t002"] = float(se["omega"][1]/R["omega"][1]);e["rm_rel_t002"] = float(se["j2"][1]/R["j2"][1])
            s_ = se["spectra"][-1];sel = s_["k"] > 100
            e["chi_k100"] = float(s_["v_compressive"][sel].sum()/s_["v_solenoidal"][sel].sum())
            mid = (s_["k"] >= 8) & (s_["k"] <= 12);e["chi_k10"] = float(np.mean(s_["v_compressive"][mid]/s_["v_solenoidal"][mid]))
            _, vs = window_spectrum(se, "v_solenoidal");_, vd = window_spectrum(se, "v_compressive")
            e["ps_k100"] = float(vs[hi].sum()/rvs[hi].sum());e["pd_k100"] = float(vd[hi].sum()/rvd[hi].sum());e["chi_k100_window"] = float(vd[hi].sum()/vs[hi].sum())
            if done(m, 512, 43):
                s2 = S[m, 512, 43];e["ek_seed_spread"] = float(s2["ek"][-1]/se["ek"][-1] - 1);e["em_seed_spread"] = float(s2["em"][-1]/se["em"][-1] - 1)
                e["re_seed_spread"] = float(s2["omega"][1:].mean()/se["omega"][1:].mean() - 1)
        if done(m, 256):
            se = S[m, 256, 42];k, v = window_spectrum(se, "v");_, b = window_spectrum(se, "b");nk = len(k)
            e["esb_u"] = bandwidth(k, v/rv[:nk], 128);e["esb_b"] = bandwidth(k, b/rb[:nk], 128)
            s1 = se["spectra"][1];r1 = R["spectra"][1]
            e["esb_u_t002"] = bandwidth(s1["k"], s1["v"]/r1["v"][:nk], 128);e["esb_b_t002"] = bandwidth(s1["k"], s1["b"]/r1["b"][:nk], 128)
            e["ratio_u"] = (v/rv[:nk]).tolist();e["ratio_b"] = (b/rb[:nk]).tolist();e["k_256"] = k.tolist()
            e["re_rel_256"] = float(se["omega"][1:].mean()/ro);e["rm_rel_256"] = float(se["j2"][1:].mean()/rj)
        for n in (256, 512):
            h, res = cost(rows, m, n);e[f"cost_h_{n}"] = h;e[f"resource_{n}"] = res
        out["setups"][m] = e
    # the spread of the kinetic energy at t = 0.2 over the finished 512^3 runs: K11's "captured by all methods"
    ek = [e["ek_ratio"] for m, e in out["setups"].items() if "ek_ratio" in e and m in MAIN]
    em = [e["em_ratio"] for m, e in out["setups"].items() if "em_ratio" in e and m in MAIN]
    out["spread"] = dict(ek=[min(ek), max(ek)], em=[min(em), max(em)], finished_512=len(ek),
                         stopped=[m for m in MAIN if not done(m, 512) and not done(m, 256)])
    return out


def fmt(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def latex(out):
    """The table for the manuscript (AASTeX deluxetable*, small type): one line per set-up."""
    lines = [r"\begin{deluxetable*}{lccccccccc}", r"\tabletypesize{\scriptsize}", r"\tablecaption{The Mach 9 decay test after Table 1 of K11: every set-up "
             r"against the reference, " + out["reference_label"].replace("_", r"\_") + r" at $512^3$.\label{tab:decay}}",
             r"\tablehead{\colhead{Set-up} & \colhead{Finished} & \colhead{$E_{\rm K}$} & \colhead{$E_{\rm M}$} & \colhead{$\langle 2\Omega+\frac{4}{3}\Delta\rangle$} & "
             r"\colhead{$\langle J^2\rangle$} & \colhead{$u$-ESB} & \colhead{$B$-ESB} & \colhead{$\bar\chi(k>100)$} & \colhead{Cost}}", r"\startdata"]
    for m, e in out["setups"].items():
        st = e["status"];done = sum(v == "complete" for v in st.values());ran = sum(v != "not run" for v in st.values())
        fin = f"{done}/{ran}" + ("" if done else f", $t\\le{max(e['reached'].values()):.2f}$")
        cost_ = e.get("cost_h_512");res = e.get("resource_512") or "";unit = "n" if "CPU" in res else "g"
        lab = pc.label(m, True).replace("&", "+").replace("_", " ")
        lines.append(f"{lab} & {fin} & {fmt(e.get('ek_ratio'), 3)} & {fmt(e.get('em_ratio'))} & {fmt(e.get('re_rel'))} & {fmt(e.get('rm_rel'))} & "
                     f"{fmt(e.get('esb_u'))} & {fmt(e.get('esb_b'))} & {fmt(e.get('chi_k100_window', e.get('chi_k100')))} & "
                     f"{'--' if cost_ is None else f'{cost_:.2g}\\,{unit}'} \\\\")
    lines += [r"\enddata", r"\tablecomments{Finished: runs that reached $t=0.2$ of those run at $256^3$ and $512^3$, two seeds (and the latest time a "
              r"stopped set-up reached). Energies at $t=0.2$ and small-scale measures (time-averaged over $t=0.02$--$0.2$) at $512^3$, relative to the "
              r"reference. Effective spectral bandwidths (ESB, K11 sec.~5.4) at $256^3$ against the reference at $512^3$, from spectra averaged over "
              f"$t={WINDOW[0]}$--${WINDOW[1]}$, tolerance $\\pm{int(100*TOL)}\\%$; $\\bar\\chi$ over the same window at $512^3$. Cost to $t=0.2$ at $512^3$: "
              r"g, hours on one H100; n, hours on 128-core nodes.}", r"\end{deluxetable*}"]
    return "\n".join(lines) + "\n"


def main():
    out = analyse()
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True);OUT_TEX.write_text(latex(out))
    print(f"reference: {out['reference']};  E_K/ref at t = 0.2: {out['spread']['ek'][0]:.3f}-{out['spread']['ek'][1]:.3f};  "
          f"E_M/ref: {out['spread']['em'][0]:.2f}-{out['spread']['em'][1]:.2f} ({out['spread']['finished_512']} set-ups finished at 512^3)")
    print(f"{'set-up':28s} {'st512':8s} {'EK':>6s} {'EM':>5s} {'Re':>5s} {'Rm':>5s} {'uESB':>5s} {'bESB':>5s} {'u.02':>5s} {'b.02':>5s} {'chi':>5s} {'cost':>8s}")
    for m, e in out["setups"].items():
        print(f"{e['label'][:28]:28s} {e['status']['512_s42'][:8]:8s} {fmt(e.get('ek_ratio'), 3):>6s} {fmt(e.get('em_ratio')):>5s} {fmt(e.get('re_rel')):>5s} "
              f"{fmt(e.get('rm_rel')):>5s} {fmt(e.get('esb_u')):>5s} {fmt(e.get('esb_b')):>5s} {fmt(e.get('esb_u_t002')):>5s} {fmt(e.get('esb_b_t002')):>5s} "
              f"{fmt(e.get('chi_k100')):>5s} {fmt(e.get('cost_h_512'), 2):>6s}{'n' if 'CPU' in (e.get('resource_512') or '') else 'g'}")
    print("wrote", OUT_JSON, OUT_TEX)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- figures (drawn by manuscript_figures.py)
def _draw_kw(m, **kw):
    st = pc.STYLE[m];return dict(color=pc.color(m), ls=st[3], **kw)


def _legend(fig, methods, ncol=4, y=0.0, extra=()):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=pc.color(m), ls=pc.STYLE[m][3], marker=pc.STYLE[m][4] or None, markerfacecolor="white", markersize=3.6, lw=1.1,
                 label=pc.label(m, True)) for m in methods] + list(extra)
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, fontsize=7, handlelength=2.6, columnspacing=1.2)


def finished(out, n=512):
    return [m for m in MAIN if out["setups"][m]["status"][f"{n}_s42"] == "complete"]


def fig_rates(sw, out=None):
    """K11 Figs. 1 and 2 for every set-up that finished at 512^3: E_K and E_M against time, then the small-scale measures
    2Ω + 4/3Δ and J^2 relative to the reference's at the same time (they span a factor of ten in time)."""
    import matplotlib.pyplot as plt
    out = out or analyse(sw);ref = out["setups"][out["reference"]];ms = finished(out)
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.6), sharex=True)
    for m in ms:
        e = out["setups"][m];t = np.array(e["t_512"]);kw = _draw_kw(m, lw=1.0)
        axs[0, 0].plot(t, e["ek_512"], **kw);axs[0, 1].plot(t, e["em_512"], **kw)
        axs[1, 0].plot(t[1:], (np.array(e["omega_512"])/np.array(ref["omega_512"]))[1:], **kw)
        axs[1, 1].plot(t[1:], (np.array(e["j2_512"])/np.array(ref["j2_512"]))[1:], **kw)
    axs[0, 0].set_yscale("log");axs[0, 0].set_ylabel(r"$E_{\rm K}=\langle u^2\rangle/2$")
    axs[0, 1].set_ylabel(r"$E_{\rm M}=\langle B^2\rangle/2$")
    axs[1, 0].set_ylabel(r"$(2\Omega+\frac{4}{3}\Delta)\,/\,$reference");axs[1, 1].set_ylabel(r"$J^2\,/\,$reference")
    for ax in axs[1]:ax.set_xlabel("$t$");ax.axhline(1, color=pc.MUTED, lw=.6, ls=":");ax.set_yscale("log");ax.set_ylim(0.25, 2)
    for ax, l in zip(axs.flat, "abcd"):ax.text(.03, .93, f"({l})", transform=ax.transAxes, fontsize=8.5);ax.tick_params(which="both", top=True, right=True)
    fig.subplots_adjust(left=.1, right=.98, top=.98, bottom=.2, hspace=.08, wspace=.28)
    _legend(fig, ms, ncol=4)
    return fig


def fig_dissipation(sw, out=None):
    """Where the numerical dissipation acts. (a) K11's two small-scale measures, time-averaged, at 512^3 (filled) and 256^3
    (open), relative to the reference at 512^3: a ranking of effective Reynolds and magnetic Reynolds numbers; the
    diagonal is the reference's magnetic Prandtl number. (b) Solenoidal and dilatational velocity power at k > 100,
    averaged over t = 0.1-0.2 at 512^3, relative to the reference's: the codes lose the solenoidal part first."""
    import matplotlib.pyplot as plt
    out = out or analyse(sw);ms = finished(out);ref = out["reference"]
    R = series(sw, ref, 512, 42)
    def win(se, key):
        return np.mean([s[key] for t, s in zip(se["t"], se["spectra"]) if WINDOW[0] - 1e-9 <= t <= WINDOW[1] + 1e-9], axis=0)
    k = R["spectra"][0]["k"];sel = k > 100;rs, rd = win(R, "v_solenoidal")[sel].sum(), win(R, "v_compressive")[sel].sum()
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.6))
    ax = axs[0]
    for m in ms:
        e = out["setups"][m];mk = pc.STYLE[m][4] or "o"
        ax.plot(e["re_rel"], e["rm_rel"], marker=mk, color=pc.color(m), ms=5, ls="none", mfc=pc.color(m))
        if "re_rel_256" in e:
            ax.plot(e["re_rel_256"], e["rm_rel_256"], marker=mk, color=pc.color(m), ms=5, ls="none", mfc="white")
            ax.plot([e["re_rel_256"], e["re_rel"]], [e["rm_rel_256"], e["rm_rel"]], color=pc.color(m), lw=.5, alpha=.6)
    x = np.array([0.1, 3]);ax.plot(x, x, color=pc.MUTED, lw=.6, ls=":")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(0.12, 1.6);ax.set_ylim(0.08, 1.6)
    ax.set_xlabel(r"$\langle 2\Omega+\frac{4}{3}\Delta\rangle\,/\,$reference  ($\propto{\rm Re}_{\rm eff}$)")
    ax.set_ylabel(r"$\langle J^2\rangle\,/\,$reference  ($\sim{\rm Rm}_{\rm eff}$)")
    ax.text(.04, .93, "(a)  filled $512^3$, open $256^3$", transform=ax.transAxes, fontsize=8)
    ax = axs[1]
    for m in ms:
        se = series(sw, m, 512, 42);mk = pc.STYLE[m][4] or "o"
        ax.plot(win(se, "v_solenoidal")[sel].sum()/rs, win(se, "v_compressive")[sel].sum()/rd, marker=mk, color=pc.color(m), ms=5, ls="none")
    ax.plot([0.01, 2], [0.01, 2], color=pc.MUTED, lw=.6, ls=":")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(0.01, 1.6);ax.set_ylim(0.05, 1.6)
    ax.set_xlabel(r"solenoidal power at $k>100$ / reference");ax.set_ylabel(r"dilatational power at $k>100$ / reference")
    ax.text(.04, .93, f"(b)  $512^3$, $t={WINDOW[0]:g}$–${WINDOW[1]:g}$", transform=ax.transAxes, fontsize=8)
    for a in axs:a.tick_params(which="both", top=True, right=True)
    fig.subplots_adjust(left=.09, right=.98, top=.97, bottom=.3, wspace=.3)
    _legend(fig, ms, ncol=4)
    return fig


def fig_bandwidth(sw, out=None):
    """K11 Fig. 4 at 256^3 with the spectra averaged over t = 0.1-0.2: each set-up's velocity and magnetic spectrum over
    the reference's at 512^3; the ±25 % band sets the effective spectral bandwidth. (c): that bandwidth against the cost
    of the 256^3 run to t = 0.2."""
    import matplotlib.pyplot as plt
    out = out or analyse(sw);ms = [m for m in MAIN if "esb_u" in out["setups"][m]]
    fig = plt.figure(figsize=(7.1, 3.9));gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.05), wspace=.32)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    for j, key in enumerate(("ratio_u", "ratio_b")):
        ax = axs[j];ax.axhspan(1 - TOL, 1 + TOL, color=pc.PANEL if hasattr(pc, "PANEL") else "#f3f5f4", lw=0)
        for m in ms:
            e = out["setups"][m];k = np.array(e["k_256"]);ax.plot(np.log10(k[1:]/1), np.array(e[key])[1:], **_draw_kw(m, lw=.9))
        ax.axhline(1, color=pc.MUTED, lw=.6);ax.set_ylim(0, 1.5);ax.set_xlim(0, np.log10(128))
        ax.set_xlabel(r"$\log_{10} k/k_{\min}$");ax.set_ylabel((r"$P(u,k)$", r"$P(B,k)$")[j] + r"$\,/\,P_{\rm ref}$, $256^3$")
        ax.text(.04, .06, f"({'ab'[j]})", transform=ax.transAxes, fontsize=8.5)
    ax = axs[2]
    for m in ms:
        e = out["setups"][m];c = e.get("cost_h_256")
        if c is None:continue
        mk = pc.STYLE[m][4] or "o";cpu = "CPU" in (e.get("resource_256") or "")
        ax.plot(c, e["esb_u"], marker=mk, color=pc.color(m), ms=5, ls="none", mfc=pc.color(m))
        ax.plot(c, e["esb_b"], marker=mk, color=pc.color(m), ms=5, ls="none", mfc="white")
    ax.set_xscale("log");ax.set_xlabel("cost to $t=0.2$ at $256^3$\n[H100 h; CPU: 128-core node h]");ax.set_ylabel("effective spectral bandwidth")
    ax.text(.04, .93, "(c)  filled $u$, open $B$", transform=ax.transAxes, fontsize=8)
    for a in axs:a.tick_params(which="both", top=True, right=True)
    fig.subplots_adjust(left=.07, right=.99, top=.97, bottom=.31)
    _legend(fig, ms, ncol=4)
    return fig
