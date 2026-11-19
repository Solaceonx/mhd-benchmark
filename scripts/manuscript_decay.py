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
SETUPS = (tuple(sd.METHODS) + tuple(m for m in sd.NEWCODES if not m.startswith(("castro", "nyx"))) + tuple(sd.RAMSES2)   # Castro/Nyx held out until complete
          + tuple(sd.FOFC))
# The reference pool and the spreads: double precision (and Abel et al.'s kernels), each code as its authors recommend --
# not AthenaPK with its first-order flux correction, an option switched on after its high-order set-up failed here
MAIN = tuple(m for m in SETUPS if m not in sd.RAMSES2_F32 and m not in sd.FOFC)
FIGS = MAIN + ("athenapk_best_fofc",)       # the figures: MAIN and that corrected high-order set-up
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


def cost(rows, m, n, seed=42, sw=None):
    """(hours on one H100, or on 128-core nodes; resource text) of a finished run, else (None, text)."""
    r = rows.get((m, n, seed))
    if r is None and sw is not None and m in sd.FOFC and (meta := sw.meta(TEST, m, n, seed)):   # not in the site's tables: its record
        r = dict(status=meta.get("status"), solver_seconds=meta.get("solve_seconds"), resource="1 H100")
    if not r or r["status"] != "complete" or not r.get("solver_seconds"):return None, (r or {}).get("resource")
    res = r.get("resource") or "";s = float(r["solver_seconds"])
    if "CPU" in res:
        ranks = int(res.split()[0]);return s*ranks/CORES/3600, res
    gpus = int(res.split()[0]) if res[:1].isdigit() else 1
    return s*gpus/3600, res


ENERGY_TOL = 0.01        # a run whose total energy moves by more than this is not used (energy is conserved; all but one run hold 1e-3)


def analyse(sw=None):
    """Every metric per seed, against the reference run of the same seed, then averaged over the seeds whose run finished and
    conserved energy (the spread between the seeds is kept). The figures draw seed 42, or 43 where 42 is not usable."""
    sw = sw or sd.Sweep()
    rows = {(r["method"], r["N"], r.get("seed", 42)): r for r in json.loads((sd.SITE/"media"/"analysis"/TEST/"data.json").read_text())["runs"]}
    S = {(m, n, s): series(sw, m, n, s) for m in SETUPS for n in (256, 512) for s in SEEDS}
    drift = {}
    for (m, n, s), se in S.items():
        d = sw.diag(TEST, m, n, s)
        if se is not None and d is not None:
            k = sw.finite_rows(TEST, m, n, s);e_ = d["energy"][:k];drift[m, n, s] = float(np.max(np.abs(e_/e_[0] - 1)))
    done = lambda m, n, s=42: S[m, n, s] is not None and len(S[m, n, s]["t"]) == 11
    valid = lambda m, n, s: done(m, n, s) and drift.get((m, n, s), 0.0) <= ENERGY_TOL
    seeds_of = lambda m, n: [s for s in SEEDS if valid(m, n, s)]
    # the reference: the largest geometric mean of the two small-scale measures (time-averaged, 512^3, mean over seeds)
    small = {m: (np.mean([S[m, 512, s]["omega"][1:].mean() for s in seeds_of(m, 512)]), np.mean([S[m, 512, s]["j2"][1:].mean() for s in seeds_of(m, 512)]))
             for m in MAIN if seeds_of(m, 512)}
    ref = max(small, key=lambda m: np.sqrt(small[m][0]*small[m][1]))
    if seeds_of(ref, 512) != list(SEEDS):raise SystemExit(f"the reference {ref} must have every seed at 512^3")
    R = {s: S[ref, 512, s] for s in SEEDS}
    W = {s: {key: window_spectrum(R[s], key)[1] for key in ("v", "b", "v_solenoidal", "v_compressive")} for s in SEEDS}
    kr = window_spectrum(R[42], "v")[0];hi = kr > 100
    out = dict(test=TEST, reference=ref, reference_label=pc.label(ref), window=list(WINDOW), tolerance=TOL, energy_tolerance=ENERGY_TOL,
               energy_violations=[dict(setup=m, N=n, seed=s, drift=v) for (m, n, s), v in drift.items() if v > ENERGY_TOL], setups={})
    mean = lambda v: float(np.mean(v)) if v else None
    for m in SETUPS:
        e = dict(label=pc.label(m), family=pc.STYLE[m][0], status={}, reached={}, energy_drift={})
        for n in (256, 512):
            for s in SEEDS:
                se = S[m, n, s];key = f"{n}_s{s}"
                e["status"][key] = ("complete" if valid(m, n, s) else "energy" if done(m, n, s) else
                                    "stopped" if se is not None or (m, n, s) in rows else "not run")
                e["reached"][key] = float(se["t"][-1]) if se is not None else 0.0
                if (m, n, s) in drift:e["energy_drift"][key] = drift[m, n, s]
        for n in (256, 512):
            ss = seeds_of(m, n);s0 = ss[0] if ss else 42;se = S[m, n, s0]
            if se is None:continue
            e[f"seed_fig_{n}"] = s0;e[f"t_{n}"] = se["t"].tolist();e[f"ek_{n}"] = se["ek"].tolist();e[f"em_{n}"] = se["em"].tolist()
            e[f"omega_{n}"] = se["omega"].tolist();e[f"j2_{n}"] = se["j2"].tolist()
            e[f"ms_{n}"] = np.sqrt(2*se["ek"]).tolist();e[f"ma_{n}"] = np.sqrt(se["kin"]/np.maximum(se["em"], 1e-300)).tolist()
            if len(se["t"]) == 11:
                e[f"ref_omega_{n}"] = R[s0]["omega"].tolist();e[f"ref_j2_{n}"] = R[s0]["j2"].tolist()
        ss = seeds_of(m, 512);e["seeds_512"] = ss
        if ss:
            per = lambda f: [f(S[m, 512, s], s) for s in ss]
            ek = per(lambda se, s: se["ek"][-1]/R[s]["ek"][-1]);em = per(lambda se, s: se["em"][-1]/R[s]["em"][-1])
            re = per(lambda se, s: se["omega"][1:].mean()/R[s]["omega"][1:].mean());rm = per(lambda se, s: se["j2"][1:].mean()/R[s]["j2"][1:].mean())
            e.update(ek_ratio=mean(ek), em_ratio=mean(em), re_rel=mean(re), rm_rel=mean(rm), ek_seeds=ek, em_seeds=em, re_seeds=re, rm_seeds=rm)
            if len(ss) == 2:      # how far two realisations of the same set-up differ (not relative to the reference)
                a_, b_ = S[m, 512, 42], S[m, 512, 43]
                e["ek_seed_raw"] = float(b_["ek"][-1]/a_["ek"][-1] - 1);e["em_seed_raw"] = float(b_["em"][-1]/a_["em"][-1] - 1)
            e["re_rel_t002"] = mean(per(lambda se, s: se["omega"][1]/R[s]["omega"][1]));e["rm_rel_t002"] = mean(per(lambda se, s: se["j2"][1]/R[s]["j2"][1]))
            def chi(se, s):
                s_ = se["spectra"][-1];sel = s_["k"] > 100;return s_["v_compressive"][sel].sum()/s_["v_solenoidal"][sel].sum()
            e["chi_k100"] = mean(per(chi))
            ps = per(lambda se, s: window_spectrum(se, "v_solenoidal")[1][hi].sum()/W[s]["v_solenoidal"][hi].sum())
            pd = per(lambda se, s: window_spectrum(se, "v_compressive")[1][hi].sum()/W[s]["v_compressive"][hi].sum())
            cw = per(lambda se, s: window_spectrum(se, "v_compressive")[1][hi].sum()/window_spectrum(se, "v_solenoidal")[1][hi].sum())
            e.update(ps_k100=mean(ps), pd_k100=mean(pd), chi_k100_window=mean(cw))
        ss = seeds_of(m, 256);e["seeds_256"] = ss
        if ss:
            def over_ref(s, key):
                v = window_spectrum(S[m, 256, s], key)[1];return v/W[s][key][:len(v)]
            ru = np.mean([over_ref(s, "v") for s in ss], axis=0);rb = np.mean([over_ref(s, "b") for s in ss], axis=0)
            k = window_spectrum(S[m, 256, ss[0]], "v")[0]
            e["esb_u"] = bandwidth(k, ru, 128);e["esb_b"] = bandwidth(k, rb, 128)
            e["esb_u_seeds"] = [bandwidth(k, window_spectrum(S[m, 256, s], "v")[1]/W[s]["v"][:len(k)], 128) for s in ss]
            e["esb_b_seeds"] = [bandwidth(k, window_spectrum(S[m, 256, s], "b")[1]/W[s]["b"][:len(k)], 128) for s in ss]
            s1 = [S[m, 256, s]["spectra"][1] for s in ss];r1 = [R[s]["spectra"][1] for s in ss]
            e["esb_u_t002"] = bandwidth(k, np.mean([a["v"]/b["v"][:len(k)] for a, b in zip(s1, r1)], axis=0), 128)
            e["esb_b_t002"] = bandwidth(k, np.mean([a["b"]/b["b"][:len(k)] for a, b in zip(s1, r1)], axis=0), 128)
            e["ratio_u"] = ru.tolist();e["ratio_b"] = rb.tolist();e["k_256"] = k.tolist()
            e["re_rel_256"] = mean([S[m, 256, s]["omega"][1:].mean()/R[s]["omega"][1:].mean() for s in ss])
            e["rm_rel_256"] = mean([S[m, 256, s]["j2"][1:].mean()/R[s]["j2"][1:].mean() for s in ss])
        for n in (256, 512):
            h, res = cost(rows, m, n, sw=sw);e[f"cost_h_{n}"] = h;e[f"resource_{n}"] = res
        out["setups"][m] = e
    ek = [e["ek_ratio"] for m, e in out["setups"].items() if e.get("ek_ratio") is not None and m in MAIN]
    em = [e["em_ratio"] for m, e in out["setups"].items() if e.get("em_ratio") is not None and m in MAIN]
    out["spread"] = dict(ek=[min(ek), max(ek)], em=[min(em), max(em)], finished_512=len(ek),
                         stopped=[m for m in MAIN if not seeds_of(m, 512) and not seeds_of(m, 256)])
    return out


def fmt(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def latex(out):
    """The table for the manuscript (AASTeX deluxetable*, small type): one line per set-up."""
    lines = [r"\begin{deluxetable*}{lccccccccc}", r"\tabletypesize{\scriptsize}", r"\tablecaption{The Mach 9 decay test after Table 1 of K11: every setup "
             r"against the reference, " + out["reference_label"].replace("_", r"\_") + r" at $512^3$.\label{tab:decay}}",
             r"\tablehead{\colhead{Setup} & \colhead{Finished} & \colhead{$E_{\rm K}$} & \colhead{$E_{\rm M}$} & \colhead{$\langle 2\Omega+\frac{4}{3}\Delta\rangle$} & "
             r"\colhead{$\langle J^2\rangle$} & \colhead{$u$-ESB} & \colhead{$B$-ESB} & \colhead{$\bar\chi(k>100)$} & \colhead{Cost}}", r"\startdata"]
    for m, e in out["setups"].items():
        st = e["status"];done = sum(v == "complete" for v in st.values());ran = sum(v != "not run" for v in st.values())
        fin = r"$\le128^3$ only" if not ran else f"{done}/{ran}" + ("" if done else f", $t\\le{max(e['reached'].values()):.2f}$")
        cost_ = e.get("cost_h_512");res = e.get("resource_512") or "";unit = "n" if "CPU" in res else "g"
        lab = pc.label(m, True).replace("&", "+").replace("_", " ") + (r"$^{\rm a}$" if "energy" in st.values() else "")
        lines.append(f"{lab} & {fin} & {fmt(e.get('ek_ratio'), 3)} & {fmt(e.get('em_ratio'))} & {fmt(e.get('re_rel'))} & {fmt(e.get('rm_rel'))} & "
                     f"{fmt(e.get('esb_u'))} & {fmt(e.get('esb_b'))} & {fmt(e.get('chi_k100_window', e.get('chi_k100')))} & "
                     f"{'--' if cost_ is None else f'{cost_:.2g}\\,{unit}'} \\\\")
    lines += [r"\enddata", r"\tablecomments{Finished: runs that reached $t=0.2$ of those run at $256^3$ and $512^3$, two seeds (and the latest time a "
              r"stopped setup reached). Energies at $t=0.2$ and small-scale measures (time-averaged over $t=0.02$--$0.2$) at $512^3$, relative to the "
              r"reference. Effective spectral bandwidths (ESB, K11 sec.~5.4) at $256^3$ against the reference at $512^3$, from spectra averaged over "
              f"$t={WINDOW[0]}$--${WINDOW[1]}$, tolerance $\\pm{int(100*TOL)}\\%$; $\\bar\\chi$ over the same window at $512^3$. Cost to $t=0.2$ at $512^3$: "
              r"g, hours on one H100; n, hours on 128-core nodes. Every metric is the mean over the seeds, each against the reference's run of "
              r"the same seed." + "".join(f" $^{{\\rm a}}$Not used: {pc.label(v['setup'], True)} at ${v['N']}^3$, seed {v['seed']}, whose total "
              f"energy moved by {100*v['drift']:.0f}\\%." for v in out["energy_violations"]) + "}", r"\end{deluxetable*}"]
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
    return [m for m in FIGS if out["setups"][m].get(f"seeds_{n}")]


def fig_rates(sw, out=None):
    """K11 Figs. 1 and 2 for every set-up that finished at 512^3: E_K and E_M against time, then the small-scale measures
    2Ω + 4/3Δ and J^2 relative to the reference's at the same time (they span a factor of ten in time)."""
    import matplotlib.pyplot as plt
    out = out or analyse(sw);ref = out["setups"][out["reference"]];ms = finished(out)
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.6), sharex=True)
    for m in ms:
        e = out["setups"][m];t = np.array(e["t_512"]);kw = _draw_kw(m, lw=1.0)
        axs[0, 0].plot(t, e["ek_512"], **kw);axs[0, 1].plot(t, e["em_512"], **kw)
        axs[1, 0].plot(t[1:], (np.array(e["omega_512"])/np.array(e["ref_omega_512"]))[1:], **kw)
        axs[1, 1].plot(t[1:], (np.array(e["j2_512"])/np.array(e["ref_j2_512"]))[1:], **kw)
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
    out = out or analyse(sw);ms = finished(out)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.6))
    ax = axs[0]
    for m in ms:
        e = out["setups"][m];mk = pc.STYLE[m][4] or "o"
        ax.plot(e["re_rel"], e["rm_rel"], marker=mk, color=pc.color(m), ms=5, ls="none", mfc=pc.color(m))
        if e.get("re_rel_256") is not None:
            ax.plot(e["re_rel_256"], e["rm_rel_256"], marker=mk, color=pc.color(m), ms=5, ls="none", mfc="white")
            ax.plot([e["re_rel_256"], e["re_rel"]], [e["rm_rel_256"], e["rm_rel"]], color=pc.color(m), lw=.5, alpha=.6)
    x = np.array([0.1, 3]);ax.plot(x, x, color=pc.MUTED, lw=.6, ls=":")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(0.12, 1.6);ax.set_ylim(0.08, 1.6)
    ax.set_xlabel(r"$\langle 2\Omega+\frac{4}{3}\Delta\rangle\,/\,$reference  ($\propto{\rm Re}_{\rm eff}$)")
    ax.set_ylabel(r"$\langle J^2\rangle\,/\,$reference  ($\sim{\rm Rm}_{\rm eff}$)")
    ax.text(.04, .93, "(a)  filled $512^3$, open $256^3$", transform=ax.transAxes, fontsize=8)
    ax = axs[1]
    for m in ms:
        e = out["setups"][m];mk = pc.STYLE[m][4] or "o"
        ax.plot(e["ps_k100"], e["pd_k100"], marker=mk, color=pc.color(m), ms=5, ls="none")
    ax.plot([0.01, 4], [0.01, 4], color=pc.MUTED, lw=.6, ls=":")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlim(0.01, 3.5);ax.set_ylim(0.05, 2)
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
    out = out or analyse(sw);ms = [m for m in FIGS if "esb_u" in out["setups"][m]]
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
