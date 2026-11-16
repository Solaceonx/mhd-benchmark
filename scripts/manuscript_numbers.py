"""Every number the manuscript quotes, computed from the records, as LaTeX macros: manuscript/numbers.tex.

    python3 scripts/manuscript_numbers.py

Reads what the other scripts wrote (no new measurement here): the run counts (build_home.count), the Mach 9 decay
analysis (progress/media/analysis/decay_k11.json, manuscript_decay.py), the order x solver study
(work_precision.table), the driven-turbulence summary (turb_results/summary.csv, sherlock_turb/summarize.py) and the
nested-grid analysis (progress/media/nested/nested.json, nested_figures.py). Run those first when their data change.
Macro names are letters only (LaTeX); each line carries a comment saying where the number comes from.
"""
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_data as sd  # noqa: E402

ROOT = sd.ROOT
OUT = ROOT/"manuscript"/"numbers.tex"
M = []          # (name, value, comment)


def put(name, value, comment=""):
    assert name.isalpha(), name
    M.append((name, value, comment))


def g(x, d=2):
    return f"{x:.{d}f}"


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def counts():
    import build_home as bh
    c = bh.count()
    put("nRuns", thousands(sum(c["campaign"].values()) + c["turb"] + c["nested"]), "every run of every campaign, turbulence and nests included (build_home.count)")
    put("nRunsBench", thousands(sum(c["campaign"].values())), "runs of the test-suite campaigns")
    put("nCodes", str(len(c["code"])), "codes")
    put("nTests", str(c["tests"]), "tests of the suite")
    put("nGridMin", str(c["grids"][0]), "coarsest grid");put("nGridMax", str(c["grids"][1]), "finest grid")
    for k, v in c["campaign"].items():put("nRuns" + "".join(ch for ch in k.capitalize().replace("2", "Two") if ch.isalpha()), thousands(v), f"runs of the {k} campaign")
    put("nTurbRuns", str(c["turb"]), "driven-turbulence runs");put("nNestedRuns", str(c["nested"]), "nested-grid runs")
    labels = set()
    for f in (sd.SITE/"media"/"analysis").glob("*/data.json"):
        d = json.loads(f.read_text())
        if isinstance(d, dict) and "runs" in d:labels |= {r["method"] for r in d["runs"] if not r["method"].endswith("_reference")}
    put("nSetups", str(len(labels)), "distinct set-ups (code, scheme, precision) with a run in the test analysis, references excluded")


def decay():
    J = json.loads((sd.SITE/"media"/"analysis"/"decay_k11.json").read_text());S = J["setups"]
    main = [m for m in S if m not in sd.RAMSES2_F32]
    put("decayRef", J["reference_label"], "the reference: least diffusive finished 512^3 set-up");put("decayFinished", str(J["spread"]["finished_512"]), "set-ups finished at 512^3")
    put("decayEKmin", g(J["spread"]["ek"][0], 3));put("decayEKmax", g(J["spread"]["ek"][1], 3), "E_K/ref at t = 0.2, 512^3")
    put("decayEMmin", g(J["spread"]["em"][0]));put("decayEMmax", g(J["spread"]["em"][1]), "E_M/ref at t = 0.2, 512^3")
    sw = sd.Sweep();runs = {}
    for m in main:
        for n in (64, 128, 256, 512):
            for s in (42, 43):
                if sw.diag("decay_supersonic", m, n, s) is not None:runs.setdefault(m, []).append(sw.finite_rows("decay_supersonic", m, n, s) == 11)
    never = [m for m in main if m in runs and not any(runs[m])]
    put("decaySetups", str(len([m for m in main if m in runs])), "set-ups that ran the decay test (fp64 and Abel et al.'s kernels)")
    put("decayNever", str(len(never)), "set-ups that finished no grid (64-512^3, two seeds)")
    put("decayNeverList", ", ".join(S[m]["label"] for m in never))
    best = [m for m in never if m.endswith("_best") or m == "cholla_ppmc"]
    put("decayNeverBest", str(len(best)), "of them the PPM-type/HLLD/RK3 'best' set-ups")
    put("decayAlways", str(sum(all(runs[m]) for m in runs)), "set-ups that finished every grid and seed they ran")
    for key, short in (("glm", "GlmLLF"), ("ct", "CTproto"), ("athenak_matched", "AthenaK"), ("athenapp_matched", "Athenapp"), ("ramses_best", "Ramses"),
                       ("ramses2_best", "RamsesTwo"), ("athenapk_matched", "AthenaPK"), ("cholla_plmc", "Cholla"), ("pluto_ct_matched", "PlutoCT"),
                       ("pluto_glm_matched", "PlutoGLM"), ("enzo_dedner", "EnzoDedner"), ("enzo_ct", "EnzoCT")):
        e = S.get(key, {})
        for f, nm in (("re_rel", "Re"), ("rm_rel", "Rm"), ("esb_u", "ESBu"), ("esb_b", "ESBb"), ("ek_ratio", "EK"), ("em_ratio", "EM"),
                      ("chi_k100_window", "Chi"), ("ps_k100", "Ps"), ("pd_k100", "Pd")):
            if e.get(f) is not None:put(f"decay{nm}{short}", g(e[f], 3 if f == "ek_ratio" else 2), f"{key}: {f}")
        if e.get("re_rel"):put(f"decayPm{short}", g(e["rm_rel"]/e["re_rel"]), f"{key}: relative magnetic Prandtl number, <J^2> over <2 Omega + 4/3 Delta>, both relative")
    fin = [m for m in main if "esb_u" in S[m]]
    for f, nm in (("esb_u", "ESBu"), ("esb_b", "ESBb")):
        v = [S[m][f] for m in fin];put(f"decay{nm}Min", g(min(v)));put(f"decay{nm}Max", g(max(v)), f"{f} over {len(v)} set-ups at 256^3")
    hll = [m for m in fin if "ps_k100" in S[m] and m not in ("glm", "ct") and S[m]["ps_k100"] < 0.9]
    put("decayPsHLLmin", g(min(S[m]["ps_k100"] for m in hll)));put("decayPsHLLmax", g(max(S[m]["ps_k100"] for m in hll)), "solenoidal power fraction at k > 100, the HLL-family codes")
    put("decayPdHLLmin", g(min(S[m]["pd_k100"] for m in hll)));put("decayPdHLLmax", g(max(S[m]["pd_k100"] for m in hll)), "dilatational, the same codes")
    seeds = [abs(S[m]["ek_seed_spread"]) for m in main if "ek_seed_spread" in S[m]]
    put("decaySeedEK", g(100*max(seeds), 1), "largest |E_K(seed 43)/E_K(seed 42) - 1| at t = 0.2, per cent")
    seeds = [abs(S[m]["em_seed_spread"]) for m in main if "em_seed_spread" in S[m]]
    put("decaySeedEM", g(100*max(seeds), 1), "the same for E_M, per cent")
    a, b = S["ramses_best"], S["ramses2_best"]
    put("decayRamsesTwinEK", g(100*abs(a["ek_ratio"]/b["ek_ratio"] - 1), 2), "RAMSES (CPU) against RAMSES II (GPU), same scheme: E_K difference, per cent")
    put("decayRamsesTwinEM", g(100*abs(a["em_ratio"]/b["em_ratio"] - 1), 2))
    c = S["ramses2_uct"].get("cost_h_512");c32 = S["ramses2f32_uct"].get("cost_h_512")
    put("decayCostUCT", g(c, 2), "H100 hours to t = 0.2 at 512^3, RAMSES II UCT-HLLD fp64");put("decayCostUCTfp", g(c32, 2), "the same in fp32")
    put("decayCostEnzoCT", g(S["enzo_ct"]["cost_h_512"], 1), "Enzo CT, 128-core node hours")


def factorial():
    import work_precision as wp
    rows = wp.table()
    def gains(test, code, key):
        cells = dict(wp.GRID)[code];f = wp.factorial(rows[test], cells);return f[key] if f else None
    codes = [c for c, _ in wp.GRID]
    for test, nm in (("gem", "Gem"), ("kh_uniform", "KH"), ("brio_wu", "BrioWu"), ("orszag_tang", "OT")):
        v = [x for x in (gains(test, c, "solver_plm") for c in codes) if x]
        put(f"facHLLD{nm}Min", g(min(v), 1));put(f"facHLLD{nm}Max", g(max(v), 1), f"{test}: cost gain of HLLD over the HLL family at PLM, three codes")
    for test, nm in (("entropy", "Entropy"), ("alfven", "Alfven"), ("field_loop_3d", "LoopThree")):
        v = gains(test, "AthenaK, RK3", "order_hll")
        if v:put(f"facPPM{nm}AthenaK", g(v, 1), f"{test}: cost gain of PPM over PLM in AthenaK (HLLE)")
    shocks = ("brio_wu", "brio_wu_rotated", "orszag_tang", "rotor", "blast_2d_b0", "blast_2d_b50")
    v = [gains(t, "Abel GLM, Float32", "order_hll") for t in shocks];v = [x for x in v if x]
    put("facPPMAbelShockMin", g(min(v)));put("facPPMAbelShockMax", g(max(v)), "Abel et al.'s kernel: PPM's cost gain on the shock tests (below 1: slower)")
    v = [gains(t, "AthenaK, RK3", "order_hll") for t in shocks];v = [x for x in v if x]
    put("facPPMAthenaKShockMin", g(min(v), 1));put("facPPMAthenaKShockMax", g(max(v), 1), "AthenaK: PPM's cost gain on the shock tests")


def turbulence():
    rows = list(csv.DictReader(open(ROOT/"turb_results"/"summary.csv")))
    def re_of(prefix_code, method, mach, n):
        v = [float(r["re_eff"]) for r in rows if r["code"] == prefix_code and r["method"] == method and r["mach_target"] == str(mach) and r["N"] == str(n)
             and r["status"] == "complete" and r["re_eff"] not in ("", "nan")]
        return statistics.mean(v) if v else None
    put("turbMachs", "3, 10, 30 and 50", "target Mach numbers (config.toml, config_highmach.toml)")
    gains = [];glm = []
    for mach in (3, 10, 30, 50):
        for n in (128, 256):
            code = "ramsesfix" if any(r["code"] == "ramsesfix" for r in rows) else "ramses"
            a, b = re_of(code, "matched", mach, n), re_of(code, "best", mach, n)
            if a and b:gains.append(b/a)
            h, l = re_of("gpu", "hydro", mach, n), re_of("gpu", "glm", mach, n)
            if h and l:glm.append(l/h)
    if gains:put("turbHLLCgainMin", g(min(gains)));put("turbHLLCgainMax", g(max(gains)), "Re_eff of RAMSES HLLC/MonCen over HLL/minmod, same grid and Mach")
    if glm:put("turbGLMlossMin", g(min(glm)));put("turbGLMlossMax", g(max(glm)), "Re_eff of Abel et al.'s MHD kernel (zero field) over their Euler kernel")
    d = [float(r["re_eff"]) for r in rows if r["code"] == "gpu" and r["method"] == "hydro" and r["mach_target"] == "10" and r["status"] == "complete"]
    r2 = [r for r in rows if r["code"] == "ramses2"]
    put("turbRamsesTwoRuns", str(len(r2)), "RAMSES II UCT-HLLD turbulence runs in the summary (0 until fetched)")


def nested():
    J = json.loads((sd.SITE/"media"/"nested"/"nested.json").read_text());C = J["capped"]["runs"];W = J["window"]
    put("nestSpeedLo", g(C["uni512"]["wall"]/C["nest128"]["wall"], 0), "uniform 512^3 wall over the 3 x 128^3 nest's, capped runs")
    put("nestSpeedHi", g(C["uni256"]["wall"]/C["nest64"]["wall"], 0), "uniform 256^3 wall over the 3 x 64^3 nest's")
    put("nestUpdatesLo", g(C["uni512"]["cell_updates"]/C["nest128"]["cell_updates"], 0));put("nestUpdatesHi", g(C["uni256"]["cell_updates"]/C["nest64"]["cell_updates"], 0))
    for pair, nm in (("nest64/uni256", "Low"), ("nest128/uni512", "High")):
        w = W[pair];put(f"nestSnaps{nm}", str(w["snapshots"]), f"{pair}: snapshots in the window t >= 0.3")
        for q, qn in (("v", "V"), ("B", "B"), ("rho", "Rho")):
            s = w[q]["shape"];put(f"nest{qn}{nm}", g(s["ratio"]));put(f"nest{qn}{nm}Lo", g(s["ci95"][0]));put(f"nest{qn}{nm}Hi", g(s["ci95"][1]), f"{pair} {q}: windowed shape ratio k = 8-96, 95 % bootstrap")
    put("nestSingleMin", g(min(W[p]["v"]["shape"]["single_min"] for p in W)));put("nestSingleMax", g(max(W[p]["v"]["shape"]["single_max"] for p in W), 1),
                                                                                   "velocity shape ratio from one pair of snapshots at one time: range")
    G = J["ergodic"];put("nestErgSeeds", str(len(G["seeds"])));put("nestErgTime", g(G["ensemble_time_mean"]));put("nestErgVolume", g(G["ensemble_volume_mean"]))
    put("nestErgMass", f"{G['ensemble_mass_mean']:+.2f}");put("nestErgZ", g(abs(G["z_ensemble"]), 1))


def suite():
    """Robustness over the whole suite (finished of planned, every test and grid) and speed at full occupancy."""
    import work_precision as wp
    sw = sd.Sweep();count = {}
    for table in (sw.rows, sw.newcodes, sw.ramses2, sw.fixed):
        for (test, label, n, seed), row in table.items():
            d, t = count.get(label, (0, 0));count[label] = (d + (row.get("state") == "done"), t + 1)
    names = dict(glm="GlmLLF", glm_hll16="GlmFSixteen", glm_fix_hll16="GlmFSixteenFixed", glm_hlld="GlmHLLD", ct="CTproto", ct_fix="CTprotoFixed",
                 athenak_matched="AthenaK", athenak_best="AthenaKBest", ramses_best="Ramses", athenapp_best="AthenappBest", ramses2_uct="RamsesTwoUCT",
                 ramses2_best="RamsesTwo", ramses2_matched="RamsesTwoHLL", enzo_ct="EnzoCT", enzo_dedner="EnzoDedner", cholla_plmc="Cholla", cholla_ppmc="CholloPPMC",
                 pluto_ct_best="PlutoCTBest", pluto_ct_matched="PlutoCT", athenapk_best="AthenaPKBest", athenapk_matched="AthenaPK", phantom_sph="Phantom")
    for m, nm in names.items():
        if m in count:
            d, t = count[m];put(f"rob{nm}", f"{d} of {t}", f"{m}: runs that reached the end, of those planned")
    for m, nm in names.items():
        r = wp.full_rate(sw, m)
        if r:
            cpu = any("CPU" in (x.get("resource") or "") for x in wp.runs_of(r[1].split()[0]).values() if x["method"] == m)
            put(f"speed{nm}", g(r[0]/1e9, 2) if not cpu else g(r[0]/1e6, 2), f"{m}: {'G' if not cpu else 'M'} cell updates per second per {'H100' if not cpu else 'core'} ({r[1]})")


def accuracy():
    """End-time error over the median of the same-order CT codes (manuscript_figures.accuracy, Our figures' claims), at 512^2."""
    import manuscript_figures as mf
    smooth = ("entropy", "alfven", "alfven_oblique", "field_loop_novz")
    def ratios(method, n=512, tests=None):
        out = []
        for test, _, kind in mf.ACCURACY:
            if tests is not None and test not in tests:continue
            err = mf.endpoint_errors(test, n, kind);base = [err[m] for m in mf.MATCHED if m in err]
            if len(base) >= 2 and method in err:out.append(err[method]/statistics.median(base))
        return out
    nonlinear = [t for t, _, _ in mf.ACCURACY if t not in smooth]
    for m, nm in (("glm", "GlmLLF"), ("glm_hlld", "GlmHLLD"), ("glm_fix_hll16", "GlmFSixteenFixed"), ("ct_fix", "CTprotoFixed"), ("glm_hll16", "GlmFSixteen")):
        for tests, tn in ((nonlinear, "Nonlin"), (smooth, "Smooth")):
            r = ratios(m, 512, tests)
            if r:
                f = lambda x: f"{x:.2g}" if x < 10 else f"{x:.0f}"
                put(f"acc{nm}{tn}Min", f(min(r)));put(f"acc{nm}{tn}Max", f(max(r)), f"{m}: end error over the same-order CT codes' median, 512^2, {tn.lower()} tests")
    put("accNonlinTests", str(len(nonlinear)), "nonlinear planar tests of the accuracy figure")


def main():
    counts();decay();factorial();turbulence();nested();suite();accuracy()
    lines = ["% Written by scripts/manuscript_numbers.py from the records; do not edit by hand."]
    for name, value, comment in M:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f"  % {comment}" if comment else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True);OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(M)} numbers -> {OUT}")


if __name__ == "__main__":
    main()
