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
import re
import subprocess
import statistics

import numpy as np
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_data as sd  # noqa: E402

ROOT = sd.ROOT
OUT = ROOT/"manuscript"/"numbers.tex"
M = []          # (name, value, comment)


def pc_label(m):
    import paper_common as pc
    return pc.plain(pc.STYLE[m][1])


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
    sys.path.insert(0, str(ROOT/"sherlock_newcodes"));import tasks as tk
    put("nPhysTests", str(len(tk.TESTS)), "physics tests of the suite (timing and the time-step-jump variant excluded)")
    sys.path.insert(0, str(ROOT/"sherlock_sweep"/"codes"));import external as ex
    flat = sum(ex.planar(ex.load_config(ROOT/"sherlock_sweep"/"configs"/f"{t}.toml")) for t in tk.TESTS)
    put("nPlanarTests", str(flat), "planar tests");put("nCubeTests", str(len(tk.TESTS) - flat), "3D tests")
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
    fam = {S[m]["family"] for m in main if S[m].get("ek_ratio") is not None}
    put("decayFinishedCodes", {9: "nine", 10: "ten", 8: "eight", 11: "eleven", 12: "twelve", 13: "thirteen"}.get(len(fam), str(len(fam))), "codes among them")
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
    # By Riemann solver: the two-wave (HLL/HLLE) set-ups, the HLLD set-ups other than the reference's scheme, and RAMSES
    groups = (("HLL", ("athenak_matched", "athenapp_matched", "athenapk_matched", "pluto_ct_matched", "pluto_glm_matched", "enzo_dedner")),
              ("HLLD", ("cholla_plmc", "enzo_ct")))
    for nm, ms in groups:
        ms = [m for m in ms if m in S and "ps_k100" in S[m]]
        for f, q in (("ps_k100", "Ps"), ("pd_k100", "Pd"), ("chi_k100_window", "Chi")):
            v = [S[m][f] for m in ms if S[m].get(f) is not None]
            put(f"decay{q}{nm}min", g(min(v)));put(f"decay{q}{nm}max", g(max(v)), f"{f} over the {nm} set-ups {ms}")
    # The spread of the relative measures between the two seeds (each seed against the reference's run of that seed)
    for f, nm in (("em_seeds", "EM"), ("ek_seeds", "EK"), ("re_seeds", "Re"), ("rm_seeds", "Rm")):
        v = [abs(S[m][f][1]/S[m][f][0] - 1) for m in main if len(S[m].get(f) or []) == 2 and m not in ("ramses2_uct",)]
        if v:put(f"decayRelSeed{nm}", g(100*max(v), 1), f"largest relative difference between the seeds of {f} (each against the reference of its seed), per cent")
    v = [abs(S[m][f][1] - S[m][f][0]) for m in main for f in ("esb_u_seeds", "esb_b_seeds") if len(S[m].get(f) or []) == 2]
    put("decayRelSeedESB", g(max(v), 3), "largest difference of a bandwidth between the seeds (fraction of Nyquist; one shell is 1/128)")
    seeds = [abs(S[m]["ek_seed_raw"]) for m in main if "ek_seed_raw" in S[m]]
    put("decaySeedEK", g(100*max(seeds), 1), "largest |E_K(seed 43)/E_K(seed 42) - 1| at t = 0.2 over the set-ups, per cent")
    seeds = [abs(S[m]["em_seed_raw"]) for m in main if "em_seed_raw" in S[m]]
    put("decaySeedEM", g(100*max(seeds), 1), "the same for E_M, per cent")
    v = J["energy_violations"]
    put("decayEnergyViolations", str(len(v)), "runs whose total energy moved by more than 1 % (not used)")
    if v:put("decayEnergyWorst", g(100*max(x["drift"] for x in v), 0), "the largest such change, per cent")
    put("decayEnergyTolerance", g(100*J["energy_tolerance"], 0))
    if v:                 # how far the excluded run's kinetic energy strays from the reference's run of the same seed
        import manuscript_decay as md
        w = v[0];a_ = md.series(sw, w["setup"], w["N"], w["seed"]);r_ = md.series(sw, J["reference"], w["N"], w["seed"])
        put("decayEnergyBadSetup", S[w["setup"]]["label"].split(",")[0]);put("decayEnergyBadEK", g(100*float(np.max(a_["ek"]/r_["ek"] - 1)), 0),
            "the excluded run: largest excess of E_K over the reference's, per cent")
        put("decayEnergyBadRe", g(float(a_["omega"][1:].mean()/r_["omega"][1:].mean()), 2), "and its 2 Omega + 4/3 Delta relative to the reference")
    others = [d for e in S.values() for d in e.get("energy_drift", {}).values() if d <= J["energy_tolerance"]]
    put("decayEnergyOthers", g(100*max(others), 2), "the largest change of total energy of every other run, per cent")
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
    """The driven-turbulence pilot (sherlock_turb/summarize.py -> turb_results/summary.csv): complete runs only. RAMSES as
    released and with its forcing projection corrected share the code name; the run name tells them apart."""
    rows = [r for r in csv.DictReader(open(ROOT/"turb_results"/"summary.csv")) if r["status"] == "complete"]
    kind = lambda r: r["run"].split("-")[0] + "-" + r["method"]          # ramses-best, ramsesfix-best, gpu-hydro, ramses2-uct, ...
    def mean_of(k, col, mach, n):
        v = [float(r[col]) for r in rows if kind(r) == k and r["mach_target"] == str(mach) and r["N"] == str(n) and r[col] not in ("", "nan")]
        return statistics.mean(v) if v else None
    machs = sorted({int(r["mach_target"]) for r in rows})
    put("turbMachs", ", ".join(map(str, machs[:-1])) + f" and {machs[-1]}", "target Mach numbers")
    grow = [mean_of(k, "re_eff", m, 256)/mean_of(k, "re_eff", m, 128) for k in {kind(r) for r in rows} for m in machs
            if mean_of(k, "re_eff", m, 256) and mean_of(k, "re_eff", m, 128)]
    put("turbReGrowMin", g(min(grow), 1));put("turbReGrowMax", g(max(grow), 1), "Re_eff(256^3)/Re_eff(128^3), every code, Mach number")
    hllc = [mean_of("ramsesfix-best", "re_eff", m, n)/mean_of("ramsesfix-matched", "re_eff", m, n) for m in machs for n in (128, 256)
            if mean_of("ramsesfix-best", "re_eff", m, n) and mean_of("ramsesfix-matched", "re_eff", m, n)]
    put("turbHLLCgainMin", g(min(hllc)));put("turbHLLCgainMax", g(max(hllc)), "RAMSES (corrected forcing): Re_eff of HLLC/MonCen over HLL/minmod")
    hll = [mean_of("gpu-hydro", "re_eff", m, n)/mean_of("ramsesfix-matched", "re_eff", m, n) for m in machs for n in (128, 256)
           if mean_of("gpu-hydro", "re_eff", m, n) and mean_of("ramsesfix-matched", "re_eff", m, n)]
    put("turbHydroOverRamsesMin", g(min(hll)));put("turbHydroOverRamsesMax", g(max(hll)), "Abel et al.'s Euler kernel over RAMSES HLL (corrected), Re_eff")
    glm = [mean_of("gpu-glm", "re_eff", m, n)/mean_of("gpu-hydro", "re_eff", m, n) for m in machs for n in (128, 256)
           if mean_of("gpu-glm", "re_eff", m, n) and mean_of("gpu-hydro", "re_eff", m, n)]
    put("turbGLMlossMin", g(min(glm)));put("turbGLMlossMax", g(max(glm)), "Re_eff of Abel et al.'s MHD kernel (zero field) over their Euler kernel")
    proj = [mean_of("gpu-hydro_ramsesproj", "mach", m, 128)/mean_of("gpu-hydro", "mach", m, 128) for m in (3, 10) if mean_of("gpu-hydro_ramsesproj", "mach", m, 128)]
    put("turbProjMachMin", g(100*(1 - max(proj)), 0));put("turbProjMachMax", g(100*(1 - min(proj)), 0),
        "per cent lower Mach number with RAMSES's released projection in Abel et al.'s kernel, same a_rms, 128^3")
    fl = [float(r["floor_cells"])/int(r["N"])**3 for r in rows if int(r["mach_target"]) >= 30 and r["floor_cells"] not in ("", "nan")]
    put("turbFloorMaxPermille", g(1000*max(fl), 1), "largest share of cells at the density floor at Mach 30-50, per mille")
    put("turbRamsesTwoRuns", str(sum(r["run"].startswith("ramses2-") for r in rows)), "RAMSES II runs in the summary (0 until fetched)")
    out = subprocess.run([sys.executable, str(ROOT/"sherlock_turb"/"forcing_compressive_share.py")], capture_output=True, text=True).stdout
    share = float(re.search(r"comp_frac 0\.000: .* released RAMSES ([\d.]+)", out).group(1))
    put("turbProjCompressive", g(100*share, 0), "per cent of the released RAMSES forcing's power that is compressive at comp_frac = 0")


def nested():
    J = json.loads((sd.SITE/"media"/"nested"/"nested.json").read_text());C = J["capped"]["runs"];W = J["window"]
    put("nestSpeedLo", g(C["uni512"]["wall"]/C["nest128"]["wall"], 0), "uniform 512^3 wall over the 3 x 128^3 nest's, capped runs")
    put("nestSpeedHi", g(C["uni256"]["wall"]/C["nest64"]["wall"], 0), "uniform 256^3 wall over the 3 x 64^3 nest's")
    put("nestUpdatesLo", g(C["uni512"]["cell_updates"]/C["nest128"]["cell_updates"], 0));put("nestUpdatesHi", g(C["uni256"]["cell_updates"]/C["nest64"]["cell_updates"], 0))
    for pair, nm in (("nest64_long/uni256_long", "Low"), ("nest128_long/uni512_long", "High")):
        w = W[pair];put(f"nestSnaps{nm}", str(w["snapshots"]), f"{pair}: snapshots in the window t >= 0.3")
        for q, qn in (("v", "V"), ("B", "B"), ("rho", "Rho")):
            s = w[q]["shape"];put(f"nest{qn}{nm}", g(s["ratio"]));put(f"nest{qn}{nm}Lo", g(s["ci95"][0]));put(f"nest{qn}{nm}Hi", g(s["ci95"][1]), f"{pair} {q}: windowed shape ratio k = 8-96, 95 % bootstrap")
    put("nestSingleMin", g(min(W[p]["v"]["shape"]["single_min"] for p in W)));put("nestSingleMax", g(max(W[p]["v"]["shape"]["single_max"] for p in W), 1),
                                                                                   "velocity shape ratio from one pair of snapshots at one time: range")
    for pair, nm in (("nest64_long/uni256_long", "Low"), ("nest128_long/uni512_long", "High")):
        put(f"nestEnd{nm}", g(W[pair]["times"][-1], 1), f"{pair}: last snapshot of the window")
    G = J["ergodic"];put("nestErgSeeds", str(len(G["seeds"])));put("nestErgTime", g(G["ensemble_time_mean"]));put("nestErgVolume", g(G["ensemble_volume_mean"]))
    put("nestErgMass", f"{G['ensemble_mass_mean']:+.2f}");put("nestErgZ", g(abs(G["z_ensemble"]), 1))


def suite():
    """Robustness over the whole suite (finished of planned, every test and grid) and speed at full occupancy."""
    import work_precision as wp
    sw = sd.Sweep();count = {}
    for table in (sw.rows, sw.newcodes, sw.ramses2, sw.fixed):
        for (test, label, n, seed), row in table.items():
            ok = row.get("state") == "done" and sw.went_bad(test, label, n, seed) is None     # reached the end with a finite state
            d, t = count.get(label, (0, 0));count[label] = (d + ok, t + 1)
    names = dict(glm="GlmLLF", glm_hll16="GlmFSixteen", glm_fix_hll16="GlmFSixteenFixed", glm_hlld="GlmHLLD", ct="CTproto", ct_fix="CTprotoFixed",
                 athenak_matched="AthenaK", athenak_best="AthenaKBest", ramses_best="Ramses", athenapp_best="AthenappBest", ramses2_uct="RamsesTwoUCT",
                 ramses2_best="RamsesTwo", ramses2_matched="RamsesTwoHLL", ramses2f32_uct="RamsesTwoFpUCT", enzo_ct="EnzoCT", enzo_dedner="EnzoDedner", cholla_plmc="Cholla", cholla_ppmc="CholloPPMC",
                 pluto_ct_best="PlutoCTBest", pluto_ct_matched="PlutoCT", athenapk_best="AthenaPKBest", athenapk_matched="AthenaPK", phantom_sph="Phantom")
    for m, nm in names.items():
        if m in count:
            d, t = count[m];put(f"rob{nm}", f"{d} of {t}", f"{m}: runs that reached the end, of those planned")
    for m, nm in names.items():
        r = wp.full_rate(sw, m)
        if r:
            cpu = any("CPU" in (x.get("resource") or "") for x in wp.runs_of(r[1].split()[0]).values() if x["method"] == m)
            put(f"speed{nm}", g(r[0]/1e9, 2) if not cpu else g(r[0]/1e6, 2), f"{m}: {'G' if not cpu else 'M'} cell updates per second per {'H100' if not cpu else 'core'} ({r[1]})")


def nodes_per_gpu():
    """How many 128-core nodes one H100 replaces for the same scheme: AthenaK (GPU) against Athena++ (CPU), PLM/HLLE; RAMSES II
    against RAMSES, MUSCL-Hancock/HLLD with CT (work_precision.full_rate: per H100, and per core on CPUs)."""
    import work_precision as wp
    sw = sd.Sweep();r = lambda m: wp.full_rate(sw, m)[0]
    put("nodesAthena", g(r("athenak_matched")/(128*r("athenapp_matched")), 1), "AthenaK PLM/HLLE per H100 over Athena++ PLM/HLLE per 128 cores")
    put("nodesRamses", g(r("ramses2_best")/(128*r("ramses_best")), 0), "RAMSES II HLLD per H100 over RAMSES HLLD per 128 cores")


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


def gem_precision():
    """RAMSES II UCT-HLLD on GEM in fp32 against fp64: the relative difference of the reconnected flux over time."""
    sw = sd.Sweep();out = {}
    for n in (256, 512):
        a, b = sw.diag("gem", "ramses2_uct", n), sw.diag("gem", "ramses2f32_uct", n)
        rel = np.abs(b["recon_flux_lower"] - a["recon_flux_lower"])/np.maximum(np.abs(a["recon_flux_lower"]), 1e-12);out[n] = (a["time"], rel)
    t, r = out[512];first = t[np.argmax(r > 1e-5)]
    put("gemFpSplitT", g(float(first), 1), "GEM 512^2: time at which fp32 and fp64 reconnected flux first differ by 1e-5")
    put("gemFpEndFiveTwelve", g(100*float(r[-1]), 0), "their difference at the end, per cent")
    put("gemFpMaxTwoFiftySix", g(100*float(out[256][1].max()), 1), "the largest difference at 256^2, per cent")


def turbulence_ramses2():
    """RAMSES II (UCT-HLLD, zero field, corrected projection) in the driven-turbulence set-up, against RAMSES's hydro
    solvers (corrected forcing) and Abel et al.'s kernels at the same Mach number and grid (summary.csv, seed means)."""
    rows = [r for r in csv.DictReader(open(ROOT/"turb_results"/"summary.csv")) if r["status"] == "complete"]
    kind = lambda r: r["run"].split("-")[0] + "-" + r["method"]
    def mean_of(k, col, mach, n):
        v = [float(r[col]) for r in rows if kind(r) == k and r["mach_target"] == str(mach) and r["N"] == str(n) and r[col] not in ("", "nan")]
        return statistics.mean(v) if v else None
    machs = sorted({int(r["mach_target"]) for r in rows if kind(r) == "ramses2-uct"})
    pairs = [(m, n) for m in machs for n in (128, 256)]
    best = [mean_of("ramses2-uct", "re_eff", m, n)/mean_of("ramsesfix-best", "re_eff", m, n) for m, n in pairs if mean_of("ramsesfix-best", "re_eff", m, n)]
    put("turbTwoOverBestMin", g(min(best)));put("turbTwoOverBestMax", g(max(best)), "Re_eff RAMSES II UCT-HLLD (MHD, B = 0) over RAMSES HLLC/MonCen (hydro), corrected forcing")
    glm = [mean_of("ramses2-uct", "re_eff", m, n)/mean_of("gpu-glm", "re_eff", m, n) for m, n in pairs if mean_of("gpu-glm", "re_eff", m, n)]
    put("turbTwoOverGLMMin", g(min(glm), 1));put("turbTwoOverGLMMax", g(max(glm), 1), "Re_eff RAMSES II UCT-HLLD over Abel et al.'s MHD kernel (both MHD, B = 0)")
    hyd = [mean_of("ramses2-uct", "re_eff", m, n)/mean_of("gpu-hydro", "re_eff", m, n) for m, n in pairs if mean_of("gpu-hydro", "re_eff", m, n)]
    put("turbTwoOverHydroMin", g(min(hyd), 1));put("turbTwoOverHydroMax", g(max(hyd), 1), "Re_eff RAMSES II UCT-HLLD over Abel et al.'s Euler kernel")
    mach = [abs(mean_of("ramses2-uct", "mach", m, n)/m - 1) for m, n in pairs]
    put("turbTwoMachDev", g(100*max(mach), 0), "largest |Mach/target - 1| of RAMSES II, per cent (seed means)")
    neg = [float(r["floor_cells"])/int(r["N"])**3 for r in rows if kind(r) == "ramses2-uct" and int(r["mach_target"]) >= 30 and r["floor_cells"] not in ("", "nan")]
    put("turbTwoFloorPermille", g(1000*max(neg), 1), "RAMSES II: largest share of cells at or below the density floor at Mach 30-50, per mille")
    put("turbTwoRe", f"{mean_of('ramses2-uct', 're_eff', 50, 256):,.0f}".replace(",", "{,}"), "RAMSES II Re_eff at Mach 50, 256^3")


def precision():
    """AthenaK and AthenaPK in single precision (2026-11-12 campaign) against the same set-ups in double precision: the
    end-time error on the tests with an exact reference (recorded in run.toml), the speed of the 3D runs, and how many
    runs finish. sweep_data.Sweep resolves both the sweep's and the new codes' records."""
    import tomllib
    sw = sd.Sweep()
    def rec(t, m, n, seed=42):
        p = sw.path(t, m, n, "run.toml", seed)
        return tomllib.loads(p.read_text()) if p and p.is_file() else None
    def end_err(r, field):
        if not r or r.get("status") != "complete":return None
        fr = [f for f in r.get("frames", []) if f.get("errors") and abs(f["time"] - r["t_end_requested"]) <= 1e-6*max(1, r["t_end_requested"])]
        v = fr[-1]["errors"].get(field + "_relative_L1") if fr else None
        return v if v and v > 0 else None
    smooth = ("entropy", "alfven", "alfven_oblique")
    ratios, speed, done = {}, {}, {}
    for code in ("athenak", "athenapk"):
        for v in ("matched", "best"):
            a, b = f"{code}_{v}", f"{code}f32_{v}";rs, sp, nd = [], [], [0, 0]
            for (t, m, n, seed) in sw.by:
                if m != b:continue
                rb = rec(t, b, n, seed);ra = rec(t, a, n, seed)
                if rb is not None:nd[1] += 1;nd[0] += rb.get("status") == "complete"
                if t in smooth and n >= 64 and seed == 42:
                    field = "density" if t == "entropy" else "magnetic"        # the field the wave carries
                    ea, eb = end_err(ra, field), end_err(rb, field)
                    if ea and eb:rs.append((eb/ea, t, n, field, ea, eb))
                if ra and rb and ra.get("status") == rb.get("status") == "complete" and not rb.get("planar") and n >= 256 \
                        and ra.get("zone_cycles_per_second") and rb.get("zone_cycles_per_second"):
                    sp.append(rb["zone_cycles_per_second"]/ra["zone_cycles_per_second"])
                elif ra and rb and ra.get("status") == rb.get("status") == "complete" and n >= 256 and rb.get("geometry") == "cube" \
                        and ra.get("zone_cycles_per_second") and rb.get("zone_cycles_per_second") and code == "athenak":
                    sp.append(rb["zone_cycles_per_second"]/ra["zone_cycles_per_second"])
            ratios[b], speed[b], done[b] = rs, sp, nd
    name = {"athenakf32_matched": "AthenaKMatched", "athenakf32_best": "AthenaKBest", "athenapkf32_matched": "AthenaPKMatched", "athenapkf32_best": "AthenaPKBest"}
    for b, key in name.items():
        rs = ratios[b]
        if rs:
            put(f"fpErrMax{key}", g(max(r[0] for r in rs)), f"{b}: largest fp32/fp64 end-time L1 error on the smooth waves, N >= 64")
            put(f"fpErrMed{key}", g(statistics.median(r[0] for r in rs)), f"{b}: median of that ratio")
        if speed[b]:put(f"fpSpeed{key}", g(statistics.median(speed[b])), f"{b}: median fp32/fp64 cell-update rate, 3D runs at N >= 256 ({len(speed[b])})")
        put(f"fpDone{key}", f"{done[b][0]} of {done[b][1]}", f"{b}: runs complete of those recorded")
    # RAMSES II, single against double precision, every planar test and grid of the accuracy figure (data.json errors)
    import manuscript_figures as mf
    r2 = []
    for test, _, kind in mf.ACCURACY:
        for n in (64, 128, 256, 512):
            e = mf.endpoint_errors(test, n, kind)
            for a in ("ramses2_uct", "ramses2_best", "ramses2_matched"):
                b = a.replace("ramses2", "ramses2f32")
                if a in e and b in e:r2.append((e[b]/e[a], test, n))
    put("fpTwoRuns", str(len(r2)), "RAMSES II runs compared in both precisions");put("fpTwoWithin", str(sum(abs(x[0] - 1) <= .1 for x in r2)), "of them within 10 %")
    put("fpTwoEntropy", g(max(x[0] for x in r2 if x[1] == "entropy" and x[2] == 512), 1), "largest fp32/fp64 on the entropy wave at 512^2")
    # The round-off floor of the high-order set-up: AthenaPK PPM/HLLD/RK3 on the entropy wave, fp32 against fp64, by grid
    pk = {r[2]: r for r in ratios["athenapkf32_best"] if r[1] == "entropy"}
    tex = lambda x, d: (lambda m, e: f"{m}\\times10^{{{int(e)}}}")(*f"{x:.{d}e}".split("e"))
    put("fpFloorLo", tex(pk[64][5], 1), "AthenaPK best fp32 entropy error at 64^2");put("fpFloorHi", tex(pk[512][5], 1), "at 512^2")
    put("fpDoubleHi", tex(pk[512][4], 0), "the same set-up in fp64 at 512^2")
    put("fpFloorGrow", g(pk[512][5]/pk[64][5], 1), "growth of the fp32 error from 64^2 to 512^2 (round-off accumulates over the steps)")
    st512 = rec("entropy", "athenapkf32_best", 512)["steps"]
    put("fpStepsHi", thousands(st512), "steps of that run at 512^2");put("fpUlpSteps", f"{st512*2.0**-24:.1e}".replace("e-0", "\\times10^{-") + "}", "2^-24 x steps")
    put("fpUlpFrac", g(pk[512][5]/(st512*2.0**-24), 2), "measured floor over 2^-24 x steps")
    m2 = [r[0] for b in ("athenakf32_matched", "athenapkf32_matched") for r in ratios[b]]
    put("fpMatchedMin", g(min(m2)));put("fpMatchedMax", g(max(m2)), "second-order (matched) set-ups: fp32/fp64 error range on the smooth waves, 64-512")


def mhdale():
    """MHD-ALE: the published runs (cell values as initial state) against the projected initial state (ABEL_IC=l2) and
    the same with degree-4/3 elements four cells wide (pilot 2026-11-17, mhdale_l2_results/): the end-time error
    (test_diagnostics: recorded where exact, else against AthenaK restricted) and its rank among every set-up at that grid."""
    import tomllib
    import test_diagnostics as td
    D = td.Data()
    def endpoint(test, n, v):
        base = D.sw.results if v == "ale" else ROOT/"mhdale_l2_results"
        p = base/f"{test}-mhdale-{v}-run-N{n}-s42/data/mhdale_{v}_r1"
        if not (p/"run.toml").is_file():return None
        m = tomllib.loads((p/"run.toml").read_text());field = "magnetic" if test == "field_loop_novz" else "density"
        end = float(D.sw.configs[test]["t_end"]);mode, _, _ = td.metric(test, field)
        fr = [f for f in m.get("frames", []) if abs(f["time"] - end) <= 1e-6*max(1, end)]
        if not fr:return None
        f = fr[-1]
        if f.get("errors") and f["errors"].get(f"{field}_{mode}_L1") is not None:return f["errors"][f"{field}_{mode}_L1"]
        ref, _ = D.reference(test, n, 42, f["time"]);q = p/f["file"]
        if ref is None or not q.is_file():return None
        a = np.fromfile(q, dtype="<f4").reshape(ref.shape, order="F")
        return td.compute_errors(a, ref)[f"{field}_{mode}_L1"] if np.isfinite(a).all() else None
    def rank(test, n, e):
        field = "magnetic" if test == "field_loop_novz" else "density"
        others = [x for r in D.select(test, n) if r.method != "mhdale_ale" and (x := D.endpoint(r, field)) is not None]
        return sum(o < e for o in others) + 1, len(others) + 1
    tests = {"entropy": "Entropy", "blast_2d_b50": "Blast", "rotor": "Rotor", "orszag_tang": "OT", "brio_wu": "BrioWu", "gem": "Gem",
             "field_loop_novz": "Loop", "kh_uniform": "KH"}
    for t, key in tests.items():
        for v, vk in (("ale", "Pub"), ("ale_l2", "Ltwo"), ("ale_l2_q4", "Qfour")):
            for n in (128, 256):
                e = endpoint(t, n, v)
                if e is None:continue
                r, of = rank(t, n, e)
                put(f"ale{key}{vk}{'Lo' if n == 128 else 'Hi'}", str(r), f"MHD-ALE {v}: rank of the end-time error on {t} at {n}^2 (of {of})")
                put(f"ale{key}{vk}{'Lo' if n == 128 else 'Hi'}Of", str(of))
    gain = [endpoint("entropy", n, "ale")/endpoint("entropy", n, "ale_l2") for n in (64, 128, 256)]
    put("aleEntropyGainMin", g(min(gain), 0));put("aleEntropyGainMax", g(max(gain), 0), "entropy wave: published error over the projected initial state's, 64-256^2")
    put("aleEntropyPub", f"{100*endpoint('entropy', 128, 'ale'):.3f}", "published MHD-ALE entropy error at 128^2, per cent")
    put("aleEntropyLtwo", f"{100*endpoint('entropy', 128, 'ale_l2'):.4f}", "projected initial state, 128^2, per cent")


BENCH_KEYS = {"glm": "GlmLLF", "glm_hll16": "GlmFSixteen", "glm_hlld": "GlmHLLD", "glm_fix_hll16": "GlmFSixteenFixed", "glm_fix_hlld": "GlmHLLDFixed",
              "glm_fix_hll32": "GlmFThirtyTwoFixed", "ct": "CTproto", "ct_fix": "CTprotoFixed", "athenak_matched": "AthenaKMatched",
              "athenak_best": "AthenaKBest", "athenapk_matched": "AthenaPKMatched", "athenapk_best": "AthenaPKBest", "cholla_plmc": "ChollaPLMC",
              "cholla_ppmc": "ChollaPPMC", "pluto_ct_matched": "PlutoCTMatched", "pluto_ct_best": "PlutoCTBest", "pluto_glm_best": "PlutoGLMBest",
              "ramses2_uct": "RamsesTwoUct", "ramses2_best": "RamsesTwoCT", "ramses2_matched": "RamsesTwoHLL", "ramses2f32_uct": "RamsesTwoFpUct",
              "ramses2f32_best": "RamsesTwoFpCT", "athenapp_matched": "AthenappMatched", "athenapp_best": "AthenappBest", "ramses_matched": "RamsesHLL",
              "ramses_best": "RamsesHLLD", "enzo_dedner": "EnzoDedner", "enzo_ct": "EnzoCT", "arepo_moving": "ArepoMoving", "arepo_static": "ArepoStatic",
              "gizmo_mfm": "Gizmo", "mhdale_ale": "Mhdale", "phantom_sph": "Phantom"}


def benchmark():
    """Time to a given accuracy (progress/media/analysis/work_precision.json, as the Benchmark page): for each set-up the
    median, over the tests on which it reached the error of AthenaK PLM/HLLE/RK2 at 256^2, of its time over the fastest
    set-up's on that test; the tests reached and finished; and its measured order on the smooth waves (slope between its
    two finest grids). One H100 against 128 CPU cores."""
    import build_benchmark as bb
    rows = json.loads(bb.DATA.read_text());tests, methods, fastest = bb.setups(rows)
    put("benchTests", str(len(tests)), "tests of the time-to-accuracy comparison")
    for m, key in BENCH_KEYS.items():
        if m not in methods:continue
        med, n, ran = bb.time_ratio(rows, tests, fastest, m)
        if med is not None:
            put(f"bench{key}", f"{med:.1f}" if med < 10 else f"{med:.0f}" if med < 1e3 else f"{med:,.0f}".replace(",", "{,}"), f"{m}: median time to the accuracy over the fastest's")
        put(f"bench{key}Reached", f"{n} of {ran}", f"{m}: tests on which it reached the accuracy, of those it finished")
        o = [rows[t]["methods"][m]["order"] for t in ("entropy", "alfven", "alfven_oblique") if t in rows and (rows[t]["methods"].get(m) or {}).get("order") is not None]
        if o:put(f"order{key}", g(statistics.median(o), 1), f"{m}: median measured order on the smooth waves ({len(o)} tests)")
    second = ("athenak_matched", "athenapp_matched", "athenapk_matched", "cholla_plmc", "pluto_ct_matched", "ramses_matched", "ramses_best",
              "ramses2_matched", "ramses2_best", "ramses2_uct", "enzo_dedner", "glm", "glm_fix_hll16", "ct_fix")
    os_ = [statistics.median(o) for m in second if m in methods and (o := [rows[t]["methods"][m]["order"] for t in ("entropy", "alfven", "alfven_oblique")
                                                                          if t in rows and (rows[t]["methods"].get(m) or {}).get("order") is not None])]
    put("orderSecondMin", g(min(os_), 1));put("orderSecondMax", g(max(os_), 1), "median smooth-wave order over the second-order grid set-ups")
    for m, key in (("ramses2f32_uct", "RamsesTwoFpUct"), ("ramses2_uct", "RamsesTwoUct"), ("cholla_ppmc", "ChollaPPMC"), ("athenak_best", "AthenaKBest")):
        rs = [rows[t]["methods"][m]["cost"]/fastest[t] for t in tests if (rows[t]["methods"].get(m) or {}).get("cost")]
        put(f"bench{key}Lo", g(np.percentile(rs, 25), 1));put(f"bench{key}Hi", g(np.percentile(rs, 75), 1), f"{m}: interquartile range of the per-test time ratio")
    best = sorted(((bb.time_ratio(rows, tests, fastest, m)[0], m) for m in methods if bb.time_ratio(rows, tests, fastest, m)[0]), key=lambda x: x[0])
    put("benchFastestName", pc_label(best[0][1]), "the set-up with the least median time to the accuracy")


SCORE = (("Abel et al.\\ kernels (Julia, H100)", (("glm", "GLM, LLF"), ("glm_hll16", "GLM f16/HLL (production)"), ("glm_fix_hll16", "GLM f16/HLL, fixed"),
                                                     ("glm_fix_hlld", "GLM HLLD, fixed"), ("ct_fix", "CT prototype, fixed"))),
         ("GPU codes (H100)", (("athenak_matched", "AthenaK PLM/HLLE/RK2"), ("athenak_best", "AthenaK PPM4/HLLD/RK3"), ("athenapk_matched", "AthenaPK PLM/HLLE/VL2"),
                               ("athenapk_best", "AthenaPK PPM/HLLD/RK3"), ("cholla_plmc", "Cholla PLMC/HLLD"), ("cholla_ppmc", "Cholla PPMC/HLLD"),
                               ("pluto_ct_matched", "gPLUTO linear/HLL, CT"), ("pluto_ct_best", "gPLUTO parabolic/HLLD, CT"), ("pluto_glm_best", "gPLUTO parabolic/HLLD, GLM"),
                               ("ramses2_matched", "RAMSES II HLL, CT"), ("ramses2_best", "RAMSES II HLLD, CT"), ("ramses2_uct", "RAMSES II UCT-HLLD"),
                               ("ramses2f32_uct", "RAMSES II UCT-HLLD, fp32"))),
         ("CPU codes (128 cores)", (("athenapp_matched", "Athena++ PLM/HLLE/VL2"), ("athenapp_best", "Athena++ PPM/HLLD/RK3"), ("ramses_matched", "RAMSES HLL"),
                                    ("ramses_best", "RAMSES HLLD"), ("enzo_dedner", "Enzo Dedner PLM/HLL"), ("enzo_ct", "Enzo CT PLM/HLLD"),
                                    ("arepo_moving", "AREPO moving mesh"), ("arepo_static", "AREPO static mesh"), ("gizmo_mfm", "GIZMO MFM"),
                                    ("phantom_sph", "Phantom SPMHD"), ("mhdale_ale", "MHD-ALE"))))


def equal_steps():
    """Per set-up: the median, over tests and grids, of its steps over AthenaK PLM/HLLE/RK2's on the same test and grid;
    and its time to accuracy if it had taken AthenaK's steps (the cost model with AthenaK's step count at each grid; the
    error is that of the set-up's own run): {method: (step ratio, median equal-step time ratio, tests reached)}."""
    import work_precision as wp
    sw = sd.Sweep();ratio, eq = {}, {}
    for test, kind in wp.TESTS:
        c = wp.curves(sw, test, kind);base = c.get("athenak_matched")
        if not base:continue
        ref = {x["N"]: x["steps"] for x in base["points"]};eps = next(x["error"] for x in base["points"] if x["N"] == 256)
        d_ = 3 if test.endswith("_3d") else 2;costs = {}
        for m, cur in c.items():
            pts = []
            for x in cur["points"]:
                if x["N"] in ref:
                    ratio.setdefault(m, []).append(x["steps"]/ref[x["N"]])
                    y = dict(x);y["cost"] = ref[x["N"]]*x["N"]**d_/cur["rate"]/(wp.CORES if x["cpu"] else 1);pts.append(y)
            hit = wp.at_error(dict(points=pts), eps) if pts else None
            if hit:costs[m] = hit[1]
        if costs:
            mn = min(costs.values())
            for m, v in costs.items():eq.setdefault(m, []).append(v/mn)
    return {m: (statistics.median(ratio[m]), statistics.median(eq[m]) if m in eq else None, len(eq.get(m, []))) for m in ratio}


def scorecard():
    """manuscript/tables/scorecard.tex: per set-up its measured order on the smooth waves, its median time to the accuracy of
    AthenaK PLM/HLLE/RK2 at 256^2 over the fastest set-up's (work_precision.json), the runs it finished of those planned
    (every test and grid) and its rate of cell updates at full occupancy (work_precision.full_rate)."""
    import build_benchmark as bb
    import work_precision as wp
    rows = json.loads(bb.DATA.read_text());tests, methods, fastest = bb.setups(rows)
    EQ = equal_steps()
    for m, key in BENCH_KEYS.items():
        if m in EQ:
            put(f"steps{key}", g(EQ[m][0], 1), f"{m}: median steps over AthenaK PLM/HLLE's, same test and grid")
            if EQ[m][1] is not None:put(f"benchEq{key}", f"{EQ[m][1]:.1f}" if EQ[m][1] < 10 else f"{EQ[m][1]:.0f}", f"{m}: median time to accuracy at AthenaK's step count")
    sw = sd.Sweep();count = {}
    for table in (sw.rows, sw.newcodes, sw.ramses2, sw.fixed):
        for (test, label, n, seed), row in table.items():
            ok = row.get("state") == "done" and sw.went_bad(test, label, n, seed) is None     # reached the end with a finite state
            d, t = count.get(label, (0, 0));count[label] = (d + ok, t + 1)
    import tomllib
    ale = [tomllib.loads(f.read_text()).get("status") for f in sw.results.glob("*-mhdale-ale-run-N*-s*/data/mhdale_ale_r1/run.toml")]
    count["mhdale_ale"] = (sum(x == "complete" for x in ale), len(ale))
    def fmt_ratio(x):
        return "--" if x is None else f"{x:.1f}" if x < 10 else f"{x:.0f}" if x < 1e3 else f"{x/1e3:.0f}\\,k"
    out = ["\\begin{deluxetable*}{llcccccc}", "\\tabletypesize{\\footnotesize}",
           "\\tablecaption{Every setup in one line. Order: the median slope of the $L_1$ error between the two finest grids on the three smooth "
           "waves. Time to accuracy: the median, over the tests on which the setup reached the end-time error of AthenaK PLM/HLLE/RK2 at $256^2$, of "
           "its cost to that error over the least cost of any setup on that test, with the number of tests reached of those it finished (one H100 "
           "against one 128-core node; Section~\\ref{sec:cost}). Steps: the median number of steps over that of AthenaK PLM/HLLE/RK2 on the same test "
           "and grid. Equal steps: the time to accuracy had the setup taken AthenaK's steps. Finished: runs that reached their end time with a finite "
           "state, of those planned on every test and grid. Rate: cell updates per second at full occupancy, per H100 or per 128-core "
           "node. Medians over fewer tests are taken over a different subset and are not directly comparable.\\label{tab:scorecard}}",
           "\\tablehead{\\colhead{Setup} & \\colhead{Precision} & \\colhead{Order} & \\colhead{Time to accuracy} & \\colhead{Steps} & \\colhead{Equal steps} & \\colhead{Finished} & \\colhead{Rate ($10^9\\,{\\rm s}^{-1}$)}}",
           "\\startdata"]
    for head, sets in SCORE:
        out.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{head}}}}} \\\\")
        for m, label in sets:
            med, n, ran = bb.time_ratio(rows, tests, fastest, m) if m in methods else (None, 0, 0)
            o = [rows[t]["methods"][m]["order"] for t in ("entropy", "alfven", "alfven_oblique") if t in rows and (rows[t]["methods"].get(m) or {}).get("order") is not None]
            r = wp.full_rate(sw, m);cpu = "CPU" in label or m.startswith(("athenapp", "ramses_", "enzo", "arepo", "gizmo", "phantom", "mhdale"))
            rate = "--" if not r else f"{r[0]*(128 if cpu else 1)/1e9:.2f}" if r[0]*(128 if cpu else 1) >= 1e7 else f"{r[0]*(128 if cpu else 1)/1e9:.3f}"
            prec = "fp32" if "f32" in m else "fp32, f16 tiles" if "hll16" in m else "fp32" if m.startswith(("glm", "ct")) else "fp64"
            d, t = count.get(m, (None, None))
            sr, eqr, _ = EQ.get(m, (None, None, 0))
            stp = "--" if sr is None else f"{sr:.1f}";eqs = fmt_ratio(eqr)
            out.append(f"\\quad {label} & {prec} & {statistics.median(o):.1f} & {fmt_ratio(med)} ({n}/{ran}) & {stp} & {eqs} & {f'{d}/{t}' if t else '--'} & {rate} \\\\"
                       if o else f"\\quad {label} & {prec} & -- & {fmt_ratio(med)} ({n}/{ran}) & {stp} & {eqs} & {f'{d}/{t}' if t else '--'} & {rate} \\\\")
    out[-1] = out[-1].rstrip(" \\\\")
    out += ["\\enddata", "\\end{deluxetable*}"]
    (ROOT/"manuscript"/"tables"/"scorecard.tex").write_text("\n".join(out) + "\n")


def fofc():
    """AthenaPK's density floor (dfloor keeps the momentum of a cell it refills, so the velocity and, through the pressure
    floor, the total energy jump) and its first-order flux correction (newcodes_results/fofc_test/, job 52536645): the
    largest |E/E0 - 1| of the Mach 9 decay with and without it, and how far the high-order set-up gets."""
    def drift(path):
        rows = list(csv.DictReader(open(path)));e0 = float(rows[0]["energy"])
        return max(abs(float(r["energy"])/e0 - 1) for r in rows if r["energy"] not in ("", "nan", "NaN")), float(rows[-1]["time"]), rows[-1]["finite"]
    base = ROOT/"newcodes_results"
    d0, _, _ = drift(base/"mhd-results-newcodes"/"decay_supersonic-athenapk-matched-run-N512-s42"/"data"/"athenapk_matched_r1"/"diagnostics.csv")
    d1, t1, _ = drift(base/"fofc_test"/"decay_supersonic-athenapk-matched_fofc-run-N512-s42"/"data"/"athenapk_matched_fofc_r1"/"diagnostics.csv")
    put("fofcDriftOff", g(100*d0, 0), "AthenaPK PLM/HLLE 512^3 seed 42 without FOFC: largest |E/E0 - 1|, per cent")
    put("fofcDriftOn", f"{d1:.0e}".replace("e-0", "\\times10^{-") + "}", "the same with first-order flux correction")
    b0 = ROOT/"newcodes_results"/"mhd-results-newcodes"/"decay_supersonic-athenapk-best-run-N256-s42"/"data"/"athenapk_best_r1"/"diagnostics.csv"
    rows = list(csv.DictReader(open(b0)));fin = [r for r in rows if r["finite"] == "true"]
    put("fofcBestOffT", g(float(fin[-1]["time"]) if fin else 0.0, 2), "AthenaPK PPM/HLLD 256^3 seed 42 without FOFC: last finite output time")
    d2, t2, f2 = drift(base/"fofc_test"/"decay_supersonic-athenapk-best_fofc-run-N256-s42"/"data"/"athenapk_best_fofc_r1"/"diagnostics.csv")
    put("fofcBestOnT", g(t2, 1), "with FOFC: last output time")
    import tomllib
    a = tomllib.loads((base/"fofc_test"/"decay_supersonic-athenapk-matched_fofc-run-N512-s42"/"data"/"athenapk_matched_fofc_r1"/"run.toml").read_text())
    b = tomllib.loads((base/"mhd-results-newcodes"/"decay_supersonic-athenapk-matched-run-N512-s42"/"data"/"athenapk_matched_r1"/"run.toml").read_text())
    put("fofcRate", g(a["zone_cycles_per_second"]/b["zone_cycles_per_second"], 2), "FOFC over default: cell-update rate, matched 512^3 seed 42 (different nodes)")
    put("fofcSteps", g(a["steps"]/b["steps"], 2), "FOFC over default: steps to t = 0.2");put("fofcTime", g(a["solve_seconds"]/b["solve_seconds"], 2), "and solver time");put("fofcBestOnDrift", f"{d2:.0e}".replace("e-0", "\\times10^{-") + "}", "its largest |E/E0 - 1|")


def selfconv():
    """Convergence without a truth (scripts/self_convergence.py -> self_convergence.json): how much the error against a
    set-up's own finest grid (512^2) underestimates the error against the truth, at 256^2 and 128^2, over every set-up and
    planar test; and the gain of Richardson extrapolation (error of the finest grid over that of the extrapolation)."""
    d = json.loads((sd.SITE/"media"/"analysis"/"self_convergence.json").read_text())
    for N, nm in (("256", "Half"), ("128", "Quarter")):
        u = [1 - r["own"][N]/r["truth"][N] for t, v in d.items() for r in v["methods"].values()
             if r.get("finest") == 512 and r.get("own", {}).get(N) and r.get("truth", {}).get(N)]
        put(f"selfOwn{nm}", g(100*np.median(u), 0), f"median underestimate of the own-finest error at {N}^2, per cent ({len(u)} runs)")
        put(f"selfOwn{nm}Lo", g(100*np.percentile(u, 25), 0));put(f"selfOwn{nm}Hi", g(100*np.percentile(u, 75), 0), "its interquartile range")
    for smooth, nm in ((True, "Smooth"), (False, "Nonlin")):
        gg = [r["richardson"]["gain"] for t, v in d.items() for r in v["methods"].values()
              if (t in ("entropy", "alfven", "alfven_oblique")) == smooth and isinstance((r.get("richardson") or {}).get("gain"), (int, float))]
        put(f"richGain{nm}", g(np.median(gg)), f"median Richardson gain, {'smooth waves' if smooth else 'nonlinear tests'} ({len(gg)})")


def refcheck():
    """How good the references are: the two independent references, AthenaK at 4096^2 and Athena++ at 2048^2, averaged onto
    the same grid, differ by refdiff; set against the median end-time error of every set-up on that grid. And the time to
    accuracy with the current sheet left out (its references disagree by more than the errors)."""
    import test_diagnostics as td
    import build_benchmark as bb
    D = td.Data();sw = D.sw
    def ref_of(test, code, n):
        big = sw.reference_n(test, code);m = sw.meta(test, code, big, 42) if big else None
        if not m:return None
        end = float(sw.configs[test]["t_end"])
        f = next((f for f in m.get("frames", []) if abs(f["time"] - end) <= 1e-6*max(1, end)), None)
        name = f and f.get("restricted", {}).get(str(n))
        return D.load_array(str(sw.path(test, code, big, name, 42)), n) if name else None
    keys = {"brio_wu": "BW", "brio_wu_rotated": "BWR", "blast_2d_b50": "Blast", "kh_uniform": "KH", "gem": "Gem", "current_sheet": "CS"}
    fr = []
    for test, key in keys.items():
        field = "magnetic" if test == "gem" else "density"
        for n in (512, 256):
            a, b = ref_of(test, "athenak_reference", n), ref_of(test, "athenapp_reference", n)
            if a is None or b is None:continue
            d = td.compute_errors(b, a)[f"{field}_relative_L1"]
            errs = [x for r in D.select(test, n) if (x := D.endpoint(r, field)) is not None]
            put(f"ref{key}", g(d/np.median(errs), 2), f"{test} at {n}^2: AthenaK 4096^2 vs Athena++ 2048^2 over the median error of {len(errs)} set-ups")
            put(f"ref{key}N", str(n));put(f"ref{key}Min", g(d/min(errs), 2), "over the smallest error")
            if test != "current_sheet":fr.append(d/np.median(errs))
            break
    put("refFracMin", g(100*min(fr), 0));put("refFracMax", g(100*max(fr), 0), "reference difference over the median error, per cent, tests other than the current sheet")
    rows = json.loads(bb.DATA.read_text());tests, methods, fastest = bb.setups(rows)
    t2 = [t for t in tests if t != "current_sheet"];ch = []
    for m in methods:
        a, _, _ = bb.time_ratio(rows, tests, fastest, m);b, _, _ = bb.time_ratio(rows, t2, fastest, m)
        if a and b:ch.append(abs(b/a - 1))
    put("benchNoCSchange", g(100*max(ch), 0), "largest change of a set-up's median time to accuracy when the current sheet is left out, per cent")


def main():
    counts();decay();factorial();turbulence();turbulence_ramses2();nested();suite();accuracy();gem_precision();precision();mhdale();benchmark();scorecard();fofc();selfconv();refcheck();nodes_per_gpu()
    lines = ["% Written by scripts/manuscript_numbers.py from the records; do not edit by hand."]
    for name, value, comment in M:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f"  % {comment}" if comment else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True);OUT.write_text("\n".join(lines) + "\n")
    print(f"{len(M)} numbers -> {OUT}")


if __name__ == "__main__":
    main()
