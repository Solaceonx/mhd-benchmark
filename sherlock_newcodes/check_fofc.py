"""AthenaPK with and without first-order flux correction on the Mach 9 decay (fofc_test.sbatch): per frame, total energy
drift, finite fields, minimum density and steps.      python3 check_fofc.py      (on Sherlock, python module loaded)"""
import csv
import os

S = os.path.expandvars("$SCRATCH")
RUNS = [("best", 256), ("matched", 512)]
for v, n in RUNS:
    for tag, root, var in (("default", "mhd-results-newcodes", v), ("fofc", "mhd-results-newcodes-fofc", v + "_fofc")):
        f = f"{S}/{root}/decay_supersonic-athenapk-{var}-run-N{n}-s42/data/athenapk_{var}_r1/diagnostics.csv"
        if not os.path.exists(f):print(f"{v} {n}^3 {tag}: no record");continue
        rows = list(csv.DictReader(open(f)));e0 = float(rows[0]["energy"])
        print(f"{v} {n}^3 {tag}:")
        for r in rows:
            e = float(r["energy"]);drift = (e - e0)/e0 if e == e else float("nan")
            print(f"   t = {float(r['time']):.2f}  steps {int(r['steps']):6d}  energy drift {drift:+.2e}  finite {r['finite']:5s}"
                  f"  rho_min {float(r['rho_min']):.3g}  E_K {float(r['kinetic']):.4g}")
