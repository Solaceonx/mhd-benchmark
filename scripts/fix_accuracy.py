"""Accuracy of the sweep-fixes kernels on the shock tests, beside Tom Abel's original kernels and AthenaK.

    python3 scripts/fix_accuracy.py [FETCHED ...]

FETCHED: local copies of $SCRATCH/mhd-results-fixes and mhd-results-verify (default: fix_results/mhd-results-*,
fetched with the command in sherlock_fixes/README.md). The measure is the one of the test pages and the Our figures
page (test_diagnostics.py, compute_errors): relative L1 of the density and of B at the last output, against the
AthenaK 4096² reference restricted to the run's grid. The originals and AthenaK are read from sweep_results/.
"""
import glob
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT/"sweep_results"
TESTS = ("brio_wu", "brio_wu_rotated", "rotor", "orszag_tang")
GRIDS = (128, 256)
# each fixed kernel beside the original with the same Riemann solver (glm_fct32: HLL, so the HLL production kernel)
ORIGINAL = {"glm_fix": "glm", "glm_fix_hll16": "glm_hll16", "glm_fix_hlld": "glm_hlld", "glm_fix_ppm": "glm_ppm",
            "glm_fct16": "glm_hll16", "glm_fct32": "glm_hll16", "ct_fix": "ct"}
AROUND = (("athenak_matched", "athenak-matched-run"), ("athenak_best", "athenak-best-run"))


def state(path, n):
    """(n, n, 9) state of a planar run, or None (missing, wrong size, or not finite)."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size != n*n*9*4:return None
    u = np.fromfile(p, dtype="<f4").reshape((n, n, 9), order="F")
    return u if np.isfinite(u).all() else None


def errors(u, ref):
    """Relative L1 of the density and of B (as test_diagnostics.compute_errors)."""
    out = []
    for cols in (slice(0, 1), slice(5, 8)):
        e = np.linalg.norm(u[..., cols].astype(float) - ref[..., cols].astype(float), axis=-1)
        out.append(float(e.mean()/np.linalg.norm(ref[..., cols].astype(float), axis=-1).mean()))
    return out


def complete(run_dir):
    try:
        return tomllib.loads((run_dir/"run.toml").read_text()).get("status") == "complete"
    except (OSError, tomllib.TOMLDecodeError):
        return False


def main(argv):
    fetched = [Path(a) for a in argv] or [Path(p) for p in sorted(glob.glob(str(ROOT/"fix_results"/"mhd-results-*")))]
    print("Relative L1 error at the end against the AthenaK 4096² reference: density / B. 'stopped' = did not finish.\n")
    for test in TESTS:
        end = tomllib.loads((ROOT/"sherlock_sweep"/"configs"/f"{test}.toml").read_text())["frames"] - 1
        for n in GRIDS:
            refs = glob.glob(str(SWEEP/f"{test}-athenak-reference-reference-N*-s42"/"data"/"athenak_reference_r1"/f"ref_N{n}_{end:03d}.bin"))
            ref = state(refs[0], n) if refs else None
            if ref is None:print(f"{test} N={n}: no reference here\n");continue
            print(f"{test}, N = {n}")
            print(f"  {'kernel':<15}{'fixed':>22}   {'original':<11}{'':>20}   fixed/original")
            for k, o in ORIGINAL.items():
                d = next((f/f"{test}-N{n}-s42"/"data"/f"{k}_r1" for f in fetched if (f/f"{test}-N{n}-s42"/"data"/f"{k}_r1").is_dir()), None)
                fu = state(d/f"state_{end:03d}.bin", n) if d is not None and complete(d) else None
                od = next((SWEEP/x/"data"/f"{o}_r1" for x in (f"{test}-gpu-N{n}-s42-all", f"{test}-gpu-N{n}-s42-{o}") if (SWEEP/x/"data"/f"{o}_r1").is_dir()), None)
                ou = state(od/f"state_{end:03d}.bin", n) if od is not None and complete(od) else None
                fe = errors(fu, ref) if fu is not None else None;oe = errors(ou, ref) if ou is not None else None
                show = lambda e, there: f"{e[0]:.3e} / {e[1]:.3e}" if e else ("stopped" if there else "not run")
                ratio = f"{fe[0]/oe[0]:.2f} / {fe[1]/oe[1]:.2f}" if fe and oe else ""
                print(f"  {k:<15}{show(fe, d is not None):>22}   {o:<11}{show(oe, od is not None):>20}   {ratio}")
            for label, stem in AROUND:
                ad = SWEEP/f"{test}-{stem}-N{n}-s42"/"data"/f"{label}_r1";au = state(ad/f"state_{end:03d}.bin", n)
                if au is not None:e = errors(au, ref);print(f"  {label:<15}{f'{e[0]:.3e} / {e[1]:.3e}':>22}")
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
