"""One line per GIZMO run on scratch: status, time reached, steps, wall time, the end-time error where the record has
one, and why a run stopped.   python3 summarize.py RESULTS_DIR [PATTERN]"""
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1]);pattern = sys.argv[2] if len(sys.argv) > 2 else "*-gizmo-*-run-N*-s*"
for d in sorted(root.glob(pattern)):
    recs = sorted(d.glob("data/gizmo_*_r1/run.toml"))
    if not recs:print(f"{d.name:<52} no record yet");continue
    m = tomllib.loads(recs[0].read_text());err = (m.get("frames") or [{}])[-1].get("errors", {})
    e = next((f"{k} {v:.3g}" for k, v in err.items() if k.endswith("relative_L1") and k.startswith(("magnetic", "density")) and v), "")
    print(f"{d.name:<52} {m.get('status', '?'):<9} t={m.get('time_reached', 0):<8.4g} steps={m.get('steps', 0):<7} "
          f"{m.get('wall_seconds', 0):.0f} s {e}" + ("" if m.get("status") == "complete" else "  | " + str(m.get("gizmo_stopped") or str(m.get("error", "")).split(chr(10))[0])[:110]))
