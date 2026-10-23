"""One line per mini-ramses-ism run on scratch: status, time reached, steps, speed, and the end-time error where the
record has one.   python3 summarize.py RESULTS_DIR [PATTERN]"""
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1]);pattern = sys.argv[2] if len(sys.argv) > 2 else "*-miniramses-*-run-N*-s*"
for d in sorted(root.glob(pattern)):
    recs = sorted(d.glob("data/miniramses_*_r1/run.toml"))
    if not recs:print(f"{d.name:<58} no record yet");continue
    m = tomllib.loads(recs[0].read_text());err = (m.get("frames") or [{}])[-1].get("errors", {})
    e = next((f"{k} {v:.3g}" for k, v in err.items() if k.endswith("relative_L1") and k.startswith(("magnetic", "density")) and v), "")
    rate = m.get("million_cell_steps_per_second")
    print(f"{d.name:<58} {m.get('status', '?'):<9} t={m.get('time_reached', 0):<8.4g} steps={m.get('steps', 0):<7} "
          + (f"{rate/1e3:.3g} G upd/s " if rate else "") + e
          + ("" if m.get("status") == "complete" else "  | " + str(m.get("error", "")).split(chr(10))[0][:100]))
