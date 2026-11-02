"""Each finished run's own clock, from the end of its code's log, for the records written before newcodes_run.py read it
(their solve_seconds was the wall time of srun, start-up included). One line per run: directory, part, seconds, timer.

    module load python/3.12.1 py-numpy/2.2.6_py312; python3 code_clocks.py > code_clocks.tsv     (login node: reads 40 kB a run)

The Mac reads it in scripts/sweep_data.py (NEWCODE_CLOCKS).
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newcodes_run import CODE_TIMERS  # noqa: E402

RESULTS = Path(os.environ.get("NEWCODES_RESULTS", Path(os.environ["SCRATCH"])/"mhd-results-newcodes"))
LOGS = dict(athenapk=("athenapk.log",), cholla=("cholla.log",), pluto=("pluto.0.log", "pluto.log"), enzo=("enzo.log",), phantom=("phantom.log",))

for part in sorted(RESULTS.glob("*-run-N*-s*/data/*_r1")):
    code = part.name.split("_")[0]
    if code not in LOGS or not (part/"run.toml").is_file():continue
    for name in LOGS[code]:
        log = part/name
        if not log.is_file():continue
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 40000));tail = f.read().decode(errors="replace")
        m = re.findall(CODE_TIMERS[code][0], tail)
        if m:
            print(f"{part.parent.parent.name}\t{part.name}\t{float(m[-1])!r}\t{CODE_TIMERS[code][1]}");break
