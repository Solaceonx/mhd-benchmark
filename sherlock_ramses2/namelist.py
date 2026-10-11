"""Namelist of one speed run: python3 namelist.py LEVEL SOLVER [dump] > run.nml

3D Orszag-Tang (build with INIT=OT and the patch of prepare_patch.py) on a uniform periodic 2^LEVEL cube
(levelmin = levelmax), one timestep for the whole grid, run to t = 0.05: the end time of the sweep's throughput test
(throughput_ot3d), so the energies the code prints at the end can be checked against the sweep's records. SOLVER:
hlld (riemann = riemann2d = hlld, Courant 0.8: the sweep's RAMSES "best") or hll (hll, Courant 0.5: "matched").
MonCen slopes in both, as in the sweep. No output files, unless `dump` is given: then the code writes its snapshots
at t = 0 and t = 0.05 (kept for later validation of the fields).
"""
import sys

SOLVERS = dict(hlld=("hlld", "hlld", 0.8), hll=("hll", "hll", 0.5))


T_END = 0.05


def namelist(level, solver, dump=False):
    riemann, riemann2d, courant = SOLVERS[solver]
    octs = sum(8**(l - 1) for l in range(1, level + 1))          # every level from 1 to LEVEL holds its octs
    return f"""&RUN_PARAMS
hydro=.true.
ncontrol=1
nsubcycle=20*1
nstepmax=1000000
nsuperoct=0
verbose=.false.
/

&AMR_PARAMS
levelmin={level}
levelmax={level}
ngridmax={int(1.15*octs) + 1000}
boxlen=1.0
/

&INIT_PARAMS
/

&OUTPUT_PARAMS
{'' if dump else 'foutput=0' + chr(10)}tout={T_END}
bkp_time_hrs=1.0d6
/

&HYDRO_PARAMS
gamma=1.6666666666666667
courant_factor={courant}
slope_type=2
riemann='{riemann}'
riemann2d='{riemann2d}'
/
"""


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[1] not in SOLVERS or (len(a) == 3 and a[2] != "dump"):sys.exit(__doc__)
    sys.stdout.write(namelist(int(a[0]), a[1], dump=len(a) == 3))
