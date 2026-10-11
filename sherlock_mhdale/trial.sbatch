#!/bin/bash
# Does MHD-ALE (T. Abel's fork) work here, with the suite added? (see README.md)
#   1. build everything;  2. the fork's own regression suite;  3. the starting state of every test of the suite, on
#   the suite's grid, against the suite's own;  4. a few steps with remaps in the periodic box, on two ranks.
# Ends with exit code 0 only if all four are clean: the production array (mhdale.sbatch) waits on that.
# Submitted by check.sh. The report is logs/trial_report.txt (bash -l status.sh shows it).
#SBATCH --job-name=mhdale-trial
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
here="$SLURM_SUBMIT_DIR"; report="$here/logs/trial_report.txt"; : > "$report"; bad=0
say() { echo "$@" | tee -a "$report"; }
export SWEEP_BUNDLE="${SWEEP_BUNDLE:-/scratch/users/lyxia/mhd-study/sherlock_sweep}"
cd "$SWEEP_BUNDLE" && source jobs/common.sh && prepare_environment && prepare_codes || { say "the sweep bundle's environment could not be set up"; exit 2; }
load_modules "${MHDALE_COMPILER_MODULES:-gcc/12.4.0 openmpi/4.1.6}"
root="$MHD_CODES/mhdale-yipihey"; trial="${MHDALE_TRIAL_DIR:-$SCRATCH/mhd-results-mhdale-trial/yipihey}"; mkdir -p "$trial"
say "== MHD-ALE (yipihey fork) trial, job ${SLURM_JOB_ID:-?} on $(hostname), $(date -u)"

# ---- 1. build
bash "$here/build.sh" 2>&1 | tee -a "$report"
[[ -x "$root/build-stock/MHD" ]] || { say "TRIAL FAILED: the fork did not build"; exit 1; }

# ---- 2. the fork's own regression suite (tests/regression.py: about 28 short runs on one and two ranks)
say ""; say "== 2. the fork's regression suite"
( cd "$root/build-stock" && rm -rf "$trial/regression" && MPIEXEC="${MHDALE_MPIEXEC:-bash $here/srun_np.sh}" python3 tests/regression.py --output "$trial/regression" ) > "$trial/regression.log" 2>&1; rc=$?
say "   exit code $rc; $(grep -c ' PASS$' "$trial/regression.log") runs passed, $(grep -c 'FAIL' "$trial/regression.log") lines with FAIL; last line: $(grep -v '^ *$' "$trial/regression.log" | tail -n 1 | cut -c1-200)"
if [[ $rc -ne 0 ]]; then bad=1; grep -n -i -E "FAIL|error|abort" "$trial/regression.log" | head -n 8 | cut -c1-220 | tee -a "$report"; fi

[[ -x "$root/build-abel/MHD" ]] || { say "TRIAL FAILED: the fork with the suite added did not build (see above)"; exit 1; }

# ---- 3. the starting state of each test, on the suite's grid, against the suite's own
N="${MHDALE_TRIAL_N:-64}"; mesh="$trial/unit_$((N / 2)).mesh"; python3 "$here/make_mesh.py" "$((N / 2))" "$mesh"
source "$here/options.sh"                               # the run options, shared with the production runs
prepare() {      # test -> spec, state and output times in $trial/$test; sets prefix, t_end, times
    local test="$1" config="$SWEEP_BUNDLE/configs/$1.toml"
    prefix="$trial/$test/ic"; mkdir -p "$trial/$test"
    "$MHD_JULIA" --startup-file=no "$SWEEP_BUNDLE/tools/write_spec.jl" "$config" "$N" "$prefix" --seed 42 --state --plane > "$trial/$test/write_spec.log" 2>&1 || return 1
    times="$trial/$test/output_times.dat"
    t_end="$(python3 -c "
import sys, tomllib
c = tomllib.load(open(sys.argv[1], 'rb')); f = int(c['frames']); t = float(c['t_end'])
open(sys.argv[2], 'w').write(''.join(repr(t*k/(f - 1)) + '\n' for k in range(1, f)))
print(repr(t))" "$config" "$times")"
}
run_abel() {     # ranks, test, extra options... -> $out
    local ranks="$1" test="$2"; shift 2
    out="$trial/$test/periodic${ABEL_TAG:-}"; mkdir -p "$out"; rm -f "$out"/abel_frame_*
    ABEL_SPEC="$prefix.spec" ABEL_BOX=periodic ABEL_N="$N" srun -n "$ranks" "$root/build-abel/MHD" -p 100 -dim 2 -m "$mesh" -rs 0 $MHDALE_ORDERS \
        $MHDALE_PERIODIC_ALE -tf "$t_end" -ptf "$times" -od "$out" "$@" > "$out/MHD.out" 2>&1
}
say ""; say "== 3. starting states at $N^2 ($((N / 2))^2 elements), periodic box: MHD-ALE's, sampled on the suite's grid, against the suite's own"
for test in $MHDALE_TESTS; do
    prepare "$test" || { say "$test: the suite's state could not be written (see $trial/$test/write_spec.log)"; bad=1; continue; }
    ABEL_ONLY_INITIAL=1 run_abel 1 "$test"; rc=$?
    if [[ -f "$out/abel_frame_000.bin" ]]; then python3 "$here/ic_check.py" "$test" "$prefix" "$N" "$out" 2>&1 | tee -a "$report"; [[ ${PIPESTATUS[0]} -eq 0 ]] || bad=1
    else bad=1; say "$(printf '%-34s' "$test") NO STATE WRITTEN (exit code $rc): $(grep -v '^ *$' "$out/MHD.out" | tail -n 3 | tr '\n' ' ' | cut -c1-260)"; fi
done
say "   tests with an out-of-plane velocity or field (the 2D code has neither; expected to be refused):"
for test in uniform alfven alfven_oblique field_loop; do
    prepare "$test" || { say "$test: the suite's state could not be written"; continue; }
    ABEL_ONLY_INITIAL=1 run_abel 1 "$test"
    say "   $(printf '%-18s' "$test") $(grep -m1 'NOT REPRESENTABLE' "$out/MHD.out" | cut -c1-150 || true)$([[ -f "$out/abel_frame_000.bin" ]] && echo ' BUT A STATE WAS WRITTEN')"
done

# ---- 4. a few steps with remaps in the periodic box, on two ranks
steps="${MHDALE_TRIAL_STEPS:-30}"
say ""; say "== 4. $steps steps on 2 ranks in the periodic box, remap every 10 steps"
for test in orszag_tang rotor brio_wu field_loop_novz; do
    prepare "$test" || { bad=1; continue; }
    t0=$(date +%s.%N); ABEL_TAG=_steps ABEL_DUMP_LAST=1 run_abel 2 "$test" -ms "$steps"; rc=$?; t1=$(date +%s.%N)
    python3 - "$out" "$rc" "$t0" "$t1" "$test" "$steps" <<'PY' 2>&1 | tee -a "$report"
import re, sys
from pathlib import Path
out, rc, t0, t1, label, want = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6])
log = (out/"MHD.out").read_text(errors="replace");steps = re.findall(r"^step (\d+),\s*t = ([0-9.eE+-]+),\s*dt = ([0-9.eE+-]+)", log, re.M)
info = lambda f: dict(l.split(None, 1) for l in (out/f).read_text().splitlines() if l.strip()) if (out/f).is_file() else {}
a, b = info("abel_frame_000.txt"), info("abel_frame_999.txt")
line = f"   {label:18s} exit code {rc}, {len(steps)} steps in {t1 - t0:.1f} s"
if steps:line += f", t = {float(steps[-1][1]):.4g}, last dt {float(steps[-1][2]):.3g}, {log.count('-----REMAP DONE-----')} remaps, {log.count('Repeating step')} repeated steps"
if a and b:line += f"; mass x{float(b['mean_rho'])/float(a['mean_rho']):.6f}, energy x{float(b['mean_E'])/float(a['mean_E']):.6f}, missing samples {float(b['samples_missing']):.0f}"
print(line)
# The step limit (-ms) counts attempts, repeated ones included, and a run whose last attempt is repeated ends without
# writing its last state: neither is a fault. Clean = the code ended by itself, without an abort, having stepped.
abort = re.search(r"MFEM abort|Verification failed|MPI_ABORT was invoked|Segmentation fault", log)
ok = rc == 0 and not abort and len(steps) + log.count("Repeating step") >= want - 1 and len(steps) >= want//2
if a and not b:print("      (no last state written: the last attempt within the step limit was a repeated one)")
if not ok:print("      NOT CLEAN; end of its output: " + " | ".join(l.strip()[:150] for l in log.splitlines()[-6:] if l.strip()))
sys.exit(0 if ok else 1)
PY
    [[ ${PIPESTATUS[0]} -eq 0 ]] || bad=1
    [[ -s "$out/B_divergence_error.dat" ]] && say "      its own divergence error of B, first and last step: $(head -n 1 "$out/B_divergence_error.dat" | awk '{print $2}'), $(tail -n 1 "$out/B_divergence_error.dat" | awk '{print $2}')"
done
say ""
if [[ $bad -eq 0 ]]; then say "== TRIAL PASSED $(date -u): the production runs (mhdale.sbatch) may start"; exit 0; fi
say "== TRIAL NOT CLEAN $(date -u): the production runs do not start"; exit 1
