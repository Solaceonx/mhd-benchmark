# Comprehensive MHD sweep bundle (Sherlock)

Stage 1 of the comprehensive sweep: every test at every grid with every code. **Start with [SWEEP_RUN.md](SWEEP_RUN.md).**
What was checked before shipping is in [VALIDATION.md](VALIDATION.md); the whole plan (stages 1–4) is `../SWEEP.md`
and the progress site's Sweep plan page.

| Path | What |
|---|---|
| `vendor/` | T. Abel's GLM kernels (`glmmhd_turb.jl`) and CT prototype (`ct_mhd.jl`), byte-identical to his repository (`PROVENANCE.json`) |
| `src/run_suite.jl` | GPU driver for those kernels: time loop, outputs, checkpoint/resume |
| `src/problems.jl` | initial conditions and exact references of every test (shared with the C++ codes via `codes/common/abel_problems.hpp`) |
| `src/lean.jl`, `src/spectra.jl` | diagnostics, error norms, block averages and power spectra, computed on the GPU (or CPU) |
| `codes/external.py` | runs AthenaK, RAMSES, Athena++ (with resume) and converts their outputs |
| `tools/postprocess_external.jl` | the same diagnostics and spectra for the external codes |
| `codes/plan.py`, `codes/estimate.py` | every run, how runs are packed into Slurm jobs, their resources; the calibrated cost model |
| `tools/sweep.py`, `tools/start_sweep.sh`, `jobs/20_sweep_coordinator.sbatch` | gates, waves under the submit limit, resubmission |
| `configs/` | one TOML per test (`tools/make_sweep_configs.py` wrote them from the campaign configs); `gate_*` for the resume gate |
| `jobs/` | Slurm scripts (setup, builds, smoke, resume check, GPU driver, external codes, coordinator) |
