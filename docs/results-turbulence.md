# Driven turbulence

A pilot: hydrodynamic turbulence driven to a steady state, in Tom Abel's Euler and MHD kernels (the MHD kernel with
zero field) on one H100 and in RAMSES on CPUs. Bundle and full description: `sherlock_turb/README.md`. Figures:
the notebook's Turbulence page (`scripts/turb_figures.py`, `scripts/build_turbulence.py`). Data: `turb_results/`.

## Set-up

- Box of side 1, mean density 1, sound speed 1. Nearly isothermal by γ − 1 = 10⁻⁵ with no cooling, so energy is
  conserved and the slope of the total energy is the power taken up from the driving, which in a steady state is
  the dissipation.
- One driving for every code: an Ornstein–Uhlenbeck force on wavenumbers 1 < |k| < 3 with a parabolic weight,
  solenoidal, memory time T = 1/(2 Mach), rescaled to an exact rms every step, no ramp.
- The driving strength is found once per Mach number with RAMSES at 64³ and reused everywhere, so a run is named by
  its target Mach number and what it reaches is a result.
- Runs last 10 T. Averages start at 2 T. A run counts as steady if the second half of that window agrees with the
  first within 10 %.
- One measuring code for all runs (`sherlock_turb/measure.jl`): Mach number, spectra, compressive share, density
  distribution, power, and an effective Reynolds number from the power and the velocity gradients on the grid.

## Mach 3 and 10, 128³ and 256³

32 runs, all steady (second half against first half: 0.1 % to 2.6 %).

- **RAMSES's driving is not what it says.** With the solenoidal setting, one fifth of the force is compressive
  ([issues found](issues-found.md#ramses)). As released, RAMSES sits 2 to 6 % lower in Mach number and 13 to 19 %
  lower in power than Tom's Euler kernel. With three lines corrected it is within 1 % in Mach number and 5 % in
  power, and two seeds of one code differ by as much.
- **Two codes, one scheme, one answer.** The velocity spectra of the Euler kernel and of RAMSES with HLL differ by
  at most 2 % at any k ≥ 4.
- **The solver sets the small scales.** At a quarter of the grid's largest wavenumber RAMSES with HLLC holds about
  twice, and the MHD kernel (LLF) about half, the power of HLL.
- **Effective Reynolds number** roughly doubles per doubling of the grid: at Mach 3 the Euler kernel goes 1,158 to
  2,489, RAMSES HLL 1,119 to 2,423, RAMSES HLLC 1,742 to 3,776.
- **Density distribution.** The width gives b = 0.41 to 0.43 at Mach 3 and 0.46 to 0.55 at Mach 10 on 256³, above
  the 1/3 usually quoted for solenoidal driving, and at Mach 10 it still falls with resolution (0.63 to 0.55 for
  the Euler kernel), so it is not converged.
- **Speed with driving.** Euler kernel 3.5 to 4.3 × 10⁹ cell updates per second, MHD kernel 1.4 × 10⁹; RAMSES
  0.4 to 0.6 × 10⁶ per core. All sixteen GPU runs took 18 minutes together.
- The half-precision kernels cannot run this set-up: an energy flux of order 10⁵ overflows half precision.

## Mach 30 and 50

`sherlock_turb/config_highmach.toml`: γ − 1 = 10⁻⁶, RAMSES with the corrected projection only. The calibration hit
Mach 29.3 and 48.6 at 64³ at the first try. RAMSES ran, and leaves negative densities in its conserved state (a few
hundred to a few thousand cells at 128³, [issues found](issues-found.md#ramses)), which first broke the measurement;
the measurement now treats such cells as being at the floor and counts them. Partial series before that: Mach 29.2
and 48.4. The measured results and the GPU runs are not fetched yet, so these Mach numbers are not on the page.

## Not done

Subsonic driving, compressive driving fractions, 512³, MHD turbulence with a mean field, AthenaK and Athena++
drivers, RAMSES II.
