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

32 runs, all steady (second half against first half: 0.1 % to 2.6 %), plus the 4 runs of the reverse test.

- **RAMSES's driving is not what it says.** With the solenoidal setting, one fifth of the force is compressive
  ([issues found](issues-found.md#ramses)). As released, RAMSES sits 2 to 6 % lower in Mach number and 13 to 19 %
  lower in power than Tom's Euler kernel. With three lines corrected it is within 1 % in Mach number and 5 % in
  power, and two seeds of one code differ by as much. The reverse test agrees: Tom's Euler kernel driven with
  RAMSES's released projection lands on released RAMSES (128³: Mach 3.07 to 3.08 against 3.07, and 9.73 to 9.89
  against 9.84 to 9.93).
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

## Mach 30 and 50, 128³ and 256³

`sherlock_turb/config_highmach.toml`: γ − 1 = 10⁻⁶, RAMSES with the corrected projection only. 24 runs, all steady
(second half against first half within 4.8 %).

- **The codes still agree.** Mach number reached: Euler kernel 29.9 to 30.5 and 49.3 to 49.9, MHD kernel 29.1 to
  29.7 and 48.1 to 49.4, RAMSES 29.4 to 29.9 and 48.8 to 49.3. Power: 2.9 to 3.2 × 10⁴ at Mach 30 and 1.37 to
  1.50 × 10⁵ at Mach 50, RAMSES a few per cent above the kernels at Mach 50.
- **Spectra:** the two HLL codes differ by at most 5 % at any k ≥ 4, and HLLC still holds 1.8 times the power of
  HLL at a quarter of the grid's largest wavenumber.
- **Effective Reynolds number** as at lower Mach numbers: Euler kernel and RAMSES HLL within 5 % of each other
  (Mach 50: 2,074 to 2,133 against 2,033 on 128³, 4,816 to 4,894 against 4,787 on 256³), RAMSES HLLC 1.4 to 1.5
  times higher, the MHD kernel about 20 % lower.
- **Every code sits on the density floor.** Up to 0.5 % of the cells of Tom's kernels are at the floor of 10⁻⁶ on
  256³, and up to 2 % of RAMSES's, where up to 42,000 cells are at or below zero
  ([issues found](issues-found.md#ramses)). The width of the density distribution is then set by the floor (b comes
  out between 0.6 and 1.9 and jumps from seed to seed), so the density statistics of these runs are not usable.
  Mach number, power and effective Reynolds number are much less affected.
- **Heating stays small:** the sound speed ends 0.2 % (Mach 30) and 0.6 % (Mach 50) above its start.
- Speed is as before: 3.6 to 4.3 × 10⁹ cell updates per second for the Euler kernel, 1.4 × 10⁹ for the MHD kernel.

## Not done

Subsonic driving, compressive driving fractions, 512³, MHD turbulence with a mean field, AthenaK and Athena++
drivers, RAMSES II.
