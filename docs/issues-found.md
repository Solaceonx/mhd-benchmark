# Issues found in the codes

Each entry says what was seen, where it comes from, and how far it was checked. Nothing in Tom Abel's source was
changed; the fixes for his kernels are proposals, tried on the CPU in `scripts/solver_checks/` with his functions
loaded unedited.

## RAMSES

**Turbulence driving: the solenoidal projection is wrong.** `find_unitk` in `turb/turb_force_utils.f90` takes the
direction of a wave vector from the absolute values of its components (`ii = TURB_GS - i`) where it needs the signed
ones. Modes whose components differ in sign get a wrong direction. With `comp_frac = 0`, 20 % of the force power is
compressive; it sits in 48 of the 86 driven modes. Present at commit `5f0ed46`, on the `dev` branch on 2026-10-10, and
in RAMSES II (`turb/turb_commons.f90`, commit `aafd2ab`). Checked on saved force fields, then on Sherlock: with the
three lines corrected, RAMSES agrees with Tom's Euler kernel to 1 % in Mach number
([turbulence](results-turbulence.md)). `sherlock_turb/build_ramses.sh fixed` applies the correction to a copy and
refuses unless exactly three lines change. Not reported upstream from this repository yet.

**Negative densities at Mach 30 and 50.** RAMSES applies its density floor to the primitive variables only, so the
conserved state it writes holds negative densities (down to −6 × 10⁻⁷, a few hundred to a few thousand cells at
128³). Anything that takes a square root or a logarithm of the density has to guard against it.

**HLL stalls at Mach 9.** The matched set-up's timestep collapses in `decay_supersonic` at 64³ and 128³; the 256³
and 512³ runs of that set-up were skipped. HLLD gets through.

## Tom's kernels

1. **The CT prototype is first order because a slope is halved twice** (`ct_mhd.jl`: the slope function returns the
   half slope, and the predictor and the interface states halve it again). With the full slope the error falls 17
   to 45 times; with a predictor in all directions it is second order and about 90 times better at 128².
2. **Half-precision tiles lose convergence.** The production kernel stores primitives, interface states and fluxes
   in half precision. Storing differences from a reference value per tile restores second order at the same storage.
3. **Half precision overflows at 65,504.** On magnetised blasts the flux of the cleaning potential exceeds it, and
   at Mach 9 the energy flux does. This is why those kernels stop on blasts, on Mach 9, and on the turbulence
   pilot's nearly isothermal gas.
4. **HLLD's test for the degenerate case never triggers in single precision** (`glmmhd_turb.jl`, the tolerance
   `1.0f-12`). A tolerance relative to the total pressure lets the current sheet and GEM run and changes nothing
   else.
5. **The cleaning speed follows 1/Δt**, so it jumps when a step is shortened to land on an output time. Taking it
   from the CFL step avoids that.
6. **Courant number.** The sweep used 0.25, the code's default is 0.7. On the CPU checks 0.4 is safe for all three
   Riemann solvers, with 1.6 times fewer steps and lower error.
7. **Leakage is not a matter of the cleaning parameters** (damping from none to 0.33 cells changes it by less than
   a factor of two). A cell-centred CT update removes most of it; it needs one more layer of ghost cells, and its
   cost on a GPU was not measured.

Not examined: why PPM stops more often at 512, anything in 3D, the GPU speed of any of these changes.

## AthenaK

**PPM4/HLLD/RK3 stalls at Mach 9.** With no first-order fallback the timestep collapses by about 10⁶ in
`decay_supersonic` and the job runs out of time.

## Athena++

**PPM/HLLD ends in NaN** in the Mach 9 decay, and on 3D Orszag–Tang at 512³ after t = 0.6. The time recorded for
that run is therefore not a time to solution.

## AREPO

**Seven static-mesh runs stop themselves** with AREPO's "time-step of integer size 1 not allowed": `orszag_tang`,
`field_loop` and `field_loop_novz` at 256² and 512², `current_sheet` at 256². The moving mesh finishes every test,
and so does the static mesh at 128². Orszag–Tang at 256² stops at the same point on one rank, and only slightly
later at half the Courant number, so it is neither a parallel artefact nor cured by a smaller step. In the field
loop the magnetic energy has grown from 1.4 × 10⁻⁷ to 2.6 × 10⁻² by then (`paper_figures/arepo.pdf`). The cause is
not established.

## MHD-ALE

Read from the source and partly confirmed by a trial build on Sherlock: in 2D there is no v_z or B_z, so 4 of the 17
planar tests cannot run; the original code's remap allowed periodicity in y only, which the fork extends; one line
takes the wave speed for the timestep from |B|/√ρ without squaring the Alfvén speed, which looks too small for
strong fields and is a question for the author. It builds on Sherlock and its regression suite passes.

## RAMSES II

nvfortran 25.1 fails with an internal compiler error on `cooling/cooling_module.f90` at the Makefile's `-O4`. The
bundle rebuilds a failing object at lower optimisation and records it; the trial build passed with that one object
at `-O1`.

## Sherlock

Collected in [running on Sherlock](sherlock.md#pitfalls).
