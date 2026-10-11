# Tests

One file per test in `sherlock_sweep/configs/`; the initial conditions are in `sherlock_sweep/src/problems.jl`.

## Planar tests (17)

| Test | What it checks |
|---|---|
| `uniform` | A uniform state stays uniform |
| `entropy` | Advection of a density wave: order of accuracy against the exact answer |
| `alfven`, `alfven_oblique` | Circularly polarised Alfvén waves, along the grid and oblique to it |
| `brio_wu`, `brio_wu_rotated` | The Brio–Wu shock tube, along the grid and at 45° |
| `orszag_tang` | The Orszag–Tang vortex |
| `rotor` | The MHD rotor |
| `field_loop`, `field_loop_novz` | Advection of a weak field loop, with and without motion out of the plane |
| `blast_2d_b0`, `blast_2d_b50`, `blast_2d_b100` | A blast wave in fields of increasing strength |
| `kh_uniform`, `kh_localized` | Kelvin–Helmholtz with the field everywhere, and with a field-free layer (Tomida et al. 2026) |
| `current_sheet` | A current sheet with field reversals |
| `gem` | The GEM reconnection challenge (Birn et al. 2001) |

`kh_localized_dtjump` (Tom's kernels only) repeats the localized test and at t = 3 cuts the timestep to 1/40 for 100
steps, as Tomida et al. do, to show what a cleaning speed tied to the timestep does.

## 3D tests

| Test | What it checks |
|---|---|
| `orszag_tang_3d` | Orszag–Tang in three dimensions |
| `field_loop_3d` | The field loop in three dimensions |
| `blast_3d_b0`, `blast_3d_b50`, `blast_3d_b100` | 3D blasts |
| `decay` | Decaying turbulence from Mach 2, three seeds |
| `decay_supersonic` | Decaying turbulence at Mach 9 with a mean field (after Kritsuk et al. 2011, from a smooth random start), two seeds |
| `throughput_ot3d` | Timing only, GPU codes: 3D Orszag–Tang for a short time, 64³ to 512³ including grids divisible by 6 |

`gate_resume` and `gate_resume_3d` are not science: Orszag–Tang at 64 cells a side is stopped after a checkpoint,
resumed, and compared output by output with a run left alone. They must pass before the other runs are released.

## Grids and references

| Tests | Grids | Reference |
|---|---|---|
| Planar | 32² to 512² (Kelvin–Helmholtz from 64²) | exact where there is one, otherwise AthenaK at 4096² |
| 3D | 32³ to 512³ | AthenaK at 512³ for grids up to 256³ |
| Mach 9 decay | 64³ to 512³ | none finer was run |
| Timing | 64³, 96³, 128³, 192³, 256³, 384³, 512³ | |

Timing is repeated 5 times up to 128, 3 times at 256 and once from 512.

## Driven turbulence

A separate pilot ([results](results-turbulence.md)): isothermal hydrodynamic turbulence driven to a steady state at
Mach 3 and 10 on 128³ and 256³, then Mach 30 and 50.

## Planned and not built

Explicit resistivity for GEM, a driven small-scale dynamo (after Pakmor et al. 2026), and 1024³ references on several
GPUs. See [open items](status.md).
