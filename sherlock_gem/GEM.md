# GEM magnetic reconnection (2026-09-30)

The GEM reconnection challenge (Birn et al. 2001, JGR 106, 3715) in its MHD form: no Hall term and no
explicit resistivity, so the reconnection rate is set by each code's numerical resistivity. That is the
point of running it here: it separates GLM from CT (and from AthenaK) in the place where ∇·B cleaning
is most stressed, the X-point.

**Setup** (`problems.jl` "gem", mirrored in `codes/common/abel_problems.hpp`): Harris sheet
Bx = B0 tanh(z/λ), n = n0 sech²(z/λ) + 0.2 n0, n0(Ti+Te) = B0²/2, λ = 0.5 d_i, perturbation
ψ = 0.1 B0 d_i cos(2πx/Lx) cos(πz/Lz), Lx × Lz = 25.6 × 12.8 d_i, run to Ωᵢt = 40. Both solvers only
support the periodic unit square, so it is the usual periodic double sheet: sheets at y = 1/4 and 3/4
(each half box is 25.6 × 12.8), all lengths and times divided by 25.6 (t_end = 1.5625). The field comes
from A_z, so ∇·B = 0 to round-off at t = 0, and both X-points sit at x = ½ (the perturbation is ψ0 cos 2π(x − ½)). Checked on the Mac: total
pressure 0.6 apart from the perturbation's own ±0.025, reconnected flux 0.2 B0 d_i (= 2ψ0) at t = 0 in
Julia and in the C++ port, C++/Julia initial states agree to 1.7e-7.

**Measured:** reconnected flux Φ(t) = ½∫|B_y| dx along each sheet (`recon_flux_lower/upper` in
`diagnostics.csv`, box units; × 25.6 for B0 d_i), plus the usual diagnostics and snapshots.

| Config | What | Grids | Outputs |
|---|---|---|---|
| `gem` | GLM + CT (5 repeats), AthenaK matched and best, AthenaK reference 2048² (λ = 40 cells) | 64², 128², 256² (λ = 1.25–5 cells) | 41 (every Ωᵢ⁻¹) |
| `movie_gem__glm`, `movie_gem__ct` | video runs; truth AthenaK 2048² → 512² | 512² (λ = 10 cells) | 241 |

The accuracy config uses the original GLM c_h policy, as in the rest of the campaign; the movie GLM
config keeps the CFL-step c_h on steps shortened to hit outputs (`glm_ch_at_outputs = "cfl"`).

**Jobs:** 9 (smoke; AthenaK rebuilt with the GEM header into `$SCRATCH/mhd-codes-gem`; 3 GPU-driver
jobs; 4 AthenaK jobs), on `gpu,owners`, H100 SXM5, `--requeue`. About 5 GPU-h. About 7.5 GB returned
(6.8 GB of it movie frames, deleted after rendering). Results in `$SCRATCH/mhd-results-gem`.

## Commands

Mac:

```
cd /Users/lyxia/Projects/research/abel_mhd
scp sherlock_gem.tar.gz sherlock_gem.tar.gz.sha256 lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/
```

Sherlock (clones the pinned sources into `$SCRATCH/mhd-codes-gem` on first use, then submits):

```
cd /scratch/users/lyxia/mhd-study && sha256sum -c sherlock_gem.tar.gz.sha256 && tar -xzf sherlock_gem.tar.gz && cd sherlock_gem && sha256sum -c --quiet BUNDLE_SHA256SUMS && unset MHD_CODES MHD_RESULTS && bash tools/submit_gem.sh
```

Retry single parts: `bash tools/submit_gem.sh --retry gem:gpu gem:athenak:best gem:reference movie_gem__ct:gpu ...`

Download (Mac), then `scripts/rebuild_gpu_page.sh` and
`MOVIE_ARCHIVES=sherlock_gem/returns scripts/render_movies_from_archives.sh --delete-archives`:

```
rsync -a lyxia@login.sherlock.stanford.edu:/scratch/users/lyxia/mhd-study/sherlock_gem/returns/ /Users/lyxia/Projects/research/abel_mhd/sherlock_gem/returns/
```
