"""The "Nested turbulence" page: Tom Abel's nested-grid code (NestedMHD) on Sherlock's H100s, a nest against a uniform grid
at its finest resolution in driven Mach-10 MHD.
    python3 scripts/build_nested.py [--page-only]
Draws the figures and movies first (scripts/nested_figures.py) unless --page-only; every number on the page comes from
progress/media/nested/nested.json, which that script writes from the records in nested_results/."""
import html
import json
import struct
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"progress"
MEDIA = SITE/"media"/"nested"
sys.path.insert(0, str(ROOT/"scripts"))
from publish_site import PAGES  # noqa: E402

PAGE = "nested.html"
e = html.escape


def png_width(path):
    head = path.read_bytes()[:24]
    return struct.unpack(">I", head[16:20])[0] if head[:8] == b"\x89PNG\r\n\x1a\n" else 0


def figure(i, name, title, caption, finding=None):
    f = MEDIA/f"{name}.png"
    if not f.is_file():
        return ""
    width = png_width(f) // 2; src = f"media/nested/{name}.png?v={int(f.stat().st_mtime)}"
    out = [f'<article class="card paper-fig" id="{name}"><div class="fig-head"><span class="fig-no">{i}</span><h3>{e(title)}</h3></div>',
           f'<div class="fig-pair"><figure class="ours" style="flex-basis:{int(.8 * width)}px;max-width:{width}px"><a class="image-link" href="{src}">'
           f'<img src="{src}" alt="{e(title)}" loading="lazy"></a></figure></div><p>{caption}</p>']
    if finding:
        out.append(f'<p class="finding"><strong>In these runs:</strong> {finding}</p>')
    return "".join(out) + "</article>"


def movie(case, m, title, text, rerun=True):
    mp4 = MEDIA/f"{case}-movie.mp4"
    if not mp4.is_file():
        return ""
    v = int(mp4.stat().st_mtime); src = f"media/nested/{case}-movie"
    poster = f"{src}.jpg" if (MEDIA/f"{case}-movie.jpg").is_file() else f"{src}.png"
    note = (" This is a rerun of the same set-up with outputs every 0.002; stopping that often changes the time steps slightly, so it does not follow the runs "
            "of the figures below in detail.") if rerun else ""
    return f'''<article class="card test-movie" id="movie-{case}"><h3>{e(title)}</h3>
<video controls loop muted playsinline preload="none" poster="{poster}?v={v}" style="width:100%;height:auto;max-height:80vh;object-fit:contain"
aria-label="{e(title)}"><source src="{src}.mp4?v={v}" type="video/mp4"><a href="{src}.mp4">Download the movie</a></video>
<p>{text} {m["frames"]} frames, one every {m["dt"]:g} from t = 0 to {m["t_last"]:g}{"" if m["finished"] else " (the run stopped there)"}, played at 25 per second.
Every level is drawn in its own place at the finest resolution (coarser levels as blocks); dashed squares mark the finer levels. The colour scales are fixed.{note}</p>
<p><a href="{src}.mp4?v={v}" download>Download MP4 · {m["mb"]:g} MB</a></p></article>'''


def f2(x, d=2):
    return f"{x:.{d}f}"


def build(J):
    R = J["runs"]; tu = J["t_uniform"]; cu = J["cell_updates"]; dt = J["dt_finest"]; vc, vb = J["variance_share_central"], J["variance_share_box"]
    pc = lambda c, q: f"{100 * vc[c][q]:.0f} %" if vc[c][q] >= 0.1 else f"{100 * vc[c][q]:.1f} %"
    wall = {c: R[c]["wall"] for c in R}
    agree = J["gpu"]["agreement"]; speed = J["gpu"]["speed"]
    unif32 = [s["rate"] for s in speed if s["setup"].startswith("uniform") and s["T"] == "Float32"]
    nest_ct = next(s["rate"] for s in speed if s["setup"].startswith("nest") and s["T"] == "Float32" and s["mode"] == "ct")
    nest_glm = next(s["rate"] for s in speed if s["setup"].startswith("nest") and s["T"] == "Float32" and s["mode"] == "glm")
    ratio_wall = J["wall_02_to_tu"]["uni256"] / J["wall_02_to_tu"]["nest64"]
    ratio_steps = J["steps_to_tu"]["uni256"][0] / J["steps_to_tu"]["nest64"][-1]
    bt = J["brms_box_at_tu"]; bc = J["brms_central_range"]; mb = J["mach_box"]
    number = next(i for i, (src, _, _) in enumerate(PAGES, 1) if src == PAGE)
    W = J["window"]; C = J["capped"]; G = J["ergodic"]; B = J["big"]; cr = C["runs"]; sh = C["share_central"]; sr = C["spectrum_ratio"]; ln = C["lnrho_central"]
    cpc = lambda c, q: f"{100 * sh[c][q]:.0f} %" if sh[c][q] >= 0.1 else f"{100 * sh[c][q]:.1f} %"
    chips = ["3–4 levels, refinement ratio 2", "driven Mach 10, plasma β 200", "constrained transport, HLL, Courant 0.25",
             f"nest 3 × 64³ to t = 1 in {cr['nest64']['wall'] / 60:.0f} min", f"uniform 256³ to t = 1 in {cr['uni256']['wall'] / 60:.0f} min",
             f"big nest (1024³-equivalent) to t = {B['t']:g} in {B['wall'] / 3600:.1f} h", f"{len(G['seeds'])}-seed ergodicity test",
             f"div B ≤ {max(max(J['divb'].values()), B['divb']):.0e}", "one H100 per run"]
    out = [f'''
<div class="page-heading"><div><div class="eyebrow"><span>{number:02d} · Nested turbulence</span><span class="badge">nests against uniform grids</span></div>
<h1>Nested turbulence: Tom Abel's NestedMHD on a GPU</h1>
<p>A nest puts a finer grid inside a coarser one: here three or four levels, each twice as fine as the one around it, centred in a periodic box, so the centre
of the box is computed at four or eight times the resolution of the rest. Tom's question: does the finest level of a nest reproduce what a uniform grid at
that resolution gives, for much less computing? We ran his code (<a href="https://github.com/yipihey/NestedTurbulence">NestedTurbulence</a>, MIT) on
Sherlock's H100s, which it had not run on before, fixed what stopped it in 3-D, and set nests against uniform grids in driven Mach-10 MHD turbulence.</p></div>
<a class="method-link" href="../sherlock_nested/README.md">Set-up, fixes and jobs <span>↗</span></a></div>
<div class="chips">{"".join(f'<span class="chip">{e(c)}</span>' for c in chips)}</div>
<div class="card"><h3>The short answer</h3><ul>
<li><strong>The nest is {cr["uni512"]["wall"] / cr["nest128"]["wall"]:.0f}–{cr["uni256"]["wall"] / cr["nest64"]["wall"]:.0f} times faster to the same time.</strong> With the same cap on the Alfvén speed in both
(30 c<sub>s</sub>; it adds at most {max(r["mass_added"] for r in cr.values()):.0e} of the mass), every run now reaches its end: the 3 × 64³ nest t = 1 in
{cr["nest64"]["wall"]:.0f} s against {cr["uni256"]["wall"] / 60:.0f} min for the uniform 256³ grid, the 3 × 128³ nest t = 0.5 in {cr["nest128"]["wall"] / 60:.0f} min
against {cr["uni512"]["wall"] / 3600:.1f} h for the uniform 512³ grid, with {cr["uni256"]["cell_updates"] / cr["nest64"]["cell_updates"]:.0f} and
{cr["uni512"]["cell_updates"] / cr["nest128"]["cell_updates"]:.0f} times fewer cell updates (figure 2).</li>
<li><strong>Density agrees.</strong> Over the central cube from t = {C["t_stat"]:g} to 1, ln ρ has mean {ln["nest64"]["mean"]:.2f} and spread {ln["nest64"]["sigma"]:.2f} in the
nest and {ln["uni256"]["mean"]:.2f} and {ln["uni256"]["sigma"]:.2f} in the uniform grid, and k = 32–96 holds {cpc("nest64", "rho")} and {cpc("uni256", "rho")} of
the density variance. The gap we reported first ({pc("nest64", "rho")} against {pc("uni256", "rho")}) came from the early outputs alone: before t ≈ 0.35 the finest
level has not yet built its small scales.</li>
<li><strong>Velocity may be short at small scales; it takes a window of time to tell.</strong> Each snapshot's spectrum divided by its variance in the central
cube, then averaged over every snapshot from t = {C["t_stat"]:g} on (one per crossing time): over k = 8–96 the nest's velocity is
{W["nest64/uni256"]["v"]["shape"]["ratio"]:.2f} of the uniform grid's at 256³ (95 %: {W["nest64/uni256"]["v"]["shape"]["ci95"][0]:.2f}–{W["nest64/uni256"]["v"]["shape"]["ci95"][1]:.2f},
{W["nest64/uni256"]["snapshots"]} snapshots) and {W["nest128/uni512"]["v"]["shape"]["ratio"]:.2f} at 512³ ({W["nest128/uni512"]["v"]["shape"]["ci95"][0]:.2f}–{W["nest128/uni512"]["v"]["shape"]["ci95"][1]:.2f},
{W["nest128/uni512"]["snapshots"]} snapshots); field and density are within their intervals of 1. A single pair of snapshots at one time gives anything from
{min(W[p]["v"]["shape"]["single_min"] for p in W):.1f} to {max(W[p]["v"]["shape"]["single_max"] for p in W):.1f} for the velocity: compared at one instant the two grids
can look equal or very different (figure 3). Longer windows are set up (to t = 3 at 256³, 1.5 at 512³).</li>
<li><strong>One point, over time, samples the volume (ergodicity).</strong> In {len(G["seeds"])} runs with different driving seeds, the time mean of ln ρ at one point is
{G["ensemble_time_mean"]:.2f} and the volume mean {G["ensemble_volume_mean"]:.2f} ({abs(G["z_ensemble"]):.1f} standard errors apart); the mass-weighted mean,
{G["ensemble_mass_mean"]:+.2f}, is far off. The correlation time at the point is {G["tau_s"]:.3f}, about a tenth of a crossing time (figure 4).</li>
<li><strong>A big nest runs too.</strong> Root 128³ plus three levels of 192³ (1024³-equivalent in the central 3/16 of the box) reached t = {B["t"]:g} in
{B["wall"] / 3600:.1f} h on one H100 (figure 5).</li></ul></div>''']
    M = J.get("movies", {})
    if "deep192" in M:
        out.append(movie("deep192", M["deep192"], "Movie: the big nest, root 128³ + 3 levels of 192³ (finest 1024³-equivalent)",
                         "Density and field strength in the mid-plane z = 1/2; left the whole box with every level pasted in, right the finest level alone.", rerun=False))
    if "nest128" in M:
        out.append(movie("nest128", M["nest128"], "Movie: nest 3 × 128³ (finest 512³-equivalent)",
                         "Density and field strength in the mid-plane z = 1/2, from the uniform start to developed turbulence."))
    if "nest64" in M:
        out.append(movie("nest64", M["nest64"], "Movie: nest 3 × 64³ (finest 256³-equivalent)",
                         "Density and field strength in the mid-plane z = 1/2, from the uniform start to developed turbulence."))
    if not M:
        out.append('<div class="card"><h3>Movies</h3><p>The movie runs (a mid-plane slice of every level every 0.002, from t = 0 to 1) are on Sherlock now; '
                   'they are added here when they finish.</p></div>')
    sg = G["seeds"]; within = sum(abs(r["time_mean"] - r["volume_mean"]) <= r["err"] for r in sg)
    figs_new = [
        ("nested-overview", "What the 3 × 128³ nest looks like",
         "The mid-plane at t = 0.5. Left: the whole box, level 1 (128³ across the box) with level 3 (the central quarter, 512³-equivalent) pasted in; level 2 "
         "is computed but was not saved in these runs (the movies show it). Right: level 3 alone. Top ln ρ, bottom log₁₀ |B|.", None),
        ("nested-capped", "Nest against uniform grid, both capped",
         "The comparison again, now with the same safeguard in both: where the Alfvén speed B/√ρ would exceed 30 c<sub>s</sub>, the density is raised to bring it "
         "down, at the same velocity. Dashed: finest resolution 256³ (3 × 64³ nest and uniform 256³, to t = 1). Solid: finest 512³ (3 × 128³ nest and uniform 512³, "
         f"to t = 0.5). Top: each field's variance per unit ln k in the central cube (Hann-windowed, the shaded wave numbers are set by the window), averaged over the "
         f"snapshots from t = {C['t_stat']:g} on. Bottom: the density PDF of the central cube over the same outputs, the central cube's rms field, and the wall-clock time.",
         f"the density PDFs lie on top of each other at 256³; at 512³ the nest's is a little narrower ({ln['nest128']['sigma']:.2f} against {ln['uni512']['sigma']:.2f}) "
         f"over only {cr['nest128']['snapshots']} snapshots. The velocity spectra of the nests fall below the uniform grids' from k ≈ 15 on, by about "
         f"{100 * (1 - sr['nest64/uni256']['v']['median']):.0f} % at 256³ and {100 * (1 - sr['nest128/uni512']['v']['median']):.0f} % at 512³: the finest level is "
         "fed through its boundary by coarser levels that do not carry those scales. The central cube's rms field swings by a factor of 3–4 in time in every run, "
         "so its spectra need longer runs than these to compare. The cap adds "
         f"{cr['uni512']['mass_added']:.0e} of the mass in the uniform 512³ grid and {cr['nest64']['mass_added']:.0e} or less in the others."),
        ("nested-window", "Spectra over a window of time, with their errors",
         "The capped runs again, every snapshot from t = 0.3 on (one per crossing time, t = L/(2·10 c<sub>s</sub>) = 0.05). Each snapshot's spectrum in the central cube "
         "is divided by its own variance there (the cube's total variance swings with the large eddies passing through it), then the snapshots are averaged. "
         "Intervals: bootstrap over the snapshots of each run (neighbouring snapshots are uncorrelated: lag-1 correlation "
         f"{min(min(W[p][q]['lag1'].values()) for p in W for q in ('v', 'B', 'rho')):+.2f} to {max(max(W[p][q]['lag1'].values()) for p in W for q in ('v', 'B', 'rho')):+.2f}). "
         "Top: the windowed spectra, 95 % intervals. Middle and bottom: the nest's over the uniform grid's for each pair; dark, the windowed ratio with its 95 % interval; "
         "pale, the 10–90 % range of the ratio of one snapshot pair taken at the same time.",
         f"averaged over the window, the nests' velocity over k = 8–96 is {W['nest64/uni256']['v']['shape']['ratio']:.2f} of the uniform grid's at 256³ "
         f"(95 %: {W['nest64/uni256']['v']['shape']['ci95'][0]:.2f}–{W['nest64/uni256']['v']['shape']['ci95'][1]:.2f}) and {W['nest128/uni512']['v']['shape']['ratio']:.2f} at 512³ "
         f"({W['nest128/uni512']['v']['shape']['ci95'][0]:.2f}–{W['nest128/uni512']['v']['shape']['ci95'][1]:.2f}): short at 256³, just outside the interval, and not measurable at 512³ "
         f"with {W['nest128/uni512']['snapshots']} snapshots. The field ({W['nest64/uni256']['B']['shape']['ratio']:.2f} and {W['nest128/uni512']['B']['shape']['ratio']:.2f}) and density "
         f"({W['nest64/uni256']['rho']['shape']['ratio']:.2f} and {W['nest128/uni512']['rho']['shape']['ratio']:.2f}) are within their intervals of 1. One pair of snapshots "
         f"at one time puts the velocity ratio anywhere from {min(W[p]['v']['shape']['single_min'] for p in W):.2f} to {max(W[p]['v']['shape']['single_max'] for p in W):.2f}, "
         f"and the raw power (not divided by the variance) from {min(W[p]['v']['raw']['single_min'] for p in W):.2f} to {max(W[p]['v']['raw']['single_max'] for p in W):.1f}: "
         "the comparison only means something over a window. To shrink the interval by two needs four times the snapshots: the long runs (55 snapshots at 256³, 25 at 512³) are ready to submit."),
        ("nested-ergodic", "Ergodicity: one point over time against the volume",
         f"Tom's test (his report, chapter 7) in 3-D: {len(sg)} runs of the 3 × 64³ nest to t = 1, identical but for the random seed of the driving, each with a probe "
         "in the finest-level cell at the box centre, read every second step of that level. Left: the PDF of s = ln ρ at the probe over time (all seeds pooled) and over "
         f"the finest level's volume. Middle: per seed, the probe's time mean of s with its error σ/√N<sub>eff</sub>, where N<sub>eff</sub> = T/2τ<sub>int</sub> "
         f"(about {G['n_eff']:.0f} independent samples per run), against the volume mean of the finest level and the mass-weighted mean. Right: the autocorrelation of "
         "s at the probe.",
         f"the time mean matches the volume mean within its error in {within} of {len(sg)} runs, and the {len(sg)} time means scatter by {G['spread_time_means']:.2f}, "
         f"no more than the errors predict ({G['mean_err']:.2f}). So a point in the turbulence samples the volume-weighted distribution, not the mass-weighted one "
         f"(mean {G['ensemble_mass_mean']:+.2f}). The correlation time, {G['tau_s']:.3f}, is about a tenth of the crossing time L / (10 c<sub>s</sub>) = 0.1. "
         f"Each run took {G['wall'] / 60:.1f} min."),
        ("nested-big", "The big nest: root 128³ + 3 levels of 192³",
         "A four-level nest with each refined level wider than its parent's half: root 128³ over the box, then 192³ cells over 3/4, 3/8 and 3/16 of the box "
         f"(the finest at 1024³-equivalent), {B['precision']}, Alfvén speed capped at {B['va_max']:g} c<sub>s</sub>, to t = {B['t']:g}. Top: each level's own "
         "spectrum (Hann-windowed for the refined levels), averaged over t = 0.2–0.5. Bottom: the mean time step of each level against the finest level of the "
         "uncapped 3 × 128³ nest, the mass the cap and the floor add, and the rms Mach number.",
         f"{B['wall'] / 3600:.1f} h on one H100 for {B['cell_updates']:.1e} cell updates, checkpointed every half hour. With the cap the finest step never fell "
         f"below {B['dt_min'][str(B['levels'])]:.0e} (the uncapped 3 × 128³ nest went to {dt['nest128']['min']:.0e}), and the cap added {B['mass_added']:.0e} of the mass. "
         f"Each finer level holds more of its variance at k = 32–96: the density {', '.join(f'{100 * B['share'][f'rho_L{l}']:.0f} %' for l in range(2, B['levels'] + 1))} "
         f"on levels 2–{B['levels']}."),
    ]
    figs_old = [
        ("nested-slices", f"Nest and uniform grid at the same time (t = {tu:g})",
         f"Mid-plane slices of the 3 × 64³ nest (top) and the uniform 256³ grid (bottom): the whole box at 64³ (the uniform grid averaged over 4³ blocks) and the central "
         f"cube at 256³ cells across the box, the nest's finest level. Both runs have the same driving with the same random seed.",
         "the pictures differ: a turbulent flow is chaotic, so small differences between the two grids grow until the flows no longer match point by point, within "
         "a few crossing times (L / 10 c<sub>s</sub> = 0.1). They are compared by their statistics."),
        ("nested-stats", "Statistics",
         "Top: PDFs over the central cube (the nest's finest level, and the same cells of the uniform grid). Bottom: rms speed and field over the whole box at the "
         "nest's root resolution, and the rms field of the central cube.",
         f"the Mach number holds at about 10 in every run (whole-box averages {f2(mb['nest64'])}, {f2(mb['uni256'])} and {f2(mb['nest128'])}). The whole-box field grows "
         f"in the same way in the nest and the uniform grid ({f2(bt['nest64'])} and {f2(bt['uni256'])} at t = {tu:g}). The central cube is too small to compare its field: "
         f"its rms swings between {f2(bc['nest64'][0])} and {f2(bc['nest64'][1])} in the nest and {f2(bc['uni256'][0])} and {f2(bc['uni256'][1])} in the uniform grid. "
         "The uniform grid has more weak-field cells (log₁₀ |B| below −2)."),
        ("nested-spectra", "Normalized power spectra",
         "Each field's variance per unit ln k, k E(k)/σ² (that is 4π k³P(k)/σ² for the 3-D power P(k)), so the area under a curve over ln k is 1 and runs "
         "with different total power compare by where their variance lives. Velocity, magnetic field and density, averaged over the snapshots (every 0.05). "
         "Top: the whole periodic box at the nest's root resolution. Bottom: the central cube at the finest resolution; it is not periodic, so it is multiplied "
         "by a Hann window first, which sets the lowest wave numbers (shaded). The nest 3 × 128³ is averaged to t = 0.5, the other two to the uniform run's last "
         "output; the field is still growing, so its spectra are not compared across those times.",
         f"over the whole box, outside its finer levels the nest is a 64³ simulation: k = 8–24 holds {100 * vb['nest64']['v']:.1f} % of its velocity variance "
         f"against {100 * vb['uni256']['v']:.1f} % in the uniform grid. In the central cube, at the same resolution, k = 32–96 holds {pc('nest64', 'rho')} of the "
         f"density variance in the nest's finest level against {pc('uni256', 'rho')} in the uniform grid ({pc('nest128', 'rho')} in the 3 × 128³ nest, twice as fine); "
         f"the field differs less ({pc('nest64', 'B')} against {pc('uni256', 'B')}). Gas crosses the central cube (a quarter of the box) in about 0.025 at Mach 10, "
         "probably too short for the cascade to rebuild the small scales the coarser levels never had. The capped runs to t = 1 (figure 2) show the density gap closes "
         "after t ≈ 0.35; the velocity stays somewhat short."),
        ("nested-cost", "Cost and time step",
         "Left: wall-clock time on one H100. Middle: the mean time step of the finest grid between outputs. Right: the lowest density in the central cube, and over "
         "the whole uniform 256³ grid.",
         f"to t = {tu:g} the uniform 256³ grid took {J['steps_to_tu']['uni256'][0]:,} steps, {ratio_steps:.1f} times as many as the nest's finest level "
         f"({J['steps_to_tu']['nest64'][-1]:,}). Its step fell to {dt['uni256']['min']:.0e} while the 3 × 64³ nest's never went below {dt['nest64']['min']:.0e}. "
         f"The cause is nearly empty cells: where ρ approaches the floor (10⁻⁶) the Alfvén speed B/√ρ becomes huge. The 3 × 128³ nest, with a finer finest level, "
         f"meets them as well (its step fell to {dt['nest128']['min']:.0e} near t = 0.29), so they come with resolution, not with the uniform grid itself."),
    ]
    out += [figure(i + 1, *f) for i, f in enumerate(figs_new)]
    out.append('<div class="card"><h3>The first comparison, without the cap</h3><p>The figures below are from the runs without a cap on the Alfvén speed. '
               f'There the uniform 256³ grid stalled at t = {tu:g} and the uniform 512³ grid reached no output: nearly empty cells made the time step collapse. '
               'They are kept for the record; the capped runs above replace them.</p></div>')
    out += [figure(len(figs_new) + i + 1, *f) for i, f in enumerate(figs_old)]
    figs = figs_new + figs_old
    def row(name, finest, prec, r, steps, cu):
        return (name, finest, prec, f"{r['t']:g}" if r.get("t") is not None else "–", f"{r['wall']:,.0f} s", " / ".join(f"{s:,}" for s in steps),
                f"{cu:.2e}", f"{cu / r['wall']:.2e}")
    rows = [row("nest 3 × 64³, capped", "256³ in the central 1/4", "Float64", cr["nest64"], cr["nest64"]["steps"], cr["nest64"]["cell_updates"]),
            row("uniform 256³, capped", "256³ everywhere", "Float64", cr["uni256"], cr["uni256"]["steps"], cr["uni256"]["cell_updates"]),
            row("nest 3 × 128³, capped", "512³ in the central 1/4", "Float32", cr["nest128"], cr["nest128"]["steps"], cr["nest128"]["cell_updates"]),
            row("uniform 512³, capped", "512³ everywhere", "Float32", cr["uni512"], cr["uni512"]["steps"], cr["uni512"]["cell_updates"]),
            row(f"big nest {B['root']}³ + {B['levels'] - 1} × {B['n']}³, capped", "1024³ in the central 3/16", B["precision"], B, B["steps"], B["cell_updates"]),
            (f"ergodicity, {len(sg)} seeds of 3 × 64³", "256³ in the central 1/4", "Float64", "1", f"{G['wall']:.0f} s each", "–", "–", "–"),
            ("nest 3 × 64³", "256³ in the central 1/4", "Float64", "1", f"{wall['nest64']:.0f} s", " / ".join(f"{s:,}" for s in R["nest64"]["steps"]),
             f"{cu['nest64']:.2e}", f"{cu['nest64'] / wall['nest64']:.2e}"),
            ("uniform 256³", "256³ everywhere", "Float64", f"{R['uni256']['t']:g} (stalled)", f"{wall['uni256']:.0f} s", f"{R['uni256']['steps'][0]:,}",
             f"{R['uni256']['steps'][0] * 256 ** 3:.2e}", f"{R['uni256']['steps'][0] * 256 ** 3 / wall['uni256']:.2e}"),
            ("nest 3 × 128³", "512³ in the central 1/4", "Float32", "0.5", f"{wall['nest128']:.0f} s", " / ".join(f"{s:,}" for s in R["nest128"]["steps"]),
             f"{cu['nest128']:.2e}", f"{cu['nest128'] / wall['nest128']:.2e}"),
            ("uniform 512³", "512³ everywhere", "Float32", "below 0.2 (no output)", "–", "–", "–", "–")]
    cols = ("run", "finest resolution", "precision", "reached t", "wall", "steps per level (coarsest first)", "cell updates", "cell updates / s")
    out.append(f'<article class="card paper-fig" id="nested-runs"><div class="fig-head"><span class="fig-no">{len(figs) + 1}</span><h3>Every run</h3></div>'
               '<p>One H100 each; the last four rows are the first comparison, without the cap. The nests advance each level with its own time step, so the finest level takes the most steps. The rates include the driving and '
               'the outputs every 0.01; small grids do not fill the GPU.</p><div class="table-scroll"><table class="metrics-table sweep-text paper-table"><thead><tr>'
               + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr></thead><tbody>"
               + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div></article>")
    out.append(f'''<div class="card"><h3>Set-up</h3>
<p>Box of side 1, mean density 1, sound speed 1, isothermal. A uniform field along x with plasma β = 200 (B₀ = 0.1). A solenoidal random acceleration on
1 ≤ |k| ≤ 3, scaled continuously to hold the rms Mach number at 10, random seed 1 in every run. Tom's MUSCL-Hancock scheme with HLL fluxes and constrained
transport, Courant 0.25, density floor 10⁻⁶. The nests: each level has as many cells as the root (64³ or 128³) and covers the central half of the level
around it; each level takes its own time step (Berger–Oliger), with refluxing and an EMF correction at the level boundaries, so mass and div B = 0 are kept.</p></div>
<div class="card"><h3>What it took to run in 3-D on a GPU</h3><ul>
<li><strong>The GPU agrees with the CPU</strong> to {max(a["diff"] for a in agree if a["T"] == "Float64"):.0e} in double precision and {max(a["diff"] for a in agree if a["T"] == "Float32"):.0e}
in single, with the same number of steps on every level (a nested Orszag–Tang, CT and GLM). Speed: a uniform grid {min(unif32) / 1e9:.1f}–{max(unif32) / 1e9:.1f} × 10⁹ cell updates
per second in single precision; the 3 × 128³ nest {nest_ct / 1e9:.2f} × 10⁹ with CT and {nest_glm / 1e9:.2f} × 10⁹ with GLM.</li>
<li><strong>Courant 0.4 is unstable in 3-D</strong> with these kernels, even on a uniform grid (they add the three directions' signal speeds); 0.25 is used. Tom's own runs were 2-D.</li>
<li><strong>Ghost cells could go negative.</strong> Filling a fine level's ghost cells from its parent added three independently limited slopes, so next to a nearly empty
cell a corner ghost could reach −½ of its parent's density; every 3-D nest failed at t ≈ 0.034. Fixed by one factor that scales all the slopes of a cell so the fine
density stays above the lowest of its six coarse neighbours.</li>
<li><strong>Faster driving.</strong> The driving's totals are now summed on the GPU instead of copying the root level to the CPU every step.</li></ul>
<p>The fixes are on our branch <code>gpu-3d-fixes</code> of his code; they are not in Tom's repository yet.</p></div>
<div class="card"><h3>Open</h3><ul>
<li>Why the finest level is short of small-scale velocity: a run whose finest level is wider than a quarter of the box, against a uniform grid at its resolution,
would tell whether it is the time gas spends inside the level.</li>
<li>Longer windows for the time-averaged spectra: capped runs to t = 3 at 256³ (55 snapshots) and t = 1.5 at 512³ (25), set up in <code>compare.jl</code>
(cases <code>*_long</code>), not yet submitted. With 15 and 5 snapshots the 95 % intervals of the spectrum ratios are ±25–50 %.</li>
<li>The ergodicity test with the probe near a level boundary, and for the field.</li></ul></div>
<div class="card paper-state"><h3>Where the data comes from</h3><p>nested_results/ (copied from Sherlock): compare/ and compare_cap/ (histograms every 0.01, snapshots every 0.05),
ergodic/ (probe series, volume PDFs), big/ (the big nest), movie/ (mid-plane slices every 0.002), logs/ and gpu_check/. Figures: <code>python3 scripts/nested_figures.py</code>; page: <code>python3 scripts/build_nested.py</code>.</p></div>''')
    return "".join(out)


def render(main_html):
    nav = "".join(f'<a href="{src}"' + (' aria-current="page"' if src == PAGE else "") + f'><span>{i:02d}</span>{e(label)}</a>'
                  for i, (src, _, label) in enumerate(PAGES, 1))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="Tom Abel's nested-grid MHD code on a GPU: a nest against a uniform grid in driven Mach-10 turbulence.">
<title>Nested turbulence · MHD Field Notes</title><link rel="stylesheet" href="style.css"></head>
<body data-test="nested"><a class="skip-link" href="#main">Skip to the figures</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> NESTED TURBULENCE</div></header>
<nav class="test-nav" aria-label="Pages">{nav}</nav>
<main id="main">{main_html}</main>
<footer><span>Drawn by scripts/nested_figures.py from nested_results/ (copied from Sherlock; sherlock_nested/README.md).</span><span>Built {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")}</span></footer>
</body></html>'''


def main(argv):
    if "--page-only" not in argv:
        import nested_figures
        nested_figures.main()
    J = json.loads((MEDIA/"nested.json").read_text())
    (SITE/PAGE).write_text(render(build(J)))
    print("Nested turbulence page:", SITE/PAGE)


if __name__ == "__main__":
    main(sys.argv[1:])
