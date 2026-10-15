"""Build an annotated reading of the pinned, unmodified Abel et al. solver.

The source stays outside this repository. Excerpts are extracted at build time,
with source line numbers and a checksum guard; commentary is our own.
"""
import hashlib
import html
import json
import re
from pathlib import Path

from solver_equations import EQUATIONS, render_equations

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sherlock_sweep/vendor/glmmhd_turb.jl"
COMMIT = "15ce37d47782afc9b4304b5cccaf847a4f7a4ae0"
UPSTREAM = f"https://github.com/yipihey/GLMMHDTurb/blob/{COMMIT}/glmmhd_turb.jl"
e = html.escape

# slug, displayed term, plain-language definition, where to see it
TERMS = [
    ("eulerian", "Eulerian grid", "The cells stay fixed in space. Fluid flows across their faces; the grid does not follow individual parcels of gas.", "state"),
    ("average", "Cell average", "The volume average of a quantity inside one cell. The conserved array represents these averages; reconstruction estimates values at the faces.", "state"),
    ("conserved", "Conserved state U", "Here: density, three momentum densities, total energy density, three magnetic components, and the GLM variable ψ. Conserved fluxes balance between neighboring cells; floors and source terms need separate accounting.", "state"),
    ("primitive", "Primitive state q", "Density, three velocities, gas pressure, three magnetic components, and ψ. These are convenient for reconstruction and wave-speed estimates.", "state"),
    ("energy", "Total energy E", "Internal plus kinetic plus magnetic energy per volume. In this source E = p/(γ−1) + ρ|v|²/2 + |B|²/2; it has no ψ-energy term.", "state"),
    ("floor", "Density / pressure floor", "A numerical lower bound used to avoid nonphysical states. Converting to primitives with a floor and resetting a conserved cell are different operations; a cell reset can change global mass, momentum, and energy.", "update"),
    ("flux", "Physical flux F(U)", "The rate a conserved quantity crosses a unit area in a given direction, for a single state. A Riemann flux combines the two reconstructed states at a face.", "flux"),
    ("pressure", "Magnetic pressure / tension", "The flux uses total pressure p + |B|²/2 and directional magnetic terms such as −Bx² and −BxBy. Magnetic pressure resists compression; tension couples the direction of the field to forces and transport.", "flux"),
    ("plm", "PLM reconstruction", "Piecewise linear method: estimate a limited slope in each cell, then extrapolate to its two faces. In this code prim_slope returns half the cell-to-cell slope, so q ± s are the edge states.", "slopes"),
    ("limiter", "Slope limiter / MonCen", "Monotonized-central limiting keeps the centered slope when it is safe, reduces steep slopes, and sets the slope to zero when the two one-sided differences change sign. It helps prevent spurious oscillations.", "slopes"),
    ("minmod", "Minmod limiter", "Returns zero for opposite-sign differences; otherwise keeps the smaller magnitude with their common sign. The optional lean LLF path uses minmod. The original non-lean and production PLM paths shown here use MonCen.", "slopes"),
    ("tvd", "Total variation diminishing (TVD)", "In the setting where a method is TVD, the sum of absolute jumps between neighboring values does not grow. A familiar scalar TVD limiter does not by itself prove that the full multidimensional MHD algorithm is TVD.", "slopes"),
    ("predictor", "MUSCL–Hancock predictor", "Reconstruct spatial edges and evolve the cell state by half a timestep. The predicted state is used to build the left and right face states for the Riemann solve.", "predictor"),
    ("unsplit", "Unsplit update", "Contributions from x, y, and z enter one predictor and one final update. Dimensional splitting would advance the solution in successive directional substeps.", "predictor"),
    ("glm", "GLM / Dedner cleaning", "Generalized Lagrange Multiplier cleaning adds ψ. Coupling ψ to the normal magnetic field transports divergence errors; damping reduces ψ. This controls ∇·B errors rather than keeping a discrete divergence exactly zero.", "cleaning"),
    ("psi", "ψ (psi)", "The auxiliary GLM cleaning field: component 9 in both state tuples. At a face, F[Bn] = ψ* and F[ψ] = ch² Bn*. The cell update then damps ψ.", "cleaning"),
    ("ch", "Cleaning speed ch", "The GLM pair propagates at ±ch relative to this fixed grid. In the usual launcher, ch scales as 1/Δt; the constant-ch benchmark variant adjusts the driver to hold it fixed.", "timing"),
    ("normal", "Normal field Bn", "The component perpendicular to the face: Bx at an x face, By at a y face, Bz at a z face. The GLM solve supplies a shared interface value Bn* to both sides of the MHD solve.", "cleaning"),
    ("fast", "Fast magnetosonic speed cf", "The speed of the fastest physical ideal-MHD wave family relative to the gas in a chosen direction. It combines gas and magnetic restoring forces and depends on the angle to B. Add the normal fluid velocity to obtain grid-frame signal speeds.", "speeds"),
    ("signal", "Signal speeds", "Estimated speeds of the edges or internal waves of the Riemann fan, measured in the grid frame. HLL uses two outer estimates; LLF uses a single maximum absolute bound.", "speeds"),
    ("riemann", "Riemann problem / solver", "Two constant states meet at a face. Their evolution forms a wave pattern; an approximate solver estimates the flux through that face from the two states.", "llf"),
    ("fan", "Wave fan", "The wave pattern emerging from an initial jump. Ideal MHD has seven physical characteristic families: two fast, two Alfvén, two slow, and the contact. GLM adds its cleaning pair. Approximate solvers retain only part of this structure.", "hlld"),
    ("llf", "LLF / Rusanov", "Local Lax–Friedrichs flux: average the physical fluxes and subtract a term proportional to the state jump and a local maximum signal speed. That second term supplies numerical diffusion.", "llf"),
    ("hll", "HLL", "Harten–Lax–van Leer flux: approximate the fan with two outer waves and a single intermediate state. The HLL function here estimates those speeds from the left and right fast magnetosonic speeds.", "hll"),
    ("hlle", "HLLE", "The Einfeldt variant of HLL uses wave-speed estimates designed for robust, positivity-preserving behavior under appropriate conditions. The function on this page is named HLL; it should not be relabeled HLLE without checking its estimates and guarantees.", "hll"),
    ("contact", "Contact discontinuity / HLLC", "A contact travels with the normal fluid velocity and can carry a density jump. HLLC restores a contact wave in hydrodynamics. These benchmark GLM configurations use LLF, HLL, or HLLD, rather than HLLC.", "hlld"),
    ("hlld", "HLLD", "An MHD approximate solver retaining five waves: two outer fast waves, two rotational Alfvén waves, and a contact, with four intermediate states. It does not resolve both slow-wave families separately.", "hlld"),
    ("alfven", "Alfvén / rotational waves", "Waves that couple transverse velocity and magnetic-field changes through magnetic tension. In the HLLD intermediate states, their speeds are SM ± |Bn|/√ρ* in the code's magnetic units.", "hlld"),
    ("conservation", "Conservation form", "Update a cell with incoming flux minus outgoing flux. A shared face flux leaves one cell and enters the next, so internal faces cancel in a global sum, up to rounding. Boundary fluxes, forcing, and numerical resets must also be counted.", "update"),
    ("courant", "Courant number / CFL timestep", "Controls how far information may travel during one step relative to the cell width. The benchmark driver sums |v| + cf over all three directions before choosing Δt, so its numerical Courant value is not directly equivalent to every other code's convention.", "timing"),
    ("tile", "GPU tile / shared memory", "A block of threads loads nearby cells and reuses them from fast on-chip memory. Neighboring halo cells supply the reconstruction stencil. This organizes the computation without moving the Eulerian mesh.", "gpu"),
    ("float", "Float16 / Float32", "16-bit and 32-bit floating-point representations. In the production path shown here, shared primitives, face states, and fluxes are stored in Float16, then read into Float32 arithmetic; global conserved arrays remain Float32. Rounding and Float16's finite range still matter.", "gpu"),
    ("julia", "Julia notation", "q[1] is the first entry (Julia indexes from 1). ntuple(f, 9) constructs nine entries. 0.5f0 is a Float32 literal. ! conventionally marks a mutating function. @inline encourages inlining; @fastmath permits floating-point transformations and relaxed exceptional-value semantics.", "state"),
    ("ppm", "PPM reconstruction", "Piecewise parabolic method. This source uses compact three-point parabolic edges, monotonicity constraints, Hancock time prediction, and a shock fallback. It is not automatically the same algorithm or accuracy order as AthenaK PPM4.", "ppm"),
]


def note(title, text, terms, lo, hi=None):
    return dict(title=title, text=text, terms=terms.split(), lo=lo, hi=hi or lo)


# Excerpts remain verbatim. Each tuple is an inclusive range in the pinned source.
SECTIONS = [
    dict(id="state", title="What each cell stores", intro="Start with the two representations of the same gas and field. The conserved state is evolved; primitives make reconstruction and wave estimates convenient.", ranges=[(63, 64), (68, 86)], notes=[
        note("Read the nine slots", "Slots 2–4 change from momentum density ρv to velocity v; slot 5 changes from total energy density E to gas pressure p. B and ψ keep their slots. Magnetic units absorb μ₀, giving |B|²/2 for magnetic energy density.", "average conserved primitive julia", 63, 64),
        note("Recover pressure from energy", "Subtract kinetic and magnetic energy from E to recover internal energy, then multiply by γ−1. max(..., pfl) imposes a pressure floor on this primitive state. It does not by itself rewrite the stored conserved energy.", "energy floor", 69, 76),
        note("Convert back", "Build total energy from gas pressure, kinetic energy, and magnetic energy. Multiply velocities by density to obtain momentum densities. All these quantities are per volume, rather than totals for the whole cell.", "conserved energy", 79, 86),
    ]),
    dict(id="flux", title="The equations become a flux", intro="This is the ideal-MHD flux through an x face for one state. Later, the Riemann solver combines the left and right states into one numerical face flux.", ranges=[(98, 115)], notes=[
        note("Pressure includes the magnetic field", "ptot is p + |B|²/2. The −Bx² and −BxBy terms in the momentum flux encode directional magnetic stresses. The energy flux also includes magnetic work through −Bx(v·B).", "flux pressure", 101, 110),
        note("Nine flux components", "The tuple follows the conserved-state order. Its first entry is mass flux ρvx. The Bx and ψ entries are placeholders: the GLM pair fills them separately, while the transverse magnetic fluxes come from the induction equation.", "flux conserved glm", 106, 114),
    ]),
    dict(id="slopes", title="Reconstruct the cell edges", intro="PLM estimates a line inside each cell. MonCen limits that line so sharp jumps do not produce large new overshoots.", ranges=[(145, 155), (157, 158)], notes=[
        note("Compare the two neighboring differences", "dl = M−L and dr = R−M. If their signs disagree, the cell is at an extremum and the limiter returns zero. Otherwise it caps the centered slope using twice the smaller one-sided difference.", "limiter minmod tvd", 145, 150),
        note("These are half-slopes", "prim_slope applies MonCen to all nine primitive quantities, then multiplies by 1/2. That is why padd(q, s, +1) reaches one face and padd(q, s, −1) reaches the other; s already includes the half-cell distance.", "plm primitive", 153, 158),
    ]),
    dict(id="predictor", title="Predict half a timestep", intro="The Hancock predictor advances the reconstructed cell state to the middle of the timestep, before neighboring states meet at a face.", ranges=[(165, 172)], notes=[
        note("The half-step", "dtdx means Δt/Δx. Multiplying by 0.5 gives the half-step factor. Each direction contributes a difference of physical fluxes evaluated at that cell’s reconstructed edges.", "predictor flux", 165, 170),
        note("All directions contribute together", "The x, y, and z differences enter a single predicted conserved state. This is the unsplit predictor used here. These are local physical fluxes for prediction; the later corrector uses numerical Riemann fluxes shared by cells.", "unsplit conservation predictor", 168, 172),
    ]),
    dict(id="cleaning", title="Solve the GLM cleaning pair", intro="Before the physical MHD solve, the code reconciles the left and right normal magnetic fields using the auxiliary cleaning field ψ.", ranges=[(175, 179)], notes=[
        note("One shared normal field", "bns is Bn*, the interface magnetic field after solving the two-wave linear cleaning subsystem. A jump in ψ changes this field; a jump in Bn changes ψ*. Both sides of the following MHD solve receive the same Bn*.", "glm normal psi", 175, 179),
        note("What cleaning transports", "The pair corresponds to ∂t Bn + ∂n ψ = 0 and ∂t ψ + ch² ∂n Bn = 0 before damping, with wave speeds ±ch. It controls numerical divergence errors; it does not impose exact discrete ∇·B = 0.", "glm ch psi", 176, 178),
    ]),
    dict(id="speeds", title="Estimate how fast information travels", intro="The fast magnetosonic speed is relative to the gas. The Riemann solver adds the fluid velocity to estimate speeds relative to the fixed grid.", ranges=[(89, 95)], notes=[
        note("Sound and Alfvén contributions", "c2 is cs² = γp/ρ; b2 is vA² = |B|²/ρ in these units. bn is the magnetic component normal to this face, so the result depends on the propagation direction.", "fast normal", 89, 93),
        note("Choose the fast root", "The nested square roots give the fast branch of the magnetosonic relation. The max prevents a small negative discriminant from roundoff. Parallel to B, cf = max(cs, vA); perpendicular to B, cf = √(cs²+vA²).", "fast signal", 94),
    ]),
    dict(id="llf", title="LLF: average flux plus diffusion", intro="This is the Rusanov / local Lax–Friedrichs flux used by the original benchmark path. L and R are reconstructed, predicted primitive states with a shared cleaned normal field.", ranges=[(182, 190)], notes=[
        note("Choose one speed bound", "smax bounds the absolute fluid-frame-shifted fast signals on both sides and also includes ch. A larger bound increases the coefficient of numerical diffusion in this LLF implementation.", "llf signal ch", 184, 187),
        note("The whole LLF formula", "The first term averages the two physical fluxes. The second subtracts half the signal bound times the conserved-state jump. It damps unresolved jumps; uR−uL is a difference of conserved states, not primitive states.", "llf riemann conserved", 188),
        note("Restore the cleaning fluxes", "The Bn and ψ flux entries are replaced with fbn = ψ* and fpsi = ch²Bn*. The physical MHD calculation and the GLM pair together supply all nine entries.", "glm psi normal", 189),
    ]),
    dict(id="hll", title="HLL: keep two outer wave speeds", intro="HLL approximates the physical fan with two outer waves and one state between them. This source computes the two bounds directly from the left and right fast speeds.", ranges=[(335, 343)], notes=[
        note("Left and right bounds", "SL takes the smallest left-going estimate; SR takes the largest right-going estimate. Clamping SL ≤ 0 and SR ≥ 0 lets one expression also give the upwind physical flux when the physical waves all travel in the same direction.", "hll signal hlle", 336, 337),
        note("The HLL numerical flux", "The expression weights the left and right physical fluxes by the two wave bounds and adds a conserved-jump term. LLF is recovered for symmetric bounds SL = −a and SR = +a, with the same a and cleaning treatment.", "hll llf riemann", 338, 342),
    ]),
    dict(id="hlld", title="HLLD: restore contact and Alfvén waves", intro="Selected lines from hlld_x show the extra wave speeds. The full function also builds single-star and double-star states and chooses the face flux; follow the source link for that algebra.", ranges=[(203, 206), (212, 222)], notes=[
        note("The outer fast waves", "cfL and cfR set the outer bounds. This HLLD implementation uses a somewhat broader estimate than hll_x: min(uL,uR)−max(cfL,cfR) and max(uL,uR)+max(cfL,cfR), clamped around zero.", "hlld signal fast", 203, 206),
        note("The contact wave", "SM estimates the contact speed and pts the shared star-region total pressure. dLs and dRs are the densities behind the outer fast waves, obtained from the jump conditions.", "contact hlld fan", 213, 219),
        note("Two internal Alfvén waves", "SLs and SRs add the rotational waves on either side of the contact. HLLD retains five waves in the physical MHD fan; it does not explicitly restore both slow magnetosonic waves.", "alfven hlld fan", 220, 222),
    ]),
    dict(id="dispatch", title="Assemble one face solve", intro="The original dispatcher rotates any face into an x-normal frame, applies floors and GLM cleaning, selects LLF or HLLD, then rotates the flux back. The production HLL path has its own dispatcher with the same cleaning pattern.", ranges=[(275, 293)], notes=[
        note("Rotate components, not the mesh", "rot_to permutes velocity and magnetic components so the same x-flux routine can serve y and z faces. These are coordinate permutations; the Eulerian grid stays fixed.", "normal eulerian", 278),
        note("Make the two interface states physical", "The reconstructed density and pressure are floored before solving. Cleaning supplies Bn*, ψ*, and their two fluxes. Lc and Rc then share the same normal magnetic field.", "floor glm normal", 279, 286),
        note("Select the physical solver", "use_hlld=false forces LLF; otherwise positive density/pressure thresholds can request LLF in configured low-state regions. This is a preselected threshold fallback, not a general retry after every invalid HLLD result.", "llf hlld riemann", 287, 292),
    ]),
    dict(id="update", title="Update the cell by flux balance", intro="After every face has its flux, the original tiled kernel updates all nine cell quantities using incoming minus outgoing fluxes.", ranges=[(718, 727), (736, 738)], notes=[
        note("Conservation in one line", "Fxl−Fxh is the net incoming x flux, and likewise for y and z. Multiplying by Δt/Δx gives the cell-average change. Neighboring cells use the shared face flux with opposite signs, so internal contributions cancel in a domain sum up to numerical rounding.", "conservation average unsplit", 718, 722),
        note("A floor can change conserved totals", "If the updated density is below smallr, this branch resets density, all three momenta, and energy. That operation is separate from conservative flux balance and can change global totals. Completion alone does not establish accuracy or conservation.", "floor conservation", 723, 726),
        note("Damp ψ, then write the new state", "Component 9 is multiplied by glm_fac, an exponential damping factor. Components 1–8 are unchanged by this damping line. The excerpt then skips the optional turbulence-forcing block and shows the write to unew.", "psi glm conserved", 727, 738),
    ]),
    dict(id="timing", title="Set the cleaning speed and damping", intro="The step launcher sets these parameters before starting the GPU kernel. The sweep driver chooses the timestep; it is separate from Tom’s kernel source.", ranges=[(749, 752)], notes=[
        note("Cleaning speed follows the timestep", "ch = C Δx / (NDIM Δt) times glm_ch_scale, with NDIM=3. Cutting Δt while keeping the other parameters fixed raises ch. The constant-ch benchmark variant compensates through glm_ch_scale and also caps Δt for cleaning stability.", "ch courant", 749, 751),
        note("Exponential ψ damping", "For positive glm_cp_coef, glm_fac = exp[−ch Δt/(glm_cp_coef × boxlen)]. Otherwise it is one, giving no damping. The code applies this after the conservative flux update.", "psi glm ch", 752),
    ]),
    dict(id="gpu", title="What “tiled” and “f16” mean", intro="The production path changes the storage type of shared memory. It still evolves fluid on a fixed grid, and it is not an entirely Float16 calculation.", ranges=[(778, 781), (787, 791)], notes=[
        note("Read shared values as Float32", "_sgT explicitly converts the nine stored values to Float32. _spT writes into the array’s element type, which rounds to Float16 when HALF=true. The conserved global state is Float32 in these benchmark runs.", "float julia", 778, 781),
        note("Shared primitives, edges, and fluxes", "SP holds primitive states; LX/RX and their y/z counterparts hold face states. The left arrays are later reused for numerical fluxes. All use ST, so the production Float16 path rounds primitives, interface states, and fluxes in shared memory.", "tile float", 787, 791),
    ]),
    dict(id="ppm", title="The compact PPM alternative", intro="This variant estimates parabolic edges instead of linear edges. In this benchmark it is paired with Float16 shared storage and HLL; it is still time-predicted with Hancock.", ranges=[(917, 925)], notes=[
        note("Limit the parabola", "ppm_mono flattens an extremum or adjusts an edge when the parabola would violate its monotonicity constraints. ql and qr are candidate left and right edge values around cell average q0.", "ppm limiter", 917, 922),
        note("Add curvature, then predict in time", "ppm_edges uses the two neighbors and center to build a slope and curvature. mhd_face9 combines the spatial edge displacement ed−m0 with the half-time predicted state mh. This compact PPM is distinct from a full PPM4 scheme.", "ppm predictor", 923, 925),
    ]),
]


TOKEN = re.compile(r"#[^\n]*|\"[^\"]*\"|@[A-Za-z_]+|\b(?:function|end|return|if|else|elseif|for|in|where|const)\b|\b\d+(?:\.\d+)?(?:f-?\d+)?\b")


def syntax(line):
    out = []; last = 0
    for m in TOKEN.finditer(line):
        token = m.group(); kind = "comment" if token.startswith("#") else "string" if token.startswith('"') else "macro" if token.startswith("@") else "number" if token[0].isdigit() else "keyword"
        out += [e(line[last:m.start()]), f'<span class="syn-{kind}">{e(token)}</span>']; last = m.end()
    return "".join(out) + e(line[last:])


def main():
    raw = SOURCE.read_bytes()
    expected = json.loads((SOURCE.parent / "PROVENANCE.json").read_text())["files"][SOURCE.name]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("Solver checksum changed: review all excerpts and annotations before rebuilding.")
    lines = raw.decode().splitlines(); sections = []; note_data = {}
    equations = render_equations(ROOT)
    known = {t[0] for t in TERMS}
    for index, s in enumerate(SECTIONS, 1):
        links = []; blocks = []
        for j, n in enumerate(s["notes"]):
            assert set(n["terms"]) <= known
            key = f'{s["id"]}-{j}'; note_data[key] = n | {"section": s["id"], "equations": equations[key]}
            links.append(f'<button type="button" class="annotation" data-note="{key}" aria-pressed="false"><span>{j+1}</span>{e(n["title"])}</button>')
        for lo, hi in s["ranges"]:
            rendered = []
            for num in range(lo, hi+1):
                matches = [j for j, n in enumerate(s["notes"]) if n["lo"] <= num <= n["hi"]]
                attr = f' data-note="{s["id"]}-{matches[0]}" tabindex="0" role="button" aria-label="Line {num}: {e(s["notes"][matches[0]]["title"])}"' if matches else ""
                rendered.append(f'<span class="source-line" data-line="{num}"{attr}><span class="line-no" aria-hidden="true">{num}</span><span>{syntax(lines[num-1]) or " "}</span></span>')
            blocks.append(f'<pre class="source"><code>{"".join(rendered)}</code></pre>')
        fallback = "".join(f'<details class="inline-note"><summary>{e(n["title"])}</summary>{equations[s["id"]+"-"+str(j)]}<p>{e(n["text"])}</p></details>' for j, n in enumerate(s["notes"]))
        sections.append(f'''<section class="code-section" id="{s['id']}" data-default-note="{s['id']}-0">
<div class="section-heading"><span class="step-number">{index:02d}</span><h2>{e(s['title'])}</h2><a class="source-link" href="{UPSTREAM}#L{s['ranges'][0][0]}-L{s['ranges'][-1][1]}" target="_blank" rel="noopener">Source ↗</a></div>
<p class="section-intro">{e(s['intro'])}</p><div class="code-card">{('<div class="excerpt-gap">… intervening source omitted …</div>').join(blocks)}</div>
<div class="annotation-list" aria-label="Annotations for {e(s['title'])}">{''.join(links)}</div><div class="fallback-notes">{fallback}</div></section>''')
    assert set(note_data) == set(EQUATIONS), "Each annotation must have matching equations."
    glossary = "".join(f'<details class="term" id="term-{slug}" data-search="{e((title+" "+text).lower())}"><summary>{e(title)}</summary><p>{e(text)}</p><a class="term-source" href="#{section}">See it in the code ↑</a></details>' for slug, title, text, section in TERMS)
    toc = "".join(f'<a href="#{s["id"]}"><span>{i:02d}</span>{e(s["title"])}</a>' for i, s in enumerate(SECTIONS, 1))
    variants = [
        ("GLM tiled LLF", "step_tiled!", "MonCen PLM", "LLF", "Float32", "Δt dependent"),
        ("GLM f16/HLL", "step_plm!", "MonCen PLM", "HLL", "Float16", "Δt dependent"),
        ("GLM HLLD", "step_plm!", "MonCen PLM", "HLLD", "Float32", "Δt dependent"),
        ("GLM PPM", "step_plm!", "Compact PPM", "HLL", "Float16", "Δt dependent"),
        ("GLM constant ch", "step_plm!", "MonCen PLM", "HLL", "Float16", "Held constant by driver"),
    ]
    rows = "".join("<tr>"+"".join(f'<{"th scope=\"row\"" if i==0 else "td"}>{e(v)}</{"th" if i==0 else "td"}>' for i,v in enumerate(row))+"</tr>" for row in variants)
    css_version = hashlib.sha256((ROOT / "progress/solver-walkthrough.css").read_bytes()).hexdigest()[:12]
    js_version = hashlib.sha256((ROOT / "progress/solver-walkthrough.js").read_bytes()).hexdigest()[:12]
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex, nofollow"><meta name="color-scheme" content="light">
<meta name="description" content="Read the central pieces of the Abel et al. GLM-MHD solver with line annotations and a glossary.">
<title>Inside the solver · MHD Field Notes</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="solver-walkthrough.css?v={css_version}"></head>
<body class="solver-page"><a class="skip-link" href="#main">Skip to the content</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta">ABEL ET AL. · CODE WALKTHROUGH</div></header>
<nav class="test-nav" aria-label="Pages"><a href="tests.html">Tests</a><a href="solver.html" aria-current="page">Inside the solver</a></nav>
<main id="main"><div class="page-heading"><div><div class="eyebrow"><span>11 · Inside the solver</span><span class="badge">Julia · GLM-MHD</span></div>
<h1>From cell averages to the next timestep.</h1><p>Read the central pieces of Tom Abel’s Eulerian MHD code. Choose an annotation below a code block, or click a highlighted line, for matching equations and an explanation beside it. The glossary stays on the right as you read.</p></div></div>
<p class="provenance">Verbatim excerpts from <a href="{UPSTREAM}">Abel et al., GLMMHDTurb ↗</a> · benchmark snapshot <code>{COMMIT[:12]}</code>. Source order is rearranged for teaching; gaps are marked. Annotations are ours.</p>
<details class="reading-map" open><summary>Follow one timestep</summary><nav aria-label="Walkthrough sections">{toc}</nav></details>
<section class="variant-card" aria-labelledby="variants-title"><h2 id="variants-title">Which version am I reading?</h2><p>The core physics helpers are shared. The main update excerpt follows the original tiled LLF path; the other sections show the alternative solvers and production storage.</p>
<div class="table-scroll"><table><thead><tr><th>Benchmark label</th><th>Entry point</th><th>Reconstruction</th><th>Flux</th><th>Shared storage</th><th>Cleaning speed</th></tr></thead><tbody>{rows}</tbody></table></div><p class="variant-foot">All five use a fixed grid and Hancock prediction. Float16 here describes shared storage; global conserved states remain Float32. Configuration names describe the runs, not a claim of equal accuracy.</p></section>
<div class="reading-layout"><div class="code-column">{''.join(sections)}
<section class="driver-note"><h2>Where the timestep comes from</h2><p>The benchmark driver takes the maximum over cells of Σ<sub>d=x,y,z</sub>(|v<sub>d</sub>| + c<sub>f,d</sub>), then uses Δt = C Δx / that maximum. The main sweep used C = 0.25 and a unit box. Its constant-ch variant holds the initial cleaning speed fixed and adds a cleaning timestep cap.</p><p>This is driver behavior in <code>sherlock_sweep/src/run_suite.jl</code> (signals scan and <code>timestep</code>), separate from the unmodified solver excerpts above.</p></section></div>
<aside class="reading-sidebar" aria-label="Annotations and glossary"><div class="sidebar-inner">
<div class="active-note" id="reading-note" aria-live="polite" aria-atomic="true"><div class="sidebar-kicker">Reading notes <span id="note-location"></span></div><h2 id="note-title"></h2><div id="note-equations"></div><p id="note-body"></p><div id="related-terms" aria-label="Related glossary terms"></div><a id="note-code" href="#state">Go to these lines ↑</a></div>
<div class="glossary"><h2>Glossary <span>{len(TERMS)} terms</span></h2><label for="glossary-search">Find a term or symbol</label><input id="glossary-search" type="search" placeholder="Try ψ, HLL, pressure…" autocomplete="off"><p id="glossary-empty" hidden>No matching terms.</p><div class="glossary-terms">{glossary}</div></div></div></aside></div>
</main><footer><span>Source: Abel et al. · commentary: MHD Field Notes</span><a href="{UPSTREAM}">Read the full pinned source ↗</a></footer>
<script type="application/json" id="walkthrough-notes">{json.dumps(note_data, ensure_ascii=False).replace('<', chr(92)+'u003c')}</script><script src="solver-walkthrough.js?v={js_version}"></script></body></html>'''
    path = ROOT / "progress/solver.html"; path.write_text(page)
    print(f"Wrote {path.name}: {len(SECTIONS)} code sections, {len(note_data)} annotations, {len(TERMS)} glossary terms.")


if __name__ == "__main__":
    main()
