"""Publish the complete benchmark problem definitions with reading notes and configs.

Uses our tracked problem/driver files, never the vendored solver implementation.
Run with .venv-movie/bin/python scripts/build_problem_definitions.py.
"""
import hashlib
import html
import json
from pathlib import Path
import tomllib

from build_solver_walkthrough import syntax

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'progress'
SOURCE = ROOT/'sherlock_sweep/src/problems.jl'
e = html.escape

# Every line of problems.jl appears once, including helpers and diagnostics.
# Markers fail explicitly if a source edit requires reviewing the reading map.
# id, title, marker, explanation, glossary terms, optional setup equations
SECTIONS = [
 ('state', 'Interface, units and defaults', 'module Problems',
  'setup returns a primitive-state function prim, a vector potential A0, a uniform magnetic field B0, and a cell subsampling count sub. The default gamma can be overridden in the TOML config. Planar GPU tests are repeated through a cube; external codes can use true 2D grids.', 'primitive potential gamma average periodic', None),
 ('uniform', 'Uniform state', '    if name=="uniform"',
  'Density, pressure, velocity and magnetic field are constant. With periodic boundaries, a correct update preserves this state to rounding.', 'primitive periodic', 'uniform'),
 ('entropy', 'Entropy wave', '    elseif name=="entropy"',
  'A density wave moves through constant pressure and velocity. sinc(1/n)^2 gives the exact cell average of this sinusoid, rather than its value only at the cell centre. t translates the analytic target.', 'average primitive', 'entropy'),
 ('alfven', 'Aligned Alfvén wave', '    elseif name=="alfven"',
  'The transverse velocity and magnetic field rotate around a constant longitudinal field. A0 supplies the varying field through its curl. The later reference routine explains the discretization mismatch of this projected wave.', 'potential curl', 'alfven'),
 ('alfven-oblique', 'Oblique Alfvén wave', '    elseif name=="alfven_oblique"',
  'The wave travels along (1,2), rather than a grid axis. phase defines its direction and translation; the transverse basis rotates velocity and magnetic components consistently.', 'potential curl', 'alfven_oblique'),
 ('brio-wu', 'Aligned and rotated Brio–Wu tubes', '    elseif name in ("brio_wu","brio_wu_rotated")',
  'Two primitive states meet in a periodic left/right arrangement. The rotated version changes the normal direction to (1,1)/sqrt(2). square_integral builds the reversing transverse field; the normal field remains constant.', 'primitive periodic potential', 'brio_wu'),
 ('orszag-tang', 'Orszag–Tang in 2D and 3D', '    elseif name in ("orszag_tang","orszag_tang_3d")',
  'Crossing sinusoidal velocity and magnetic patterns generate shocks and interacting vortices. epsilon introduces z dependence only for the 3D variant. These are the Athena-normalized initial values.', 'primitive potential', 'orszag_tang'),
 ('rotor', 'Rotor', '    elseif name=="rotor"',
  'A dense, rotating central disc sits in a uniform horizontal field. r0 and r1 define its core and transition ring. sub=8 averages the sharp geometry over each cell instead of relying on a single point.', 'average primitive', 'rotor'),
 ('current-sheet', 'Current sheet', '    elseif name=="current_sheet"',
  'The magnetic direction reverses across two sheets, and a sinusoidal velocity perturbs them. square_integral makes the corresponding periodic vector potential. No explicit resistivity is added here.', 'potential periodic', 'current_sheet'),
 ('field-loop', 'Field loop, with or without vertical flow', '    elseif name=="field_loop"',
  'A conical Az creates a weak circular magnetic loop. field_loop_velocity reads vz from the config; field_loop_novz therefore uses this same branch with vz=0. The loop is advected by the prescribed uniform initial velocity.', 'potential beta', 'field_loop'),
 ('field-loop-3d', 'Tilted field loop in 3D', '    elseif name=="field_loop_3d"',
  'The loop wraps around an oblique cylinder. Searching periodic images gives distance from its axis while keeping the geometry compatible with the periodic cube.', 'periodic potential', 'field_loop_3d'),
 ('blast', 'Planar and spherical magnetic blasts', '    elseif name=="blast"',
  'A high-pressure disc or sphere expands into a low-pressure medium. The config chooses dim=2 or 3 and bx=0, 50 or 100; bx is divided by sqrt(4pi) here. One branch defines all six blast tests.', 'primitive gamma average', 'blast_2d_b50'),
 ('kelvin-helmholtz', 'Kelvin–Helmholtz: uniform or localized field', '    elseif name in ("kh_uniform","kh_localized")',
  'Two smoothed, oppositely moving layers are perturbed vertically. The localized case uses a periodic sech-squared band around the lower layer, including a small nonzero tail at the upper layer. The timestep-jump test uses this same initial state; its time control appears below.', 'potential beta periodic', 'kh_localized'),
 ('gem', 'GEM double current sheet', '    elseif name=="gem"',
  'Two Harris sheets and a small vector-potential perturbation seed reconnection. Density and pressure balance the sheet field approximately in the adapted periodic box. Length and time units are rescaled by gem_lx. This is the ideal-MHD adaptation, without Hall physics or explicit resistivity.', 'potential periodic', 'gem'),
 ('decay', 'Decaying turbulence: Mach 2 or Mach 9', '    elseif name=="decay"',
  'Both tests begin with a smooth random velocity field. The config sets Mach number, gamma and the mean magnetic field; decay_supersonic maps to this decay branch. It does not start from a pre-driven saturated turbulence snapshot. No continuing acceleration is applied.', 'mach seed gamma', 'decay_supersonic'),
 ('decay-modes', 'Reproducible solenoidal velocity modes', '"Solenoidal decay modes;',
  'A seeded generator chooses Fourier coefficients. Subtracting the component parallel to k makes each velocity mode solenoidal. The same mode coefficients are reused across grids; the finite-grid normalization is applied later.', 'seed solenoidal mach', None),
 ('portable-spec', 'Export the setup for the other codes', 'Plain-text problem spec',
  'spec_text writes the resolved parameters and random coefficients needed by the C++ initial-condition implementation. This separates the common setup from a code-specific solver. Config overrides remain essential: function defaults alone do not identify a named benchmark run.', 'seed primitive', None),
 ('faces', 'Construct face fields from a discrete curl', 'Lower periodic face fields',
  'The magnetic field is a finite-difference curl of edge samples of A, plus B0. Matching difference operators make the face divergence cancel before rounding. These face fields approximate continuum face averages; a continuous analytic field and its finite-grid representation are not identical.', 'potential curl', None),
 ('centers', 'Average face fields to cell centres', 'function center_faces',
  'Each component is averaged between its two neighboring faces. CT evolves face fields; the GLM conserved array stores the resulting cell-centred field. Their discrete divergence measurements therefore use different operators.', 'average curl', None),
 ('averages', 'Build the conserved initial state', 'Initial (or, for translating exact solutions, time-t)',
  'Subcell samples average density, momentum, internal energy and kinetic energy. Decay removes mean momentum and scales its rms to the requested Mach number. Magnetic energy is then added from the cell-centred field. The ninth slot, psi, starts at zero.', 'average conserved mach', None),
 ('reference', 'Which tests have an analytic target?', 'Reference target at time t, or `nothing`.',
  'Uniform flow and entropy advection have exact targets here. The Alfvén targets include initialization projection error. Periodic-return targets for weak field loops are approximate because the field still exerts Lorentz forces. Other tests return no independent analytic target.', 'average reference', None),
 ('diagnostics', 'Leakage, reconnection and shared diagnostics', 'Rows of y cells in the leakage band',
  'These routines measure field growth in the upper Kelvin–Helmholtz band, reconnected GEM flux, energies, positivity, psi and magnetic divergence. The centre and face divergence columns are different discrete measurements. A completion flag alone does not establish accuracy or positivity.', 'curl conserved reference', None),
]

TERMS = [
 ('primitive', 'Primitive state', 'Density rho, velocity v and gas pressure p describe the fluid locally. The setup returns these before converting them to conserved quantities.', 'state'),
 ('conserved', 'Conserved state', 'The array stores density, momentum densities, total energy density, magnetic components and psi. Energy includes internal, kinetic and magnetic terms.', 'averages'),
 ('average', 'Cell average', 'The volume average inside a grid cell. Sampling the centre alone generally differs from averaging a curved profile or a sharp boundary.', 'averages'),
 ('potential', 'Vector potential A', 'A field used to construct B = curl A + B0. Adding B0 supplies a uniform mean field that a periodic potential cannot represent alone.', 'faces'),
 ('curl', 'Discrete curl and divergence', 'Finite-difference counterparts of curl and divergence. Compatible operators make divergence of a discrete curl cancel, apart from rounding.', 'faces'),
 ('periodic', 'Periodic boundaries', 'Opposite box faces connect. A structure leaving one face re-enters the opposite face; the initial state must be consistent across that connection.', 'state'),
 ('gamma', 'Adiabatic index gamma', 'Relates pressure to internal energy: p = (gamma−1) times internal-energy density. A gamma close to one approximates isothermal behavior here, without being an exact isothermal equation of state.', 'state'),
 ('beta', 'Plasma beta', 'The ratio of gas pressure to magnetic pressure: beta = 2p / |B|² in these magnetic units. Large beta means the initial magnetic pressure is weak.', 'kelvin-helmholtz'),
 ('mach', 'Mach number', 'Fluid speed relative to sound speed. Decay normalizes the initial rms velocity; driven runs calibrate forcing to a target Mach and measure the Mach actually achieved.', 'decay'),
 ('seed', 'Random seed', 'Selects a reproducible random realization. Reusing a seed across different implementations does not guarantee identical forcing histories.', 'decay-modes'),
 ('solenoidal', 'Solenoidal', 'A vector field with zero divergence. Fourier projection removes the component parallel to each wave vector k.', 'decay-modes'),
 ('reference', 'Reference target', 'A comparison state. An analytic continuum solution, its grid projection, and a high-resolution numerical reference carry different errors and assumptions.', 'reference'),
]


def section_starts(lines):
    starts = []
    for ident, title, marker, note, terms, setup in SECTIONS:
        found = [i for i, line in enumerate(lines) if line.startswith(marker)]
        if len(found) != 1:raise ValueError(f'{ident}: expected one source marker, found {len(found)}')
        start = found[0]
        if start and lines[start-1] == '"""':start -= 1
        starts.append(start)
    assert starts == sorted(set(starts)) and starts[0] == 0
    return starts


def setup_equations(test, setups):
    if not test:return ''
    # Reuse the website's existing locally typeset setup equations.
    rows = setups.get('test/'+test, [])
    out = []
    for row in rows:
        parts = []
        for part in row['parts']:
            if isinstance(part, str):parts.append(e(part))
            else:
                tex, image, height, depth = part
                parts.append(f'<img src="media/setups/{e(image)}" alt="{e(tex)}" style="max-width:100%;height:auto;vertical-align:middle">')
        out.append('<p>'+''.join(parts)+'</p>')
    return '<div class="problem-equations">'+''.join(out)+'</div>' if out else ''


def render_section(ident, title, lines, lo, hi, note, terms, equations, source, prefix='P', number=''):
    key = ident+'-0'
    code = ''.join(f'<span class="source-line" id="{prefix}{n}" data-line="{n}" data-note="{key}" tabindex="0" role="button" aria-label="Line {n}: {e(title)}">'
                   f'<span class="line-no" aria-hidden="true">{n}</span><span>{syntax(lines[n-1]) or " "}</span></span>' for n in range(lo, hi+1))
    block = f'''<section class="code-section" id="{ident}" data-default-note="{key}">
<div class="section-heading"><span class="step-number">{number}</span><h2>{e(title)}</h2><a class="source-link" href="../{source}" download>Source file ↗</a></div>
<p class="section-intro">Lines {lo}–{hi} of <code>{e(Path(source).name)}</code></p><div class="code-card"><pre class="source"><code>{code}</code></pre></div>
<div class="annotation-list"><button class="annotation" type="button" data-note="{key}" aria-pressed="false">Read the setup notes</button></div>
<div class="fallback-notes"><details class="inline-note"><summary>{e(title)} explained</summary><p>{e(note)}</p>{equations}</details></div></section>'''
    return block, {key:dict(title=title, text=note, terms=terms.split(), lo=lo, hi=hi, section=ident, equations=equations)}


def branch_of(config):
    name = config.get('problem',config['name'])
    return {'alfven_oblique':'alfven-oblique', 'brio_wu':'brio-wu', 'brio_wu_rotated':'brio-wu',
            'orszag_tang':'orszag-tang', 'orszag_tang_3d':'orszag-tang', 'current_sheet':'current-sheet',
            'field_loop':'field-loop', 'field_loop_3d':'field-loop-3d',
            'kh_uniform':'kelvin-helmholtz', 'kh_localized':'kelvin-helmholtz'}.get(name,name)


def config_catalog():
    rows = []
    for path in sorted((ROOT/'sherlock_sweep/configs').glob('*.toml')):
        if path.name.startswith('gate_'):continue
        text = path.read_text();cfg = tomllib.loads(text);branch = branch_of(cfg)
        assert branch in {s[0] for s in SECTIONS}, (path.name,branch)
        result = SITE/f'test-{path.stem.replace("_","-")}.html'
        title = f'<a href="{result.name}">{e(path.stem)}</a>' if result.is_file() else e(path.stem)
        rows.append(f'<tr id="config-{path.stem}"><td>{title}</td><td><a href="#{branch}">{e(cfg.get("problem",cfg["name"]))}</a></td>'
                    f'<td>{e(str(cfg["t_end"]))}</td><td><details><summary>{e(path.name)}</summary><pre>{e(text)}</pre>'
                    f'<a href="../sherlock_sweep/configs/{path.name}" download>Download TOML</a></details></td></tr>')
    return '<div class="table-scroll problem-configs"><table class="metrics-table sweep-text"><thead><tr><th>Test / results</th><th>Source branch</th><th>End time</th><th>Exact configuration</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>', len(rows)


def main():
    raw = SOURCE.read_bytes();lines = raw.decode().splitlines();starts = section_starts(lines)
    setups = json.loads((SITE/'setups.json').read_text())
    blocks, notes = [], {}
    for i, ((ident,title,marker,note,terms,setup), start) in enumerate(zip(SECTIONS,starts)):
        stop = starts[i+1] if i+1 < len(starts) else len(lines)
        block, data = render_section(ident,title,lines,start+1,stop,note,terms,setup_equations(setup,setups),
                                     'sherlock_sweep/src/problems.jl',number=f'{i+1:02d}')
        blocks.append(block);notes.update(data)
    assert sum((starts[i+1] if i+1<len(starts) else len(lines))-start for i,start in enumerate(starts)) == len(lines)

    driver = ROOT/'sherlock_sweep/src/run_suite.jl';dl = driver.read_text().splitlines()
    lo = next(i for i,line in enumerate(dl,1) if 'jump=get(config,"dt_jump"' in line)
    hi = next(i for i,line in enumerate(dl,1) if i>lo and 'advance!(s,pe,dt,t)' in line)
    block,data = render_section('timestep-jump','The timestep jump is a driver setting',dl,lo,hi,
        'kh_localized_dtjump uses the localized-field initial condition. Its config sets dt_jump=[3.0,0.025,100]: at t=3 the driver reduces 100 timesteps by a factor of 40. This changes time stepping, not the gas initial state. Output-time clipping and cleaning-speed policy are also driver decisions.',
        'reference',setup_equations('kh_localized_dtjump',setups),'sherlock_sweep/src/run_suite.jl',prefix='R',number='23')
    blocks.append(block);notes.update(data)
    driven = ROOT/'sherlock_turb/turb_drive.jl';tl = driven.read_text().splitlines()
    params_lo=next(i for i,line in enumerate(tl,1) if line.startswith('function params('))
    params_hi=next(i for i,line in enumerate(tl,1) if i>params_lo and line=='end')
    force_lo=next(i for i,line in enumerate(tl,1) if line.startswith('mutable struct Drive'))
    initial_lo=next(i for i,line in enumerate(tl,1) if line.startswith('function initial!'))
    force_hi=next(i for i,line in enumerate(tl,1) if i>initial_lo and line=='end')
    for ident,title,lo,hi,note in [
        ('driven','Driven turbulence: gas and forcing parameters',params_lo,params_hi,
         'Driven Mach 3/10/30/50 is a separate hydro experiment. b0=0 starts without a magnetic field. gamma is the Float32 value nearest 1+gamma_minus_one, and T=1/(2 Mach) in the unit box. The forcing amplitude a_rms comes from calibration; target Mach is not an exact constraint on the evolved gas.'),
        ('driven-force','Driven turbulence: force realization and initial gas',force_lo,force_hi,
         'Drive constructs and normalizes the time-dependent large-scale force, with a signed-wave-number projection for production runs. initial! initializes a uniform gas at rest. The forcing spectrum, correlation time and projection matter in addition to the random seed. The solver implementation called here is documented separately under Inside the solver.')]:
        block,data=render_section(ident,title,tl,lo,hi,note,'mach seed solenoidal gamma','',
                                  'sherlock_turb/turb_drive.jl',prefix='D',number='24' if ident=='driven' else '25')
        blocks.append(block);notes.update(data)
    known={t[0] for t in TERMS}
    assert all(set(n['terms'])<=known for n in notes.values())
    catalog,count = config_catalog()
    registry=json.loads((SITE/'tests.json').read_text())
    nav=''.join(f'<a href="{e(t["page"])}"><span>{e(t["number"])}</span>{e(t["nav"])}</a>' for t in registry)
    toc=''.join(f'<a href="#{ident}"><span>{i+1:02d}</span>{e(title)}</a>' for i,(ident,title,*_) in enumerate(SECTIONS))
    glossary=''.join(f'<details class="term" id="term-{slug}" data-search="{e((title+" "+text).lower())}"><summary>{e(title)}</summary><p>{e(text)}</p><a class="term-source" href="#{section}">See it in the code ↑</a></details>' for slug,title,text,section in TERMS)
    sha=hashlib.sha256(raw).hexdigest()
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex, nofollow"><meta name="color-scheme" content="light"><title>Test definitions · MHD Field Notes</title>
<link rel="stylesheet" href="style.css"><link rel="stylesheet" href="solver-walkthrough.css"><link rel="stylesheet" href="problem-definitions.css"></head>
<body class="solver-page problem-page"><a class="skip-link" href="#main">Skip to the definitions</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta">LOCAL RESEARCH NOTEBOOK / TEST DEFINITIONS</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav}<a href="problems.html" aria-current="page">Test definitions</a></nav>
<main id="main"><div class="page-heading"><div><div class="eyebrow"><span>Test definitions</span></div><h1>Inside the test problems</h1>
<p>The complete <code>problems.jl</code>, with line numbers, setup equations and reading notes. Click a source line to see its explanation on the right.</p></div></div>
<p class="provenance">Current benchmark source: <code>sherlock_sweep/src/problems.jl</code> · {len(lines)} lines · <a href="../sherlock_sweep/src/problems.jl" download>Download the complete Julia file</a><br>SHA-256 <code>{sha}</code></p>
<section class="variant-card"><h2>Where the complete setup lives</h2><p><code>problems.jl</code> defines the shared initial state. The named test's TOML selects the variant, parameters, grid, end time and driver controls. These are the current repository definitions; historical run records preserve the settings used by each saved run.</p>
<p><a href="#configurations">All {count} sweep configurations</a> · <a href="#timestep-jump">Timestep jump</a> · <a href="#driven">Driven Mach 3 / 10 / 30 / 50</a> · <a href="solver.html">Inside the evolution kernels</a></p>
<p><a href="../sherlock_sweep/codes/common/abel_problems.hpp" download>C++ initial-condition implementation</a> · <a href="../sherlock_sweep/src/run_suite.jl" download>Sweep driver</a> · <a href="../sherlock_turb/turb_drive.jl" download>Driven-turbulence driver</a></p></section>
<details class="reading-map"><summary>Jump to a problem or helper</summary><nav>{toc}<a href="#timestep-jump">Timestep-jump control</a><a href="#driven">Driven turbulence</a><a href="#configurations">All configurations</a></nav></details>
<div class="reading-layout"><div class="code-column">{''.join(blocks)}
<section class="driver-note" id="configurations"><h2>Configurations for every sweep test</h2><p>Open a filename to read the complete TOML. Source branch links show how variants share one initialization routine. Test names link to their results where available.</p>{catalog}
<h2>Driven-turbulence configurations</h2><p><a href="../sherlock_turb/config.toml" download>Mach 3 / 10</a> · <a href="../sherlock_turb/config_highmach.toml" download>Mach 30 / 50</a> · <a href="../sherlock_turb/config_movie.toml" download>Dense movies: Mach 3 / 10</a> · <a href="../sherlock_turb/config_movie_highmach.toml" download>Dense movies: Mach 30 / 50</a></p><p>Near-isothermal gamma differs between the low- and high-Mach campaigns. Forcing strengths are calibrated separately; achieved Mach and validity are measured in the result pages.</p></section></div>
<aside class="reading-sidebar" aria-label="Setup annotations and glossary"><div class="sidebar-inner"><div class="active-note" id="reading-note" aria-live="polite" aria-atomic="true"><div class="sidebar-kicker">Reading notes <span id="note-location"></span></div><h2 id="note-title"></h2><div id="note-equations"></div><p id="note-body"></p><div id="related-terms"></div><a id="note-code" href="#state">Go to these lines ↑</a></div>
<div class="glossary"><h2>Glossary <span>{len(TERMS)} terms</span></h2><label for="glossary-search">Find a term or symbol</label><input id="glossary-search" type="search" placeholder="Try Mach, potential, cell average…" autocomplete="off"><p id="glossary-empty" hidden>No matching terms.</p><div class="glossary-terms">{glossary}</div></div></div></aside></div></main>
<footer><span>Benchmark setup source and commentary · MHD Field Notes</span><a href="tests.html">Test results</a></footer>
<script type="application/json" id="walkthrough-notes">{json.dumps(notes,ensure_ascii=False).replace('<',chr(92)+'u003c')}</script><script src="solver-walkthrough.js"></script></body></html>'''
    (SITE/'problems.html').write_text(page)
    print(f'Problem definitions: complete {len(lines)}-line source, {len(notes)} annotations, {count} configs.')


if __name__=='__main__':main()
