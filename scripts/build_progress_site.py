"""Build the local, three-column research notebook from saved simulation artifacts.

No simulations are run. HTML and data work through file:// or a local HTTP server.
Use --movies to create/update GLM, CT and Athena movies at matching saved times.
Use --gpu-only to refresh just the GPU suite page from downloaded Sherlock returns,
keeping the existing Orszag–Tang data and media.
"""
import argparse
import csv
from datetime import datetime
import html
import json
import os
from pathlib import Path
import tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/abel-mhd-cache")
import matplotlib
matplotlib.use("Agg")
from plot_style import COLUMN, INK, MUTED, RULE, apply_movie_style, field_map, frame
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
from PIL import Image
from compare_athena_reference import load_athena, load_local, restrict
import build_gpu_suite
apply_movie_style(10)      # after the imports: the notebook's look for every figure and movie made here

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "progress"
MEDIA = SITE / "media"
STUDY = ROOT / "scripts/out/spectral_study"
ANALYSIS = STUDY / "analysis"
TIMES = [0.0, 0.1, 0.25, 0.5]
COLORS = COLUMN                    # GLM green, CT orange, the Athena++ reference grey: as the columns of the page
MAPS = {"density": field_map("density"), "magnetic": field_map("magnetic")}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save_field(path, q, bounds, cmap):
    rgba = matplotlib.colormaps[cmap]((np.flipud(q)-bounds[0])/(bounds[1]-bounds[0]))
    Image.fromarray((255*rgba).astype("uint8")).resize((512,512),Image.Resampling.NEAREST).save(path)


def movies(states, bounds, reference_n, n, field, times):
    import imageio_ffmpeg
    for method, frames in states.items():
        path = MEDIA / f"ot-{method}-{n}-{field}.mp4"
        fig, ax = plt.subplots(figsize=(5.12,5.76),dpi=100)
        fig.subplots_adjust(left=.12,right=.94,bottom=.13,top=.87)
        im = ax.imshow(frames[0],origin="lower",extent=(0,1,0,1),cmap=MAPS[field],vmin=bounds[0],vmax=bounds[1],interpolation="nearest")
        ax.set(xlabel="x / L",ylabel="y / L");frame(ax)
        title=fig.suptitle("",fontsize=13,color=INK)
        caption=f"Athena++ {reference_n}² → {n}² · {field}" if method=='truth' else f"{n}² · {field} · {len(times)} saved states · shared scale"
        fig.text(.12,.035,caption,fontsize=9,color=MUTED)
        writer=imageio_ffmpeg.write_frames(str(path),(512,576),fps=12,codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=16,output_params=["-crf","19","-movflags","+faststart"])
        writer.send(None)
        try:
            for i in range(len(frames)):
                label='Athena++' if method=='truth' else method.upper()
                im.set_data(frames[i]); title.set_text(f"{label}  /  Orszag–Tang  /  t = {times[i]:.4f}")
                fig.canvas.draw()
                rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
                for _ in range(12 if i in (0,len(frames)-1) else 2):writer.send(rgb)
        finally:
            writer.close();plt.close(fig)
        print("Rendered",path.name,flush=True)


def plot_spectra(rows,n,method,nref,ncheck):
    fig,axes=plt.subplots(2,1,figsize=(6,5.5),layout="constrained")
    name="athena" if method=="truth" else method
    for ax,field,title in zip(axes,("kinetic_energy","magnetic_energy"),("Kinetic energy spectrum","Magnetic energy spectrum")):
        sources=[(name,COLORS[method],"-",f"Athena++ {nref}²" if method=="truth" else method.upper())]
        if method!="truth":sources.append(("athena",COLORS["truth"],"--",f"Athena++ {nref}² → {n}²"))
        elif n<=ncheck:sources.append(("check",MUTED,"--",f"Athena++ {ncheck}² → common grid"))
        for source,color,style,label in sources:
            sel=[r for r in rows if int(r['N'])==n and r['method']==source and r['time']=='0.5' and r['diagnostic']==field and 1<=int(r['k_lower'])<n//2]
            ax.loglog([float(r['k_lower'])+.5 for r in sel],[float(r['power']) for r in sel],style,color=color,lw=1.8,label=label)
        ax.set(title=title,ylabel="Shell energy",xlabel="kL · cycles per box")
        ax.set_xticks([2**j for j in range(1,10) if 2**j<=n//2]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True,color=RULE,lw=.6);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.savefig(MEDIA/f"spectra-{n}-{method}.png",dpi=130);plt.close(fig)


def plot_energies(rows,n,method):
    source="athena" if method=="truth" else method
    selected=[r for r in rows if int(r['N'])==n and r['method']==source and r['representation']=='comparison_grid']
    fig,ax=plt.subplots(figsize=(6,3.5),layout="constrained")
    # one code per figure: its colour for all three energies, told apart by the line (and the reference's saved times by dots)
    for field,style in (("kinetic","-"),("magnetic",(0,(5,2))),("thermal",(0,(1,1.5)))):
        ax.plot([float(r['time']) for r in selected],[float(r[field]) for r in selected],linestyle=style,marker="o" if method=="truth" else "",ms=4,lw=1.8,label=field.capitalize(),color=COLORS[method])
    ax.set(xlabel="Time",ylabel="Mean energy density",title="Energy partition on the comparison grid")
    ax.grid(True,color=RULE,lw=.6);ax.set_axisbelow(True);ax.legend(fontsize=9,handlelength=3.2)
    fig.savefig(MEDIA/f"energies-{n}-{method}.png",dpi=130);plt.close(fig)


def build_ot(render_movies):
    spec=read_csv(ANALYSIS/'spectra.csv');norms=read_csv(ANALYSIS/'norms.csv')
    validation=json.loads((ANALYSIS/'validation.json').read_text())
    reference_root=Path(validation['reference'])
    if not reference_root.exists():
        reference_root=ROOT/'scripts/out'/Path(validation['reference']).name
    nref=json.loads((reference_root/'run_manifest.json').read_text())['N']
    check_root=Path(validation['check_reference'])
    if not check_root.exists():check_root=ROOT/'scripts/out'/check_root.name
    ncheck=json.loads((check_root/'run_manifest.json').read_text())['N']
    full={t:load_athena(reference_root,t) for t in TIMES}
    energies=read_csv(ANALYSIS/'energies.csv')
    # Movie provenance is independent of the quantitative reference. Keep the
    # existing dense 512² trajectory even when norms use a finer reference.
    movie_root=ROOT/'scripts/out/athena_512_movie41'
    if not (movie_root/'run_manifest.json').exists():movie_root=ROOT/'scripts/out/athena_512_movie'
    if not (movie_root/'run_manifest.json').exists():
        raise ValueError(f'Missing dense-output Athena run: {movie_root}')
    movie_meta=json.loads((movie_root/'run_manifest.json').read_text())
    if not movie_meta['completed'] or not movie_meta['segments']:
        raise ValueError('Athena movie requires a completed dense-output run')
    movie_times=np.array([0.0]+[s['time'] for s in movie_meta['segments']])
    assert np.all(np.diff(movie_times)>0) and np.isclose(movie_times[-1],.5)
    movie_full=[load_athena(movie_root,float(t)) for t in movie_times]
    ot={'times':TIMES,'referenceN':nref,'checkReferenceN':ncheck,'runs':{},'norms':norms,
        'validation':validation,'movieReferenceN':movie_meta['N']}
    for n in (64,128,256,512):
        root=STUDY/f'glm_ct_{n}'
        meta=tomllib.loads((root/'metadata.toml').read_text())
        states={m:[load_local(root,m,t,n) for t in TIMES] for m in ('glm','ct')}
        states['truth']=[restrict(full[t],n) for t in TIMES]
        movie_local=ROOT/'scripts/out/spectral_study_movie41'/f'glm_ct_{n}'
        if not (movie_local/'metadata.toml').exists():movie_local=root
        movie_local_meta=tomllib.loads((movie_local/'metadata.toml').read_text())
        local_times=np.linspace(0,float(movie_local_meta['t_end']),int(movie_local_meta['frames']))
        if len(local_times)!=len(movie_times) or not np.allclose(local_times,movie_times,atol=1e-8):
            raise ValueError(f'Movie cadence mismatch: {movie_local} versus {movie_root}')
        movie_states={m:[load_local(movie_local,m,float(t),n) for t in movie_times] for m in ('glm','ct')}
        movie_states['truth']=[restrict(s,n) for s in movie_full]
        run={'meta':meta,'bounds':{},'images':{},'truthStats':[]}
        for state in states['truth']:
            run['truthStats'].append({'rhoMin':float(state['density'].min()),'rhoMax':float(state['density'].max()),'Brms':float(np.sqrt(np.mean(np.sum(state['magnetic']**2,axis=-1))))})
        for field,cmap in MAPS.items():
            fields={m:[s['density'] if field=='density' else np.linalg.norm(s['magnetic'],axis=-1) for s in states[m]] for m in states}
            lo=min(a.min() for frames in fields.values() for a in frames)
            hi=max(a.max() for frames in fields.values() for a in frames)
            movie_fields={m:[s['density'] if field=='density' else np.linalg.norm(s['magnetic'],axis=-1) for s in frames] for m,frames in movie_states.items()}
            lo=min(lo,min(a.min() for frames in movie_fields.values() for a in frames));hi=max(hi,max(a.max() for frames in movie_fields.values() for a in frames))
            run['bounds'][field]=[float(lo),float(hi)]
            run['images'][field]={}
            for m,frames in fields.items():
                run['images'][field][m]=[]
                for i,a in enumerate(frames):
                    name=f'ot-{n}-{m}-{field}-{i}.png';save_field(MEDIA/name,a,(lo,hi),cmap)
                    run['images'][field][m].append('media/'+name)
            if render_movies:movies(movie_fields,(lo,hi),movie_meta['N'],n,field,movie_times)
        for m in states:plot_spectra(spec,n,m,nref,ncheck);plot_energies(energies,n,m)
        ot['runs'][str(n)]=run
        print(f'Built {n}² snapshots and column plots',flush=True)
    return ot


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--movies',action='store_true')
    parser.add_argument('--gpu-only',action='store_true',help='rebuild only the GPU suite data; reuse the existing Orszag-Tang data')
    parser.add_argument('--ot-only',action='store_true',help='rebuild only the Orszag-Tang data and media; reuse the existing GPU suite data (needs no returns)')
    parser.add_argument('--gpu-returns',type=Path,default=build_gpu_suite.DEFAULT_RETURNS,help='directory of extracted Sherlock archives')
    parser.add_argument('--movies-only',action='store_true',help='only refresh the movie list (progress/movies.json) in the existing page data; needs no returns')
    parser.add_argument('--gpu-override',action='append',default=[],metavar='BUNDLE=RETURNS',
        help='tests configured in BUNDLE (e.g. sherlock_rerun) take their results only from RETURNS (extracted archives)')
    args=parser.parse_args()
    MEDIA.mkdir(parents=True,exist_ok=True)
    tests=json.loads((SITE/'tests.json').read_text())
    ids=[t['id'] for t in tests];pages=[t['page'] for t in tests]
    assert len(ids)==len(set(ids)) and len(pages)==len(set(pages))
    if args.movies_only:
        data=json.loads((SITE/'data.js').read_text().removeprefix('window.RESEARCH_DATA = ').rstrip().removesuffix(';'))
        data['gpu']['movies']=build_gpu_suite.movies_manifest({t['name'] for t in data['gpu']['tests']})
        (SITE/'data.js').write_text('window.RESEARCH_DATA = '+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n')
        print('Movies on the page:',len(data['gpu']['movies']));return
    if args.gpu_only:
        previous=(SITE/'data.js').read_text().removeprefix('window.RESEARCH_DATA = ').rstrip().removesuffix(';')
        ot=json.loads(previous)['ot']
    else:ot=build_ot(args.movies)
    overrides=[]
    for item in args.gpu_override:
        bundle,sep,root=item.partition('=')
        if not sep:parser.error('--gpu-override expects BUNDLE=RETURNS')
        overrides.append(((ROOT/bundle).resolve(),Path(root).resolve()))
    if args.ot_only:gpu=json.loads((SITE/'data.js').read_text().removeprefix('window.RESEARCH_DATA = ').rstrip().removesuffix(';'))['gpu']
    else:gpu=build_gpu_suite.build(args.gpu_returns.resolve(),MEDIA,overrides)
    data={'tests':tests,'ot':ot,'gpu':gpu,'built':datetime.now().astimezone().isoformat(timespec='minutes')}
    (SITE/'data.js').write_text('window.RESEARCH_DATA = '+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n')
    for test in tests:
        if test['kind'] in ('sweep','papers','turbulence','tests','ours'):continue      # static pages: scripts/build_sweep_plan.py, build_paper_figures.py, build_turbulence.py (run below)
        nav=''.join(f'<a href="{html.escape(t["page"])}"'+(' aria-current="page"' if t['id']==test['id'] else '')+f'><span>{t["number"]}</span>{html.escape(t["nav"])}</a>' for t in tests)
        page=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><meta name="description" content="Local MHD research progress: GLM, CT and independent references.">
<title>{html.escape(test['title'])} · MHD Field Notes</title><link rel="stylesheet" href="style.css">
<script defer src="data.js"></script><script defer src="sync-videos.js"></script><script defer src="app.js"></script></head>
<body data-test="{html.escape(test['id'])}"><a class="skip-link" href="#main">Skip to comparison</a>
<header class="masthead"><a class="brand" href="index.html"><span class="brand-mark">∿</span><span>MHD <strong>FIELD NOTES</strong></span></a><div class="masthead-meta"><span class="live-dot"></span>LOCAL RESEARCH NOTEBOOK <span class="separator">/</span> {'GPU · SHERLOCK' if test['kind']=='gpu' else 'CPU FIRST'}</div></header>
<nav class="test-nav" aria-label="Physics tests">{nav}</nav>
<main id="main"><div id="intro"></div><div id="controls"></div><div class="board-scroll"><div id="board" class="board" aria-label="GLM, CT and reference comparison"></div></div><div id="extra"></div></main>
<footer><span>Every claim has a run behind it. Every reference has a limit.</span><span id="build-stamp"></span></footer>
<dialog id="image-viewer"><button class="close-dialog" aria-label="Close image">×</button><img alt=""><p></p><a class="source-link" target="_blank" rel="noopener">Open original image ↗</a></dialog>
<noscript><p class="noscript">Enable JavaScript to explore the saved results. <a href="{html.escape(test['document'])}">Read this test's methods and results.</a></p></noscript>
</body></html>'''
        (SITE/test['page']).write_text(page)
    if any(t['kind']=='sweep' for t in tests):
        import build_sweep_plan;build_sweep_plan.main()
    if any(t['kind']=='papers' for t in tests):
        import build_paper_figures;build_paper_figures.main(['--page-only'])
    if any(t['kind']=='turbulence' for t in tests):
        import build_turbulence;build_turbulence.main(['--page-only'])
    if any(t['kind']=='tests' for t in tests):
        import build_test_pages;build_test_pages.main()
    if any(t['kind']=='ours' for t in tests):
        import build_our_figures;build_our_figures.main(['--page-only'])
    print('Progress site:',SITE/'index.html')


if __name__=='__main__':main()
