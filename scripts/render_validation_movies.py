"""Render saved entropy/acoustic experiments and update their notebook records."""
import json
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from build_progress_site import ROOT, SITE, MEDIA, plt, COLORS
from plot_style import MUTED, REFERENCE, RULE, RUN, field_map, frame


def encode(path, fig, draw, count):
    writer=imageio_ffmpeg.write_frames(str(path),(640,480),fps=12,codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','19','-movflags','+faststart'])
    writer.send(None)
    try:
        for i in range(count):
            draw(i);fig.canvas.draw()
            rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
            if i==count-1:fig.savefig(path.with_suffix('.png'),dpi=100)
            for _ in range(12 if i in (0,count-1) else 3):writer.send(rgb)
    finally:writer.close();plt.close(fig)


def entropy(tests):
    root=ROOT/'scripts/out/entropy_wave'
    rows=np.genfromtxt(root/'errors.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    final=rows[np.isclose(rows['time'],1)]
    result=[]
    for method in ('glm','ct'):
        sel=final[final['method']==method]
        assert np.all(np.diff(sel['L1'])<0), 'Density error must decrease on refinement'
        orders=np.log2(sel['L1'][:-1]/sel['L1'][1:])
        assert np.all(orders>1.5), 'Expected smooth-wave convergence missing'
        result.append(dict(method=method,N=sel['N'].tolist(),L1=sel['L1'].tolist(),orders=orders.tolist()))
    (root/'validation.json').write_text(json.dumps({'convergence':result,'positive_and_conservation_checks':'passed in runner',
        'saved_norm_rows':len(rows),'max_mass_drift':float(rows['mass_drift'].max()),
        'max_energy_drift':float(rows['energy_drift'].max()),'max_pressure_error':float(rows['pressure_Linf'].max()),
        'max_B_error':float(rows['B_Linf'].max()),
        'all_norms_finite':all(bool(np.isfinite(rows[k]).all()) for k in ('L1','L2','Linf','mass_drift','energy_drift','pressure_Linf','B_Linf')),
        'reference':'Exact nonlinear cell-average solution, evaluated in Float64',
        'limitation':'Constant magnetic field; not a test of magnetic wave propagation'},indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(6.4,4.8),layout='constrained')
    for r in result:ax.loglog(r['N'],r['L1'],'o-',label=r['method'].upper(),color=COLORS[r['method']])
    ax.loglog(result[0]['N'],result[0]['L1'][0]*(16/np.array(result[0]['N']))**2,':',color=MUTED,label=r'$N^{-2}$ guide')
    ax.set(xlabel='Grid N (N × N)',ylabel='Mean |density − exact cell average|',title='Entropy advection · one period · t = 1')
    ax.grid(True,color=RULE,lw=.6);ax.set_axisbelow(True);ax.legend();fig.savefig(root/'convergence.png',dpi=150);plt.close(fig)
    n=64;x=(np.arange(n)+.5)/n;xx,yy=np.meshgrid(x,x)
    for method in ('glm','ct','truth'):
        fig,ax=plt.subplots(figsize=(6.4,4.8),dpi=100,layout='constrained')
        im=ax.imshow(np.zeros((n,n)),origin='lower',extent=(0,1,0,1),vmin=.8,vmax=1.2,cmap=field_map('density'))
        fig.colorbar(im,ax=ax,label='Density').outline.set_linewidth(.6);ax.set(xlabel='x / L',ylabel='y / L');frame(ax)
        def draw(i):
            t=i/40
            if method=='truth':q=1+.2*np.sinc(1/n)**2*np.sin(2*np.pi*(xx+yy-t))
            else:q=np.fromfile(root/f'N64/{method}_{i:03d}.bin',dtype='<f4').reshape((n,n,9),order='F')[:,:,0].T
            im.set_data(q);ax.set_title(f'{"Exact cell averages" if method=="truth" else method.upper()} · 64² · t = {t:.3f}')
        encode(MEDIA/f'entropy-{method}.mp4',fig,draw,41)
    columns=[]
    for method in ('glm','ct','truth'):
        cards=[{'title':'One full advection period','video':f'media/entropy-{method}.mp4','poster':f'media/entropy-{method}.png',
            'caption':'64² · 41 saved times · play any movie to play all three'}]
        if method=='truth':
            cards.extend([{'title':'An exact nonlinear solution','formula':'ρ = 1 + 0.2 sin[2π(x + y − t)]',
                'text':'Uniform velocity (0.5, 0.5), pressure 1 and magnetic field (0.3, 0.2) stay constant. Density translates. Exact cell averages multiply the sinusoidal amplitude by sinc(1/N)²; initial states use the same averages.'},
                {'title':'What this tests','text':'Smooth multidimensional entropy transport at finite amplitude. The uniform magnetic field does not exercise Alfvén waves, reconnection or shock capture. These still need separate tests.'}])
        else:
            r=next(r for r in result if r['method']==method)
            cards.extend([{'title':'Measured density convergence','stats':[[f'Mean error · {n}²',f'{e:.3e}'] for n,e in zip(r['N'],r['L1'])]+[['Successive orders',', '.join(f'{p:.2f}' for p in r['orders'])]],
                'text':'Order = log₂(error at N / error at 2N), measured at t = 1. Specific to this smooth test and resolution range.'},
                {'title':'Checks passed','items':['Identical initial conserved fields for GLM and CT','Positive density and pressure at every RK stage','Mass and energy drift below 10⁻⁵','Density error decreases at every refinement']}])
        cards.append({'title':'Convergence and records','image':'../scripts/out/entropy_wave/convergence.png',
            'links':[['Every saved norm · CSV','../scripts/out/entropy_wave/errors.csv'],['Validation','../scripts/out/entropy_wave/validation.json'],['Run metadata','../scripts/out/entropy_wave/metadata.toml']]})
        columns.append({'subtitle':'Exact nonlinear cell averages' if method=='truth' else 'Shared planar MHD solver · uniform magnetic field',
            'status':'Exact reference' if method=='truth' else 'Validation passed','cards':cards})
    entry={'id':'entropy-wave','page':'entropy-wave.html','kind':'generic','number':'05','title':'Magnetized entropy advection','nav':'Entropy wave',
        'status':'Exact-reference check','description':'A smooth density pattern crosses a magnetized periodic box. An exact solution lets us measure numerical error without relying on another solver.',
        'chips':['2D ideal MHD','16² → 128²','Finite amplitude 0.2','One advection period'],
        'document':'../scripts/CPU_BENCHMARKS.md','columns':columns}
    tests[:]=[entry if t['id']=='entropy-wave' else t for t in tests] if any(t['id']=='entropy-wave' for t in tests) else tests+[entry]     # in its place in the navigation


def acoustic(tests):
    root=ROOT/'scripts/out/acoustic_movie';rows=np.genfromtxt(root/'profiles.csv',delimiter=',',names=True)
    for role in ('hydro','truth'):
        fig,ax=plt.subplots(figsize=(6.4,4.8),dpi=100,layout='constrained')
        # the solver's two grids in its green (coarse dashed, fine solid); the analytical wave in the grey of references
        lines={n:ax.plot([],[],label=f'{n}³ numerical' if role=='hydro' else 'Linear analytical wave',lw=1.8,color=RUN if role=='hydro' else REFERENCE,
                         linestyle=(0,(5,2)) if (role=='hydro' and n==32) else '-')[0] for n in ((32,64) if role=='hydro' else (64,))}
        ax.set(xlim=(0,1),ylim=(-1.1,1.1),xlabel='x / L',ylabel='(Density − 1) / 0.001');ax.legend(loc='upper right');ax.grid(True,color=RULE,lw=.6);ax.set_axisbelow(True)
        def draw(i):
            for n,line in lines.items():
                sel=rows[(rows['N']==n)&(rows['frame']==i)]
                line.set_data(sel['x'],(sel['density' if role=='hydro' else 'linear_reference']-1)/.001)
            ax.set_title(f'{"Hydro solver" if role=="hydro" else "Linear approximation"} · t = {i*.01:.3f}')
        encode(MEDIA/f'acoustic-{role}.mp4',fig,draw,21)
    entry=next(t for t in tests if t['id']=='acoustic-wave')
    for idx,role in ((0,'hydro'),(2,'truth')):
        cards=entry['columns'][idx]['cards']
        cards[:]=[c for c in cards if c.get('title')!='Watch the acoustic wave']
        cards.insert(0,{'title':'Watch the acoustic wave','video':f'media/acoustic-{role}.mp4','poster':f'media/acoustic-{role}.png',
            'caption':'21 times · t = 0 → 0.2 · linked playback',
            'text':'A new run with saved output times. The analytical wave is a linear approximation; the solver evolves nonlinear hydrodynamics.',
            'links':[['Saved profiles','../scripts/out/acoustic_movie/profiles.csv']]})
    entry['columns'][0]['cards'][1]['text']='Historical check without intermediate output. The movie uses a new output-clipped run; its measurements are saved separately.'
    entry['columns'][1]['cards'][1]['text']='The entropy-wave page now adds an exact smooth magnetized test. Alfvén and magnetosonic propagation remain future tests.'
    errors={str(n):float(np.mean(np.abs(r['density']-r['linear_reference']))) for n in (32,64) if len(r:=rows[(rows['N']==n)&(rows['frame']==20)])}
    (root/'validation.json').write_text(json.dumps({'final_mean_density_error':errors,'finite':bool(np.isfinite(rows['density']).all()),'note':'New output cadence; original check preserved'},indent=2)+'\n')


if __name__=='__main__':
    tests=json.loads((SITE/'tests.json').read_text());entropy(tests);acoustic(tests)
    (SITE/'tests.json').write_text(json.dumps(tests,indent=2,ensure_ascii=False)+'\n')
    print('Validation movies and notebook records updated')
