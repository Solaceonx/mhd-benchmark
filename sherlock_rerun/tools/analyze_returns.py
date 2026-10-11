"""Analyze extracted Sherlock returns on the Mac; never launches simulations.

python analyze_returns.py RETURN_DIRECTORY --out ANALYSIS_DIRECTORY
Requires Python >=3.11, numpy and matplotlib. Uses bundled compact Athena data.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tomllib
import textwrap

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mhd-return-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/mhd-return-cache')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'font.size': 14,
})
import matplotlib.pyplot as plt
import numpy as np

BUNDLE=Path(__file__).resolve().parents[1]
FIELDS={'density':slice(0,1),'momentum':slice(1,4),'energy':slice(4,5),'magnetic':slice(5,8)}


def ordered_methods(methods):
    """Keep the main study methods stable while accepting external solvers."""
    preferred=('glm','glm_hll16','glm_hlld','glm_ppm','glm_constch','ct',
               'athenak_matched','athenak_best','ramses_matched','ramses_best',
               'athenapp_matched','athenapp_best')
    methods=set(methods)
    return [m for m in preferred if m in methods]+sorted(methods-set(preferred))


def load_state(path, meta):
    shape=tuple(meta['saved_shape']); expected=int(np.prod(shape))*4
    if path.stat().st_size != expected:raise ValueError(f'Truncated/wrong-sized snapshot: {path}')
    a=np.fromfile(path,dtype='<f4').reshape(shape,order='F')
    # Canonical numpy axes: y,x,c or z,y,x,c.
    return a.transpose(tuple(reversed(range(a.ndim-1)))+(a.ndim-1,)).astype(float)


def load_faces(directory,frame,meta):
    shape=tuple(meta['saved_shape'][:-1]);fields=[]
    for d in (1,2,3):
        p=directory/f'face{d}_{frame:03d}.bin'
        if p.stat().st_size!=int(np.prod(shape))*4:raise ValueError(f'Wrong face size: {p}')
        fields.append(np.fromfile(p,dtype='<f4').reshape(shape,order='F').transpose().astype(float))
    return fields


def error_norms(a,b):
    e=np.linalg.norm(a-b,axis=-1);r=np.linalg.norm(b,axis=-1)
    norms={}
    for name,num,den in [('L1',e.mean(),r.mean()),('L2',np.sqrt(np.mean(e*e)),np.sqrt(np.mean(r*r))),('Linf',e.max(),r.max())]:
        norms['absolute_'+name]=float(num);norms['relative_'+name]=float(num/den) if den>0 else None
    return norms


def spectrum(q,vector=False):
    dimensions=q.ndim-1 if vector else q.ndim
    axes=tuple(range(dimensions));ft=np.fft.fftn(q,axes=axes,norm='forward')
    p=np.abs(ft)**2
    if vector:p=p.sum(axis=-1)
    modes=np.meshgrid(*[np.fft.fftfreq(n)*n for n in p.shape],indexing='ij',sparse=True)
    shell=np.floor(np.sqrt(sum(k*k for k in modes))+1e-10).astype(int)
    power=np.bincount(shell.ravel(),weights=p.ravel())
    expected=np.mean(np.sum(q*q,axis=-1) if vector else q*q)
    if not np.isclose(power.sum(),expected,rtol=1e-11,atol=1e-25):raise ValueError('Parseval failure')
    return power


def reference(meta,t,directory,stamp,references):
    """Driver-written reference snapshot if present, else the bundled Athena Orszag-Tang data."""
    if stamp.get('reference_file'):
        return load_state(directory/stamp['reference_file'],meta),stamp.get('reference','driver reference')
    n=meta['N'];name=meta.get('problem',meta['test'])
    if name=='orszag_tang':
        allowed=[0.,.1,.25,.5];nearest=min(allowed,key=lambda x:abs(x-t))
        if abs(nearest-t)>1e-7:return None,'unavailable at this time'
        p=references/f'athena1024_N{n}_t{nearest:.3f}.npz'
        if not p.exists():return None,'Athena unavailable on this grid'
        with np.load(p) as r:
            u=np.zeros((n,n,9));u[...,0]=r['density'];u[...,1:4]=r['momentum'];u[...,4]=r['energy'];u[...,5:8]=r['magnetic']
        return u,'Athena 1024, volume averaged; finite numerical reference'
    return None,stamp.get('reference','No evolved independent reference')


def midplane(q):
    return q[q.shape[0]//2] if q.ndim==3 else q


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def analyze(root,out,references):
    out.mkdir(parents=True,exist_ok=True)
    runfiles=sorted(root.rglob('run.toml'))
    if not runfiles:raise ValueError(f'No run.toml files found below {root}. Extract the returned archives first.')
    timings=[];norms=[];powers=[];diagnostics=[];groups={};checks=0;failed=[]
    for path in runfiles:
        m=tomllib.loads(path.read_text());directory=path.parent
        environment=tomllib.loads((directory.parent/'environment.toml').read_text())
        group=str(directory.parent.relative_to(root))
        # Plot all available codes/variants for the same physical case together.
        # ``group`` remains archive-specific in the CSVs so timing repeats from
        # separate submissions are never mixed accidentally.
        key=(m['test'],m['N'],m['seed'],bool(environment.get('smoke',False)))
        base=dict(group=group,test=m['test'],problem=m.get('problem',m['test']),N=m['N'],seed=m['seed'],method=m['method'],repeat=m['repeat'],
            gpu=environment['device'],smoke=environment['smoke'],status=m['status'])
        timing={**base,**{k:m.get(k) for k in ('steps','stepping_seconds','cfl_seconds','solve_seconds','transfer_seconds','diagnostics_seconds','binary_output_seconds','evolution_wall_seconds','startup_including_warmup_seconds','warmup_seconds','allocated_array_bytes','million_cell_steps_per_second','snapshots_written')}}
        timings.append(timing)
        if m['status']!='complete':failed.append(dict(**base,error=m.get('error','Interrupted before completion')))
        csvpath=directory/'diagnostics.csv'
        if csvpath.exists():
            with csvpath.open() as f:diagrows=list(csv.DictReader(f))
            diagnostics.extend({**base,**r} for r in diagrows)
        else:diagrows=[]
        if not m.get('snapshots_written'):continue
        for stamp in m['frames']:
            if 'file' not in stamp:continue
            t=stamp['time'];frame=stamp['frame'];u=load_state(directory/stamp['file'],m)
            if not np.isfinite(u).all():
                failed.append(dict(**base,error=f'Nonfinite saved frame {frame}'));continue
            ref,label=reference(m,t,directory,stamp,references)
            rowbase={**base,'time':t,'frame':frame,'reference':label}
            for field,sl in FIELDS.items():
                if ref is not None:norms.append({**rowbase,'field':field,**error_norms(u[...,sl],ref[...,sl])})
            rho=u[...,0];mom=u[...,1:4];b=u[...,5:8]
            if np.any(rho<=0):continue
            quantities={'density_fluctuation':(rho-rho.mean(),False,1.),'kinetic_energy':(mom/np.sqrt(rho)[...,None],True,.5),'magnetic_energy':(b,True,.5)}
            if ref is not None:
                quantities.update(error_density=(u[...,0]-ref[...,0],False,1.),error_magnetic=(b-ref[...,5:8],True,1.))
            for name,(q,vector,factor) in quantities.items():
                power=factor*spectrum(q,vector);checks+=1
                powers.extend(dict(**rowbase,diagnostic=name,k_lower=k,k_upper=k+1,power=float(p),complete_annulus=k+1<=m['N']//2) for k,p in enumerate(power))
            if m['method']=='ct':
                faces=load_faces(directory,frame,m);div=np.zeros(rho.shape)
                for d in range(rho.ndim):div+=m['N']*(np.roll(faces[d],-1,axis=rho.ndim-1-d)-faces[d])  # planar: z faces are z invariant
                dr=next((r for r in diagrows if int(r['frame'])==frame),None)
                if dr and not np.isclose(abs(div).max(),float(dr['div_face_max']),rtol=1e-5,atol=1e-9):raise ValueError(f'Face divergence mismatch: {path}, frame {frame}')
            groups.setdefault(key,{})[m['method']]=(u,ref,label,t,m['status'])
    write_csv(out/'timings.csv',timings);write_csv(out/'norms.csv',norms);write_csv(out/'spectra.csv',powers);write_csv(out/'diagnostics.csv',diagnostics)
    summary=[]
    for group in sorted(set(r['group'] for r in timings)):
        for method in ordered_methods(r['method'] for r in timings if r['group']==group):
            sel=[r for r in timings if r['group']==group and r['method']==method and r['status']=='complete']
            if not sel:continue
            row={k:sel[0][k] for k in ('group','test','N','seed','method','gpu','smoke')};row['completed_repeats']=len(sel)
            for metric in ('stepping_seconds','solve_seconds'):
                a=np.array([r[metric] for r in sel],float)
                row.update({metric+'_median':float(np.median(a)),metric+'_q25':float(np.quantile(a,.25)),metric+'_q75':float(np.quantile(a,.75))})
            # Output exists only in repeat 1: do not mix output/no-output scopes.
            row['evolution_with_output_seconds']=next((r['evolution_wall_seconds'] for r in sel if r['snapshots_written']),None)
            summary.append(row)
    write_csv(out/'timing_summary.csv',summary)
    for (name,n,seed,smoke),methods in groups.items():
        target=out/f'{name}_N{n}_seed{seed}_{"smoke" if smoke else "full"}';target.mkdir(exist_ok=True)
        method_names=ordered_methods(methods)
        arrays=[];titles=[]
        for method in method_names:
            u,ref,label,t,status=methods[method];q=midplane(u[...,0])
            arrays.append(q);titles.append(f'{method.upper()} · {status} · t={t:.3f}')
        reference_record=next((r for r in methods.values() if r[1] is not None),None)
        if reference_record:
            arrays.append(midplane(reference_record[1][...,0]))
            titles.append(f'Reference · t={reference_record[3]:.3f}\n'+textwrap.fill(reference_record[2],45))
        actual=[a for a in arrays if a is not None]
        if not actual:continue
        lo=min(a.min() for a in actual);hi=max(a.max() for a in actual)
        if hi==lo:lo-=1e-6;hi+=1e-6
        ncols=min(4,len(arrays));nrows=(len(arrays)+ncols-1)//ncols
        fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,4*nrows),squeeze=False,layout='constrained')
        axes=axes.ravel();shown=[]
        for ax,a,title in zip(axes,arrays,titles):
            if a is not None:
                im=ax.imshow(a,origin='lower',cmap='magma',vmin=lo,vmax=hi,extent=(0,1,0,1));shown.append(ax)
            ax.set_title(title,fontsize=8);ax.set(xlabel='x/L',ylabel='y/L')
        for ax in axes[len(arrays):]:ax.set_visible(False)
        fig.colorbar(im,ax=shown,label='Density',shrink=.75)
        fig.suptitle(f'{name} · N={n}³ · seed={seed} · density');fig.savefig(target/'density.png',dpi=140);plt.close(fig)
        selected=[r for r in norms if r['test']==name and r['N']==n and r['seed']==seed and r['smoke']==smoke and r['field'] in ('density','magnetic')]
        if selected:
            fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
            for ax,field in zip(axes,('density','magnetic')):
                for method in ordered_methods(r['method'] for r in selected):
                    series=[r for r in selected if r['method']==method and r['field']==field and r['relative_L1'] is not None]
                    series.sort(key=lambda r:r['time'])
                    if series:ax.plot([r['time'] for r in series],[r['relative_L1'] for r in series],'o-',label=method.upper())
                ax.set(title=field,xlabel='Physical time',ylabel='Relative L1 difference');ax.grid(alpha=.2)
                if ax.lines:ax.legend(fontsize=8)
            fig.savefig(target/'errors.png',dpi=140);plt.close(fig)
    result={'runs_found':len(runfiles),'failed_or_incomplete':failed,'spectra_checked':checks,
        'timing_note':'Only complete runs enter medians. Compute times exclude output/analysis. End-to-end with snapshots is repeat 1 only. Smoke results are not performance benchmarks.',
        'reference_note':'Bundled Athena 1024 fields are finite numerical references. Other tests use the explicitly labeled analytic reference or have no evolved reference.',
        'missing_metrics':'Internal floor activation count and actual peak device memory are not instrumented. Recorded floor cells and explicit array bytes are different quantities.'}
    (out/'analysis_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Analyzed {len(runfiles)} runs; {len(failed)} failed/incomplete; {checks} spectra checked. Output: {out}')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('returns',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--references',type=Path,default=BUNDLE/'references');a=p.parse_args()
    analyze(a.returns.resolve(),a.out.resolve(),a.references.resolve())
