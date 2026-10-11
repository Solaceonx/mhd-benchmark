"""Assemble stock Athena++ VTK blocks and compare against local GLM/CT states."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import tomllib

os.environ.setdefault('MPLCONFIGDIR','/tmp/abel-mhd-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/abel-mhd-cache')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external'/'athena'/'vis'/'python'))
import athena_read
import numpy as np
import matplotlib
matplotlib.use('Agg')
from plot_style import apply_plot_style, field_map
apply_plot_style()
import matplotlib.pyplot as plt


def load_athena(root,time):
    manifest=json.loads((root/'run_manifest.json').read_text());n=manifest['N']
    stage=root/(manifest['segments'][0]['directory'] if time==0 else f't{time:.3f}')
    paths=[]
    for p in stage.glob('*.vtk'):
        with p.open('rb') as f:
            f.readline();header=f.readline().decode()
        stamp=float(re.search(r'time=([^\s]+)',header)[1])
        if abs(stamp-time)<1e-10:paths.append(p)
    if not paths:raise ValueError(f'No snapshot at {time} in {root}')
    fields={key:np.zeros((n,n)+shape) for key,shape in
        (('density',()),('energy',()),('momentum',(3,)),('magnetic',(3,)))}
    coverage=np.zeros((n,n),dtype=np.uint8)
    for p in paths:
        x,y,z,d=athena_read.vtk(p)
        nx,ny=len(x)-1,len(y)-1
        ix=int(round((x[0]+.5)*n));iy=int(round((y[0]+.5)*n))
        assert np.allclose(np.diff(x),1/n) and np.allclose(np.diff(y),1/n)
        assert 0<=ix<ix+nx<=n and 0<=iy<iy+ny<=n
        sl=np.s_[iy:iy+ny,ix:ix+nx];coverage[sl]+=1
        for key,source in (('density','dens'),('energy','Etot'),('momentum','mom'),('magnetic','Bcc')):
            fields[key][sl]=d[source][0]
    assert np.all(coverage==1),'Missing or overlapping MeshBlocks'
    assert all(np.isfinite(a).all() for a in fields.values())
    rho=fields['density'];pressure=(5/3-1)*(fields['energy']-(fields['momentum']**2).sum(axis=-1)/(2*rho)-(fields['magnetic']**2).sum(axis=-1)/2)
    assert rho.min()>0 and pressure.min()>0
    return fields


def restrict(fields,n):
    nf=fields['density'].shape[0];assert nf%n==0;r=nf//n
    return {key:a.reshape((n,r,n,r)+a.shape[2:]).mean(axis=(1,3)) for key,a in fields.items()}


def load_local(root,method,time,n):
    d=np.genfromtxt(root/f'{method}_diagnostics.csv',delimiter=',',names=True)
    idx=int(np.argmin(abs(d['time']-time)))
    assert abs(d['time'][idx]-time)<1e-8
    u=np.fromfile(root/f'{method}_{idx:03d}.bin',dtype='<f4').reshape((n,n,9),order='F').transpose(1,0,2).astype(float)
    return {'density':u[...,0],'energy':u[...,4],'momentum':u[...,1:4],'magnetic':u[...,5:8]}


def norms(actual,reference):
    diff=actual-reference
    if diff.ndim==3:
        err=np.linalg.norm(diff,axis=-1);scale=np.linalg.norm(reference,axis=-1)
    else:err=abs(diff);scale=abs(reference)
    return float(err.sum()/scale.sum()),float(np.sqrt((err**2).sum()/(scale**2).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--check-reference',type=Path)
    parser.add_argument('--local',type=Path,default=ROOT/'scripts/out/glm_ct_64')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();ref=args.reference.resolve();local=args.local.resolve()
    n=tomllib.loads((local/'metadata.toml').read_text())['N']
    nref=json.loads((ref/'run_manifest.json').read_text())['N']
    out=(args.out or ROOT/'scripts/out'/f'errors_athena_{nref}').resolve();out.mkdir(parents=True,exist_ok=True)
    times=[0,.1,.25,.5];rows=[];checkrows=[];final={};initial_check={}
    for t in times:
        full=load_athena(ref,t)
        if t==0:
            nf=full['density'].shape[0];coords=(np.arange(nf)+.5)/nf
            xx,yy=np.meshgrid(coords,coords);rho0=25/(36*np.pi);b0=1/np.sqrt(4*np.pi)
            expected_mom=np.stack((-rho0*np.sin(2*np.pi*yy),rho0*np.sin(2*np.pi*xx),np.zeros_like(xx)),axis=-1)
            # Athena's discrete curl gives the analytic face average (a sinc factor).
            expected_b=np.stack((-b0*np.sin(2*np.pi*yy)*np.sinc(1/nf),b0*np.sin(4*np.pi*xx)*np.sinc(2/nf),np.zeros_like(xx)),axis=-1)
            assert np.max(abs(full['momentum']-expected_mom))<1e-7
            assert np.max(abs(full['magnetic']-expected_b))<1e-7
            initial_check={'coordinate_mapping_verified':True,'max_initial_momentum_sampling_error':float(np.max(abs(full['momentum']-expected_mom))),
                'max_initial_field_discrete_curl_error':float(np.max(abs(full['magnetic']-expected_b)))}
        coarse=restrict(full,n)
        np.savez_compressed(out/f'reference_on_{n}_t{t:.3f}.npz',**coarse)
        for method in ('glm','ct'):
            state=load_local(local,method,t,n)
            for field in coarse:
                l1,l2=norms(state[field],coarse[field])
                rows.append({'method':method,'time':t,'field':field,'relative_L1':l1,'relative_L2':l2,'reference_N':nref})
            if t==times[-1]:final[method]=state
        if args.check_reference:
            fine=restrict(load_athena(args.check_reference.resolve(),t),n)
            for field in coarse:
                l1,l2=norms(coarse[field],fine[field]);checkrows.append({'time':t,'field':field,'relative_L1':l1,'relative_L2':l2})
            for method in ('glm','ct'):
                state=load_local(local,method,t,n)
                for field in fine:
                    l1,l2=norms(state[field],fine[field])
                    rows.append({'method':method,'time':t,'field':field,'relative_L1':l1,'relative_L2':l2,'reference_N':json.loads((args.check_reference/'run_manifest.json').read_text())['N']})
        if t==times[-1]:final['reference']=coarse
    for filename,records in (('errors.csv',rows),('reference_sensitivity.csv',checkrows)):
        if records:
            with (out/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'validation.json').write_text(json.dumps(initial_check,indent=2)+'\n')
    colors={'glm':'#167d9a','ct':'#ce6d27'}
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,field in zip(axes.flat,('density','magnetic','momentum','energy')):
        for method in ('glm','ct'):
            for refn in sorted(set(r['reference_N'] for r in rows)):
                selected=[r for r in rows if r['field']==field and r['method']==method and r['reference_N']==refn]
                ax.plot([r['time'] for r in selected],[100*r['relative_L1'] for r in selected],'-o' if refn==nref else '--s',color=colors[method],label=f'{method.upper()} vs {refn}²')
        if checkrows:
            selected=[r for r in checkrows if r['field']==field]
            ax.plot([r['time'] for r in selected],[100*r['relative_L1'] for r in selected],':',color='black',label='Reference-to-reference difference')
        ax.set(title=field.capitalize(),xlabel='time',ylabel='Relative L1 difference (%)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'{n}² GLM/CT error relative to independent Athena++ reference',fontsize=16)
    fig.savefig(out/'error_vs_time.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    refdensity=final['reference']['density'];vmin=min(final[m]['density'].min() for m in final);vmax=max(final[m]['density'].max() for m in final)
    for ax,name,title in zip(axes[0],('reference','glm','ct'),(f'Athena++ {nref}² averaged to {n}²',f'GLM {n}²',f'CT {n}²')):
        im=ax.imshow(final[name]['density'],origin='lower',extent=(0,1,0,1),cmap=field_map('density'),vmin=vmin,vmax=vmax)
        ax.set_title(title,fontsize=11);fig.colorbar(im,ax=ax,shrink=.8)
    maxdiff=max(abs(final[m]['density']-refdensity).max() for m in ('glm','ct'))
    for ax,m in zip(axes[1,:2],('glm','ct')):
        im=ax.imshow(final[m]['density']-refdensity,origin='lower',extent=(0,1,0,1),cmap=field_map('signed'),vmin=-maxdiff,vmax=maxdiff)
        l1,_=norms(final[m]['density'],refdensity)
        ax.set_title(f'{m.upper()} − reference · L1 {100*l1:.2f}%',fontsize=11);fig.colorbar(im,ax=ax,shrink=.8)
    ax=axes[1,2];j=n//2
    for m in ('reference','glm','ct'):
        ax.plot((np.arange(n)+.5)/n,final[m]['density'][j],'-',color='black' if m=='reference' else colors[m],label=m.upper())
    ax.set_title(f'Density slice at y={(j+.5)/n:.4f}',fontsize=11);ax.legend(fontsize=8);ax.grid(alpha=.2)
    for ax in axes.flat:ax.set_xlabel('x / L')
    fig.suptitle('Orszag–Tang at t = 0.5 · numerical-reference comparison',fontsize=16)
    fig.savefig(out/'density_errors.png',dpi=150);plt.close(fig)
    print('Error artifacts:',out)
    for row in rows:
        if row['time']==.5 and row['field'] in ('density','magnetic'):print(row)
    if checkrows:
        print('Reference sensitivity at t=0.5:',[r for r in checkrows if r['time']==.5])


if __name__=='__main__':main()
