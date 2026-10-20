"""Real shared diagnostics for each test page; no simulations and no synthetic data.

    .venv-movie/bin/python scripts/test_diagnostics.py [TEST ...]

Reads fetched sweep records and available state/reference files. Every unavailable
comparison is reported. A reference is never substituted with a same-grid solver.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import html
import json
import os
from pathlib import Path
import re
import textwrap
import tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import sweep_data as sd
import paper_common as pc
from plot_style import apply_movie_style, FAMILY

ROOT=sd.ROOT
SITE=sd.SITE
OUT=SITE/"media/analysis"
FIELDS=("density","magnetic")
NORMS=("L1","L2","Linf")
ALE_TESTS=re.search(r'^MHDALE_TESTS="([^"]+)"',(ROOT/"sherlock_mhdale/options.sh").read_text(),re.M).group(1).split()
METHODS=sd.METHODS+sd.AREPO+("mhdale_ale",)
TITLES={"01-errors":"Error norms versus physical time", "02-convergence":"Error versus resolution",
        "03-cost":"Accuracy versus computation", "04-conservation":"Conservation and energy evolution",
        "05-divergence":"Magnetic divergence", "06-spectra":"Power spectra",
        "07-validity":"Completion and physical validity", "08-runtime":"Runtime and time to solution"}


def finite(value):
    try:return np.isfinite(float(value))
    except (ValueError,TypeError):return False


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value


def style(method):
    if method=="mhdale_ale":return dict(color="#227c89",linestyle="-",marker="v",label="MHD-ALE")
    v=pc.STYLE[method]
    name=(pc.plain(v[2]).replace("GLM","Abel GLM") if method.startswith("glm") else pc.plain(v[2])).replace("cₕ",r"$c_h$")
    if method=="ct":name="Abel CT prototype"
    return dict(color=FAMILY[v[0]],linestyle=v[3],marker=v[4],label=name)


def groups(methods):
    return [("All codes on shared axes",list(methods))] if methods else []


def identify(artist,method):
    """Tag vector artists so the page can emphasize selected methods."""
    fig=artist.figure
    fig._comparison_id=getattr(fig,"_comparison_id",0)+1
    artist.set_gid(f"series--{method}--{fig._comparison_id}")
    return artist


def draw(ax,x,y,method,points=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y)
    if not valid.any():return False
    kw=style(method)
    if not points:kw.pop("marker")
    line,=ax.plot(x[valid],y[valid],**kw,lw=1.7,ms=4)
    identify(line,method)
    ax.grid(False)
    return True


def blank(ax,message):
    ax.set_xticks([]);ax.set_yticks([])
    ax.text(.5,.5,textwrap.fill(message,48),ha="center",va="center",transform=ax.transAxes,
            fontsize=12,color="#64716f")


def logarithmic(ax):
    values=np.concatenate([np.asarray(l.get_ydata(),float) for l in ax.lines]) if ax.lines else np.array([])
    positive=values[np.isfinite(values)&(values>0)]
    if len(positive):ax.set_yscale("symlog",linthresh=max(1e-16,float(positive.min())/3,float(positive.max())*1e-5))


def compute_errors(actual,reference):
    if actual.shape!=reference.shape:raise ValueError("State and reference must have identical shapes")
    result={}
    for field,cols in (("density",slice(0,1)),("magnetic",slice(5,8))):
        e=np.linalg.norm(actual[...,cols].astype(float)-reference[...,cols].astype(float),axis=-1)
        b=np.linalg.norm(reference[...,cols].astype(float),axis=-1)
        for norm,num,den in (("L1",e.mean(),b.mean()),("L2",np.sqrt(np.mean(e*e)),np.sqrt(np.mean(b*b))),("Linf",e.max(),b.max())):
            result[f"{field}_absolute_{norm}"]=float(num)
            result[f"{field}_relative_{norm}"]=float(num/den) if den>0 else None
    return result


def metric(test,field):
    absolute=field=="magnetic" and test in ("blast_2d_b0","blast_3d_b0")
    return ("absolute","[code units]",1) if absolute else ("relative","[%]",100)


@dataclass
class Run:
    test:str
    method:str
    n:int
    seed:int
    path:Path
    meta:dict
    diag:dict
    planned:dict

    @property
    def finite_count(self):
        if not self.diag or "time" not in self.diag:return 0
        good=np.ones(len(self.diag["time"]),bool)
        for key in ("mass","energy","kinetic","magnetic","thermal"):
            if key in self.diag:good &= np.isfinite(self.diag[key])
        if "finite" in self.diag:good &= self.diag["finite"]!=0
        return len(good) if good.all() else int(np.argmin(good))

    @property
    def completed(self):
        return (self.meta.get("status")=="complete" and self.finite_count>0
                and self.finite_count==len(self.diag["time"])
                and abs(self.diag["time"][-1]-self.meta.get("t_end_requested",self.diag["time"][-1]))
                <=1e-5*max(1,abs(self.diag["time"][-1])))

    @property
    def gpu(self):return self.method in sd.GPU_METHODS or self.method.startswith("athenak")

    @functools.cached_property
    def environment(self):
        p=self.path.parent/"environment.toml"
        return tomllib.loads(p.read_text()) if p.is_file() else {}

    @property
    def ranks(self):
        launcher=self.environment.get("launcher","")
        match=re.search(r"(?:-n|--ntasks(?:=|\s))\s*(\d+)",launcher)
        return int(match.group(1)) if match else self.planned.get("ranks") or None

    @property
    def ranks_source(self):return "environment" if re.search(r"(?:-n|--ntasks(?:=|\s))\s*\d+",self.environment.get("launcher","")) else "plan"


class Data:
    def __init__(self):
        self.sw=sd.Sweep();self.runs={};self.missing=set()
        for key,planned in self.sw.by.items():
            test,method,n,seed=key
            if method not in METHODS:continue
            path=self.sw.path(test,method,n,"run.toml",seed)
            if path and path.is_file():
                self.runs[key]=Run(test,method,n,seed,path.parent,self.sw.meta(*key),self.sw.diag(*key) or {},planned)
        # Add-on MHD-ALE uses the same files but is outside the original sweep plan.
        roots=[self.sw.results]+[Path(p) for p in os.environ.get("MHD_ANALYSIS_EXTRA_RESULTS","").split(os.pathsep) if p]
        for root in roots:
            for p in root.glob("*-mhdale-ale-run-N*-s*/data/mhdale_ale_r1/run.toml"):
                m=tomllib.loads(p.read_text());test=m.get("test")
                if test not in self.sw.configs:continue
                n=int(m["N"]);seed=int(m.get("seed",42));key=(test,"mhdale_ale",n,seed)
                self.runs[key]=Run(test,"mhdale_ale",n,seed,p.parent,m,sd.read_csv(p.parent/"diagnostics.csv") or {},{})

    def methods(self,test):
        out=self.sw.labels(test)+[m for m in sd.AREPO if test in sd.AREPO_TESTS]
        if test in ALE_TESTS:out.append("mhdale_ale")
        return list(dict.fromkeys(out))

    def select(self,test,n=None,seed=42,methods=None):
        return [r for (t,m,grid,s),r in self.runs.items() if t==test and (seed is None or s==seed)
                and (n is None or grid==n) and (methods is None or m in methods)]

    @functools.lru_cache(maxsize=12)
    def load_array(self,path,n):
        p=Path(path)
        if not p.is_file():self.missing.add(str(p.relative_to(self.sw.results)) if p.is_relative_to(self.sw.results) else str(p));return None
        count=p.stat().st_size//4
        if count==n*n*9:shape=(n,n,9)
        elif count==n**3*9:shape=(n,n,n,9)
        else:return None
        a=np.fromfile(p,dtype="<f4").reshape(shape,order="F")
        return a if np.isfinite(a).all() else None

    def reference(self,test,n,seed,time):
        big=self.sw.reference_n(test,"athenak_reference")
        m=self.sw.meta(test,"athenak_reference",big,seed) if big else None
        if not m or big<=n:return None,None
        stamp=next((f for f in m.get("frames",[]) if abs(f["time"]-time)<=1e-6*max(1,abs(time))),None)
        if not stamp:return None,None
        name=stamp.get("restricted",{}).get(str(n))
        if not name:return None,None
        path=self.sw.path(test,"athenak_reference",big,name,seed)
        return self.load_array(str(path),n),f"AthenaK {big}{'²' if m.get('planar') else '³'} → {n}"

    @functools.lru_cache(maxsize=None)
    def errors(self,test,method,n,seed=42):
        run=self.runs.get((test,method,n,seed));out=[]
        if not run:return out
        last=run.diag.get("time",np.array([]))[:run.finite_count]
        if not len(last):return out
        for f in run.meta.get("frames",[]):
            time=f["time"]
            if time>last[-1]+1e-6*max(1,abs(time)):continue
            values=f.get("errors",{})
            if values:
                out.append(dict(time=time,values=values,reference=f.get("reference","recorded reference"),scope=run.meta.get("diagnostics_scope","")))
                continue
            if not f.get("file"):continue
            ref,name=self.reference(test,n,seed,time)
            if ref is None:continue
            a=self.load_array(str(run.path/f["file"]),n)
            # A 3D slice is never presented as a full-volume error norm.
            planar=run.meta.get("planar",False)
            if a is None or a.shape!=ref.shape or (not planar and a.ndim!=4):continue
            values=compute_errors(a,ref)
            out.append(dict(time=time,values=values,reference=name,scope="full volume" if not planar else "plane"))
        return out

    def endpoint(self,run,field):
        if not run.completed:return None
        errors=self.errors(run.test,run.method,run.n,run.seed)
        end=float(self.sw.configs[run.test]["t_end"])
        rows=[f for f in errors if abs(f["time"]-end)<=1e-6*max(1,abs(end))]
        if not rows:return None
        mode,_,_=metric(run.test,field)
        v=rows[-1]["values"].get(field+"_"+mode+"_L1")
        return float(v) if finite(v) else None

    def spectral(self,run,time):
        p=run.path/"spectra.csv"
        if p.is_file():
            d=sd.read_csv(p)
            if d:
                keep=np.abs(d["time"]-time)<=1e-6*max(1,abs(time))
                if keep.any():return {k:v[keep] for k,v in d.items()},"full-volume recorded spectrum"
        if not run.meta.get("planar"):return None,"full-volume spectrum not fetched"
        stamp=next((f for f in run.meta.get("frames",[]) if abs(f["time"]-time)<=1e-6*max(1,abs(time)) and f.get("file")),None)
        if not stamp:return None,"state not saved at selected time"
        a=self.load_array(str(run.path/stamp["file"]),run.n)
        if a is None or a.ndim!=3 or np.any(a[...,0]<=0):return None,"plane state not fetched or invalid"
        return planar_spectrum(a),"computed from cell-centred plane"


def variance_per_lnk(k,e):
    """Normalized variance per logarithmic interval of k: k E(k) / sum E, where E is the shell-summed power (so
    k E(k) = 4 pi k^3 P(k) in 3D and 2 pi k^2 P(k) in 2D). Its area over ln k is 1: a curve shows where the
    variance sits, whatever the total."""
    k=np.asarray(k,float);e=np.asarray(e,float);ok=np.isfinite(e)&(e>0)&(k>=1)
    total=e[ok].sum()
    return (k*e/total) if total>0 else np.full_like(e,np.nan)


def planar_spectrum(a):
    """Match the sweep shell convention: round(|k|), shell sums, 1..N/2."""
    n=a.shape[0];k=np.fft.fftfreq(n,d=1/n);kx,ky=np.meshgrid(k,k,indexing="ij")
    shell=np.rint(np.hypot(kx,ky)).astype(int)
    def energy(q):
        h=np.fft.fftn(q,axes=(0,1))/n**2
        power=np.abs(h)**2
        if q.ndim==3:power=power.sum(axis=-1)
        return .5*np.bincount(shell.ravel(),weights=power.ravel(),minlength=n//2+1)[1:n//2+1]
    return dict(k=np.arange(1,n//2+1),v=energy(a[...,1:4]/a[...,0,None]),b=energy(a[...,5:8]))


class Figures:
    def __init__(self,data,test):
        self.data=data;self.test=test;self.config=data.sw.configs[test];self.methods=data.methods(test)
        self.entries=[];self.root=OUT/test;self.root.mkdir(parents=True,exist_ok=True)
        available=[r for r in data.select(test) if r.diag]
        grids=sorted({r.n for r in available})
        self.n=128 if 128 in grids else (grids[0] if grids else 128)
        # Accuracy can need a different grid because reference files were fetched selectively.
        counts={n:sum(bool(data.errors(test,m,n)) for m in self.methods) for n in grids}
        self.error_n=max(grids,key=lambda n:(counts[n],n==128,-abs(n-128))) if grids else self.n

    def save(self,fig,category,suffix,group,caption,details=None):
        ident=f"{category}-{suffix}";title=TITLES[category]
        width,height=fig.get_size_inches();fig.set_size_inches(width,max(6.8,height))
        fig.suptitle(f"{title} · {self.test}",x=.075,ha="left",fontsize=18,y=.99)
        fig.text(.075,.94,group,fontsize=11,color="#64716f")
        bars=category=="08-runtime" or (category=="07-validity" and fig.axes and fig.axes[0].patches)
        fig.subplots_adjust(left=.22 if bars else .10,right=.975,bottom=.14,top=.72,wspace=.34,hspace=.55)
        # Both legend blocks describe curves on the same axes.
        handles={}
        for ax in fig.axes:
            h,l=ax.get_legend_handles_labels()
            handles.update(dict(zip(l,h)))
        known={style(m)["label"]:m for m in METHODS}
        legends=[[],[]]
        for label,handle in handles.items():
            method=known.get(label)
            ours=method in sd.GPU_METHODS if method else label.startswith("Abel")
            short=label.replace("Abel et al.","").replace("Abel ","").strip()
            legends[0 if ours else 1].append((handle,short,method))
        for j,entries in enumerate(legends):
            if not entries:continue
            entries.sort(key=lambda entry:METHODS.index(entry[2]) if entry[2] else len(METHODS))
            x=.10 if j==0 else .52
            fig.text(x,.888,"Abel et al." if j==0 else "External codes / reference",fontsize=11,fontweight="bold")
            legend=fig.legend([h for h,_,_ in entries],[l for _,l,_ in entries],loc="upper left",
                bbox_to_anchor=(x,.874),ncol=min(3,len(entries)),fontsize=9,frameon=False,
                borderaxespad=0,columnspacing=1.15,handlelength=2.6,handletextpad=.55)
            for (handle,_,method),swatch,label in zip(entries,legend.legend_handles,legend.get_texts()):
                if method:
                    identify(swatch,method);identify(label,method)
        fig.text(.075,.04,"Measured run records · available outputs only",fontsize=9,color="#64716f")
        for ext in ("png","svg"):fig.savefig(self.root/f"{ident}.{ext}",dpi=160,facecolor="white")
        plt.close(fig)
        self.entries.append(dict(category=category,title=title,group=group,file=f"media/analysis/{self.test}/{ident}.png",
                                 vector=f"media/analysis/{self.test}/{ident}.svg",aspect=float(fig.get_size_inches()[0]/fig.get_size_inches()[1]),caption=caption,details=details or []))

    def selected(self,methods,n=None):return self.data.select(self.test,self.n if n is None else n,methods=methods)

    def error_histories(self):
        for gi,(name,methods) in enumerate(groups(self.methods)):
            fig,axs=plt.subplots(2,3,figsize=(13,8))
            sources=set();missing=[]
            for j,field in enumerate(FIELDS):
                mode,units,mult=metric(self.test,field)
                for k,norm in enumerate(NORMS):
                    ax=axs[j,k];plotted=False
                    for m in methods:
                        rows=self.data.errors(self.test,m,self.error_n)
                        vals=[f["values"].get(f"{field}_{mode}_{norm}") for f in rows]
                        if not any(finite(v) for v in vals):continue
                        plotted |= draw(ax,[f["time"] for f in rows],[float(v)*mult if finite(v) else np.nan for v in vals],m,points=True)
                        sources.update(f["reference"] for f in rows)
                    if not plotted:blank(ax,"No valid reference comparison fetched at this grid/time. More state and reference snapshots are needed.")
                    else:logarithmic(ax);ax.set_xlabel("Physical time t");ax.set_ylabel(f"{field.title()} error {units}")
                    ax.set_title(norm.replace("Linf",r"$L_\infty$"))
            for m in methods:
                if not self.data.errors(self.test,m,self.error_n):missing.append(style(m)["label"]+": no paired reference data")
            self.save(fig,"01-errors",str(gi),f"{name} · N={self.error_n}, seed 42",
                "Relative L1, L2, and Linf use the matching reference norm; zero-field blasts use absolute magnetic error. Magnetic error is a vector difference. Markers are outputs, not a complete history when sparse. Symlog retains zero with a linear band at least 10^-5 of the panel peak, so roundoff does not stretch the scale. Reference: "+("; ".join(sorted(sources)) or "not available")+".",missing)

    def convergence_cost(self):
        for gi,(name,methods) in enumerate(groups(self.methods)):
            fig,axs=plt.subplots(1,2,figsize=(13,5.8));has=False
            for j,field in enumerate(FIELDS):
                mode,units,mult=metric(self.test,field)
                ax=axs[j]
                for method in methods:
                    runs=sorted(self.data.select(self.test,methods=[method]),key=lambda r:r.n)
                    pairs=[(r.n,self.data.endpoint(r,field)) for r in runs]
                    pairs=[(n,e) for n,e in pairs if e is not None]
                    if pairs:has |= draw(ax,*zip(*[(n,e*mult) for n,e in pairs]),method,points=True)
                if not ax.lines:blank(ax,"No endpoint reference errors available. Finished runs alone do not establish accuracy.")
                else:ax.set_xscale("log",base=2);logarithmic(ax);ax.set_xlabel("Cells per side, N");ax.set_ylabel(field.title()+f" {mode} L1 error {units}")
            self.save(fig,"02-convergence",str(gi),name+" · endpoint, seed 42",
                "Only completed, finite runs at the requested physical endpoint. Analytic and restricted high-resolution numerical references are labelled in the error-history figure. Numerical-reference convergence has not been established for every case.")
        fig,axs=plt.subplots(2,2,figsize=(13,8))
        for j,field in enumerate(FIELDS):
            mode,units,mult=metric(self.test,field)
            for c,gpu in enumerate((True,False)):
                ax=axs[j,c]
                for method in self.methods:
                    pairs=[]
                    for r in self.data.select(self.test,methods=[method]):
                        e=self.data.endpoint(r,field);sec=timing(self.data,r)[0]
                        if r.gpu!=gpu or e is None or not sec or (not gpu and not r.ranks):continue
                        cost=sec*(1 if gpu else r.ranks)/3600
                        pairs.append((cost,e*mult))
                    if pairs:draw(ax,*zip(*sorted(pairs)),method,points=True)
                if not ax.lines:blank(ax,"No completed endpoint errors with comparable recorded resource costs.")
                else:ax.set_xscale("log");logarithmic(ax);ax.set_xlabel("H100 GPU-hours" if gpu else "CPU core-hours");ax.set_ylabel(field.title()+f" {mode} L1 error {units}")
                ax.set_title("GPU" if gpu else "CPU")
            # GPU/CPU resource units differ, but accuracy must use the same scale.
            axes=axs[j]
            values=np.concatenate([np.asarray(line.get_ydata(),float) for ax in axes for line in ax.lines]) if any(ax.lines for ax in axes) else np.array([])
            positive=values[np.isfinite(values)&(values>0)]
            if len(positive):
                threshold=max(1e-16,float(positive.min())/3,float(positive.max())*1e-5)
                for ax in axes:ax.set_yscale("symlog",linthresh=threshold)
                limits=[ax.get_ylim() for ax in axes if ax.lines]
                for ax in axes:ax.set_ylim(min(v[0] for v in limits),max(v[1] for v in limits))
        self.save(fig,"03-cost","all","GPU and CPU resource costs separated · seed 42",
                "The same endpoint L1 errors as convergence, with matched accuracy scales across GPU/CPU panels. One H100 per GPU run; CPU costs use environment rank counts, or explicitly planned ranks when environment records are missing. Median valid completed repeats are used. GPU stepping+CFL and external evolution-loop timings have different output scopes; see run metadata.")

    def conservation(self):
        for gi,(name,methods) in enumerate(groups(self.methods)):
            fig,axs=plt.subplots(2,3,figsize=(13,8))
            for r in self.selected(methods):
                d=r.diag;k=r.finite_count
                if not k:continue
                t=d["time"][:k];mass=d.get("mass");energy=d.get("energy")
                if mass is not None and mass[0]!=0:draw(axs[0,0],t,(mass[:k]-mass[0])/abs(mass[0]),r.method)
                if energy is not None and energy[0]!=0:draw(axs[0,1],t,(energy[:k]-energy[0])/abs(energy[0]),r.method)
                mom=[d.get("momentum_"+x) for x in "xyz"]
                if all(v is not None for v in mom) and mass is not None:
                    scale=abs(mass[0])*np.sqrt(float(r.meta.get("gamma",5/3))*max(d.get("thermal",np.array([0]))[0],0)*(float(r.meta.get("gamma",5/3))-1)/max(mass[0],1e-300))
                    if scale>0:draw(axs[0,2],t,np.linalg.norm(np.array([v[:k]-v[0] for v in mom]),axis=0)/scale,r.method)
                for j,key in enumerate(("kinetic","magnetic","thermal")):
                    if key in d and energy is not None and energy[0]!=0:draw(axs[1,j],t,d[key][:k]/abs(energy[0]),r.method)
            labels=["Signed mass drift / initial mass","Signed total-energy drift / initial energy",r"Momentum drift / $M_0c_{s,0}$","Kinetic / initial total energy","Magnetic / initial total energy","Thermal / initial total energy"]
            for ax,label in zip(axs.flat,labels):
                if not ax.lines:blank(ax,"This diagnostic is not in the fetched records.")
                else:ax.set_xlabel("Physical time t");ax.set_ylabel(label);ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3))
            self.save(fig,"04-conservation",str(gi),f"{name} · N={self.n}, seed 42",
                "These sweep tests are periodic and unforced. Component energies may change physically; mass and total-energy drift reveal budget errors. Momentum uses initial mass × initial sound speed, so zero initial momentum is safe. Curves stop at the first nonfinite output.")

    def divergence(self):
        for gi,(name,methods) in enumerate(groups(self.methods)):
            fig,axs=plt.subplots(1,3,figsize=(13,5.8))
            for r in self.selected(methods):
                d=r.diag;k=r.finite_count
                if not k:continue
                t=d["time"][:k];brms=np.sqrt(sum(d.get(x+"_rms",np.zeros(len(d["time"])))[:k]**2 for x in ("bx","by","bz")))
                for j,key in enumerate(("div_center_rms","div_center_max")):
                    if key in d:
                        y=np.divide(d[key][:k],r.n*brms,out=np.full(k,np.nan),where=brms>0)
                        draw(axs[j],t,y,r.method)
                if "div_face_rms" in d:draw(axs[2],t,d["div_face_rms"][:k],r.method)
            for ax,label in zip(axs,[r"Common RMS: $\Delta x\,\|\nabla\cdot\mathbf{B}\|_2/B_{rms}$",r"Common max: $\Delta x\,\|\nabla\cdot\mathbf{B}\|_\infty/B_{rms}$","Native face divergence RMS"]):
                if not ax.lines:blank(ax,"No applicable divergence data. Relative divergence is undefined for a zero magnetic field; native face data exist only for applicable CT codes.")
                else:logarithmic(ax);ax.set_xlabel("Physical time t");ax.set_ylabel(label)
            self.save(fig,"05-divergence",str(gi),f"{name} · N={self.n}, seed 42",
                "The common diagnostic uses the shared cell-centred operator. Native face divergence is shown separately in code units; it is not directly comparable to projected cell-centred divergence. AREPO and MHD-ALE projection/sampling can introduce divergence on the analysis grid.")

    def spectra(self):
        if self.test=="uniform":
            fig,ax=plt.subplots(figsize=(12,4.8));blank(ax,"Uniform-state preservation has no nonzero-wavenumber structure. A power-spectrum comparison is not meaningful for this test.")
            self.save(fig,"06-spectra","none","Not applicable","Uniform-state errors are quantified by the error histories instead.");return
        end=float(self.config["t_end"])
        # Choose one grid with maximum actual spectral coverage; do not mix grid sizes.
        counts={}
        for r in self.data.select(self.test):
            if not r.completed:continue
            d,_=self.data.spectral(r,end)
            if d:counts[r.n]=counts.get(r.n,0)+1
        n=max(counts,key=lambda g:(counts[g],g==self.n,-abs(g-self.n))) if counts else self.n
        ref_run=None;big=self.data.sw.reference_n(self.test,"athenak_reference")
        ref_meta=self.data.sw.meta(self.test,"athenak_reference",big) if big else None
        ref=None;ref_name="No independent reference spectrum at the selected time"
        if ref_meta and big>n:
            p=self.data.sw.path(self.test,"athenak_reference",big,"run.toml").parent
            rr=Run(self.test,"athenak_reference",big,42,p,ref_meta,self.data.sw.diag(self.test,"athenak_reference",big) or {},{})
            if rr.completed:ref,_=self.data.spectral(rr,end);ref_name=f"AthenaK reference at N={big}"
            if ref is None:
                a,tag=self.data.reference(self.test,n,42,end)
                if a is not None and a.ndim==3 and (a[...,0]>0).all():ref=planar_spectrum(a);ref_name=tag+" (restricted spectrum)"
        for gi,(name,methods) in enumerate(groups(self.methods)):
            fig,axs=plt.subplots(2,2,figsize=(13,8));missing=[]
            for j,key in enumerate(("v","b")):
                if ref is not None and key in ref:
                    good=np.isfinite(ref[key])&(ref[key]>0)&(ref["k"]>=1)
                    if good.any():axs[j,0].plot(ref["k"][good],variance_per_lnk(ref["k"],ref[key])[good],color="#52514e",lw=1.5,label="Reference")
                for r in self.selected(methods,n):
                    if not r.completed:continue
                    d,why=self.data.spectral(r,end)
                    if d is None:missing.append(style(r.method)["label"]+": "+why);continue
                    if key not in d:continue
                    keep=(d["k"]>=1)&(d[key]>0)&np.isfinite(d[key])
                    if keep.any():draw(axs[j,0],d["k"][keep],variance_per_lnk(d["k"],d[key])[keep],r.method)
                    if ref is not None and key in ref:
                        common=np.intersect1d(d["k"],ref["k"])
                        a=d[key][np.searchsorted(d["k"],common)];b=ref[key][np.searchsorted(ref["k"],common)]
                        good=np.isfinite(a)&np.isfinite(b)&(b>max(np.max(ref[key])*1e-12,0))
                        if good.any():draw(axs[j,1],common[good],a[good]/b[good],r.method)
                for c,ax in enumerate(axs[j]):
                    if not ax.lines:blank(ax,"No independent reference spectrum for ratios." if c else "No nonzero spectrum fetched at this grid/time; a saved state or full-volume spectrum is needed.")
                    else:
                        ax.set_xscale("log");ax.set_xlabel(r"Wavenumber k [2π/L]")
                        if c==0:ax.set_ylim(bottom=0);ax.set_ylabel(("Velocity" if key=="v" else "Magnetic")+r" variance per ln k, $k\,E(k)/\sum E$")
                        else:ax.axhline(1,color="#888888",ls="--",lw=.8);ax.set_ylabel("Code / reference")
            self.save(fig,"06-spectra",str(gi),f"{name} · N={n}, t={end:g}, seed 42",
                "Left: normalized variance per logarithmic interval of k, k E(k) divided by the total, where E(k) is the shell-summed power (round(|k|) bins, mean excluded); k E(k) is 4πk³P(k) in 3D and 2πk²P(k) in 2D, and each curve has area 1 over ln k, so the plot shows where the variance sits. Right: ratio of the unnormalized power to the reference. For 3D, recorded full-volume spectra are used; 2D spectra come from actual plane states. Small reference power is excluded from ratios. "+ref_name+". Late chaotic flows need statistical interpretation.",sorted(set(missing)))

    def validity(self):
        ns=sorted(set(self.config["grids"]));fig,ax=plt.subplots(figsize=(12,max(5,len(self.methods)*.34+2)))
        reasons=[]
        for j,method in enumerate(self.methods):
            for i,n in enumerate(ns):
                runs=self.data.select(self.test,n,methods=[method])
                run=runs[0] if runs else None
                if not run:color="#e1e5e4";mark="·"
                elif run.completed:color=style(method)["color"];mark="✓"
                else:
                    color="#f4ddd5";reached=run.diag.get("time",np.array([run.meta.get("time_reached",0)]))[max(0,run.finite_count-1)]
                    mark=f"{100*min(1,float(reached)/float(self.config['t_end'])):.0f}%"
                    reasons.append(f"{style(method)['label']} N={n}: "+str(run.meta.get("error",run.meta.get("status","unknown"))).splitlines()[-1][:240])
                identify(ax.add_patch(plt.Rectangle((i-.46,j-.40),.92,.8,color=color,alpha=.9)),method)
                identify(ax.text(i,j,mark,ha="center",va="center",fontsize=12,color="white" if run and run.completed else "#202a2a"),method)
        ax.set_xlim(-.6,len(ns)-.4);ax.set_ylim(len(self.methods)-.5,-.5);ax.set_xticks(range(len(ns)),labels=ns)
        ax.set_yticks(range(len(self.methods)),labels=[style(m)["label"] for m in self.methods],fontsize=10)
        for method,label in zip(self.methods,ax.get_yticklabels()):identify(label,method)
        ax.set_xlabel("Cells per side, N");ax.spines[["left","bottom"]].set_visible(False)
        # Matrix needs more left margin than line plots.
        self.save(fig,"07-validity","matrix","Seed 42 · all planned grids",
            "✓: complete and finite. Percentage: last finite physical time / requested time. Grey dot: no fetched record (not a solver failure). Other seeds and full status are available in the data table.",reasons)
        fig,axs=plt.subplots(1,3,figsize=(13,5.8))
        for r in self.selected(self.methods):
            d=r.diag;k=r.finite_count
            if not k:continue
            t=d["time"][:k]
            # Abel's kernels diagnose their full N³ allocation even on planar
            # tests. External planar codes diagnose an N²×1 state.
            cells=r.n**(3 if r.method in sd.GPU_METHODS or not r.meta.get("planar") else 2)
            for j,key in enumerate(("density_floor_cells","pressure_floor_cells","nonpositive_pressure_cells")):
                if key in d:draw(axs[j],t,100*d[key][:k]/cells,r.method)
        for ax,label in zip(axs,["Density-floor cells [%]","Pressure-floor cells [%]","Nonpositive-pressure cells [%]"]):
            if not ax.lines:blank(ax,"This cell count is not in the fetched diagnostics.")
            else:ax.set_xlabel("Physical time t");ax.set_ylabel(label)
        self.save(fig,"07-validity","floors",f"N={self.n}, seed 42",
            "Fractions use the actual measured domain: Abel GPU diagnostics count N³ cells, including planar tests; external planar diagnostics count N² cells. Floor observations are sampled counts, not every internal intervention. Nonfinite tails are omitted.")

    def runtime(self):
        for gi,(name,methods) in enumerate(groups(self.methods)):
            selected=[r for r in self.selected(methods) if r.completed]
            fig,axs=plt.subplots(1,2,figsize=(13,max(6.8,len(selected)*.48+3.2)))
            rows=[]
            for r in selected:
                sec,scope,reps=timing(self.data,r)
                if sec:rows.append((r,sec,scope,reps))
            rows.sort(key=lambda row:row[1])
            for i,(r,sec,scope,reps) in enumerate(rows):
                for bar in axs[0].barh(i,sec,color=style(r.method)["color"],height=.55):identify(bar,r.method)
                # Additional time only when a matching record measures total evolution time.
                wall=r.meta.get("evolution_wall_seconds",r.meta.get("wall_seconds"))
                if reps==1 and finite(wall) and float(wall)>=sec:
                    for bar in axs[0].barh(i,float(wall)-sec,left=sec,color=style(r.method)["color"],alpha=.3,hatch="///",height=.55):identify(bar,r.method)
                identify(axs[0].text(sec,i,f" {sec:.3g}s",va="center",fontsize=9),r.method)
                # Dense timing histories are not recorded; plot only the actual endpoint.
                draw(axs[1],[sec],[self.config["t_end"]],r.method,points=True)
            if rows:
                labels=[style(r.method)["label"]+"\n"+("1 H100" if r.gpu else f"{r.ranks or '?'} CPU ranks ({r.ranks_source})") for r,*_ in rows]
                axs[0].set_yticks(range(len(rows)),labels=labels,fontsize=9);axs[0].invert_yaxis()
                for (r,*_),label in zip(rows,axs[0].get_yticklabels()):identify(label,r.method)
                axs[0].set_xscale("log");axs[0].set_xlim(min(sec for _,sec,*_ in rows)*.6,max(max(sec,float(r.meta.get("evolution_wall_seconds",r.meta.get("wall_seconds",0)) or 0)) for r,sec,*_ in rows)*1.4)
                axs[1].set_xscale("log")
                axs[0].set_xlabel("Recorded execution time [s]")
                axs[1].set_xlabel("Execution time to endpoint [s]");axs[1].set_ylabel("Requested physical endpoint t")
            else:
                for ax in axs:blank(ax,"No valid completed timing records at the selected grid.")
            self.save(fig,"08-runtime",str(gi),f"{name} · N={self.n}, seed 42",
                "All codes sorted by recorded execution time, on logarithmic time axes. Solid: recorded solver/evolution-loop time. Hatched: measured additional wall time only for a single matched timing record. Endpoint dots are measured; no intermediate wall-clock progress is inferred. GPU and CPU ranks differ; resource and timing scopes appear in the run table.")


def timing(data,run):
    values=[];scopes=[]
    if not run.completed:return None,"incomplete or invalid",0
    repeats=int(data.sw.configs[run.test].get("repeats_by_grid",{}).get(str(run.n),data.sw.configs[run.test].get("repeats",1)))
    for rep in range(1,repeats+1):
        p=run.path.parent/f"{run.method}_r{rep}/run.toml"
        if not p.is_file():continue
        m=tomllib.loads(p.read_text())
        if m.get("status")!="complete" or m.get("timing_valid") is False:continue
        d=sd.read_csv(p.parent/"diagnostics.csv")
        if d and any(not np.isfinite(d[k]).all() for k in ("energy","mass") if k in d):continue
        sec=m.get("solve_seconds")
        if finite(sec) and sec>0:values.append(float(sec));scopes.append(m.get("timing_scope","scope not recorded"))
    return (float(np.median(values)),"; ".join(sorted(set(scopes))),len(values)) if values else (None,"valid timing not recorded",0)


def table(data,test):
    rows=[]
    for r in sorted(data.select(test,seed=None),key=lambda r:(METHODS.index(r.method),r.n,r.seed)):
        sec,scope,reps=timing(data,r)
        rows.append(dict(method=r.method,label=style(r.method)["label"],N=r.n,seed=r.seed,status=r.meta.get("status"),
            finite_complete=r.completed,physical_time=r.meta.get("time_reached"),solver_seconds=sec,timing_repeats=reps,
            timing_scope=scope,resource="1 H100" if r.gpu else (f"{r.ranks} CPU ranks ({r.ranks_source})" if r.ranks else "CPU ranks not fetched"),
            precision=r.meta.get("precision"),cfl=r.meta.get("cfl",r.meta.get("settings",{}).get("cfl",r.meta.get("settings",{}).get("courant"))),
            source=str(r.path.relative_to(ROOT)) if r.path.is_relative_to(ROOT) else str(r.path),error=r.meta.get("error")))
    return rows


def section(test, figure_prefix=None, data_test=None):
    manifest=SITE/"test_analysis.json"
    if not manifest.is_file():return ""
    registry=json.loads(manifest.read_text());item=registry.get("tests",{}).get(test)
    if not item:return ""
    if figure_prefix:
        item=dict(item,figures=[f for f in item["figures"] if f["file"].startswith(figure_prefix)])
    data_test=data_test or test
    e=html.escape
    parts=['<section class="test-analysis" id="analysis"><div class="page-heading"><div><h2>Shared quantitative diagnostics</h2><p>Abel et al. and the external codes share axes for direct comparison. Colours identify code families; line styles identify variants. Measured results only.</p></div></div>']
    parts.append('<p class="small">'+" · ".join(f'<a href="#analysis-{category}">{e(category[:2]+" "+title)}</a>' for category,title in TITLES.items())+'</p>')
    methods=item.get("methods",[])
    if methods:
        options="".join(f'<option value="{e(m)}">{e(style(m)["label"].replace("Abel ","").replace("$c_h$","cₕ"))}</option>' for m in methods if m in sd.GPU_METHODS)
        families=list(dict.fromkeys(m.split("_")[0] for m in methods if m not in sd.GPU_METHODS))
        names=dict(athenak="AthenaK",ramses="RAMSES",athenapp="Athena++",arepo="AREPO",mhdale="MHD-ALE")
        external="".join(f'<option value="{e(m)}">{e(names[m])}</option>' for m in families)
        parts.append('<div class="comparison-controls" hidden style="position:sticky;top:0;z-index:4;background:var(--paper);padding:8px 0"><div class="toolbar" style="flex-wrap:wrap;justify-content:flex-start;margin-bottom:6px">'
            '<label class="control-group">Abel kernel <select class="field-select" name="abel-kernel"><option value="">All kernels</option>'+options+'</select></label>'
            '<label class="control-group">External code <select class="field-select" name="external-code"><option value="">All external codes</option>'+external+'</select></label>'
            '<button class="field-select" type="button">Show all</button></div><p class="small comparison-status" aria-live="polite">All codes are shown on shared axes.</p></div>')
    if item.get("warnings"):parts.append('<article class="card"><h3>Data coverage</h3><ul>'+"".join(f"<li>{e(w)}</li>" for w in item["warnings"])+"</ul></article>")
    for category,title in TITLES.items():
        figures=[f for f in item["figures"] if f["category"]==category]
        parts.append(f'<article class="card analysis-card" id="analysis-{category}"><h3>{e(category[:2])} · {e(title)}</h3>')
        for f in figures:
            version=re.sub(r"\D","",registry.get("built_utc",""))
            png=e(f["file"]+"?v="+version);svg=e(f["vector"]+"?v="+version)
            parts.append(f'<h4>{e(f["group"])}</h4><div class="analysis-plot" data-vector="{svg}" data-aspect="{f.get("aspect",13/8)}"><a href="{png}"><img loading="lazy" style="width:100%;height:auto" src="{png}" alt="{e(title+": "+f["group"])}"></a></div><p>{e(f["caption"])}</p><p class="small">Download all codes: <a href="{png}">PNG</a> · <a href="{svg}">Vector SVG</a></p>')
            if f["details"]:parts.append('<details><summary>Coverage and stopped runs</summary><ul>'+"".join(f"<li>{e(v)}</li>" for v in f["details"])+"</ul></details>")
        parts.append("</article>")
    build_command=".venv-movie/bin/python scripts/driven_test_diagnostics.py" if test=="sherlock_turb" else f".venv-movie/bin/python scripts/test_diagnostics.py {e(test)}"
    parts.append(f'<article class="card"><h3>Source records and reproduction</h3><p><a href="media/analysis/{e(data_test)}/data.json">Numerical data and run metadata (JSON)</a> · <a href="media/analysis/{e(data_test)}/runs.csv">Run table (CSV)</a></p><p class="small">Build: <code>{build_command}</code>. Runtime comparisons state each measurement scope.</p></article></section>')
    if methods:parts.append('<script>'+ (ROOT/"scripts/test_analysis_view.js").read_text()+'</script>')
    return "\n".join(parts)


def build(data,test):
    f=Figures(data,test)
    for fn in (f.error_histories,f.convergence_cost,f.conservation,f.divergence,f.spectra,f.validity,f.runtime):fn()
    wanted={Path(entry[key]).name for entry in f.entries for key in ("file","vector")}
    for pattern in ("0[1-8]-*.png","0[1-8]-*.svg"):
        for path in f.root.glob(pattern):
            if path.name not in wanted:path.unlink()
    rows=table(data,test);warnings=[]
    if test in ALE_TESTS and not any(r.method=="mhdale_ale" for r in data.select(test)):warnings.append("MHD-ALE is applicable but its records have not been fetched; grey entries are missing data, not failed runs.")
    if not any(data.errors(test,m,f.error_n) for m in f.methods):
        warnings.append("This timestep-jump experiment has no independent spatial truth reference. Its comparison against the no-jump control belongs in the test-specific analysis; spatial error and accuracy/cost panels are unavailable."
                        if test=="kh_localized_dtjump" else
                        "No paired spatial reference data available locally for this test. Error histories and accuracy comparisons require a valid independent reference and matched state snapshots.")
    elif not any(len(data.errors(test,m,f.error_n))>2 for m in f.methods):warnings.append("Only a few paired snapshots are local. Error points are sparse; a continuous history requires more matched reference snapshots.")
    warnings.append(f"History diagnostics use N={f.n}; reference-error histories use N={f.error_n}. Seed 42 is plotted consistently; the run table includes all fetched seeds.")
    runs=data.select(test,seed=None)
    numerical=dict(test=test,config=data.sw.configs[test],runs=rows,diagnostics=[dict(method=r.method,N=r.n,seed=r.seed,series=r.diag,errors=data.errors(test,r.method,r.n,r.seed)) for r in runs])
    (f.root/"data.json").write_text(json.dumps(clean(numerical),allow_nan=False,separators=(",",":"))+"\n")
    with (f.root/"runs.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]) if rows else ["method"]);w.writeheader();w.writerows(rows)
    return dict(figures=f.entries,warnings=warnings,records=len(rows),methods=f.methods,history_grid=f.n,error_grid=f.error_n)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("tests",nargs="*")
    args=parser.parse_args();apply_movie_style(11);data=Data()
    from build_test_pages import TITLES as TESTS
    todo=args.tests or list(TESTS)
    target=SITE/"test_analysis.json";manifest=json.loads(target.read_text()) if target.is_file() else dict(tests={})
    for test in todo:
        if test not in TESTS:parser.error(f"Unknown test {test}")
        manifest["tests"][test]=build(data,test)
        manifest.update(built_utc=datetime.now(timezone.utc).isoformat(),synthetic=False)
        target.write_text(json.dumps(clean(manifest),indent=1,allow_nan=False)+"\n")
        print(f"{test}: {len(manifest['tests'][test]['figures'])} figures, {manifest['tests'][test]['records']} records",flush=True)
        data.load_array.cache_clear()
    OUT.mkdir(parents=True,exist_ok=True)
    missing_file=OUT/"missing_snapshots.txt"
    if args.tests and missing_file.is_file():
        # A subset rebuild must retain other tests' still-missing inputs.
        data.missing.update(p for p in missing_file.read_text().splitlines()
                            if p and not (data.sw.results/Path(p)).is_file())
    missing_file.write_text("\n".join(sorted(data.missing))+"\n")


if __name__=="__main__":main()
