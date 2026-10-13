"""The shared diagnostic sections for driven turbulence, grouped by target Mach.

    .venv-movie/bin/python scripts/driven_test_diagnostics.py

These runs have no spatial truth reference. Accuracy norms are explicitly N/A;
statistical comparisons on the existing turbulence page remain the main science.
"""
import csv
from datetime import datetime,timezone
import json
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
import matplotlib.pyplot as plt
import numpy as np

import turb_figures as tf
from test_diagnostics import Figures, OUT, SITE, clean, blank, logarithmic, TITLES
from plot_style import apply_movie_style,FAMILY


def style(kind):
    old=tf.KINDS[kind]
    dash={"gpu-hydro":(0,(5,3)),"gpu-glm":(0,(1,1)),"gpu-hydro_ramsesproj":(0,(5,2,1,2)),
          "ramsesfix-matched":(0,(5,3)),"ramsesfix-best":"-","ramses-matched":(0,(2,2)),"ramses-best":(0,(7,2,1,2))}[kind]
    labels={"gpu-hydro":"Abel Euler HLL","gpu-glm":"Abel GLM LLF (B = 0)",
            "gpu-hydro_ramsesproj":"Abel Euler, released projection",
            "ramsesfix-matched":"RAMSES HLL, corrected","ramsesfix-best":"RAMSES HLLC, corrected",
            "ramses-matched":"RAMSES HLL, released","ramses-best":"RAMSES HLLC, released"}
    return dict(color=FAMILY["glm" if kind.startswith("gpu") else "ramses"],linestyle=dash,label=labels[kind])


def draw(ax,r,x,y):
    x,y=np.asarray(x),np.asarray(y);good=np.isfinite(x)&np.isfinite(y)
    kw=style(r.kind)
    if r.seed!=1:kw["label"]="_nolegend_"
    ax.plot(x[good],y[good],**kw,lw=1.6,alpha=.9 if r.seed==1 else .45)
    ax.grid(False)


class Driven(Figures):
    def __init__(self,mach,runs):
        self.test=f"driven_M{mach:g}";self.root=OUT/self.test;self.root.mkdir(parents=True,exist_ok=True)
        self.entries=[];self.runs=runs;self.mach=mach
        ns=sorted({r.n for r in runs});self.n=128 if 128 in ns else ns[0]
        self.shown=[r for r in runs if r.n==self.n]

    def absent_accuracy(self):
        for category in ("01-errors","02-convergence","03-cost"):
            fig,ax=plt.subplots(figsize=(12,4.8))
            blank(ax,"No analytic or converged spatial reference exists for these independently driven random realizations. Pointwise L1/L2/Linf error, convergence in those errors, and error-versus-cost are unavailable. Use the turbulence page's spectra, Mach statistics, power, PDFs and seed spread instead.")
            self.save(fig,category,"none",f"Mach {self.mach:g} · not applicable without a spatial reference",
                "Different random-number generators and force realizations make pointwise subtraction unsuitable as a correctness ranking. The released RAMSES projection also differs from the corrected solenoidal force; those results are labelled separately.")

    def conservation(self):
        fig,axs=plt.subplots(1,3,figsize=(13,5.8))
        for r in self.shown:
            d=r.series;t=d["time_over_T"]
            draw(axs[0],r,t,(d["mass"]-d["mass"][0])/d["mass"][0])
            draw(axs[1],r,t,d["energy"]-d["energy"][0])
            if "momentum" in d:draw(axs[2],r,t,d["momentum"])
        for ax,label in zip(axs,["Signed mass drift / initial mass","Total energy gained [code units]","Total momentum magnitude [code units]"]):
            if not ax.lines:blank(ax,"This diagnostic was not recorded.")
            else:ax.set_xlabel("Physical time / driving time T");ax.set_ylabel(label);ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,3))
        self.save(fig,"04-conservation","all",f"Mach {self.mach:g} · {self.n}³ · all available seeds",
            "The force injects energy and momentum, so their changes are not conservation errors. Exact stepwise injected-work records are unavailable, preventing an energy-budget residual. Mass should remain conserved. Fainter curves show a second seed.")

    def divergence(self):
        fig,ax=plt.subplots(figsize=(12,4.8))
        blank(ax,"Hydrodynamic experiment: the magnetic field is initialized to zero. A relative divergence measure divides by zero and is undefined. Spatial divergence was not recorded by this driver; it must not be inferred from total magnetic energy.")
        self.save(fig,"05-divergence","none",f"Mach {self.mach:g} · zero-field experiment",
            "The separate MHD decay tests supply magnetic-divergence comparisons. No magnetic-divergence values are manufactured here.")

    def spectra(self):
        fig,axs=plt.subplots(1,2,figsize=(13,5.8))
        for r in self.shown:
            d=r.spectra();good=np.isfinite(d["v"])&(d["v"]>0)
            draw(axs[0],r,d["k"][good],d["v"][good])
        axs[0].set_xscale("log");axs[0].set_yscale("log");axs[0].set_xlabel("Wavenumber k [2π/L]");axs[0].set_ylabel("Mean velocity power")
        blank(axs[1],"No converged truth spectrum for a code/reference accuracy ratio. Pairwise statistical spectrum ratios are already shown elsewhere on this page. Magnetic power is zero for this hydrodynamic experiment.")
        self.save(fig,"06-spectra","all",f"Mach {self.mach:g} · {self.n}³ · time average from 2T to 10T",
            "Full-volume spectra averaged over the recorded steady-state window. Fainter curves show seed spread; it is not a formal confidence interval. Corrected and released projection runs are labelled distinctly.")

    def validity(self):
        fig,axs=plt.subplots(1,3,figsize=(13,5.8))
        for i,r in enumerate(self.shown):
            d=r.series;t=d["time_over_T"]
            axs[0].barh(i,min(1,t[-1]/r.meta.get("t_end_T",10)),color=style(r.kind)["color"])
            if "density_floor_cells" in d:draw(axs[1],r,t,100*d["density_floor_cells"]/r.n**3)
            if "negative_density_cells" in d:draw(axs[2],r,t,100*d["negative_density_cells"]/r.n**3)
        axs[0].set_yticks(range(len(self.shown)),labels=[style(r.kind)["label"]+f" s{r.seed}" for r in self.shown],fontsize=8);axs[0].invert_yaxis();axs[0].set_xlabel("Fraction of requested physical duration")
        for ax,label in zip(axs[1:],["Density-floor cells [%]","Nonpositive density cells [%]"]):
            if not ax.lines:blank(ax,"Cell counts were not recorded for these runs. Missing counts are not zero.")
            else:ax.set_xlabel("Physical time / driving time T");ax.set_ylabel(label)
        self.save(fig,"07-validity","all",f"Mach {self.mach:g} · {self.n}³",
            "All listed production runs were recorded as complete. Completion does not imply positive density; recorded nonpositive cells are shown where available. Trial kernels rejected before production are excluded. Floor counts are sampled observations.")

    def runtime(self):
        fig,axs=plt.subplots(1,2,figsize=(13,6.3));rows=[]
        for r in self.shown:
            perf=r.meta.get("perf",{})
            sec=r.meta.get("step_seconds",perf.get("code_seconds"))
            wall=r.meta.get("wall_seconds",perf.get("wall_seconds"))
            if not sec:continue
            rows.append((r,float(sec),wall))
        rows.sort(key=lambda row:row[1])
        for i,(r,sec,wall) in enumerate(rows):
            axs[0].barh(i,sec,color=style(r.kind)["color"])
            if wall is not None and wall>=sec:axs[0].barh(i,wall-sec,left=sec,color=style(r.kind)["color"],alpha=.3,hatch="///")
            # Only endpoint timings exist. Do not infer an intermediate trajectory.
            draw(axs[1],r,[sec],[r.series["time"][-1]])
            axs[1].scatter([sec],[r.series["time"][-1]],color=style(r.kind)["color"],s=30)
        if rows:
            axs[0].set_yticks(range(len(rows)),labels=[style(r.kind)["label"]+f" s{r.seed}" for r,*_ in rows],fontsize=8);axs[0].invert_yaxis();axs[0].set_xlabel("Execution time [s]")
            axs[0].set_xscale("log");axs[0].set_xlim(min(sec for _,sec,_ in rows)*.6,max(max(sec,float(wall or 0)) for _,sec,wall in rows)*1.4)
            axs[1].set_xscale("log")
            axs[1].set_xlabel("Recorded execution time [s]");axs[1].set_ylabel("Requested physical endpoint t")
        else:
            for ax in axs:blank(ax,"Timing breakdown unavailable in the fetched metadata.")
        self.save(fig,"08-runtime","all",f"Mach {self.mach:g} · {self.n}³",
            "GPU step time includes the timestep scan and kernel, excluding forcing update and measurement. RAMSES code-loop time has a different scope. Hatched: measured additional wall time. GPU cards and CPU ranks differ; exact scopes/devices are in the JSON records.")


def main():
    apply_movie_style(11);runs=tf.load();all_entries=[];records=[]
    for mach in sorted({r.mach for r in runs}):
        selected=[r for r in runs if r.mach==mach]
        f=Driven(mach,selected)
        f.absent_accuracy();f.conservation();f.divergence();f.spectra();f.validity();f.runtime()
        rows=[dict(run=r.name,kind=r.kind,N=r.n,seed=r.seed,metadata=r.meta,series=r.series) for r in selected]
        (f.root/"data.json").write_text(json.dumps(clean(rows),allow_nan=False,separators=(",",":"))+"\n")
        all_entries+=f.entries;records+=rows
        print(f"driven Mach {mach}: {len(f.entries)} figures, {len(rows)} records",flush=True)
    root=OUT/"sherlock_turb";root.mkdir(parents=True,exist_ok=True)
    (root/"data.json").write_text(json.dumps(clean(records),allow_nan=False,separators=(",",":"))+"\n")
    with (root/"runs.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=["run","kind","N","seed","device","precision","timing_scope"]);w.writeheader()
        for r in records:w.writerow(dict(run=r["run"],kind=r["kind"],N=r["N"],seed=r["seed"],device=r["metadata"].get("device"),precision=r["metadata"].get("precision"),timing_scope=r["metadata"].get("timing_scope","RAMSES perf fields; see JSON")))
    target=SITE/"test_analysis.json";m=json.loads(target.read_text()) if target.is_file() else dict(tests={})
    m["tests"]["sherlock_turb"]=dict(figures=all_entries,records=len(records),warnings=["No spatial truth exists for independently forced realizations. Shared accuracy/error-convergence sections are explicitly unavailable; use the statistical comparisons on this page.","Plots are separated by target Mach number. Corrected and released RAMSES driving are different forcings and are labelled separately."])
    m.update(built_utc=datetime.now(timezone.utc).isoformat(),synthetic=False);target.write_text(json.dumps(clean(m),indent=1,allow_nan=False)+"\n")


if __name__=="__main__":main()
