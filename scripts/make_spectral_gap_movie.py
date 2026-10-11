"""Compare decay from a driven box with and without an imposed velocity gap."""
import argparse
import os
from pathlib import Path
import tomllib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/abel-mhd-cache")
import matplotlib
matplotlib.use("Agg")
from plot_style import (ARROW_ON_MAGNETIC, INK, MUTED, PANEL, REFERENCE, RULE, RUN, apply_movie_style, ffmpeg_path, field_map, frame)
apply_movie_style()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    meta = tomllib.loads((root / "metadata.toml").read_text())
    n, nf = meta["N"], meta["frames"]
    lo, hi = meta["gap_lower"], meta["gap_upper_exclusive"]
    data, spectra, density = {}, {}, {}
    for name in ("control", "gap"):
        data[name] = np.genfromtxt(root/f"{name}_diagnostics.csv", delimiter=",", names=True)
        s = np.genfromtxt(root/f"{name}_spectra.csv", delimiter=",", names=True)
        ns = len(s)//nf
        spectra[name] = {key: s[key].reshape(nf, ns) for key in ("velocity_power", "kinetic_power")}
        assert len(data[name]) == nf
        density[name] = np.stack([np.fromfile(root/f"{name}_density_{i:03d}.bin", dtype="<f4").reshape(n,n) for i in range(nf)])
        assert np.isfinite(density[name]).all() and density[name].min() > 0
    tau = data["control"]["t_cross"]
    assert np.allclose(tau, data["gap"]["t_cross"], atol=1e-6)
    kval = np.arange(ns) + .5
    # The notebook's look (scripts/plot_style.py). Both branches are runs of one code: the unchanged branch, which the
    # other is compared with, in the grey of reference runs; the branch with the gap in the code's green.
    colors = {"control": REFERENCE, "gap": RUN}
    styles = {"control": "-", "gap": (0, (5, 2))}
    labels = {"control": "Unchanged", "gap": "Velocity gap"}
    fig = plt.figure(figsize=(14,8.5), dpi=100)
    gs = fig.add_gridspec(2,3,left=.065,right=.95,bottom=.15,top=.78,wspace=.30,hspace=.54)
    axc, axg, axv = [fig.add_subplot(gs[0,j]) for j in range(3)]
    axe, axb, axk = [fig.add_subplot(gs[1,j]) for j in range(3)]
    fig.text(.065,.943,"Decay after removing velocity modes",fontsize=21)
    fig.text(.065,.902,f"Driven {n}³ hydro box  ·  fixed sound speed  ·  driving off in both branches  ·  gap {lo} ≤ k < {hi}",color=MUTED,fontsize=12)
    fig.text(.065,.858,f"At the split: identical density and pressure; gap changes kinetic energy by {-100*meta['fraction_kinetic_removed']:+.1f}%. No rescaling.",fontsize=10.5)
    stamp=fig.text(.065,.816,"",fontsize=12,color=INK)
    minimum = min(x.min() for x in density.values()); maximum = max(x.max() for x in density.values())
    norm=LogNorm(10**np.floor(np.log10(minimum)),10**np.ceil(np.log10(maximum)))
    ims=[]
    for ax,name in ((axc,"control"),(axg,"gap")):
        im=ax.imshow(density[name][0],origin="lower",extent=(0,1,0,1),cmap=field_map("density"),norm=norm,interpolation="nearest")
        ax.set_title(labels[name]+": density slice",loc="left");frame(ax)
        # The shared colorbar occupies the space below these two panels.
        ax.set(ylabel="y / L")
        ims.append(im)
    # Shared color scale for a fair branch comparison.
    cax=fig.add_axes([.35,.482,.20,.014])
    cb=fig.colorbar(ims[0],cax=cax,orientation="horizontal")
    cb.set_label("density / mean  (same logarithmic scale)",fontsize=9);cb.outline.set_linewidth(.6)
    lines={}
    for ax,key,title in ((axv,"velocity_power","Velocity spectrum"),(axk,"kinetic_power","Density-weighted kinetic spectrum")):
        allvalues=np.concatenate([spectra[name][key][:,1:n//2] for name in data])
        ymax=allvalues.max()*2
        ax.set(xscale="log",yscale="log",xlim=(1,n/2),ylim=(max(ymax*1e-7,1e-10),ymax))
        ax.axvspan(lo,hi,color=PANEL,lw=0)
        ax.set_title(title,loc="left")
        ax.set_xlabel("wavenumber k  (wavelength = L/k)",fontsize=10)
        ax.set_ylabel("power per unit-width shell",fontsize=10)
        ax.grid(True,color=RULE,lw=.6); ax.set_axisbelow(True)
        for name in data:
            line,=ax.plot([],[],lw=1.8,color=colors[name],ls=styles[name],label=labels[name])
            lines[name,key]=line
        ax.legend(fontsize=9,loc="lower left")
    axv.text(.98,.96,"Shaded band: removed velocity modes",transform=axv.transAxes,ha="right",va="top",fontsize=9,color=MUTED)
    axk.text(.98,.96,"Density weighting mixes modes",transform=axk.transAxes,ha="right",va="top",fontsize=9,color=MUTED)
    refke=data["control"]["kinetic"][0]
    refband=data["control"]["velocity_gap_power"][0]
    series={}
    for ax,key,title in ((axe,"kinetic","Total turbulent kinetic energy"),(axb,"velocity_gap_power","Velocity power inside the gap")):
        reference=refke if key=="kinetic" else refband
        values=[data[name][key]/reference for name in data]
        ax.set(xlim=(0,tau[-1]),ylim=(0,max(x.max() for x in values)*1.08))
        ax.set_title(title,loc="left")
        ax.set_xlabel("initial control box crossing times",fontsize=10)
        ax.set_ylabel("relative to unchanged branch at t = 0",fontsize=9)
        ax.grid(True,color=RULE,lw=.6); ax.set_axisbelow(True)
        for name in data:
            line,=ax.plot([],[],lw=2,color=colors[name],ls=styles[name],label=labels[name])
            point,=ax.plot([],[],"o",ms=4,color=colors[name])
            series[name,key]=(line,point,data[name][key]/reference)
        ax.legend(fontsize=9)
    fig.text(.065,.055,"The gap is imposed on velocity. The density-weighted spectrum need not have an empty band.",fontsize=10.5)
    fig.text(.065,.024,"Exploratory 64³ experiment: small scales are affected by numerical dissipation; gap recovery alone does not measure energy flux.",fontsize=9.5,color=MUTED)

    def update(i):
        for im,name in zip(ims,("control","gap")):
            im.set_data(density[name][i])
        for (name,key),line in lines.items():
            line.set_data(kval[1:],np.maximum(spectra[name][key][i,1:],1e-20))
        for line,point,values in series.values():
            line.set_data(tau[:i+1],values[:i+1]); point.set_data([tau[i]],[values[i]])
        stamp.set_text(f"Time {tau[i]:.2f} crossing times    ·    Mach {data['control']['mach'][i]:.2f} unchanged / {data['gap']['mach'][i]:.2f} with gap")
    sequence=[0]*24+list(range(nf))+[nf-1]*24
    anim=FuncAnimation(fig,update,frames=sequence,interval=1000/12,blit=False)
    matplotlib.rcParams["animation.ffmpeg_path"]=ffmpeg_path()
    anim.save(root/"spectral_gap_comparison.mp4",writer=FFMpegWriter(fps=12,codec="libx264",extra_args=["-crf","19","-pix_fmt","yuv420p","-movflags","+faststart"]))
    anim.save(root/"spectral_gap_comparison.gif",writer=PillowWriter(fps=12),dpi=70)
    update(0); fig.savefig(root/"initial_comparison.png",dpi=120)
    update(min(10,nf-1)); fig.savefig(root/"preview.png",dpi=120)
    plt.close(fig)

    # Space-time spectra retain all frames in a single inspectable figure.
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True,layout="constrained")
    fig.suptitle("How the spectra evolve after driving stops",fontsize=15)
    for row,key in enumerate(("velocity_power","kinetic_power")):
        high=max(spectra[name][key][:,1:n//2].max() for name in data)
        for col,name in enumerate(("control","gap")):
            ax=axes[row,col]
            im=ax.pcolormesh(kval[1:n//2],tau,np.maximum(spectra[name][key][:,1:n//2],1e-20),
                shading="nearest",cmap=field_map("density"),norm=LogNorm(high*1e-6,high))
            ax.set_xscale("log"); ax.set_xlim(1,n/2); frame(ax)
            ax.axvline(lo,color=ARROW_ON_MAGNETIC,ls=":",lw=1.2); ax.axvline(hi,color=ARROW_ON_MAGNETIC,ls=":",lw=1.2)
            ax.set_title(labels[name]+(" · velocity" if row==0 else " · density-weighted kinetic"))
            ax.set_xlabel("wavenumber k"); ax.set_ylabel("initial control crossing times")
        fig.colorbar(im,ax=list(axes[row,:]),label="power per shell",shrink=.85)
    fig.savefig(root/"spectra_vs_time.png",dpi=140); plt.close(fig)
    # The initial gap closes quickly; zoom in to make its recovery inspectable.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    early=tau<=.600001
    ratio=spectra["gap"]["velocity_power"]/np.maximum(spectra["control"]["velocity_power"],1e-30)
    im=axes[0].pcolormesh(kval[1:n//2],tau[early],ratio[early,1:n//2],shading="nearest",cmap=field_map("signed"),vmin=0,vmax=2)
    axes[0].set(xscale="log",xlim=(1,n/2),ylim=(0,.6),xlabel="wavenumber k",ylabel="initial control crossing times",title="Velocity power: gap / unchanged"); frame(axes[0])
    axes[0].axvline(lo,color="white",lw=1,ls=":"); axes[0].axvline(hi,color="white",lw=1,ls=":")
    fig.colorbar(im,ax=axes[0],label="power ratio (black: 1 = same)",extend="max")
    bandratio=data["gap"]["velocity_gap_power"]/data["control"]["velocity_gap_power"]
    axes[1].plot(tau[early],bandratio[early],color=colors["gap"],lw=2)
    axes[1].axhline(1,color=colors["control"],ls="-",lw=1.2,label="Same band power as control")
    for threshold in (.5,.9):
        i=np.flatnonzero(bandratio>=threshold)[0]
        axes[1].plot(tau[i],bandratio[i],"o",color=colors["gap"])
        axes[1].annotate(f"{threshold:.0%} exceeded at {tau[i]:.2f}",(tau[i],bandratio[i]),xytext=(8,-18),textcoords="offset points",fontsize=9)
    axes[1].set(xlim=(0,.6),ylim=(0,1.2),xlabel="initial control crossing times",ylabel="band power / control band power at the same time",title=f"Recovery within {lo} ≤ k < {hi}")
    axes[1].grid(True,color=RULE,lw=.6); axes[1].set_axisbelow(True); axes[1].legend(fontsize=9,loc="lower right")
    fig.savefig(root/"gap_recovery.png",dpi=140); plt.close(fig)
    prep=np.genfromtxt(root/"preparation.csv",delimiter=",",names=True)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout="constrained")
    for ax,key,label in zip(axes,("mach","kinetic","log_density_std"),("RMS Mach","Kinetic energy","Std. of log density")):
        ax.axvspan(prep["t_cross"][-1]-4,prep["t_cross"][-1],color=PANEL,lw=0)
        ax.plot(prep["t_cross"],prep[key],color=RUN,lw=1.5)
        ax.set_ylabel(label); ax.grid(True,color=RULE,lw=.6); ax.set_axisbelow(True)
    axes[0].set_title("Driven preparation · shaded interval used for stationarity check",loc="left")
    axes[-1].set_xlabel("nominal box crossing times (L / 10 cs)")
    fig.savefig(root/"preparation.png",dpi=130); plt.close(fig)
    print(f"Rendered {root/'spectral_gap_comparison.mp4'}")


if __name__=="__main__":
    main()
