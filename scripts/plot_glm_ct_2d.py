"""Render a matched CPU Orszag–Tang GLM/CT comparison."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/abel-mhd-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/abel-mhd-cache")
import argparse
from pathlib import Path
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
from plot_style import apply_plot_style, field_map
apply_plot_style()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import imageio_ffmpeg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    root=parser.parse_args().directory.resolve()
    meta=tomllib.loads((root/"metadata.toml").read_text())
    n,nf=meta["N"],meta["frames"]
    u={};diag={}
    for name in ("glm","ct"):
        # Julia order is x,y,component. imshow needs y,x.
        u[name]=np.stack([np.fromfile(root/f"{name}_{i:03d}.bin",dtype="<f4").reshape((n,n,9),order="F").transpose(1,0,2) for i in range(nf)])
        diag[name]=np.genfromtxt(root/f"{name}_diagnostics.csv",delimiter=",",names=True)
        assert np.isfinite(u[name]).all() and (u[name][...,0]>0).all()
    assert np.array_equal(u['glm'][0],u['ct'][0])
    t=diag['glm']['time'];assert np.array_equal(t,diag['ct']['time'])
    rho={key:value[...,0] for key,value in u.items()}
    bmag={key:np.sqrt(value[...,5]**2+value[...,6]**2) for key,value in u.items()}
    delta=rho['glm']-rho['ct']
    rho_range=(min(x.min() for x in rho.values()),max(x.max() for x in rho.values()))
    b_range=(0,max(x.max() for x in bmag.values()))
    difference_max=max(abs(delta.min()),abs(delta.max()))
    colors={'glm':'#68ddd4','ct':'#f5ad7b'}
    plt.rcParams.update({'text.color':'#dce6f0','axes.labelcolor':'#bac9da',
        'xtick.color':'#a6b7cb','ytick.color':'#a6b7cb',
        'axes.edgecolor':'#465366','axes.facecolor':'#111c2a','figure.facecolor':'#09121e',
        'savefig.facecolor':'#09121e','legend.facecolor':'#111c2a','legend.edgecolor':'#465366'})
    fig,axes=plt.subplots(2,3,figsize=(13.5,9),dpi=100)
    fig.subplots_adjust(left=.065,right=.95,bottom=.13,top=.80,wspace=.38,hspace=.48)
    fig.text(.065,.943,'ORSZAG–TANG: GLM AND CT ON THE CPU',fontsize=21,weight='bold')
    fig.text(.065,.902,f'{n} × {n} cells  ·  periodic 2D MHD  ·  γ = 5/3  ·  matched PLM / HLL / RK2',fontsize=12,color='#a6b7cb')
    stamp=fig.text(.065,.848,'',fontsize=12,color='#e6d191')
    images=[]
    for row,fields,limits,label,cmap in ((0,rho,rho_range,'Density',field_map('density')),(1,bmag,b_range,'Magnetic-field magnitude',field_map('magnetic'))):
        for col,name in enumerate(('glm','ct')):
            ax=axes[row,col]
            im=ax.imshow(fields[name][0],origin='lower',extent=(0,1,0,1),vmin=limits[0],vmax=limits[1],cmap=cmap,interpolation='nearest')
            ax.set(title=f'{name.upper()} · {label}',xlabel='x / L',ylabel='y / L')
            fig.colorbar(im,ax=ax,fraction=.047,pad=.025)
            images.append((im,fields[name]))
    ax=axes[0,2]
    im=ax.imshow(delta[0],origin='lower',extent=(0,1,0,1),vmin=-difference_max,vmax=difference_max,cmap=field_map('signed'),interpolation='nearest')
    ax.set(title='Density difference · GLM − CT',xlabel='x / L',ylabel='y / L')
    fig.colorbar(im,ax=ax,fraction=.047,pad=.025);images.append((im,delta))
    ax=axes[1,2];lines=[]
    for name in ('glm','ct'):
        line,=ax.plot([],[],color=colors[name],lw=2,label=name.upper())
        lines.append((line,diag[name]['normalized_div_center_rms']))
    ymax=max(x[1].max() for x in lines)
    ax.set(xlim=(0,t[-1]),ylim=(0,ymax*1.1),xlabel='time',ylabel='Δx × rms(∇·B) / rms(B)',title='Same cell-centered divergence operator')
    ax.grid(alpha=.15);ax.legend()
    fig.text(.065,.068,'CT preserves face-based divergence. Cell-centered interpolation uses a different discrete operator.',fontsize=10)
    fig.text(.065,.035,'All image scales are fixed in time and shared between methods. Differences do not identify which solution is more accurate.',fontsize=9,color='#a6b7cb')

    def update(i):
        for image,field in images:image.set_data(field[i])
        for line,values in lines:line.set_data(t[:i+1],values[:i+1])
        l1=np.abs(delta[i]).mean()/rho['ct'][i].mean()
        stamp.set_text(f"Time {t[i]:.3f}    ·    density difference / mean density {100*l1:.2f}%    ·    CT face max |∇·B| {diag['ct']['div_face_max'][i]:.2e}")
    update(nf-1);fig.savefig(root/'comparison.png',dpi=130)
    anim=FuncAnimation(fig,update,frames=[0]*12+list(range(nf))+[nf-1]*24,interval=1000/12,blit=False)
    matplotlib.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    anim.save(root/'glm_ct_comparison.mp4',writer=FFMpegWriter(fps=12,codec='libx264',extra_args=['-crf','19','-pix_fmt','yuv420p','-movflags','+faststart']))
    anim.save(root/'glm_ct_comparison.gif',writer=PillowWriter(fps=12),dpi=70)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for name in ('glm','ct'):
        d=diag[name]
        axes[0].plot(t,d['div_center_rms'],label=f'{name.upper()} cell-centered',color=colors[name])
        axes[1].plot(t,(d['mass']/d['mass'][0]-1),label=name.upper(),color=colors[name])
        axes[2].plot(t,(d['energy']/d['energy'][0]-1),label=name.upper(),color=colors[name])
    axes[0].plot(t,diag['ct']['div_face_max'],ls='--',color='#e6d191',label='CT face max')
    for ax,title in zip(axes,('Divergence: operators labeled separately','Relative mass drift','Relative total-energy drift')):
        ax.set_title(title,fontsize=10);ax.set_xlabel('time');ax.grid(alpha=.15);ax.legend(fontsize=8)
    axes[0].set_yscale('symlog',linthresh=1e-6)
    fig.savefig(root/'diagnostics.png',dpi=140);plt.close(fig)
    print('Rendered',root/'glm_ct_comparison.mp4')


if __name__=='__main__':main()
