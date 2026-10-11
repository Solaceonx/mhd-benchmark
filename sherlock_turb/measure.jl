# Measurements of a driven-turbulence state: one implementation for every code. turb_drive.jl calls it on the device
# state of Tom Abel's kernels, turb_measure.jl on the frames RAMSES wrote. It uses the reductions of the sweep's
# src/lean.jl and the spectra of its src/spectra.jl, which must be included first (as tools/postprocess_external.jl does).
module TurbMeasure
using KernelAbstractions, Printf
import ..Lean, ..Spectra
const KA=KernelAbstractions
const F=Float64

"The state as nine arrays (rho, mx, my, mz, E, bx, by, bz, psi) of an (n, n, n, 9) array."
fields(u)=ntuple(c->view(u,:,:,:,c),9)

# Both codes floor the density at 1e-6 of the mean where they form velocities (smallr in RAMSES and in Tom's
# kernels). Tom's kernels also reset a cell that falls below it; RAMSES does not touch the conserved density, and at
# Mach 30 and 50 leaves some cells below the floor, a few of them below zero. The measurements treat such a cell as
# the codes do, at the floor (velocity, kinetic energy, ln rho, spectra), and count them. Mass and the density
# extremes are of the density as it is.
const FLOOR=1.0e-6
@inline fl(r)=max(F(r),FLOOR)
root(x)=x>=0 ? sqrt(x) : NaN          # a state too broken for a number gives NaN in its line, not an error that loses the run
@inline vel(m,r,a,b,c)=@inbounds F(m[a,b,c])/fl(r[a,b,c])
# |curl v|^2 and (div v)^2 with periodic centred differences, h = n/2 (grid spacing 1/n), as Lean's divergences.
@kernel function velgrad_kernel!(w2,d2,rho,mx,my,mz,h)
    i,j,k=@index(Global,NTuple)
    nx,ny,nz=size(w2)
    ip=ifelse(i==nx,1,i+1);im=ifelse(i==1,nx,i-1)
    jp=ifelse(j==ny,1,j+1);jm=ifelse(j==1,ny,j-1)
    kp=ifelse(k==nz,1,k+1);km=ifelse(k==1,nz,k-1)
    @inbounds begin
        dxx=h*(vel(mx,rho,ip,j,k)-vel(mx,rho,im,j,k));dxy=h*(vel(mx,rho,i,jp,k)-vel(mx,rho,i,jm,k));dxz=h*(vel(mx,rho,i,j,kp)-vel(mx,rho,i,j,km))
        dyx=h*(vel(my,rho,ip,j,k)-vel(my,rho,im,j,k));dyy=h*(vel(my,rho,i,jp,k)-vel(my,rho,i,jm,k));dyz=h*(vel(my,rho,i,j,kp)-vel(my,rho,i,j,km))
        dzx=h*(vel(mz,rho,ip,j,k)-vel(mz,rho,im,j,k));dzy=h*(vel(mz,rho,i,jp,k)-vel(mz,rho,i,jm,k));dzz=h*(vel(mz,rho,i,j,kp)-vel(mz,rho,i,j,km))
        w2[i,j,k]=(dzy-dyz)^2+(dxz-dzx)^2+(dyx-dxy)^2
        d2[i,j,k]=(dxx+dyy+dzz)^2
    end
end

"""
    stats(q; gamma)

The numbers of one time of a turbulence run, from `q = fields(u)` on any device. Box length 1, so a volume mean is a
mean over cells. Mach numbers use the sound speed of the gas as it is now, `cs = sqrt(gamma <p>/<rho>)`.
"""
function stats(q;gamma)
    rho,mx,my,mz,E,bx,by,bz=q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8]
    n=size(rho,1);N=length(rho);g=F(gamma)
    mean(f,args...)=Lean.fsum(f,args...)/N
    ke(r,a,b,c)=(F(a)^2+F(b)^2+F(c)^2)/(2fl(r))
    me(a,b,c)=(F(a)^2+F(b)^2+F(c)^2)/2
    pr(r,a,b,c,e,x,y,z)=(g-1)*(F(e)-ke(r,a,b,c)-me(x,y,z))
    v2(r,a,b,c)=(F(a)^2+F(b)^2+F(c)^2)/fl(r)^2
    cs2(r,a,b,c,e,x,y,z)=g*pr(r,a,b,c,e,x,y,z)/fl(r)    # explicit arguments: a named f(x...) does not compile for the GPU
    lnr(x)=log(fl(x))
    P=(rho,mx,my,mz,E,bx,by,bz)
    mass=mean(F,rho);kinetic=mean(ke,rho,mx,my,mz);magnetic=mean(me,bx,by,bz);pmean=mean(pr,P...)
    v2vol=mean(v2,rho,mx,my,mz);c2=g*pmean/mass
    s1=mean(x->lnr(x)-log(mass),rho);s2=mean(x->(lnr(x)-log(mass))^2,rho)
    w2=similar(rho,F,size(rho));d2=similar(rho,F,size(rho))
    velgrad_kernel!(KA.get_backend(w2))(w2,d2,rho,mx,my,mz,F(n)/2;ndrange=size(w2))
    res=(mass=mass,kinetic=kinetic,thermal=pmean/(g-1),magnetic=magnetic,energy=mean(F,E),
        cs=root(c2),v_rms=root(v2vol),mach_vol=root(v2vol/c2),mach_mass=root(2kinetic/mass/c2),
        v_max=root(Lean.fmax(v2,rho,mx,my,mz)),rho_min=Lean.fmin(F,rho),rho_max=Lean.fmax(F,rho),
        sigma_rho=sqrt(max(mean(x->F(x)^2,rho)/mass^2-1,0.0)),s_mean=s1,sigma_s=sqrt(max(s2-s1^2,0.0)),
        cs2_min=Lean.fmin(cs2,P...),cs2_max=Lean.fmax(cs2,P...),
        enstrophy=mean(identity,w2),dilatation=mean(identity,d2),
        enstrophy_rho=mean((r,w)->fl(r)*w,rho,w2),dilatation_rho=mean((r,d)->fl(r)*d,rho,d2),
        momentum=sqrt(mean(F,mx)^2+mean(F,my)^2+mean(F,mz)^2),
        density_floor_cells=Lean.fcount(x->F(x)<=1.00001e-6,rho),negative_density_cells=Lean.fcount(x->F(x)<=0,rho),
        finite=all(x->Lean.fall(isfinite,x),P))
    finalize(w2);finalize(d2)        # free the Float64 temporaries now (GPU memory), not at the next GC
    res
end

# Density distribution: s = ln(rho/<rho>) in bins of S_STEP from S_LO; volume and mass fraction in each bin.
const S_LO=-16.0;const S_STEP=0.1;const S_BINS=280
function density_pdf(rho::Array,mean_rho)
    vol=zeros(Int,S_BINS);mass=zeros(F,S_BINS)
    for x in rho
        s=log(max(F(x),floatmin(F))/mean_rho);b=clamp(floor(Int,(s-S_LO)/S_STEP)+1,1,S_BINS)
        vol[b]+=1;mass[b]+=x
    end
    vol./length(rho),mass./sum(mass)
end

"The files of one run directory: series.csv (every measured time), spectra.csv and pdf.csv (frames)."
mutable struct Writer
    dir::String
    series::IOStream
    spectra::IOStream
    pdf::IOStream
    header::Bool
    spectra_header::Bool
end
function Writer(dir)
    mkpath(dir);w=Writer(dir,open(joinpath(dir,"series.csv"),"w"),open(joinpath(dir,"spectra.csv"),"w"),open(joinpath(dir,"pdf.csv"),"w"),false,false)
    println(w.pdf,"frame,time,s_low,volume_fraction,mass_fraction");w
end
Base.close(w::Writer)=foreach(close,(w.series,w.spectra,w.pdf))

"One line of series.csv: `lead` (index, time, ...) followed by the stats."
function series!(w::Writer,lead::NamedTuple,row::NamedTuple)
    all=merge(lead,row)
    w.header || (println(w.series,join(string.(keys(all)),','));w.header=true)
    println(w.series,join(values(all),','));flush(w.series)
end

"""
    frame!(w, frame, time, u, mean_rho; slice, state)

Spectra and density distribution of the (n, n, n, 9) state `u` on its device; optionally its midplane as
slice_###.bin and the whole state as `state` (both Float32, as the sweep writes them). Returns what was written.
"""
function frame!(w::Writer,frame,time,u,mean_rho;slice=false,state=nothing)
    q=fields(u);n=size(u,1);out=Dict{String,Any}("frame"=>frame,"time"=>time)
    rho=max.(q[1],Float32(FLOOR))         # at the floor where the code left less (see FLOOR): the spectra take roots of it
    sp=Spectra.spectra(rho,q[2:4],q[6:8]);cols=[k for k in keys(sp) if k!==:k]
    w.spectra_header || (println(w.spectra,join(["frame","time","k",string.(cols)...],','));w.spectra_header=true)
    for i in eachindex(sp.k);println(w.spectra,join((frame,time,sp.k[i],(getfield(sp,c)[i] for c in cols)...),','));end
    vol,mass=density_pdf(Array(rho),mean_rho);finalize(rho)
    for b in 1:S_BINS;vol[b]>0 && println(w.pdf,join((frame,time,@sprintf("%.1f",S_LO+(b-1)*S_STEP),vol[b],mass[b]),','));end
    flush(w.spectra);flush(w.pdf)
    if slice
        out["slice"]=@sprintf("slice_%03d.bin",frame);write(joinpath(w.dir,out["slice"]),Array(u[:,:,n÷2+1,:]))
    end
    if state!==nothing
        out["state"]=state;write(joinpath(w.dir,state),Array(u))
    end
    out
end
end
