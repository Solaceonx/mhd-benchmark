# CPU emulation of the GLM kernels of vendor/glmmhd_turb.jl for z-invariant (planar) problems.
# The physics functions are Tom Abel's, unedited (loaded from the vendored file by vendor_physics.jl). This file
# re-implements only the loop structure of integrator_tiled! / integrator_plm! / integrator_plm_ppm! on an N x N plane
# (all z-neighbours identical, so z slopes are zero and the two z fluxes of a cell cancel), with the same points at which
# values pass through shared storage:
#   primitives -> store -> slopes, Hancock -> interface states -> store -> Riemann -> flux -> store -> f32 update.
# `st` is the storage of primitives and interface states, `fst` that of fluxes: :f32 or :f16 (the production kernel
# has both :f16). The timestep is the sweep driver's (src/run_suite.jl: timestep): courant/(N * max over cells of the
# sum over three directions of |v_d| + c_fast,d), c_h = courant*dx/(3 dt)*scale, psi damped by exp(-c_h dt/(cp*L)).
# It reproduces the sweep's recorded errors of the Float32 kernels to three digits (see README.md).
module Emu
include("vendor_physics.jl")
module G end          # the vendored GLM physics, unedited
load_physics(G,"glmmhd_turb.jl",GLM_LINES)
module GH end         # the same with one line of hlld_x changed: a relative tolerance on the star-state denominator
load_physics(GH,"glmmhd_turb.jl",GLM_LINES;edit="small = abs(denK) < 1.0f-12"=>"small = abs(denK) < 1.0f-4*pts")
using Printf
const T9=NTuple{9,Float32}
@inline r16(q::T9)=ntuple(v->Float32(Float16(q[v])),9)
# Storage of a value that passes through the shared tile. :f32 keeps it; :f16 rounds it to half precision (the production
# kernel); :a16 is the proposed encoding: half precision of the difference from an anchor a (a Float32 value that every
# tile computes identically: the tile's first cell for cell and interface states, that cell's own flux for fluxes).
@inline keep(::Val{:f32},q::T9,a::T9)=q
@inline keep(::Val{:f16},q::T9,a::T9)=r16(q)
@inline keep(::Val{:a16},q::T9,a::T9)=ntuple(v->a[v]+Float32(Float16(q[v]-a[v])),9)
const Z9=ntuple(_->0f0,9)
@inline sc9(f::T9,c::Float32)=(f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9]*c)
@inline w(i,n)=i<1 ? i+n : (i>n ? i-n : i)

@inline rsolve(::Val{:llf},L,R,d,g,ch,sr,pf)=G.riemann(L,R,d,g,ch,sr,pf,0f0,0f0,false)      # step_tiled!(use_hlld=false)
@inline rsolve(::Val{:hll},L,R,d,g,ch,sr,pf)=G.riemann_hll(L,R,d,g,ch,sr,pf)
@inline rsolve(::Val{:hlld},L,R,d,g,ch,sr,pf)=G.riemann(L,R,d,g,ch,sr,pf,0f0,0f0,true)
"HLLD with the relative tolerance (module GH); pass as `rsolver` with riem = :hlld"
hlld_fixed(rv,L,R,d,g,ch,sr,pf)=GH.riemann(L,R,d,g,ch,sr,pf,0f0,0f0,true)

Base.@kwdef mutable struct Run
    n::Int
    gamma::Float32
    courant::Float32=0.25f0
    ch_scale::Float32=1f0
    cp_coef::Float32=0.18f0
    smallr::Float32=1f-6
    pfl::Float32=1f-6
    riem::Symbol=:llf
    st::Symbol=:f32
    fst::Symbol=:f32
    ppm::Bool=false
    tb::Int=4                       # tile size (anchors of :a16)
    fct::Bool=false                 # cell-centred constrained transport (Toth 2000, flux-CT) in place of GLM: see step!
    fscale::Bool=false              # fluxes pass through storage in units of 1/1024 of the grid's largest flux of each component
    psiflux::Symbol=:raw            # what passes through storage as the flux of psi: :raw = c_h^2 Bn* (original); :bn = Bn*, multiplied by c_h^2 after
    policy::Symbol=:stop            # :stop = the driver stops at a nonpositive pressure; :floor = it floors it in the CFL scan
    damp::Symbol=:box               # :box = exp(-c_h dt/(cp_coef L)) (original); :cell = exp(-alpha c_h dt/dx)
    alpha::Float32=0.1f0
    ch_from::Symbol=:dt             # :dt = c_h from the actual dt (original); :cfl = from the CFL dt even when dt is clipped
    rsolver::Function=rsolve        # replaced in tests of a changed Riemann solver
    track::Bool=false               # record the largest |flux| of each component in `peak`
    peak::Vector{Float32}=zeros(Float32,9)
end

function signal(u::Matrix{T9},r::Run)
    vmax=0f0;g=r.gamma
    for c in u
        rho=c[1];p=(g-1f0)*(c[5]-(c[2]^2+c[3]^2+c[4]^2)/(2f0*rho)-(c[6]^2+c[7]^2+c[8]^2)/2f0)
        r.policy===:floor && isfinite(p) && (p=max(p,r.pfl))
        (isfinite(rho) && rho>0 && isfinite(p) && p>0 && isfinite(c[6]+c[7]+c[8])) || return Inf32
        q=(rho,c[2]/rho,c[3]/rho,c[4]/rho,p,c[6],c[7],c[8],0f0)
        vmax=max(vmax,abs(q[2])+G.fast_speed(q,g,c[6])+abs(q[3])+G.fast_speed(q,g,c[7])+abs(q[4])+G.fast_speed(q,g,c[8]))
    end
    vmax
end

function step!(un::Matrix{T9},u::Matrix{T9},W,r::Run,dt::Float32,ch::Float32,::Val{RV},::Val{ST},::Val{FST}) where {RV,ST,FST}
    n=r.n;g=r.gamma;sr=r.smallr;pf=r.pfl;dtdx=dt*n
    P,XP,XM,YP,YM,FX,FY=W;rv=Val(RV);st=Val(ST);fst=Val(FST);tb=r.tb;nt=n÷tb
    glm_fac=r.damp===:box ? exp(-(ch/(r.cp_coef*1f0))*dt) : exp(-r.alpha*ch*dt*n)
    A=[G.cons2prim(u[(ti-1)*tb+1,(tj-1)*tb+1],g,sr,pf) for ti in 1:nt,tj in 1:nt]      # anchors: first cell of each tile
    AX=[r.rsolver(rv,a,a,1,g,ch,sr,pf) for a in A];AY=[r.rsolver(rv,a,a,2,g,ch,sr,pf) for a in A]
    @inline tile(i)=(i-1)÷tb+1
    c9=r.psiflux===:bn ? 1f0/(ch*ch) : 1f0
    Threads.@threads for j in 1:n
        for i in 1:n;@inbounds P[i,j]=keep(st,G.cons2prim(u[i,j],g,sr,pf),A[tile(i),tile(j)]);end
    end
    Threads.@threads for j in 1:n
        jm=w(j-1,n);jp=w(j+1,n)
        @inbounds for i in 1:n
            im=w(i-1,n);ip=w(i+1,n);m0=P[i,j];an=A[tile(i),tile(j)]
            mxl=P[im,j];mxr=P[ip,j];myl=P[i,jm];myr=P[i,jp]
            sx=G.prim_slope(mxl,m0,mxr);sy=G.prim_slope(myl,m0,myr)
            mh=G.cons2prim(G.hancock(m0,sx,sy,Z9,dtdx,g),g,sr,pf)
            (mh[1]<=sr || mh[5]<=pf) && (mh=m0)
            if r.ppm
                up=!(G.spj(mxl[5],m0[5],mxr[5]) || G.spj(myl[5],m0[5],myr[5]) || G.spj(m0[5],m0[5],m0[5]))
                pmx,ppx=G.ppm9(mxl,m0,mxr);pmy,ppy=G.ppm9(myl,m0,myr)
                XP[i,j]=keep(st,G.iflo(up ? G.mhd_face9(mh,ppx,m0) : G.padd(mh,sx,1f0),m0,sr,pf),an)
                XM[i,j]=keep(st,G.iflo(up ? G.mhd_face9(mh,pmx,m0) : G.padd(mh,sx,-1f0),m0,sr,pf),an)
                YP[i,j]=keep(st,G.iflo(up ? G.mhd_face9(mh,ppy,m0) : G.padd(mh,sy,1f0),m0,sr,pf),an)
                YM[i,j]=keep(st,G.iflo(up ? G.mhd_face9(mh,pmy,m0) : G.padd(mh,sy,-1f0),m0,sr,pf),an)
            else
                XP[i,j]=keep(st,G.iflo(G.padd(mh,sx,1f0),m0,sr,pf),an);XM[i,j]=keep(st,G.iflo(G.padd(mh,sx,-1f0),m0,sr,pf),an)
                YP[i,j]=keep(st,G.iflo(G.padd(mh,sy,1f0),m0,sr,pf),an);YM[i,j]=keep(st,G.iflo(G.padd(mh,sy,-1f0),m0,sr,pf),an)
            end
        end
    end
    Threads.@threads for j in 1:n
        jp=w(j+1,n)
        @inbounds for i in 1:n
            ip=w(i+1,n)
            FX[i,j]=r.rsolver(rv,XP[i,j],XM[ip,j],1,g,ch,sr,pf)
            FY[i,j]=r.rsolver(rv,YP[i,j],YM[i,jp],2,g,ch,sr,pf)
        end
    end
    if r.track
        for F in (FX,FY),f in F,v in 1:9;a=abs(f[v]);a>r.peak[v] && (r.peak[v]=a);end
    end
    # the fluxes pass through storage divided by S (one number per component for the whole grid): 1, or c_h^2 for the
    # flux of psi (psiflux = :bn), or 1/1024 of the largest flux of that component on the grid (fscale)
    S=r.fscale ? ntuple(v->(m=max(maximum(abs(f[v]) for f in FX),maximum(abs(f[v]) for f in FY));m>0 ? m/1024f0 : 1f0),9) :
                 ntuple(v->v==9 ? 1f0/c9 : 1f0,9)
    Si=ntuple(v->1f0/S[v],9)
    Threads.@threads for j in 1:n
        @inbounds for i in 1:n
            ax=AX[tile(i),tile(j)];ay=AY[tile(i),tile(j)]
            fx=keep(fst,ntuple(v->FX[i,j][v]*Si[v],9),ntuple(v->ax[v]*Si[v],9));FX[i,j]=ntuple(v->fx[v]*S[v],9)
            fy=keep(fst,ntuple(v->FY[i,j][v]*Si[v],9),ntuple(v->ay[v]*Si[v],9));FY[i,j]=ntuple(v->fy[v]*S[v],9)
        end
    end
    Threads.@threads for j in 1:n
        jm=w(j-1,n)
        @inbounds for i in 1:n
            im=w(i-1,n);u0=u[i,j];a=FX[im,j];b=FX[i,j];c=FY[i,jm];d=FY[i,j]
            r1=ntuple(v->u0[v]+dtdx*((a[v]-b[v])+(c[v]-d[v])),9)
            if r1[1]<sr
                emag=0.5f0*(r1[6]^2+r1[7]^2+r1[8]^2);r1=(sr,0f0,0f0,0f0,pf/(g-1f0)+emag,r1[6],r1[7],r1[8],r1[9])
            end
            un[i,j]=(r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7],r1[8],r1[9]*glm_fac)
        end
    end
    if r.fct
        # Flux-CT (Toth 2000, sec. 4.3) on the cell-centred field: one EMF per cell corner, the average of the four face
        # fluxes around it (Fx[By] = -Ez, Fy[Bx] = +Ez), and central differences of it. The central-difference
        # divergence of B does not change. psi stays zero and is not used.
        Ez=[0.25f0*(-FX[i,j][7]-FX[i,w(j+1,n)][7]+FY[i,j][6]+FY[w(i+1,n),j][6]) for i in 1:n,j in 1:n]
        Threads.@threads for j in 1:n
            jm=w(j-1,n)
            @inbounds for i in 1:n
                im=w(i-1,n);u0=u[i,j];c=un[i,j]
                bx=u0[6]-0.5f0*dtdx*((Ez[i,j]+Ez[im,j])-(Ez[i,jm]+Ez[im,jm]))
                by=u0[7]+0.5f0*dtdx*((Ez[i,j]+Ez[i,jm])-(Ez[im,j]+Ez[im,jm]))
                un[i,j]=(c[1],c[2],c[3],c[4],c[5],bx,by,c[8],0f0)
            end
        end
    end
    nothing
end

# Advance u to t_end as the sweep driver does (src/run_suite.jl): `frames` outputs equally spaced from 0, a Float64 clock
# stepped in Float32 dt, steps shortened to land on outputs. Returns (u, t, steps, why it stopped, smallest dt / CFL dt).
function evolve(u0::Matrix{T9},r::Run,t_end;frames=2,each=nothing,maxsteps=10^7)
    u=copy(u0);un=similar(u);n=r.n;W=ntuple(_->Matrix{T9}(undef,n,n),7)
    t=0.0;steps=0;rv=Val(r.riem);st=Val(r.st);fst=Val(r.fst);shortest=1.0
    for target in range(0,Float64(t_end);length=frames)
        while t<target-1e-10
            vmax=signal(u,r)
            isfinite(vmax) && vmax>0 || return u,t,steps,"nonphysical state in the CFL scan",shortest
            dtc=r.courant/(n*vmax);dt=Float32(min(dtc,target-t))
            ch=r.fct ? 1f-20 : r.courant/(n*(r.ch_from===:dt ? dt : dtc))/3f0*r.ch_scale
            shortest=min(shortest,dt/dtc)
            step!(un,u,W,r,dt,ch,rv,st,fst);u,un=un,u;t+=Float64(dt);steps+=1
            each===nothing || each(u,t,steps)
            steps>=maxsteps && return u,t,steps,"step limit",shortest
        end
    end
    u,t,steps,"",shortest
end

# ---- initial data: cell-centred fields from a vector potential A_z(x,y) plus a uniform field, as src/problems.jl ----
function initial(n,gamma,prim;Az=(x,y)->0.0,B0=(0.0,0.0,0.0),sub=1,Bz=nothing)
    c(i)=(i-.5)/n;f(i)=(i-1)/n;wr(i)=mod1(i,n)
    A=[Az(f(i),f(j)) for i in 1:n,j in 1:n]
    bxf=[n*(A[i,wr(j+1)]-A[i,j])+B0[1] for i in 1:n,j in 1:n]         # lower x face
    byf=[-n*(A[wr(i+1),j]-A[i,j])+B0[2] for i in 1:n,j in 1:n]        # lower y face
    off=((1:sub).-.5)./sub.-.5;u=Matrix{T9}(undef,n,n)
    for j in 1:n,i in 1:n
        rho=0.0;m=zeros(3);th=0.0;ke=0.0
        for oy in off,ox in off
            q=prim((i-.5+ox)/n,(j-.5+oy)/n);rho+=q[1];m.+=q[1].*(q[2],q[3],q[4]);th+=q[5]/(gamma-1);ke+=q[1]*(q[2]^2+q[3]^2+q[4]^2)/2
        end
        s=sub^2;rho/=s;m./=s;th/=s;ke/=s
        bx=Float32((Float32(bxf[i,j])+Float32(bxf[wr(i+1),j]))/2f0);by=Float32((Float32(byf[i,j])+Float32(byf[i,wr(j+1)]))/2f0);bz=Bz===nothing ? Float32(B0[3]) : Float32(Bz((i-.5)/n,(j-.5)/n))
        u[i,j]=(Float32(rho),Float32(m[1]),Float32(m[2]),Float32(m[3]),Float32(th+ke+(Float64(bx)^2+Float64(by)^2+Float64(bz)^2)/2),bx,by,bz,0f0)
    end
    u
end
comp(u,v)=[Float64(c[v]) for c in u]
"rms of the divergence of B at cell corners (the one cell-centred CT keeps) and at cell centres (central differences), times dx"
function divs(u)
    n=size(u,1);a=0.0;b=0.0
    for j in 1:n,i in 1:n
        ip=mod1(i+1,n);jp=mod1(j+1,n);im=mod1(i-1,n);jm=mod1(j-1,n)
        dc=((Float64(u[ip,j][6])+u[ip,jp][6])-(Float64(u[i,j][6])+u[i,jp][6])+(Float64(u[i,jp][7])+u[ip,jp][7])-(Float64(u[i,j][7])+u[ip,j][7]))/2
        dd=(Float64(u[ip,j][6])-u[im,j][6]+Float64(u[i,jp][7])-u[i,jm][7])/2
        a+=dc^2;b+=dd^2
    end
    sqrt(a/n^2),sqrt(b/n^2)
end
end
