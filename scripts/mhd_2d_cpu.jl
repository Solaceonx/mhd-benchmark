"""Shared 2D CPU MHD driver: existing GLM flux physics, or face-staggered CT.

Planar MHD only: vz=Bz=0. Both paths use the same MonCen PLM, HLL flux,
SSP-RK2 integrator, ideal-gas EOS and periodic grid. CT uses the arithmetic
Balsara–Spicer edge EMF from GLMMHDTurb/ct_mhd.jl. Unlike the old cubic drivers,
storage and work scale as N^2. The original CUDA/SIMD implementations are untouched.
"""
module MHD2DCPU
include(joinpath(@__DIR__,"..","GLMMHDTurb","glmmhd_turb.jl"))
const G=GLMMHDTurb
using Base.Threads

Base.@kwdef struct Params
    n::Int=64
    gamma::Float32=5f0/3f0
    cfl::Float32=0.35f0
    floor::Float32=1f-7
    cleaning_damping::Float32=0.18f0 # decay time = coefficient * L/ch, as existing GLM
end
struct State
    u::Array{Float32,3} # rho,mx,my,mz,E,Bx,By,Bz,psi (CT Bx/By are derived)
    bx::Matrix{Float32} # CT lower-x face
    by::Matrix{Float32} # CT lower-y face
end
State(n)=State(zeros(Float32,n,n,9),zeros(Float32,n,n),zeros(Float32,n,n))
struct Workspace
    stage::State
    out::State
    rhs0::State
    rhs1::State
    q::Array{Float32,3}
    sx::Array{Float32,3}
    sy::Array{Float32,3}
    fx::Array{Float32,3} # lower face flux
    fy::Array{Float32,3}
    ez::Matrix{Float32} # lower-left corner electric field
end
Workspace(n)=Workspace(State(n),State(n),State(n),State(n),
    zeros(Float32,n,n,9),zeros(Float32,n,n,9),zeros(Float32,n,n,9),
    zeros(Float32,n,n,9),zeros(Float32,n,n,9),zeros(Float32,n,n))
@inline wrap(i,n)=mod1(i,n)
@inline load(a,i,j)=ntuple(d->@inbounds(a[i,j,d]),Val(9))
@inline function put!(a,q,i,j)
    @inbounds for d in 1:9; a[i,j,d]=q[d]; end
end
function centered!(s,p)
    n=p.n
    @inbounds for j in 1:n,i in 1:n
        s.u[i,j,6]=0.5f0*(s.bx[i,j]+s.bx[wrap(i+1,n),j])
        s.u[i,j,7]=0.5f0*(s.by[i,j]+s.by[i,wrap(j+1,n)])
    end
end
function initialize(p;method=:glm,uniform=false)
    s=State(p.n); n=p.n; sb=1f0/sqrt(4f0*Float32(pi))
    @inbounds for j in 1:n,i in 1:n
        x=Float32((i-.5)/n); y=Float32((j-.5)/n)
        if uniform
            q=(1f0,.2f0,-.1f0,0f0,1f0,.3f0,.2f0,0f0,0f0)
        else
            q=(25f0/(36f0*Float32(pi)),-sin(2f0*Float32(pi)*y),sin(2f0*Float32(pi)*x),0f0,
               5f0/(12f0*Float32(pi)),-sb*sin(2f0*Float32(pi)*y),sb*sin(4f0*Float32(pi)*x),0f0,0f0)
        end
        put!(s.u,G.prim2cons(q,p.gamma),i,j)
        # OT Bx depends only on y, By only on x: face and cell values coincide.
        s.bx[i,j]=q[6]; s.by[i,j]=q[7]
    end
    method===:ct && centered!(s,p)
    s
end
function reconstruct!(w,s,p)
    n=p.n
    @threads for j in 1:n
        @inbounds for i in 1:n
            put!(w.q,G.cons2prim(load(s.u,i,j),p.gamma,p.floor,p.floor),i,j)
        end
    end
    @threads for j in 1:n
        @inbounds for i in 1:n
            q=load(w.q,i,j)
            sx=G.prim_slope(load(w.q,wrap(i-1,n),j),q,load(w.q,wrap(i+1,n),j))
            sy=G.prim_slope(load(w.q,i,wrap(j-1,n)),q,load(w.q,i,wrap(j+1,n)))
            # Shared positivity fallback in reconstructed rho/p, no state floors.
            bad=min(q[1]-max(abs(sx[1]),abs(sy[1])),q[5]-max(abs(sx[5]),abs(sy[5])))<=p.floor
            put!(w.sx,bad ? ntuple(_->0f0,Val(9)) : sx,i,j)
            put!(w.sy,bad ? ntuple(_->0f0,Val(9)) : sy,i,j)
        end
    end
end
@inline function face_flux(l,r,dir,p,ch,bn,method)
    if method===:glm
        return G.riemann_hll(l,r,dir,p.gamma,ch,p.floor,p.floor)
    end
    # Identical HLL physics to GLM, but a single staggered normal field.
    l0=G.rot_to(l,dir); r0=G.rot_to(r,dir)
    lc=(l0[1],l0[2],l0[3],l0[4],l0[5],bn,l0[7],l0[8],0f0)
    rc=(r0[1],r0[2],r0[3],r0[4],r0[5],bn,r0[7],r0[8],0f0)
    G.rot_flux_from(G.hll_x(lc,rc,p.gamma,bn,0f0,0f0),dir)
end
function curl_rhs!(r,w,p)
    n=p.n
    @inbounds for j in 1:n,i in 1:n
        im=wrap(i-1,n); jm=wrap(j-1,n)
        w.ez[i,j]=0.25f0*(w.fy[i,j,6]+w.fy[im,j,6]-w.fx[i,j,7]-w.fx[i,jm,7])
    end
    @inbounds for j in 1:n,i in 1:n
        r.bx[i,j]=-n*(w.ez[i,wrap(j+1,n)]-w.ez[i,j])
        r.by[i,j]= n*(w.ez[wrap(i+1,n),j]-w.ez[i,j])
    end
end
function rhs!(r,s,w,p,ch,method)
    n=p.n
    reconstruct!(w,s,p)
    @threads for j in 1:n
        @inbounds for i in 1:n
            im=wrap(i-1,n); jm=wrap(j-1,n)
            lx=G.padd(load(w.q,im,j),load(w.sx,im,j),1f0)
            rx=G.padd(load(w.q,i,j),load(w.sx,i,j),-1f0)
            ly=G.padd(load(w.q,i,jm),load(w.sy,i,jm),1f0)
            ry=G.padd(load(w.q,i,j),load(w.sy,i,j),-1f0)
            put!(w.fx,face_flux(lx,rx,1,p,ch,s.bx[i,j],method),i,j)
            put!(w.fy,face_flux(ly,ry,2,p,ch,s.by[i,j],method),i,j)
        end
    end
    @threads for j in 1:n
        @inbounds for i in 1:n,d in 1:9
            r.u[i,j,d]=n*(w.fx[i,j,d]-w.fx[wrap(i+1,n),j,d]+w.fy[i,j,d]-w.fy[i,wrap(j+1,n),d])
        end
    end
    method===:ct && curl_rhs!(r,w,p)
end
function valid(s,p)
    @inbounds for j in 1:p.n,i in 1:p.n
        u=load(s.u,i,j)
        all(isfinite,u) || return false
        u[1]>p.floor || return false
        pr=(p.gamma-1)*(u[5]-(u[2]^2+u[3]^2+u[4]^2)/(2u[1])-(u[6]^2+u[7]^2+u[8]^2)/2)
        pr>p.floor || return false
        u[4]==0 && u[8]==0 || return false # planar restriction
    end
    true
end
function damp!(s,dt,ch,p)
    if p.cleaning_damping>0
        factor=exp(-dt*ch/p.cleaning_damping)
        @views s.u[:,:,9].*=factor
    end
end
function step!(s,w,p,dt,ch,method)
    # Strang-split psi damping around shared SSP-RK2 flux integration.
    method===:glm && damp!(s,dt/2,ch,p)
    rhs!(w.rhs0,s,w,p,ch,method)
    @. w.stage.u=s.u+dt*w.rhs0.u
    if method===:ct
        @. w.stage.bx=s.bx+dt*w.rhs0.bx
        @. w.stage.by=s.by+dt*w.rhs0.by
        centered!(w.stage,p)
    end
    valid(w.stage,p) || error("Nonphysical first RK stage")
    rhs!(w.rhs1,w.stage,w,p,ch,method)
    @. w.out.u=0.5f0*s.u+0.5f0*(w.stage.u+dt*w.rhs1.u)
    if method===:ct
        @. w.out.bx=0.5f0*s.bx+0.5f0*(w.stage.bx+dt*w.rhs1.bx)
        @. w.out.by=0.5f0*s.by+0.5f0*(w.stage.by+dt*w.rhs1.by)
        centered!(w.out,p)
    else
        damp!(w.out,dt/2,ch,p)
    end
    valid(w.out,p) || error("Nonphysical final RK state")
    copyto!(s.u,w.out.u)
    if method===:ct; copyto!(s.bx,w.out.bx); copyto!(s.by,w.out.by); end
    nothing
end
function speeds(s,p)
    sx=sy=0f0
    @inbounds for j in 1:p.n,i in 1:p.n
        q=G.cons2prim(load(s.u,i,j),p.gamma,p.floor,p.floor)
        sx=max(sx,abs(q[2])+G.fast_speed(q,p.gamma,q[6]))
        sy=max(sy,abs(q[3])+G.fast_speed(q,p.gamma,q[7]))
    end
    sx,sy
end
function div_center(s,p)
    n=p.n
    [Float64(n)/2*(Float64(s.u[wrap(i+1,n),j,6])-s.u[wrap(i-1,n),j,6]+
        Float64(s.u[i,wrap(j+1,n),7])-s.u[i,wrap(j-1,n),7]) for i in 1:n,j in 1:n]
end
function div_face(s,p)
    n=p.n
    [Float64(n)*(Float64(s.bx[wrap(i+1,n),j])-s.bx[i,j]+Float64(s.by[i,wrap(j+1,n)])-s.by[i,j]) for i in 1:n,j in 1:n]
end
function diagnostics(s,p,method)
    valid(s,p) || error("Invalid diagnostic state")
    mass=energy=kinetic=magnetic=0.0; rhomin=pmin=Inf
    mx=my=0.0
    @inbounds for j in 1:p.n,i in 1:p.n
        u=Float64.(load(s.u,i,j)); ke=(u[2]^2+u[3]^2)/(2u[1]); me=(u[6]^2+u[7]^2)/2
        mass+=u[1];energy+=u[5];kinetic+=ke;magnetic+=me;mx+=u[2];my+=u[3]
        rhomin=min(rhomin,u[1]);pmin=min(pmin,(p.gamma-1)*(u[5]-ke-me))
    end
    dc=div_center(s,p); face=method===:ct ? maximum(abs,div_face(s,p)) : NaN
    brms=sqrt(2magnetic/p.n^2)
    (;mass=mass/p.n^2,energy=energy/p.n^2,kinetic=kinetic/p.n^2,magnetic=magnetic/p.n^2,
      mx=mx/p.n^2,my=my/p.n^2,rhomin,pmin,div_center_max=maximum(abs,dc),
      div_center_rms=sqrt(sum(abs2,dc)/p.n^2),div_face_max=face,
      normalized_div_center_rms=sqrt(sum(abs2,dc)/p.n^2)/(p.n*brms))
end
end
