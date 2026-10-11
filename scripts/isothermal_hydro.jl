# Isothermal Euler adaptation of the Orszag–Tang CPU SIMD PLM/LLF workflow.
# Four conserved fields (rho, rho*vx, rho*vy, rho*vz); P=cs^2*rho everywhere.
using SIMD, Base.Threads, Printf
const IW = 8
const IV = Vec{IW,Float32}
const IG = 2
Base.@kwdef struct IsoParams
    N::Int = 64
    cs::Float32 = 1f0
    courant::Float32 = 0.35f0
    rho_floor::Float32 = 1f-7
end
@inline iload(a,x,j,k) = ntuple(d -> vload(IV,pointer(a,x+size(a,1)*((j-1)+size(a,2)*((k-1)+size(a,3)*(d-1))))),Val(4))
@inline function istore!(a,q,x,j,k)
    for d in 1:4
        vstore(q[d],pointer(a,x+size(a,1)*((j-1)+size(a,2)*((k-1)+size(a,3)*(d-1)))))
    end
end
@inline iprim(q) = (q[1],q[2]/q[1],q[3]/q[1],q[4]/q[1])
@inline icons(q) = (q[1],q[1]*q[2],q[1]*q[3],q[1]*q[4])
@inline iadd(q,s,c) = ntuple(d -> q[d]+c*s[d],Val(4))
@inline function imoncen(dl,dr)
    dc=0.5f0*(dl+dr)
    copysign(min(abs(dc),2f0*min(abs(dl),abs(dr))),dc)*max(sign(dl*dr),zero(dl))
end
@inline islope(l,q,r) = ntuple(d -> 0.5f0*imoncen(q[d]-l[d],r[d]-q[d]),Val(4))
@inline function iflux(q,dir,cs2)
    vn=q[dir+1]; mass=q[1]*vn
    (mass,mass*q[2]+(dir==1 ? cs2*q[1] : zero(mass)),
          mass*q[3]+(dir==2 ? cs2*q[1] : zero(mass)),
          mass*q[4]+(dir==3 ? cs2*q[1] : zero(mass)))
end
@inline function illf(l,r,dir,cs)
    fl=iflux(l,dir,cs*cs); fr=iflux(r,dir,cs*cs)
    ul=icons(l); ur=icons(r)
    a=max(abs(l[dir+1]),abs(r[dir+1]))+cs
    ntuple(d -> 0.5f0*(fl[d]+fr[d]-a*(ur[d]-ul[d])),Val(4))
end
@inline iw(i,n) = mod1(i,n)
@inline function irecon(a,x,j,k,p,dtdx)
    n=p.N; q=iprim(iload(a,x,j,k))
    sx=islope(iprim(iload(a,x-1,j,k)),q,iprim(iload(a,x+1,j,k)))
    sy=islope(iprim(iload(a,x,iw(j-1,n),k)),q,iprim(iload(a,x,iw(j+1,n),k)))
    sz=islope(iprim(iload(a,x,j,iw(k-1,n))),q,iprim(iload(a,x,j,iw(k+1,n))))
    u=icons(q)
    fxl=iflux(iadd(q,sx,-1f0),1,p.cs^2); fxr=iflux(iadd(q,sx,1f0),1,p.cs^2)
    fyl=iflux(iadd(q,sy,-1f0),2,p.cs^2); fyr=iflux(iadd(q,sy,1f0),2,p.cs^2)
    fzl=iflux(iadd(q,sz,-1f0),3,p.cs^2); fzr=iflux(iadd(q,sz,1f0),3,p.cs^2)
    uh=ntuple(d -> u[d]-0.5f0*dtdx*(fxr[d]-fxl[d]+fyr[d]-fyl[d]+fzr[d]-fzl[d]),Val(4))
    # Revert the predictor to first order in any lane whose reconstructed face
    # could be nonpositive. The same deterministic face flux is used on both sides.
    bad=uh[1] <= p.rho_floor + max(abs(sx[1]),max(abs(sy[1]),abs(sz[1])))
    safe=ntuple(d -> vifelse(bad,u[d],uh[d]),Val(4))
    z=zero(IV)
    (iprim(safe),ntuple(d->vifelse(bad,z,sx[d]),Val(4)),
                  ntuple(d->vifelse(bad,z,sy[d]),Val(4)),
                  ntuple(d->vifelse(bad,z,sz[d]),Val(4)))
end
function iso_ghost!(a,p)
    n=p.N
    @inbounds for d in 1:4,k in 1:n,j in 1:n,g in 1:IG
        a[g,j,k,d]=a[n+g,j,k,d]
        a[n+IG+g,j,k,d]=a[IG+g,j,k,d]
    end
end
function iso_step!(b,a,p,dt::Float32)
    n=p.N; h=dt*n
    @threads for k in 1:n
        @inbounds for j in 1:n,x in IG+1:IW:IG+n
            q,sx,sy,sz=irecon(a,x,j,k,p,h)
            xp,xps,_,_=irecon(a,x+1,j,k,p,h); xm,xms,_,_=irecon(a,x-1,j,k,p,h)
            yp,_,yps,_=irecon(a,x,iw(j+1,n),k,p,h); ym,_,yms,_=irecon(a,x,iw(j-1,n),k,p,h)
            zp,_,_,zps=irecon(a,x,j,iw(k+1,n),p,h); zm,_,_,zms=irecon(a,x,j,iw(k-1,n),p,h)
            fxh=illf(iadd(q,sx,1f0),iadd(xp,xps,-1f0),1,p.cs)
            fxl=illf(iadd(xm,xms,1f0),iadd(q,sx,-1f0),1,p.cs)
            fyh=illf(iadd(q,sy,1f0),iadd(yp,yps,-1f0),2,p.cs)
            fyl=illf(iadd(ym,yms,1f0),iadd(q,sy,-1f0),2,p.cs)
            fzh=illf(iadd(q,sz,1f0),iadd(zp,zps,-1f0),3,p.cs)
            fzl=illf(iadd(zm,zms,1f0),iadd(q,sz,-1f0),3,p.cs)
            u=iload(a,x,j,k)
            istore!(b,ntuple(d -> u[d]+h*(fxl[d]-fxh[d]+fyl[d]-fyh[d]+fzl[d]-fzh[d]),Val(4)),x,j,k)
        end
    end
end
function iso_stats(a,p)
    mass=ke=logsum=log2sum=0.0; mom=zeros(3); rmin=Inf; rmax=signal=0.0
    @inbounds for k in 1:p.N,j in 1:p.N,i in IG+1:IG+p.N
        r=Float64(a[i,j,k,1]); x=Float64(a[i,j,k,2]); y=Float64(a[i,j,k,3]); z=Float64(a[i,j,k,4])
        if !isfinite(r+x+y+z) || r<=p.rho_floor
            return nothing
        end
        mass+=r; mom[1]+=x; mom[2]+=y; mom[3]+=z; ke+=(x*x+y*y+z*z)/(2r)
        lr=log(r); logsum+=lr; log2sum+=lr*lr
        rmin=min(rmin,r); rmax=max(rmax,r); signal=max(signal,(abs(x)+abs(y)+abs(z))/r+3p.cs)
    end
    bulk=mom./mass; kinetic=(ke-0.5mass*sum(abs2,bulk))/p.N^3
    (; mass=mass/p.N^3, momentum=mom./p.N^3, bulk, kinetic,
       mach=sqrt(2kinetic/(mass/p.N^3))/p.cs,
       logstd=sqrt(max(0,log2sum/p.N^3-(logsum/p.N^3)^2)),rmin,rmax,signal)
end

# Reject invalid trial states; never repair density or silently add mass.
function iso_advance!(b,a,p,dt)
    h=Float32(dt); rejected=0
    for attempt in 1:13
        iso_step!(b,a,p,h)
        d=iso_stats(b,p)
        if d!==nothing
            iso_ghost!(b,p)
            return h,d,rejected
        end
        h*=0.5f0; rejected+=1
    end
    error("Could not obtain a positive finite isothermal state")
end
