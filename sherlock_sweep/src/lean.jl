# Per-output measurements on the device that holds the state (GPU, or CPU arrays in tests): the same
# quantities as Problems.diagnostics and solution_errors, computed with fused reductions instead of
# Float64 host copies of the whole volume, plus block-averaged restriction for reduced outputs.
# At 512^3 the host path needed ~50 GB of RAM and took longer than the solve; this takes well under 1 s.
module Lean
using KernelAbstractions
import ..Problems
const KA=KernelAbstractions
const F=Float64

lazy(f,args...)=Broadcast.instantiate(Broadcast.broadcasted(f,args...))
fsum(f,args...)=mapreduce(identity,+,lazy(f,args...);init=0.0)
fmax(f,args...)=mapreduce(identity,max,lazy(f,args...);init=-Inf)
fmin(f,args...)=mapreduce(identity,min,lazy(f,args...);init=Inf)
fcount(f,args...)=mapreduce(identity,+,lazy((x...)->Int(f(x...)),args...);init=0)
fall(f,args...)=mapreduce(identity,&,lazy(f,args...);init=true)

# Periodic centred and face divergences, term by term as in Problems.diagnostics (n/2 (b[i+1]-b[i-1]) and
# n (f[i+1]-f[i]) per direction, grid spacing 1/n in every direction). A single z plane gives zero z terms.
@kernel function divergence_kernel!(dc,df,bx,by,bz,f1,f2,f3,n,withfaces)
    i,j,k=@index(Global,NTuple)
    nx,ny,nz=size(dc)
    ip=ifelse(i==nx,1,i+1);im=ifelse(i==1,nx,i-1)
    jp=ifelse(j==ny,1,j+1);jm=ifelse(j==1,ny,j-1)
    kp=ifelse(k==nz,1,k+1);km=ifelse(k==1,nz,k-1)
    @inbounds begin
        h=n/2
        dc[i,j,k]=h*(F(bx[ip,j,k])-F(bx[im,j,k]))+h*(F(by[i,jp,k])-F(by[i,jm,k]))+h*(F(bz[i,j,kp])-F(bz[i,j,km]))
        if withfaces
            df[i,j,k]=n*(F(f1[ip,j,k])-F(f1[i,j,k]))+n*(F(f2[i,jp,k])-F(f2[i,j,k]))+n*(F(f3[i,j,kp])-F(f3[i,j,k]))
        end
    end
end
function divergences(bx,by,bz,faces)
    dc=similar(bx,F,size(bx));withfaces=faces!==nothing
    df=withfaces ? similar(bx,F,size(bx)) : dc
    f1,f2,f3=withfaces ? faces : (bx,by,bz)
    backend=KA.get_backend(dc)
    # No synchronize: the reductions that read dc/df run later on the same stream.
    divergence_kernel!(backend)(dc,df,bx,by,bz,f1,f2,f3,F(size(bx,1)),withfaces;ndrange=size(dc))
    dc,withfaces ? df : nothing
end

"""
    diagnostics(q, faces=nothing; gamma, problem)

`q = (rho, mx, my, mz, E, bx, by, bz, psi)`: equal-size (n, n, nz) arrays or views on one device;
`psi = nothing` for CT (reported as zero). Returns the NamedTuple of Problems.diagnostics, same keys
and order, equal to rounding (summation order differs).
"""
function diagnostics(q,faces=nothing;gamma=5/3,problem="")
    rho,mx,my,mz,E,bx,by,bz,psi=q
    n=size(rho,1);N=length(rho);g=F(gamma)
    mean(f,args...)=fsum(f,args...)/N
    ke(r,a,b,c)=(F(a)^2+F(b)^2+F(c)^2)/(2F(r))
    me(a,b,c)=(F(a)^2+F(b)^2+F(c)^2)/2
    pr(r,a,b,c,e,x,y,z)=(g-1)*(F(e)-ke(r,a,b,c)-me(x,y,z))
    P=(rho,mx,my,mz,E,bx,by,bz)
    if problem=="kh_localized"
        rows=Problems.leak_band(n);rows==first(rows):last(rows) || error("leak band is not contiguous")
        band=map(x->view(x,:,first(rows):last(rows),:),(bx,by,bz))
        leak_max=fmax((a,b,c)->sqrt(me(a,b,c)*2),band...)
        leak_rms=sqrt(fsum((a,b,c)->sqrt(me(a,b,c)*2)^2,band...)/length(band[1]))
    else
        leak_max,leak_rms=NaN,NaN
    end
    if problem=="gem" && n%4==0
        flux(j)=fsum((a,c)->abs((F(a)+F(c))/2),view(by,:,j,1),view(by,:,j+1,1))/(2n)
        flux_lower,flux_upper=flux(n÷4),flux(3n÷4)
    else
        flux_lower,flux_upper=NaN,NaN
    end
    dc,df=divergences(bx,by,bz,faces)
    brms=sqrt(mean((a,b,c)->F(a)^2+F(b)^2+F(c)^2,bx,by,bz))
    fields=psi===nothing ? P : (P...,psi)
    zvar=maximum(x->fmax((a,b)->abs(F(a)-F(b)),x,view(x,:,:,1:1)),fields)
    fin=all(x->fall(isfinite,x),fields)
    floor=1.00001e-6
    res=(mass=mean(F,rho),energy=mean(F,E),kinetic=mean(ke,rho,mx,my,mz),magnetic=mean(me,bx,by,bz),
        thermal=mean(pr,P...)/(g-1),
        momentum_x=mean(F,mx),momentum_y=mean(F,my),momentum_z=mean(F,mz),
        bx_rms=sqrt(mean(x->F(x)^2,bx)),by_rms=sqrt(mean(x->F(x)^2,by)),bz_rms=sqrt(mean(x->F(x)^2,bz)),
        bz_max=fmax(x->abs(F(x)),bz),psi_rms=psi===nothing ? 0.0 : sqrt(mean(x->F(x)^2,psi)),
        psi_max=psi===nothing ? 0.0 : fmax(x->abs(F(x)),psi),
        leak_band_bmax=leak_max,leak_band_brms=leak_rms,recon_flux_lower=flux_lower,recon_flux_upper=flux_upper,
        rho_min=fmin(F,rho),pressure_min=fmin(pr,P...),finite=fin,
        density_floor_cells=fcount(x->F(x)<=floor,rho),pressure_floor_cells=fcount((x...)->pr(x...)<=floor,P...),
        nonpositive_pressure_cells=fcount((x...)->pr(x...)<=0,P...),
        div_center_rms=sqrt(mean(x->x^2,dc)),div_center_max=fmax(abs,dc),
        normalized_div_center_rms=brms>0 ? sqrt(mean(x->x^2,dc))/(n*brms) : 0.0,
        div_face_rms=df===nothing ? NaN : sqrt(mean(x->x^2,df)),div_face_max=df===nothing ? NaN : fmax(abs,df),
        z_variation=zvar)
    finalize(dc);df===nothing || finalize(df)    # free the Float64 temporaries now (GPU memory), not at the next GC
    res
end

"""
    errors(q, ref)

solution_errors(u, ref) for the state `q` (as in `diagnostics`) against the host reference `ref`
(n, n, nz, 9), which is copied to the device first.
"""
function errors(q,ref)
    ref===nothing && return Dict{String,Float64}()
    r=similar(q[1],Float32,size(ref));copyto!(r,ref)
    R(c)=view(r,:,:,:,c);N=length(q[1]);result=Dict{String,Float64}()
    d1(a,b)=sqrt((F(a)-F(b))^2);s1(b)=sqrt(F(b)^2)
    d3(a1,a2,a3,b1,b2,b3)=sqrt((F(a1)-F(b1))^2+(F(a2)-F(b2))^2+(F(a3)-F(b3))^2)
    s3(b1,b2,b3)=sqrt(F(b1)^2+F(b2)^2+F(b3)^2)
    for (name,cols) in (("density",(1,)),("momentum",(2,3,4)),("energy",(5,)),("magnetic",(6,7,8)))
        A=map(c->q[c],cols);B=map(R,cols);d,s=length(cols)==1 ? (d1,s1) : (d3,s3)
        for (norm,num,den) in (("L1",fsum(d,A...,B...)/N,fsum(s,B...)/N),
                               ("L2",sqrt(fsum((x...)->d(x...)^2,A...,B...)/N),sqrt(fsum((x...)->s(x...)^2,B...)/N)),
                               ("Linf",fmax(d,A...,B...),fmax(s,B...)))
            result[name*"_absolute_"*norm]=num
            result[name*"_relative_"*norm]=den>0 ? num/den : NaN
        end
    end
    finalize(r)
    result
end

"""
    restrict(A, m)

Block average of the first three (spatial) dimensions of a device array `A` (n, n, nz, rest...) to
(m, m, mz, rest...) on the host, as Float32 (sums in Float64). A single z plane stays one plane.
"""
function restrict(A,m)
    n,_,nz=size(A);rest=size(A)[4:end]
    n%m==0 || error("restrict: $n is not a multiple of $m")
    r=n÷m;mz=nz==1 ? 1 : m;rz=nz÷mz
    s=sum(F,reshape(A,r,m,r,m,rz,mz,rest...);dims=(1,3,5))
    Float32.(reshape(Array(s),m,m,mz,rest...)./(r*r*rz))
end
end
