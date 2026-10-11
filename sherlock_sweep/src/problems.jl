module Problems
using Random, LinearAlgebra, Statistics, Printf

# Every problem lives on the periodic unit cube [0,1)^3 with N^3 cells, because both
# vendored solvers only support that geometry. Problems from the FLASH MHD test suite
# (FLASH user guide section 35.2) are rescaled/adapted to it; see README.
const NAMES=("uniform","entropy","alfven","alfven_oblique","brio_wu","brio_wu_rotated",
    "orszag_tang","orszag_tang_3d","rotor","current_sheet","field_loop","field_loop_3d","blast","decay",
    "kh_uniform","kh_localized","gem")
const FULL_3D=("orszag_tang_3d","field_loop_3d","decay")
wrap(i,n)=mod1(i,n)

"Gamma given as a number or as a string such as \"5/3\"."
function parse_gamma(x)
    x isa Real && return Float64(x)
    parts=split(strip(x),"/")
    length(parts)==2 ? parse(Float64,parts[1])/parse(Float64,parts[2]) : parse(Float64,x)
end
default_gamma(name)=name in ("brio_wu","brio_wu_rotated") ? 2.0 : name in ("rotor","blast") ? 1.4 : 5/3
gamma_of(name,par=Dict{String,Any}())=parse_gamma(get(par,"gamma",default_gamma(name)))
is_planar(name,par=Dict{String,Any}())=name=="blast" ? get(par,"dim",3)==2 : !(name in FULL_3D)

"Periodic primitive of a +1/-1 square wave on [0,1): +1 on [0,a) and [b,1), -1 on [a,b)."
function square_integral(x,a,b)
    x=mod(x,1.0)
    x<a ? x : x<b ? 2a-x : x+2a-2b      # periodic when b-a=1/2
end

field_loop_velocity(name,par)=name=="field_loop_3d" ? (1.0,1.0,2.0) :
    (2.0,1.0,Float64(get(par,"vz",sqrt(5.0))))

"""
Problem definition: primitive state (rho,vx,vy,vz,p) at a point, vector potential A at a
point (periodic part only), uniform field B0 and subsamples per cell dimension for the
cell averages of density, momentum and energy. `t` translates problems with exact solutions.
"""
function setup(name,n,par,seed,t)
    A0=(x,y,z)->(0.0,0.0,0.0);B0=(0.0,0.0,0.0);sub=1
    if name=="uniform"
        prim=(x,y,z)->(1.0,.2,-.1,.05,1.0);B0=(.3,.2,0.0)
    elseif name=="entropy"
        # Exact cell average of the density wave; momentum and energy are linear in rho.
        prim=(x,y,z)->(1+.2*sinc(1/n)^2*sin(2pi*(x+y-t)),.5,.5,0.0,1.0);B0=(.3,.2,0.0)
    elseif name=="alfven"
        # Right-going circularly polarized Alfven wave along x; rho=Bx=1, period 1.
        prim=(x,y,z)->(1.0,0.0,-.1*cos(2pi*(x-t)),-.1*sin(2pi*(x-t)),.1)
        A0=(x,y,z)->(0.0,-.1*cos(2pi*(x-t))/2pi,-.1*sin(2pi*(x-t))/2pi);B0=(1.0,0.0,0.0)
    elseif name=="alfven_oblique"
        # Same wave with k parallel to (1,2): wavelength 1/sqrt(5), period 1/sqrt(5).
        s5=sqrt(5.0);tx,ty=-2/s5,1/s5;phase(x,y)=2pi*(x+2y-s5*t)
        prim=(x,y,z)->(1.0,-.1*cos(phase(x,y))*tx,-.1*cos(phase(x,y))*ty,-.1*sin(phase(x,y)),.1)
        A0=(x,y,z)->(-.1*cos(phase(x,y))*tx/(2s5*pi),-.1*cos(phase(x,y))*ty/(2s5*pi),-.1*sin(phase(x,y))/(2pi*s5))
        B0=(1/s5,2/s5,0.0)
    elseif name in ("brio_wu","brio_wu_rotated")
        # Periodic L|R|L arrangement. Aligned: interfaces at x=0,1/2. Rotated: normal (1,1)/sqrt(2).
        rotated=name=="brio_wu_rotated";xi(x,y)=rotated ? mod(x+y,1.0) : x
        prim=(x,y,z)->xi(x,y)<.5 ? (1.0,0.0,0.0,0.0,1.0) : (.125,0.0,0.0,0.0,.1)
        g(x,y)=square_integral(xi(x,y),.5,1.0)        # integral of B_t=+1/-1 along the normal
        if rotated
            A0=(x,y,z)->(0.0,0.0,-g(x,y)/sqrt(2));B0=(.75/sqrt(2),.75/sqrt(2),0.0);sub=8
        else
            A0=(x,y,z)->(0.0,0.0,-g(x,y));B0=(.75,0.0,0.0)
        end
    elseif name in ("orszag_tang","orszag_tang_3d")
        # Athena normalization (sound speed 1, beta=2gamma); identical dynamics to FLASH's.
        b=1/sqrt(4pi);eps=name=="orszag_tang_3d" ? Float64(get(par,"epsilon",.2)) : 0.0
        prim=(x,y,z)->(25/(36pi),-(1+eps*sin(2pi*z))*sin(2pi*y),(1+eps*sin(2pi*z))*sin(2pi*x),eps*sin(2pi*z),5/(12pi))
        A0=(x,y,z)->(0.0,0.0,b*(cos(2pi*y)/2pi+cos(4pi*x)/4pi))
    elseif name=="rotor"
        r0=.1;r1=.115;u0=Float64(get(par,"u0",2.0))
        prim=function(x,y,z)
            r=hypot(x-.5,y-.5);f=(r1-r)/(r1-r0)
            r<=r0 ? (10.0,-u0*(y-.5)/r0,u0*(x-.5)/r0,0.0,1.0) :
            r<r1 ? (1+9f,-f*u0*(y-.5)/r,f*u0*(x-.5)/r,0.0,1.0) : (1.0,0.0,0.0,0.0,1.0)
        end
        B0=(Float64(get(par,"b0",5.0))/sqrt(4pi),0.0,0.0);sub=8
    elseif name=="current_sheet"
        # Gardiner & Stone (2005) sec. 5.5 [0,2]^2 box halved: lengths /2, speeds unchanged, times /2.
        # Their vx = v0 sin(pi y) on [0,2] (also RAMSES patch/mhd/sheet) is one wavelength per box:
        # sin(2 pi y) here. FLASH's user guide writes sin(2 pi y) on [0,2]; we follow the original.
        u0=Float64(get(par,"u0",.1))
        prim=(x,y,z)->(1.0,u0*sin(2pi*y),0.0,0.0,.1)
        A0=(x,y,z)->(0.0,0.0,-square_integral(x,.25,.75))
    elseif name=="field_loop"
        a0=1e-3;R=.3;prim=(x,y,z)->(1.0,field_loop_velocity(name,par)...,1.0)
        A0=(x,y,z)->(0.0,0.0,a0*max(R-hypot(x-.5,y-.5),0.0))
    elseif name=="field_loop_3d"
        # Gardiner & Stone (2008) loop in a unit cube: cylinder axis (-1,0,1)/sqrt(2), which
        # is a lattice vector, so the periodic images stay 0.707 apart (> loop diameter 0.6).
        a0=1e-3;R=.3;ax=(-1/sqrt(2),0.0,1/sqrt(2))
        prim=(x,y,z)->(1.0,field_loop_velocity(name,par)...,1.0)
        A0=function(x,y,z)
            d=Inf
            for sx in -1:1,sy in -1:1,sz in -1:1
                v=(x-.5+sx,y-.5+sy,z-.5+sz);p=v[1]*ax[1]+v[3]*ax[3]
                d=min(d,sqrt(max(v[1]^2+v[2]^2+v[3]^2-p^2,0.0)))
            end
            a=a0*max(R-d,0.0);(a*ax[1],0.0,a*ax[3])
        end
    elseif name=="blast"
        dim=get(par,"dim",3);dim in (2,3) || error("blast dim must be 2 or 3")
        prim=(x,y,z)->(1.0,0.0,0.0,0.0,sqrt((x-.5)^2+(y-.5)^2+(dim==3 ? (z-.5)^2 : 0.0))<.1 ? 1000.0 : .1)
        B0=(Float64(get(par,"bx",100.0))/sqrt(4pi),0.0,0.0);sub=dim==3 ? 4 : 8
    elseif name in ("kh_uniform","kh_localized")
        # Tomida et al. (2026) sec. III.1-2 parameters: rho=1, p=10, gamma=5/3, vx=+-v0 with v0=1 and
        # tanh width a=0.05, vy=0.01 sin(2 pi x) exp(-(y-y_l)^2/0.2^2), uniform Bx=0.01 (beta=2e5).
        # Their single layer with outflow y boundaries becomes a periodic double layer (y=1/4, 3/4).
        # kh_localized concentrates Bx in a sech^2 band on the lower layer, with
        # nonzero periodic tails at the upper layer. Track growth from that baseline;
        # an upper-layer field alone does not establish divergence-cleaning leakage.
        a=.05;sig=.2;pr=10.0;v0=1.0
        prim=(x,y,z)->(1.0,v0*(tanh((y-.25)/a)-tanh((y-.75)/a)-1),
            .01*sin(2pi*x)*(exp(-((y-.25)/sig)^2)+exp(-((y-.75)/sig)^2)),0.0,pr)
        if name=="kh_uniform"
            B0=(sqrt(2pr/Float64(get(par,"beta",2e5))),0.0,0.0)
        else
            # Tomida use Bx = B0 exp(-y^2/0.1^2); we use the sech^2 band of equal FWHM (w = 0.0945)
            # because its antiderivative is elementary in both Julia and C++.
            # Bx = b*sum_s sech^2((y-1/4+s)/w): uniform mean 2bw plus a periodic part Az = int(Bx - mean);
            # the tanh antiderivative telescopes over images, so Az is periodic to ~exp(-50).
            b=Float64(get(par,"b_peak",.01));w=Float64(get(par,"b_width",.1));mean_b=2b*w
            F(y)=sum(b*w*(tanh((y-.25+sh)/w)-tanh((sh-.25)/w)) for sh in -1:1)-mean_b*y
            A0=(x,y,z)->(0.0,0.0,F(y));B0=(mean_b,0.0,0.0)
        end
    elseif name=="gem"
        # GEM reconnection challenge (Birn et al. 2001), MHD version: Harris sheet Bx = B0 tanh(z/lambda),
        # n = n0 sech^2(z/lambda) + n_bg, n0 (Ti+Te) = B0^2/2, perturbation psi0 cos(2 pi x/Lx) cos(pi z/Lz),
        # Lx x Lz = 25.6 x 12.8 d_i, lambda = 0.5, psi0 = 0.1, n_bg = 0.2 (B0 = n0 = 1, times in 1/Omega_i).
        # Periodic double sheet in the unit square: sheets at y = 1/4, 3/4 (each half box is 25.6 x 12.8),
        # lengths / Lx, so t = 40/Omega_i is t = 40/Lx here. A_z = lambda ln cosh - lambda ln cosh - y is
        # periodic to ~lambda exp(-1/(2 lambda)); psi0 cos(2 pi (x - 1/2)) puts both X-points at x = 1/2.
        L=Float64(get(par,"gem_lx",25.6));lam=Float64(get(par,"gem_lambda",.5))/L
        psi0=Float64(get(par,"gem_psi0",.1))/L;nbg=Float64(get(par,"gem_nbg",.2))
        sheets(y)=sech((y-.25)/lam)^2+sech((y-.75)/lam)^2
        prim=(x,y,z)->(sheets(y)+nbg,0.0,0.0,0.0,.5*(sheets(y)+nbg))
        lncosh(u)=abs(u)+log1p(exp(-2abs(u)))-log(2.0)
        A0=(x,y,z)->(0.0,0.0,lam*lncosh((y-.25)/lam)-lam*lncosh((y-.75)/lam)-y-psi0*cos(2pi*x)*cos(2pi*(y-.25)))
        sub=8
    elseif name=="decay"
        modes=decay_modes(seed)
        prim=function(x,y,z)
            v=zeros(3)
            for (kv,a,phase) in modes;v.+=a.*sin(2pi*(kv[1]*x+kv[2]*y+kv[3]*z)+phase);end
            (1.0,v...,1/gamma_of(name,par))
        end
        B0=Tuple(Float64.(get(par,"b0_vector",[.1,0.0,0.0])))
    else
        error("Unknown problem $name")
    end
    (;prim,A0,B0,sub)
end

"Solenoidal decay modes; stored coefficients make the 3D random field reproducible across grids."
function decay_modes(seed)
    modes=Tuple{NTuple{3,Int},Vector{Float64},Float64}[]
    rng=MersenneTwister(seed)
    for kz in -2:2,ky in -2:2,kx in -2:2
        kv=(kx,ky,kz);ksq=sum(abs2,kv)
        1<=ksq<=4 || continue
        first(filter(!iszero,collect(kv)))>0 || continue
        a=randn(rng,3);k=collect(kv);a.-=k.*(dot(a,k)/ksq)
        a./=sqrt(ksq);push!(modes,(kv,a,2pi*rand(rng)))
    end
    modes
end

"""
Plain-text problem spec for the C++ port (codes/common/abel_problems.hpp): everything the
external codes need to rebuild `initial(name,n)` without re-deriving config defaults.
"""
function spec_text(name,n;seed=42,mach=2.0,par=Dict{String,Any}(),t=0.0)
    s=setup(name,n,par,seed,t)
    lines=["# generated by problems.jl spec_text; do not edit","problem $name","n $n","sub $(s.sub)",
        "planar $(Int(is_planar(name,par)))","dim $(get(par,"dim",3))",@sprintf("gamma %.17g",gamma_of(name,par)),@sprintf("t %.17g",t)]
    for (key,default) in (("vz",sqrt(5.0)),("u0",name=="current_sheet" ? .1 : 2.0),("b0",5.0),("bx",100.0),("epsilon",.2),
                          ("beta",2e5),("b_peak",.01),("b_width",.1),
                          ("gem_lx",25.6),("gem_lambda",.5),("gem_psi0",.1),("gem_nbg",.2))
        push!(lines,@sprintf("%s %.17g",key,Float64(get(par,key,default))))
    end
    push!(lines,@sprintf("b0_vector %.17g %.17g %.17g",Float64.(get(par,"b0_vector",[.1,0.0,0.0]))...))
    if name=="decay"
        # Same normalization as `initial`: mean removed, then rms scaled to `mach` (rho=1, sub=1).
        c=((1:n).-.5)./n;m=zeros(3);sq=0.0;vs=zeros(n,n,n,3)
        for k in 1:n,j in 1:n,i in 1:n;vs[i,j,k,:].=s.prim(c[i],c[j],c[k])[2:4];end
        m=[mean(@view vs[:,:,:,d]) for d in 1:3]
        for d in 1:3;vs[:,:,:,d].-=m[d];end
        push!(lines,@sprintf("mean %.17g %.17g %.17g",m...),@sprintf("scale %.17g",mach/sqrt(sum(abs2,vs)/n^3)))
        for (kv,a,phase) in decay_modes(seed)
            push!(lines,@sprintf("mode %d %d %d %.17g %.17g %.17g %.17g",kv...,a...,phase))
        end
    end
    join(lines,"\n")*"\n"
end

"""
Lower periodic face fields from edge-centred samples of a periodic vector potential A plus
a uniform field B0. The discrete divergence cancels before floating-point rounding.
Midpoint edge samples approximate line integrals; general oblique/3D face fields
are second-order approximations, not exact continuum face averages.
"""
function faces_from_potential(A,B0,n,planar;thin=false)
    nz=planar ? 1 : n;c(i)=(i-.5)/n;f(i)=(i-1)/n
    Ax=[A(c(i),f(j),f(k))[1] for i in 1:n,j in 1:n,k in 1:nz]
    Ay=[A(f(i),c(j),f(k))[2] for i in 1:n,j in 1:n,k in 1:nz]
    Az=[A(f(i),f(j),c(k))[3] for i in 1:n,j in 1:n,k in 1:nz]
    w(i)=wrap(i,n);wz(k)=wrap(k,nz)
    bx=[n*((Az[i,w(j+1),k]-Az[i,j,k])-(Ay[i,j,wz(k+1)]-Ay[i,j,k]))+B0[1] for i in 1:n,j in 1:n,k in 1:nz]
    by=[n*((Ax[i,j,wz(k+1)]-Ax[i,j,k])-(Az[w(i+1),j,k]-Az[i,j,k]))+B0[2] for i in 1:n,j in 1:n,k in 1:nz]
    bz=[n*((Ay[w(i+1),j,k]-Ay[i,j,k])-(Ax[i,w(j+1),k]-Ax[i,j,k]))+B0[3] for i in 1:n,j in 1:n,k in 1:nz]
    ntuple(d->Float32.(thin ? (bx,by,bz)[d] : repeat((bx,by,bz)[d],1,1,n÷nz)),3)
end

function center_faces(b)
    ntuple(d->(b[d].+circshift(b[d],ntuple(k->k==d ? -1 : 0,3)))./2f0,3)
end

"""
Initial (or, for translating exact solutions, time-t) conserved state and CT face fields.
plane=true returns only the z = 1 plane of a z-invariant problem ((n, n, 1, 9) and (n, n, 1) faces, identical
to that plane of the cube); used for 2D references and post-processing, where an n^3 cube would not fit.
"""
function initial(name,n;seed=42,mach=2.0,par=Dict{String,Any}(),t=0.0,plane=false)
    name in NAMES || error("Unknown problem $name")
    n>=8 && n%4==0 || error("N must be >=8 and divisible by 4")
    gamma=gamma_of(name,par);planar=is_planar(name,par);s=setup(name,n,par,seed,t)
    thin=plane && planar;b=faces_from_potential(s.A0,s.B0,n,planar;thin)
    nz=planar ? 1 : n;m=s.sub;offsets=((1:m).-.5)./m.-.5
    rho=zeros(n,n,nz);mom=zeros(n,n,nz,3);thermal=zeros(n,n,nz);kinetic=zeros(n,n,nz)
    Threads.@threads for k in 1:nz
        for j in 1:n,i in 1:n
            for oz in (planar ? (0.0,) : offsets),oy in offsets,ox in offsets
                r,vx,vy,vz,p=s.prim((i-.5+ox)/n,(j-.5+oy)/n,(k-.5+oz)/n)
                rho[i,j,k]+=r;mom[i,j,k,1]+=r*vx;mom[i,j,k,2]+=r*vy;mom[i,j,k,3]+=r*vz
                thermal[i,j,k]+=p/(gamma-1);kinetic[i,j,k]+=r*(vx^2+vy^2+vz^2)/2
            end
        end
    end
    samples=planar ? m^2 : m^3
    rho./=samples;mom./=samples;thermal./=samples;kinetic./=samples
    if name=="decay"
        for d in 1:3;mom[:,:,:,d].-=mean(@view mom[:,:,:,d]);end
        mom.*=mach/sqrt(sum(abs2,mom)/n^3)
        kinetic.=dropdims(sum(abs2,mom;dims=4);dims=4)./(2rho)
    end
    u=zeros(Float32,n,n,thin ? 1 : n,9);bc=center_faces(b)
    # z-invariant problems: fill plane 1 and copy it (bit-identical: the faces do not vary with z).
    Threads.@threads for k in 1:(planar ? 1 : n)
        kk=planar ? 1 : k
        for j in 1:n,i in 1:n
            me=sum(Float64(bc[d][i,j,k])^2 for d in 1:3)/2
            u[i,j,k,1]=rho[i,j,kk]
            for d in 1:3;u[i,j,k,1+d]=mom[i,j,kk,d];u[i,j,k,5+d]=bc[d][i,j,k];end
            u[i,j,k,5]=thermal[i,j,kk]+kinetic[i,j,kk]+me
        end
    end
    planar && for k in 2:size(u,3);u[:,:,k,:].=@view u[:,:,1,:];end
    u,b
end

"""
Reference target at time t, or `nothing`. Entropy advection is an exact cell-average
solution. Alfvén targets translate the initial-data projection; point-sampled velocity
and discrete-curl magnetic fields introduce an O(dx^2) initialization mismatch.
The projected targets therefore measure evolution plus initialization error.
"""
function reference(name,n,t,initial_state;seed=42,mach=2.0,par=Dict{String,Any}(),plane=false)
    name=="uniform" && return initial_state,"exact uniform state"
    if name in ("entropy","alfven","alfven_oblique")
        label=name=="entropy" ? "exact cell-average entropy advection" :
            "projected continuum Alfven wave; O(dx^2) initialization mismatch; not an exact discrete evolution"
        return initial(name,n;seed,mach,par,t,plane)[1],label
    end
    if name in ("field_loop","field_loop_3d")
        shift=field_loop_velocity(name,par).*t
        is_planar(name,par) && (shift=shift[1:2])       # z-invariant: z displacement is irrelevant
        all(abs(s-round(s))<1e-5 for s in shift) &&
            return initial_state,"kinematic periodic-return target; weak Lorentz forces make this approximate"
    end
    nothing,"no independent reference at this time"
end

"""
Rows of y cells in the leakage band of `kh_localized`: the field-free upper shear layer,
|y - 3/4| < 0.1, which starts with only the ~1e-6 tail of the lower-layer band.
"""
leak_band(n)=[j for j in 1:n if abs((j-.5)/n-.75)<.1]

function diagnostics(u,faces=nothing;gamma=5/3,problem="")
    n=size(u,1);q=Float64.(u);rho=q[:,:,:,1];mom=q[:,:,:,2:4];b=q[:,:,:,6:8]
    # Leakage metric (NaN for other problems, so every diagnostics.csv has the same columns).
    if problem=="kh_localized"
        bb=b[:,leak_band(n),:,:];bmag=sqrt.(dropdims(sum(abs2,bb;dims=4);dims=4))
        leak_max,leak_rms=maximum(bmag),sqrt(mean(abs2,bmag))
    else
        leak_max,leak_rms=NaN,NaN
    end
    # GEM reconnected flux (Birn et al. 2001): half the integral of |B_y| along each sheet, in box units
    # (multiply by gem_lx for B0 d_i). The sheets lie on cell faces; average the two adjacent rows.
    if problem=="gem" && n%4==0
        flux(j)=sum(abs.((b[:,j,1,2].+b[:,j+1,1,2])./2))/(2n)
        flux_lower,flux_upper=flux(n÷4),flux(3n÷4)
    else
        flux_lower,flux_upper=NaN,NaN
    end
    ke=dropdims(sum(abs2,mom;dims=4);dims=4)./(2rho)
    me=dropdims(sum(abs2,b;dims=4);dims=4)./2
    pr=(gamma-1).*(q[:,:,:,5].-ke.-me)
    div=zeros(size(rho));fd=zeros(size(rho))
    for d in 1:3
        shift=ntuple(k->k==d ? -1 : 0,3)
        bd=b[:,:,:,d];div.+=n/2 .* (circshift(bd,shift).-circshift(bd,.-shift))
        faces===nothing || (fd.+=n.*(circshift(Float64.(faces[d]),shift).-faces[d]))
    end
    brms=sqrt(mean(sum(abs2,b;dims=4)))
    return (mass=mean(rho),energy=mean(@view q[:,:,:,5]),kinetic=mean(ke),magnetic=mean(me),thermal=mean(pr)/(gamma-1),
        momentum_x=mean(@view mom[:,:,:,1]),momentum_y=mean(@view mom[:,:,:,2]),momentum_z=mean(@view mom[:,:,:,3]),
        bx_rms=sqrt(mean(abs2,@view b[:,:,:,1])),by_rms=sqrt(mean(abs2,@view b[:,:,:,2])),bz_rms=sqrt(mean(abs2,@view b[:,:,:,3])),
        bz_max=maximum(abs,@view b[:,:,:,3]),psi_rms=sqrt(mean(abs2,@view q[:,:,:,9])),psi_max=maximum(abs,@view q[:,:,:,9]),
        leak_band_bmax=leak_max,leak_band_brms=leak_rms,recon_flux_lower=flux_lower,recon_flux_upper=flux_upper,
        rho_min=minimum(rho),pressure_min=minimum(pr),finite=all(isfinite,q),
        density_floor_cells=count(<=(1.00001e-6),rho),pressure_floor_cells=count(<=(1.00001e-6),pr),nonpositive_pressure_cells=count(<=(0),pr),
        div_center_rms=sqrt(mean(abs2,div)),div_center_max=maximum(abs,div),
        normalized_div_center_rms=brms>0 ? sqrt(mean(abs2,div))/(n*brms) : 0.0,
        div_face_rms=faces===nothing ? NaN : sqrt(mean(abs2,fd)),div_face_max=faces===nothing ? NaN : maximum(abs,fd),
        z_variation=maximum(abs,q.-reshape(q[:,:,1,:],n,n,1,9)))
end

"Face-divergence tolerance for Float32 rounding of the initial faces."
div_tolerance(b,n)=1e-6*n*max(maximum(d->maximum(abs,d),b),1e-30)
end
