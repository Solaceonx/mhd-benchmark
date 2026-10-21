# Checks of the sweep-fixes branch of Tom Abel's solver (GLMMHDTurb/, branch sweep-fixes) on the CPU, before it goes
# to Sherlock. The branch's kernels and the original ones (sherlock_sweep/vendor) run as written, through kernel_cpu.jl.
#   julia -t 4 --project=../../sherlock_sweep/env fix_branch.jl
# 1. Control: with every fix switched off, the branch's kernels reproduce the original ones bit for bit (3D state).
# 2. The branch's 3D kernels on z-invariant data against the planar emulation (emu.jl, ct_driver.jl) of the fix it
#    implements, one step at a time: they must agree to rounding, so the emulation's convergence and leakage tables
#    (README.md) are tables of the branch.
# 3. What only the 3D kernels can show: mass conservation across tile edges with anchored storage, the corner
#    divergence of B under cell-centred CT on 3D data, and the magnetised blast with each storage.
include("kernel_cpu.jl")
include("emu.jl");using .Emu,Printf,Random;include("problems.jl")
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const GM=load_solver(joinpath(ROOT,"sherlock_sweep","vendor","glmmhd_turb.jl"),:GLM_original)
const GF=load_solver(joinpath(ROOT,"GLMMHDTurb","glmmhd_turb.jl"),:GLM_branch)
# the branch with only the HLLD guard put back as it was (for the control of the HLLD kernel)
const GG=let src=read(joinpath(ROOT,"GLMMHDTurb","glmmhd_turb.jl"),String),old="small = abs(denK) < 1.0f-4*pts",f=tempname()*".jl"
    count(old,src)==1 || error("guard line not found");write(f,replace(src,old=>"small = abs(denK) < 1.0f-12"));m=load_solver(f,:GLM_branch_oldguard);rm(f);m
end
const CM=load_solver(joinpath(ROOT,"sherlock_sweep","vendor","ct_mhd.jl"),:CT_original)
const CF=load_solver(joinpath(ROOT,"GLMMHDTurb","ct_mhd.jl"),:CT_branch)
const T9=NTuple{9,Float32}
failures=String[]
check(ok,what)=(println(ok ? "ok    " : "WRONG ",what);ok || push!(failures,what))

to3d(u::Matrix{T9})=(n=size(u,1);a=Array{Float32}(undef,n,n,n,9);for k in 1:n,j in 1:n,i in 1:n,v in 1:9;a[i,j,k,v]=u[i,j][v];end;a)
layer(a,k=1)=[ntuple(v->a[i,j,k,v],9) for i in 1:size(a,1),j in 1:size(a,2)]
zinvariant(a)=all(a[:,:,k,:]==a[:,:,1,:] for k in 2:size(a,3))
params(G,n,gamma)=G.Params(N=n,gamma=Float32(gamma),courant=0.25f0,smallr=1f-6,smallc=sqrt(Float32(gamma)),glm_ch_scale=1f0,glm_cp_coef=0.18f0)
mat(u)=[c[v] for c in u,v in 1:9]
reldiff(a,b)=maximum(abs.(Float64.(a).-Float64.(b)))/max(maximum(abs.(Float64.(b))),1e-30)

"A smooth, fully 3D magnetised state (all three directions, all nine variables but psi active)."
function state3d(n;seed=1)
    rng=MersenneTwister(seed);ph=2pi .* rand(rng,12);a=Array{Float32}(undef,n,n,n,9);g=5/3
    for k in 1:n,j in 1:n,i in 1:n
        x,y,z=((i,j,k).-.5)./n;s(m,c)=sin(2pi*m+ph[c])
        rho=1+.3s(x+y,1)*s(z,2);vx=.5s(y+z,3);vy=.4s(x-z,4);vz=.3s(x+2y,5);p=1+.2s(x+y+z,6)
        bx=.4+.2s(y-z,7);by=.3+.2s(z+x,8);bz=.2+.1s(x-y,9)        # cell-centred B, not divergence-free: fine for a control
        a[i,j,k,:]=Float32[rho,rho*vx,rho*vy,rho*vz,p/(g-1)+rho*(vx^2+vy^2+vz^2)/2+(bx^2+by^2+bz^2)/2,bx,by,bz,.01s(x+y-z,10)]
    end
    a
end
function signal3d(a,gamma)
    vmax=0f0
    for k in axes(a,3),j in axes(a,2),i in axes(a,1)
        c=ntuple(v->a[i,j,k,v],9);rho=c[1];p=(gamma-1f0)*(c[5]-(c[2]^2+c[3]^2+c[4]^2)/(2f0*rho)-(c[6]^2+c[7]^2+c[8]^2)/2f0)
        q=(rho,c[2]/rho,c[3]/rho,c[4]/rho,p,c[6],c[7],c[8],0f0)
        vmax=max(vmax,abs(q[2])+GM.fast_speed(q,gamma,c[6])+abs(q[3])+GM.fast_speed(q,gamma,c[7])+abs(q[4])+GM.fast_speed(q,gamma,c[8]))
    end
    vmax
end
"Advance with step(u,un,dt) for nsteps CFL steps; the arrays are swapped as the drivers do."
function advance(step,u0,p,nsteps)
    u=copy(u0);un=similar(u);af=zeros(Float32,3,64,64,64)
    for _ in 1:nsteps
        dt=Float32(p.courant/(p.N*signal3d(u,p.gamma)));step(u,un,af,dt);u,un=un,u
    end
    u
end

println("1. Control: the branch with its fixes off is the original, bit for bit (3D state, 24^3, 12 steps)")
let n=24,u0=state3d(n);pm=params(GM,n,5/3);pf=params(GF,n,5/3)
    for (what,orig,branch) in (
            ("step_plm! f16 tile 6, HLL (glm_hll16)",(u,un,af,dt)->GM.step_plm!(u,un,af,pm,dt,0f0;half=true,riemann=:hll),
                                                    (u,un,af,dt)->GF.step_plm!(u,un,af,pf,dt,0f0;half=true,riemann=:hll,anchored=false)),
            ("step_plm! f16 tile 4, HLL",(u,un,af,dt)->GM.step_plm!(u,un,af,pm,dt,0f0;tb=4,half=true,riemann=:hll),
                                         (u,un,af,dt)->GF.step_plm!(u,un,af,pf,dt,0f0;tb=4,half=true,riemann=:hll,anchored=false)),
            ("step_plm! PPM f16 (glm_ppm)",(u,un,af,dt)->GM.step_plm!(u,un,af,pm,dt,0f0;half=true,riemann=:hll,recon=:ppm),
                                           (u,un,af,dt)->GF.step_plm!(u,un,af,pf,dt,0f0;half=true,riemann=:hll,recon=:ppm,anchored=false)),
            ("step_plm! f32 tile 4, HLLD (glm_hlld), the branch with the original HLLD guard put back",
                                           (u,un,af,dt)->GM.step_plm!(u,un,af,pm,dt,0f0;tb=4,half=false,riemann=:hlld),
                                           (u,un,af,dt)->GG.step_plm!(u,un,af,params(GG,n,5/3),dt,0f0;tb=4,half=false,riemann=:hlld)),
            ("step_tiled! LLF (glm)",(u,un,af,dt)->GM.step_tiled!(u,un,af,pm,dt,0f0;use_hlld=false,lean=false),
                                     (u,un,af,dt)->GF.step_tiled!(u,un,af,pf,dt,0f0;use_hlld=false,lean=false)))
        a=advance(orig,u0,pm,12);b=advance(branch,u0,pf,12)
        check(a==b && all(isfinite,a),"$what: identical$(a==b ? "" : @sprintf(", differ by %.1e",reldiff(b,a)))")
    end
    a=advance((u,un,af,dt)->GM.step_plm!(u,un,af,pm,dt,0f0;tb=4,half=false,riemann=:hlld),u0,pm,12)
    b=advance((u,un,af,dt)->GF.step_plm!(u,un,af,pf,dt,0f0;tb=4,half=false,riemann=:hlld),u0,pf,12)
    check(reldiff(b,a)<1e-5,@sprintf("step_plm! HLLD with the new guard: differs from the original by %.1e on this smooth state",reldiff(b,a)))
    # the original driver's unlisted Float32 triples go through the launcher with its new trailing arguments
    t=(u,un,af,dt)->(ch=pf.courant/(n*dt)/3f0*pf.glm_ch_scale;gf=exp(-(ch*ch/(pf.glm_cp_coef*pf.boxlen*ch))*dt);
        GF._plm_launch(Val(4),Val(false),Val(:hll),192,n÷4,un,u,af,pf,GF.pfloor(pf),dt,GF.dxof(pf),ch,gf,0f0,false))
    o=(u,un,af,dt)->(ch=pm.courant/(n*dt)/3f0*pm.glm_ch_scale;gf=exp(-(ch*ch/(pm.glm_cp_coef*pm.boxlen*ch))*dt);
        GM._plm_launch(Val(4),Val(false),Val(:hll),192,n÷4,un,u,af,pm,GM.pfloor(pm),dt,GM.dxof(pm),ch,gf,0f0,false))
    a=advance(o,u0,pm,6);b=advance(t,u0,pf,6);check(a==b,"_plm_launch Float32 tile 4, HLL as the sweep driver calls it (glm_hll32): identical")
end

println("\n2. The branch's 3D kernels on z-invariant data against the planar emulation of each fix (one step each)")
function compare(what,ic,gamma,kernel,emurun;steps=3,tol=2e-6,ref_other=nothing,noise=false)
    n=size(ic,1);u2=copy(ic);u3=to3d(u2);p=params(GF,n,gamma);af=zeros(Float32,3,64,64,64);worst=0.0;other=0.0
    r=Emu.Run(;n=n,gamma=Float32(gamma),emurun...);W=ntuple(_->Matrix{T9}(undef,n,n),7);un2=similar(u2);un3=similar(u3)
    for _ in 1:steps
        dt=Float32(r.courant/(n*Emu.signal(u2,r)));ch=r.fct ? 1f-20 : r.courant/(n*dt)/3f0*r.ch_scale
        Emu.step!(un2,u2,W,r,dt,ch,Val(r.riem),Val(r.st),Val(r.fst));kernel(u3,un3,af,p,dt)
        zinvariant(un3) || (check(false,"$what: the 3D result is not z-invariant");return)
        worst=max(worst,reldiff(mat(layer(un3)),mat(un2)))
        if ref_other!==nothing
            uo=similar(u2);Emu.step!(uo,u2,W,Emu.Run(;n=n,gamma=Float32(gamma),ref_other...),dt,ch,Val(ref_other.riem),Val(get(ref_other,:st,:f32)),Val(get(ref_other,:fst,:f32)))
            other=max(other,reldiff(mat(uo),mat(un2)))
        end
        u2,un2=un2,u2;u3,un3=un3,u3
    end
    noise ? check(worst<other,@sprintf("%s: kernel vs emulation %.1e, below the emulation's own Float16 rounding (its Float32 run differs by %.1e)",what,worst,other)) :
    check(worst<tol,@sprintf("%s: kernel vs emulation %.1e%s",what,worst,ref_other===nothing ? "" : @sprintf(" (the emulation's plain-f16 step differs from it by %.1e)",other)))
end
let n=24
    compare("entropy wave, f16 tile 6 anchored, HLL vs emulation a16 storage, f32 fluxes",entropy(n),5/3,
        (u,un,af,p,dt)->GF.step_plm!(u,un,af,p,dt,0f0;half=true,riemann=:hll,anchored=true),(;riem=:hll,st=:a16,fst=:f32,tb=6);
        ref_other=(;riem=:hll,st=:f16,fst=:f16,tb=6))
    compare("entropy wave, PPM f16 anchored vs emulation",entropy(n),5/3,
        (u,un,af,p,dt)->GF.step_plm!(u,un,af,p,dt,0f0;half=true,riemann=:hll,recon=:ppm,anchored=true),(;riem=:hll,st=:a16,fst=:f32,tb=6,ppm=true);
        ref_other=(;riem=:hll,st=:f16,fst=:f16,tb=6,ppm=true))
    compare("Orszag-Tang, f16 tile 4 anchored, HLL vs emulation",orszag_tang(n),5/3,
        (u,un,af,p,dt)->GF.step_plm!(u,un,af,p,dt,0f0;tb=4,half=true,riemann=:hll,anchored=true),(;riem=:hll,st=:a16,fst=:f32,tb=4);steps=5,ref_other=(;riem=:hll,tb=4),noise=true)
    compare("Orszag-Tang, f16 tile 4, original storage vs emulation f16 (the emulation's own check)",orszag_tang(n),5/3,
        (u,un,af,p,dt)->GF.step_plm!(u,un,af,p,dt,0f0;tb=4,half=true,riemann=:hll,anchored=false),(;riem=:hll,st=:f16,fst=:f16,tb=4);steps=5,tol=1e-3)
    fs=zeros(Float32,n,n,n,9)
    compare("field loop, cell-centred CT, f32 tile 4, HLL vs emulation fct",field_loop(n,0.0),5/3,
        (u,un,af,p,dt)->GF.step_plm!(u,un,af,p,dt,0f0;tb=4,half=false,riemann=:hll,anchored=false,fct=true,fs=fs),(;riem=:hll,fct=true);steps=5)
    compare("Orszag-Tang, cell-centred CT, f16 tile 6 anchored, HLL vs emulation fct a16",orszag_tang(n),5/3,
        (u,un,af,p,dt)->GF.step_plm!(u,un,af,p,dt,0f0;half=true,riemann=:hll,anchored=true,fct=true,fs=fs),(;riem=:hll,st=:a16,fst=:f32,tb=6,fct=true);steps=5,ref_other=(;riem=:hll,tb=6,fct=true),noise=true)
end
# CT prototype: the branch's kernels against the emulation's full slope + all-direction predictor (ct_slope.jl)
module CTE end
Emu.load_physics(CTE,"ct_mhd.jl",Emu.CT_LINES;edit="ntuple(t->0.5f0*moncen(M[t]-L[t],R[t]-M[t]),8)"=>"ntuple(t->moncen(M[t]-L[t],R[t]-M[t]),8)")
Base.include(CTE,joinpath(@__DIR__,"ct_driver.jl"))
# The 2D emulation's state placed in the plane (p, q) of the 3D grid, invariant along r; perm = (p, q, r) must be a
# cyclic order of the axes so that cross products keep their sign. 2D x-components go to axis p, y to q, z to r.
function place3d(U,bxf,byf,bz,n,perm)
    p,q,r=perm;U3=zeros(Float32,n,n,n,5);F=ntuple(_->zeros(Float32,n,n,n),3)
    for c in 1:n,b in 1:n,a in 1:n
        I=zeros(Int,3);I[p]=a;I[q]=b;I[r]=c;u=U[a,b]
        U3[I...,1]=u[1];U3[I...,1+p]=u[2];U3[I...,1+q]=u[3];U3[I...,1+r]=u[4];U3[I...,5]=u[5]
        F[p][I...]=bxf[a,b];F[q][I...]=byf[a,b];F[r][I...]=bz[a,b]
    end
    U3,F
end
"Largest difference between the 3D kernels' state and the 2D emulation's (relative to each quantity's size), after `steps`."
function ct_vs_emulation(prim,A,B0,n,perm;steps=4)
    U,bxf,byf,bz=CTE.initial(n,prim,A,B0);U3,F=place3d(U,bxf,byf,bz,n,perm);p,q,r=perm
    sc=CF.scratch(n);worst=0.0;wb=0.0
    for _ in 1:steps
        dt=Float32(.25f0/(n*CTE.signal(U,bxf,byf,bz,n)))
        CTE.step_full!(U,bxf,byf,bz,dt,n;emf=:gs05v);CF.step!(U3,F...,sc...,dt,1f0/n,n)
        at(a,b)=(I=zeros(Int,3);I[p]=a;I[q]=b;I[r]=1;I)
        for (v,v3) in ((1,1),(2,1+p),(3,1+q),(4,1+r),(5,5))
            m=max(maximum(abs(U[a,b][v]) for a in 1:n,b in 1:n),1f-30)
            worst=max(worst,maximum(abs(U3[at(a,b)...,v3]-U[a,b][v]) for a in 1:n,b in 1:n)/max(m,1f0))
        end
        mb=max(maximum(abs,bxf),maximum(abs,byf),maximum(abs,bz),1f-30)
        for (f2,f3) in ((bxf,F[p]),(byf,F[q]),(bz,F[r]))
            wb=max(wb,maximum(abs(f3[at(a,b)...]-f2[a,b]) for a in 1:n,b in 1:n)/mb)
        end
    end
    worst,wb
end
let n=24
    for (what,prim,A,B0) in (("entropy wave",(x,y)->(1+.2*sin(2pi*(x+y)),.5,.5,0.0,1.0),(x,y)->(0.0,0.0,0.0),(.3,.2,0.0)),
                             ("Alfven wave",(x,y)->(1.0,0.0,-.1*cos(2pi*x),-.1*sin(2pi*x),.1),(x,y)->(0.0,-.1*cos(2pi*x)/2pi,-.1*sin(2pi*x)/2pi),(1.0,0.0,0.0)))
        worst,wb=ct_vs_emulation(prim,A,B0,n,(1,2,3))
        check(worst<2e-6 && wb<2e-6,@sprintf("CT prototype, %s: kernel vs emulation (full slope, all-direction predictor, upwinded edge EMF) %.1e, face B %.1e",what,worst,wb))
    end
    # the field loop in each plane of the grid: the edge EMF along z, then along x, then along y does the work
    loop(x,y)=(1.0,2.0,1.0,0.0,1.0);Aloop(x,y)=(0.0,0.0,1e-3*max(.3-hypot(x-.5,y-.5),0.0))
    for (plane,perm) in (("x-y plane (E_z)",(1,2,3)),("y-z plane (E_x)",(2,3,1)),("z-x plane (E_y)",(3,1,2)))
        worst,wb=ct_vs_emulation(loop,Aloop,(0.0,0.0,0.0),n,perm;steps=6)
        check(worst<2e-6 && wb<2e-6,@sprintf("CT prototype, field loop in the %s: kernel vs emulation %.1e, face B %.1e of its size",plane,worst,wb))
    end
end

println("\n3. 3D only")
let n=24,u0=state3d(n);p=params(GF,n,5/3);m0=sum(Float64,u0[:,:,:,1])
    for (what,anch) in (("original f16",false),("anchored f16",true))
        u=advance((u,un,af,dt)->GF.step_plm!(u,un,af,p,dt,0f0;half=true,riemann=:hll,anchored=anch),u0,p,12)
        dm=abs(sum(Float64,u[:,:,:,1])-m0)/m0
        check(dm<1e-6,@sprintf("mass after 12 steps of the 3D state, %s: relative change %.1e",what,dm))
    end
    # cell-centred CT keeps the corner divergence of B (of whatever field it starts with)
    # divergence of B at the cell corner (i+1/2,j+1/2,k+1/2), times dx: each component's difference across the corner,
    # averaged over the four cell pairs that touch it
    function cornerdiv(a)
        n=size(a,1);w(i)=mod1(i,n);D=zeros(n,n,n)
        for k in 1:n,j in 1:n,i in 1:n,e in 0:1,f in 0:1
            D[i,j,k]+=(a[w(i+1),w(j+e),w(k+f),6]-a[i,w(j+e),w(k+f),6]+a[w(i+e),w(j+1),w(k+f),7]-a[w(i+e),j,w(k+f),7]+
                       a[w(i+e),w(j+f),w(k+1),8]-a[w(i+e),w(j+f),k,8])/4
        end
        D
    end
    fs=zeros(Float32,n,n,n,9);D0=cornerdiv(u0)
    for (what,kw) in (("f16 tile 6 anchored, HLL",(;half=true,riemann=:hll,anchored=true)),("f32 tile 4, HLL",(;tb=4,half=false,riemann=:hll,anchored=false)))
        u=advance((u,un,af,dt)->GF.step_plm!(u,un,af,p,dt,0f0;kw...,fct=true,fs=fs),u0,p,12)
        dD=maximum(abs.(cornerdiv(u).-D0));m=maximum(abs.(u0[:,:,:,6:8]))
        check(dD<1e-5*m && all(isfinite,u),@sprintf("cell-centred CT, %s, 3D state: corner divergence of B (zero at the start) is %.1e of |B| after 12 steps",what,dD/m))
    end
    gm=advance((u,un,af,dt)->GF.step_plm!(u,un,af,p,dt,0f0;tb=4,half=false,riemann=:hll),u0,p,12)
    @printf("      (GLM, same kernel: %.1e of |B|)\n",maximum(abs.(cornerdiv(gm).-D0))/maximum(abs.(u0[:,:,:,6:8])))
end
let n=32,u0=to3d(blast(n,100.0))
    p=params(GF,n,7/5);println("  blast B = 100/sqrt(4pi), 32^3 (z-invariant), to t = 0.01 with 10 outputs (steps shortened to land on them):")
    for (what,kw,cfl) in (("original f16, c_h from the step taken",(;anchored=false),false),("original f16, c_h from the CFL step",(;anchored=false),true),
                          ("anchored f16, c_h from the CFL step",(;anchored=true),true))
        u=copy(u0);un=similar(u);af=zeros(Float32,3,64,64,64);t=0.0;why="reached the end";steps=0
        for target in range(0,.01;length=11)
            while t<target-1e-10 && why=="reached the end"
                v=signal3d(u,p.gamma);(isfinite(v) && v>0) || (why="nonphysical state";break)
                dtc=Float32(p.courant/(n*v));dt=Float32(min(dtc,target-t))
                GF.step_plm!(u,un,af,p,dt,0f0;half=true,riemann=:hll,dt_cfl=cfl ? dtc : dt,kw...);u,un=un,u;t+=dt;steps+=1
            end
        end
        all(isfinite,u) || (why="non-finite state")
        println(@sprintf("    %-40s t = %.4f after %d steps: %s",what,t,steps,why))
    end
end
println(isempty(failures) ? "\nall checks passed" : "\nFAILED: "*join(failures,"; "))
