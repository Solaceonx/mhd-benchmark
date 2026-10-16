# Original GPU kernels plus test/measurement adapters. No solver equations edited.
module Suite
using CUDA, TOML, SHA, Dates, Printf, Statistics
include("problems.jl")
include("errors.jl")
include("lean.jl")
include("spectra.jl")
include("../vendor/glmmhd_turb.jl")
include("../vendor/ct_mhd.jl")
const G=GLMMHDTurb
const ROOT=normpath(joinpath(@__DIR__,".."))

# CT hard-codes `const GAMMA = 5f0/3f0`. For other adiabatic indices the unmodified vendor
# file is parsed again with only that constant replaced, under a new module name. The file
# on disk is never edited; gamma=5/3 always uses the original module.
const CT_SOURCE=joinpath(ROOT,"vendor","ct_mhd.jl")
is_gamma_const(e)=e isa Expr && e.head===:const && e.args[1] isa Expr && e.args[1].head===:(=) && e.args[1].args[1]===:GAMMA
function ct_variant(ex,gamma::Float32,name::Symbol)
    ex isa Expr && ex.head===:module || return ex
    hits=Ref(0)
    walk(e)=is_gamma_const(e) ? (hits[]+=1;Expr(:const,Expr(:(=),:GAMMA,gamma))) :
        e isa Expr ? Expr(e.head,map(walk,e.args)...) : e
    out=walk(ex);out.args[2]=name
    hits[]==1 || error("Expected exactly one CT GAMMA constant, found $(hits[])")
    out
end
include(ex->ct_variant(ex,7f0/5f0,:CTMHD_gamma7_5),CT_SOURCE)
include(ex->ct_variant(ex,2f0,:CTMHD_gamma2),CT_SOURCE)
include(ex->ct_variant(ex,101f0/100f0,:CTMHD_gamma101_100),CT_SOURCE)
const CT_BY_GAMMA=Dict(CTMHD.GAMMA=>CTMHD,CTMHD_gamma7_5.GAMMA=>CTMHD_gamma7_5,CTMHD_gamma2.GAMMA=>CTMHD_gamma2,
    CTMHD_gamma101_100.GAMMA=>CTMHD_gamma101_100)

# Methods run by the GPU driver. All GLM variants are the advisor's unmodified kernels:
#   glm          original suite path: step_tiled!, Float32 tile, full Hancock, LLF (use_hlld=false)
#   glm_hll16    production step_plm!: f16 shared tile, HLL (TB=6 when N%6==0, else TB=4)
#   glm_hlld     step_plm! Float32 tile, HLLD (TB=4)
#   glm_ppm      step_plm! lean parabolic PPM, f16, HLL
#   glm_constch  glm_hll16 with a constant cleaning speed (Tomida et al. 2026 "D1c"): c_h fixed at
#                its t=0 value and dt capped so c_h dt/dx never exceeds the variable scheme's value
#   ct           constrained-transport prototype, CTMHD.step!
# Checks of what half precision does (sherlock_checks/). step_plm! lists the tile/precision/solver triples its author
# uses; the kernels behind it (integrator_plm!, integrator_plm_ppm!) are generic in all three, so these call the same
# unedited kernels for triples step_plm! does not list (plm_unlisted! below):
#   glm_hll32    the production kernel with a Float32 tile: HLL, TB=6 when N%6==0, else TB=4
#   glm_ppm32    the PPM kernel with a Float32 tile
#   glm_hlld_x   glm_hlld's own triple called through plm_unlisted!: must reproduce glm_hlld bit for bit
# The sweep-fixes branch of the kernels (sherlock_fixes/; vendor/ must hold that branch, FIXES below). Each passes the
# CFL step for c_h (dt_cfl), so a step shortened to land on an output keeps its CFL step's c_h:
#   glm_fix        glm (step_tiled!, LLF)
#   glm_fix_hll16  glm_hll16 with anchored Float16 storage
#   glm_fix_hlld   glm_hlld with the HLLD degenerate-state guard relative to the star pressure
#   glm_fix_ppm    glm_ppm with anchored Float16 storage
#   glm_fct16      glm_fix_hll16 with cell-centred constrained transport in place of GLM cleaning (no psi, no c_h)
#   glm_fct32      the production kernel with a Float32 4-cell tile and cell-centred constrained transport
#   ct_fix         the CT prototype with the full limited slope and the predictor in all three directions
# With the branch as vendor/, glm_hll16, glm_ppm and glm_constch pass anchored=false and are the original kernels bit
# for bit; glm_hlld, glm_hlld_x and ct are not available (the branch changed what they run).
const METHODS=("glm","glm_hll16","glm_hlld","glm_ppm","glm_constch","ct","glm_hll32","glm_ppm32","glm_hlld_x",
               "glm_fix","glm_fix_hll16","glm_fix_hlld","glm_fix_ppm","glm_fct16","glm_fct32","ct_fix")
const FIX_METHODS=(:glm_fix,:glm_fix_hll16,:glm_fix_hlld,:glm_fix_ppm,:glm_fct16,:glm_fct32,:ct_fix)
const FIXES=isdefined(G,:fct_scratch)       # vendor/ is the sweep-fixes branch
"Keywords that keep step_plm! the original kernel on the sweep-fixes branch (none with the original vendor/)."
const ORIGINAL_STORAGE=FIXES ? (;anchored=false) : (;)
function check_vendor(m::Symbol)
    m in FIX_METHODS && !FIXES && error("$m needs the sweep-fixes branch of the kernels in vendor/")
    m in (:glm_hlld,:glm_hlld_x,:ct) && FIXES && error("$m is the original kernel, which the sweep-fixes branch in vendor/ changed")
end
# Core study: the original tiled GLM path versus the CT prototype. Additional
# GLM kernels remain available through an explicit `methods` list.
const DEFAULT_METHODS=["glm","ct"]
is_glm(m::Symbol)=startswith(string(m),"glm")
const SOLVER_TEXT=Dict(
    :glm=>"original step_tiled!, Float32 tile TB=4, full Hancock, use_hlld=false (LLF), c_h proportional to 1/dt",
    :glm_hll16=>"original step_plm! production kernel: f16 shared tile, MonCen PLM, Hancock, HLL, c_h proportional to 1/dt",
    :glm_hlld=>"original step_plm!: Float32 tile TB=4, MonCen PLM, Hancock, HLLD, c_h proportional to 1/dt",
    :glm_ppm=>"original step_plm! recon=:ppm: lean parabolic PPM, f16, HLL, c_h proportional to 1/dt",
    :glm_constch=>"original step_plm! f16/HLL; driver holds c_h constant via glm_ch_scale and caps dt (Tomida D1c)",
    :ct=>"original CTMHD.step!, 7 kernels, transverse-free Hancock, HLL, arithmetic edge EMF",
    :glm_hll32=>"original integrator_plm! with a Float32 shared tile (a triple step_plm! does not list), MonCen PLM, Hancock, HLL, c_h proportional to 1/dt",
    :glm_ppm32=>"original integrator_plm_ppm! with a Float32 shared tile (a triple step_plm! does not list), lean parabolic PPM, HLL, c_h proportional to 1/dt",
    :glm_hlld_x=>"original integrator_plm!, Float32 tile TB=4, HLLD, launched by the driver as glm_hll32 is (control for that launch)",
    :glm_fix=>"sweep-fixes step_tiled!, Float32 tile TB=4, LLF, c_h from the CFL step",
    :glm_fix_hll16=>"sweep-fixes step_plm!: anchored f16 tile (differences from tile anchors, fluxes dt/dx*F as head+tail), HLL, c_h from the CFL step",
    :glm_fix_hlld=>"sweep-fixes step_plm!: Float32 tile TB=4, HLLD with the guard |denK| < 1e-4 p_tot*, c_h from the CFL step",
    :glm_fix_ppm=>"sweep-fixes step_plm! recon=:ppm: lean PPM, anchored f16 tile, HLL, c_h from the CFL step",
    :glm_fct16=>"sweep-fixes step_plm!: anchored f16 tile, HLL, cell-centred constrained transport (flux-CT, Toth 2000) in place of GLM",
    :glm_fct32=>"sweep-fixes step_plm!: Float32 tile, HLL, cell-centred constrained transport (flux-CT, Toth 2000) in place of GLM",
    :ct_fix=>"sweep-fixes CTMHD.step!: full MonCen slope, MUSCL-Hancock predictor in all directions, HLL, arithmetic edge EMF")
function ct_module(gamma)
    haskey(CT_BY_GAMMA,Float32(gamma)) || error("CT adapter supports gamma in $(sort(collect(keys(CT_BY_GAMMA)))); got $gamma")
    CT_BY_GAMMA[Float32(gamma)]
end

function record(path,data)
    open(path*".tmp","w") do io;TOML.print(io,data);end
    mv(path*".tmp",path;force=true)
end

mutable struct DeviceState
    method::Symbol
    ct::Module
    ch0::Float32          # constant cleaning speed for glm_constch (0 otherwise)
    u::CuArray{Float32,4}
    other::CuArray{Float32,4}
    faces::NTuple{3,CuArray{Float32,3}}
    scratch::Tuple
    signals::CuArray{Float32,3}
    force::CuArray{Float32,4}
end

function allocate(method,h,b,gamma)
    n=size(h,1);empty3=CUDA.zeros(Float32,1,1,1);empty4=CUDA.zeros(Float32,1,1,1,1);C=ct_module(gamma)
    check_vendor(method)
    if is_glm(method)
        u=CuArray(h)
        DeviceState(method,C,0f0,u,similar(u),(empty3,empty3,empty3),method in (:glm_fct16,:glm_fct32) ? (G.fct_scratch(n),) : (),
            CUDA.zeros(Float32,n,n,n),empty4)
    else
        DeviceState(method,C,0f0,CuArray(h[:,:,:,1:5]),empty4,Tuple(CuArray.(b)),C.scratch(n),
            CUDA.zeros(Float32,n,n,n),empty4)
    end
end

function reset!(s,h,b)
    if is_glm(s.method);copyto!(s.u,h)
    else
        copyto!(s.u,h[:,:,:,1:5]);for d in 1:3;copyto!(s.faces[d],b[d]);end
    end
    CUDA.synchronize()
end
function centers!(s,n)
    is_glm(s.method) || @cuda(threads=256,blocks=cld(n^3,256), s.ct.bcc_kernel!(s.scratch[1:3]...,s.faces...,n))
end

# Unfloored state validity and a common physical signal-speed rule. This adapter
# does not alter the evolution kernels or their internal floor policies.
function signals_kernel!(out,u,bx,by,bz,isct,n,gamma,pfloor)
    l=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if l<=n^3
        i=(l-1)%n+1;j=((l-1)÷n)%n+1;k=(l-1)÷(n*n)+1
        @inbounds begin
            r=u[i,j,k,1];mx=u[i,j,k,2];my=u[i,j,k,3];mz=u[i,j,k,4];e=u[i,j,k,5]
            b1=isct ? bx[i,j,k] : u[i,j,k,6];b2=isct ? by[i,j,k] : u[i,j,k,7];b3=isct ? bz[i,j,k] : u[i,j,k,8]
            p=(gamma-1f0)*(e-(mx*mx+my*my+mz*mz)/(2f0*r)-(b1*b1+b2*b2+b3*b3)/2f0)
            # pressure_policy="floor": use the solvers' own floor value instead of stopping.
            pfloor>0f0 && isfinite(p) && (p=max(p,pfloor))
            if !(isfinite(r) && r>0 && isfinite(p) && p>0 && isfinite(b1+b2+b3))
                out[i,j,k]=Inf32
            else
                q=(r,mx/r,my/r,mz/r,p,b1,b2,b3,0f0)
                out[i,j,k]=abs(q[2])+G.fast_speed(q,gamma,b1)+abs(q[3])+G.fast_speed(q,gamma,b2)+abs(q[4])+G.fast_speed(q,gamma,b3)
            end
        end
    end
    return
end
function timestep(s,p,pfloor)
    n=p.N;centers!(s,n)
    b=is_glm(s.method) ? (s.signals,s.signals,s.signals) : s.scratch[1:3]
    @cuda threads=256 blocks=cld(n^3,256) signals_kernel!(s.signals,s.u,b...,!is_glm(s.method),n,p.gamma,pfloor)
    vmax=maximum(s.signals);CUDA.synchronize()
    isfinite(vmax) && vmax>0 || error("Nonphysical/nonfinite state in CFL scan")
    # glm_constch: fix c_h at the variable scheme's t=0 value vmax/3, then cap dt at the same
    # cleaning Courant number, so c_h*dt/dx <= courant/3 (Tomida et al. 2026, D1c).
    s.method===:glm_constch && s.ch0==0f0 && (s.ch0=vmax/3f0)
    dt=p.courant/(n*vmax)
    s.method===:glm_constch ? min(dt,p.courant/(3f0*n*s.ch0)) : dt
end
"Params with a different glm_ch_scale (Params is immutable)."
with_ch_scale(p,scale)=G.Params(;Dict(f=>getfield(p,f) for f in fieldnames(G.Params))...,glm_ch_scale=Float32(scale))
"""
The kernel behind step_plm! for a (tile, precision, Riemann solver) triple that step_plm! does not list. The lines
before the launch are step_plm!'s own, copied; the launchers and kernels are the vendored ones, unedited.
"""
function plm_unlisted!(uold,unew,afield,p,dt::Float32,t::Float32,tv::Val{TB},hv::Val,rv::Val;ppm::Bool=false) where {TB}
    N=p.N;dx=G.dxof(p);pfl=G.pfloor(p)
    N%TB==0 || error("N=$N not divisible by the tile size $TB")
    ch=p.courant*dx/dt/Float32(G.NDIM)*p.glm_ch_scale
    glm_fac=p.glm_cp_coef>0 ? exp(-(ch*ch/(p.glm_cp_coef*p.boxlen*ch))*dt) : 1.0f0
    ramp=min(t/p.turb_T,1.0f0);nb=N÷TB
    (ppm ? G._plm_ppm_launch : G._plm_launch)(tv,hv,rv,192,nb,unew,uold,afield,p,pfl,dt,dx,ch,glm_fac,ramp,false)
    nothing
end
function advance!(s,p,dt,t;dt_cfl::Float32=dt)
    m=s.method
    if m===:glm
        G.step_tiled!(s.u,s.other,s.force,p,dt,Float32(t);use_hlld=false,do_turb=false,lean=false)
    elseif m===:glm_hll16
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll,ORIGINAL_STORAGE...)
    elseif m===:glm_hlld
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);tb=4,half=false,riemann=:hlld)
    elseif m===:glm_ppm
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll,recon=:ppm,ORIGINAL_STORAGE...)
    elseif m===:glm_constch
        # step_plm! sets c_h = courant*dx/(3dt)*scale; this scale makes c_h == ch0.
        pc=with_ch_scale(p,3f0*s.ch0*dt*p.N/p.courant)
        G.step_plm!(s.u,s.other,s.force,pc,dt,Float32(t);half=true,riemann=:hll,ORIGINAL_STORAGE...)
    elseif m===:glm_fix
        G.step_tiled!(s.u,s.other,s.force,p,dt,Float32(t);use_hlld=false,do_turb=false,lean=false,dt_cfl)
    elseif m===:glm_fix_hll16
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll,anchored=true,dt_cfl)
    elseif m===:glm_fix_hlld
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);tb=4,half=false,riemann=:hlld,dt_cfl)
    elseif m===:glm_fix_ppm
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll,recon=:ppm,anchored=true,dt_cfl)
    elseif m===:glm_fct16
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll,anchored=true,fct=true,fs=s.scratch[1])
    elseif m===:glm_fct32
        # tile 4 always: a Float32 tile of 6 needs 90 KB of shared memory (the limit is 48 KB)
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);tb=4,half=false,riemann=:hll,anchored=false,fct=true,fs=s.scratch[1])
    elseif m===:glm_hll32 || m===:glm_ppm32
        tile=p.N%6==0 ? Val(6) : Val(4)      # as step_plm! chooses
        plm_unlisted!(s.u,s.other,s.force,p,dt,Float32(t),tile,Val(false),Val(:hll);ppm=m===:glm_ppm32)
    elseif m===:glm_hlld_x
        plm_unlisted!(s.u,s.other,s.force,p,dt,Float32(t),Val(4),Val(false),Val(:hlld))
    end
    if is_glm(m)
        s.u,s.other=s.other,s.u
    else
        s.ct.step!(s.u,s.faces...,s.scratch...,dt,1f0/p.N,p.N)
    end
    CUDA.synchronize()
end

"The state as nine device arrays (rho, mx, my, mz, E, bx, by, bz, psi); CT: cell-centred B, psi = nothing."
function device_fields(s,n)
    is_glm(s.method) && return ntuple(c->view(s.u,:,:,:,c),9)
    centers!(s,n)
    (ntuple(c->view(s.u,:,:,:,c),5)...,s.scratch[1],s.scratch[2],s.scratch[3],nothing)
end
device_faces(s)=is_glm(s.method) ? nothing : s.faces
"Host copy of the z planes `z` (all by default) as (n, n, length(z), 9) plus CT faces."
function download(s,n;z=1:n)
    if is_glm(s.method);return (z==1:n ? Array(s.u) : Array(s.u[:,:,z,:])),nothing;end
    centers!(s,n);CUDA.synchronize()
    u=zeros(Float32,n,n,length(z),9);u[:,:,:,1:5].=Array(s.u[:,:,z,:])
    for d in 1:3;u[:,:,:,5+d].=Array(s.scratch[d][:,:,z]);end
    u,Tuple(Array(f[:,:,z]) for f in s.faces)
end
"Block average of the state to m^3 (host, (m, m, m, 9)); CT: cell-centred B, psi = 0."
function restricted(s,n,m)
    is_glm(s.method) && return Lean.restrict(s.u,m)
    centers!(s,n);r=zeros(Float32,m,m,m,9);r[:,:,:,1:5].=Lean.restrict(s.u,m)
    for d in 1:3;r[:,:,:,5+d].=Lean.restrict(s.scratch[d],m);end
    r
end

function write_snapshot(path,frame,u,b;stem=@sprintf("state_%03d",frame))
    open(joinpath(path,stem*".bin"),"w") do io;write(io,u);end
    b===nothing || for d in 1:3
        open(joinpath(path,@sprintf("face%d_%03d.bin",d,frame)),"w") do io;write(io,b[d]);end
    end
    stem*".bin"
end

problem_of(config)=get(config,"problem",config["name"])

# Checkpoint/restart (checkpoint_seconds > 0). Owner-node preemption requeues a job from the start;
# the requeued job finds <run>/checkpoint and resumes bit for bit. A checkpoint is written into
# checkpoint.tmp and renamed; state.toml is written last, so a directory without it is incomplete.
struct StopAfterCheckpoint <: Exception end
function save_checkpoint(s,dir,scalars)
    tmp=dir*".tmp";rm(tmp;force=true,recursive=true);mkpath(tmp)
    write(joinpath(tmp,"u.bin"),Array(s.u))
    is_glm(s.method) || for d in 1:3;write(joinpath(tmp,"face$d.bin"),Array(s.faces[d]));end
    record(joinpath(tmp,"state.toml"),scalars)
    rm(dir;force=true,recursive=true);mv(tmp,dir)
end
checkpoint_dir(path)=(d=joinpath(path,"checkpoint");isfile(joinpath(d,"state.toml")) ? d :
    isfile(joinpath(d*".tmp","state.toml")) ? d*".tmp" : nothing)
"""
Keep the header and the rows of a CSV for which `keep(row::Dict)` holds. A preemption can land after an
output was written but before the next checkpoint; the resumed run writes that output again.
"""
function truncate_rows(file,keep)
    isfile(file) && filesize(file)>0 || return
    lines=readlines(file);header=split(lines[1],',')
    ok(l)=(v=split(l,',');length(v)==length(header) && try keep(Dict(zip(header,v))) catch;false end)   # torn lines are dropped
    open(file*".tmp","w") do io;println(io,lines[1]);foreach(l->ok(l) && println(io,l),lines[2:end]);end
    mv(file*".tmp",file;force=true)
end
function load_checkpoint!(s,dir)
    u=Array{Float32}(undef,size(s.u)...);read!(joinpath(dir,"u.bin"),u);copyto!(s.u,u)
    if !is_glm(s.method)
        for d in 1:3;f=Array{Float32}(undef,size(s.faces[d])...);read!(joinpath(dir,"face$d.bin"),f);copyto!(s.faces[d],f);end
    end
    CUDA.synchronize();TOML.parsefile(joinpath(dir,"state.toml"))
end

function run_one(config,n,seed,method,rep,path)
    problem=problem_of(config);gamma=Problems.gamma_of(problem,config);planar=Problems.is_planar(problem,config)
    mach=get(config,"mach",2.0)
    # snapshot_frames_by_grid = {"512" = [...]}: fewer full states on large grids.
    snapframes=get(get(config,"snapshot_frames_by_grid",Dict()),string(n),get(config,"snapshot_frames",collect(0:config["frames"]-1)))
    # Runaway guard (dt collapse): max_steps_per_n * N steps, else max_steps.
    max_steps=get(config,"max_steps_per_n",0)>0 ? config["max_steps_per_n"]*n : get(config,"max_steps",100000)
    # Movie runs of 3D problems save only the z midplane at every output (slice_output = true).
    slice=get(config,"slice_output",false)===true && !planar;plane=slice ? n÷2+1 : 1
    zsave=planar ? (1:1) : slice ? (plane:plane) : (1:n)
    # Reduced outputs for large 3D runs (rep 1 only): z-midplane slices, block-averaged states, power spectra.
    nf=config["frames"];sf=get(config,"slice_frames",Int[]);slice_frames=planar ? Int[] : sf=="all" ? collect(0:nf-1) : sf
    restrict_to=planar ? Int[] : [m for m in get(config,"restrict_to",Int[]) if m<n]
    spectra_every=planar ? 0 : Int(get(config,"spectra_every",0))
    spectra_due(frame)=spectra_every>0 && (frame%spectra_every==0 || frame==nf-1)
    mkpath(path);p=G.Params(N=n,gamma=Float32(gamma),courant=Float32(config["cfl"]),
        smallr=1f-6,smallc=sqrt(Float32(gamma)),glm_ch_scale=Float32(get(config,"glm_ch_scale",1.0)),glm_cp_coef=Float32(get(config,"glm_cp_coef",.18)))
    C=ct_module(gamma);policy=get(config,"pressure_policy","stop")
    # Both solvers floor pressure at 1e-6 internally (GLM pfloor=smallr, CT SMALLP).
    pfloor=policy=="floor" ? 1f-6 : 0f0
    meta=Dict{String,Any}("status"=>"running","test"=>config["name"],"problem"=>problem,"N"=>n,"seed"=>seed,
        "method"=>string(method),"repeat"=>rep,"gamma"=>gamma,"gamma_float32"=>Float64(p.gamma),"cfl"=>Float64(p.courant),
        "parameters"=>Dict(k=>v for (k,v) in config if !(k in ("name","grids","seeds","frames","repeats","snapshot_frames","slurm_time"))),
        "precision"=>"Float32 state and shared storage; original GLM contains @fastmath",
        "solver"=>SOLVER_TEXT[method],"dt_jump"=>get(config,"dt_jump",Float64[]),
        "ct_module"=>string(nameof(C)),"ct_gamma_adapter"=>C===CTMHD ? "original module" : "vendor source re-parsed at load with only const GAMMA replaced; file unchanged",
        "glm_pressure_floor"=>"smallc=sqrt(gamma) so pfloor=smallr*smallc^2/gamma=smallr",
        "glm_ch_scale"=>Float64(p.glm_ch_scale),"glm_cp_coef"=>Float64(p.glm_cp_coef),"glm_cleaning_policy"=>get(config,"glm_ch_at_outputs","dt")=="cfl" ? "ch from the CFL dt; steps shortened to land on outputs keep that ch (glm_ch_at_outputs = cfl)" : "Original driver infers ch from the actual (possibly output-clipped) dt",
        "t_end_requested"=>config["t_end"],"field_order"=>["rho","mx","my","mz","E","Bx","By","Bz","psi"],
        "binary"=>"little-endian Float32, Julia/Fortran column major; x varies fastest",
        "planar"=>planar,"saved_shape"=>(planar || slice) ? [n,n,9] : [n,n,n,9],"periodic"=>true,
        "slice_frames"=>slice_frames,"slice_shape"=>[n,n,9],"slice_plane"=>n÷2+1,"restrict_to"=>restrict_to,
        "restricted_scope"=>"rstate_N<m>_<frame>.bin: (m,m,m,9) block averages at snapshot frames (CT: of the cell-centred B; psi 0)",
        "spectra_every"=>spectra_every,"spectra_scope"=>spectra_every>0 ? "spectra.csv: shell sums, sum_k E = <f^2>/2; rep 1; see src/spectra.jl" : "none",
        "diagnostics_scope"=>"on the device (src/lean.jl), full volume, Float64 accumulation",
        "snapshots_written"=>rep==1,"snapshot_frames"=>snapframes,
        "snapshot_scope"=>planar ? "z_index_1_plane; full-volume z variation recorded" : slice ? "z_midplane_index_$(plane)_of_$(n); diagnostics use the full volume" : "full_3d",
        "exact_reference"=>get(config,"exact_reference","none"),"frames"=>Any[],"slice"=>slice ? "z midplane" : "",
        "timing_scope"=>"Synchronized elapsed host wall time. stepping excludes CFL; solve=stepping+CFL. evolution includes diagnostics/transfers/output. Startup and warmup separate.",
        "floor_counts"=>"Observed cells at/below floors at saved times; internal floor activations are NOT instrumented",
        "pressure_policy"=>policy=="floor" ? "floor: nonpositive unfloored pressure tolerated; CFL uses p=max(p,1e-6); counts in diagnostics.csv" : "stop at first nonpositive unfloored pressure",
        "rejected_steps"=>0,"retry_policy"=>"No retries; failures retained")
    manifest=joinpath(path,"run.toml")
    ckpt_every=Float64(get(config,"checkpoint_seconds",0));ckdir=ckpt_every>0 ? checkpoint_dir(path) : nothing
    if ckdir!==nothing && isfile(manifest)
        # Resume: keep the outputs already recorded; timings of a resumed run span attempts.
        # Outputs 0 .. next_idx-2 were complete when the checkpoint was written; later ones are redone.
        old=TOML.parsefile(manifest);ck=TOML.parsefile(joinpath(ckdir,"state.toml"));kept=ck["next_idx"]-1
        meta["frames"]=filter(f->f["frame"]<kept,old["frames"])
        for f in ("diagnostics.csv","spectra.csv");truncate_rows(joinpath(path,f),r->parse(Int,r["frame"])<kept);end
        truncate_rows(joinpath(path,"dense_diagnostics.csv"),r->parse(Int,r["step"])<=ck["steps"])
        for k in ("first_nonpositive_pressure_time","allocated_array_bytes");haskey(old,k) && (meta[k]=old[k]);end
        meta["resumed_count"]=get(old,"resumed_count",0)+1
    else
        ckdir=nothing
    end
    meta["timing_valid"]=ckdir===nothing
    record(manifest,meta)
    try
        startup=time();h,b=Problems.initial(problem,n;seed,mach,par=config)
        init=Lean.diagnostics(ntuple(c->view(h,:,:,:,c),9),b;gamma)
        isfinite(init.div_face_max) && init.div_face_max<=Problems.div_tolerance(b,n) || error("Initial face divergence too large")
        # Planar problems: references and error norms on the z = 1 plane (identical norms, since z-invariance is
        # checked at every output), so a 512^2 test does not build and upload a 512^3 reference each time.
        hplane=planar ? h[:,:,1:1,:] : h
        reference(t)=Problems.reference(problem,n,t,hplane;seed,mach,par=config,plane=planar)
        errfields(q)=planar ? map(x->x===nothing ? nothing : view(x,:,:,1:1),q) : q
        s=allocate(method,h,b,gamma);CUDA.synchronize()
        arrs=is_glm(method) ? (s.u,s.other,s.signals,s.force,s.scratch...) : (s.u,s.faces...,s.scratch...,s.signals)
        meta["allocated_array_bytes"]=sum(sizeof,arrs)
        meta["memory_scope"]="allocated_array_bytes: explicit solver + adapter arrays. device_memory_*: device-wide used bytes (total - free, includes the CUDA context and pool) from CUDA.MemoryInfo, after warmup and peak over outputs"
        warm=time();dt=timestep(s,p,pfloor)
        for _ in 1:2;advance!(s,p,dt,0.0);end
        s.ch0=0f0      # constant-c_h methods re-measure c_h from the reset initial state
        Lean.diagnostics(device_fields(s,n),device_faces(s);gamma,problem);Lean.errors(errfields(device_fields(s,n)),hplane)
        spectra_every>0 && (q=device_fields(s,n);Spectra.spectra(q[1],q[2:4],q[6:8]))   # FFT plans, shell table
        meta["warmup_seconds"]=time()-warm;reset!(s,h,b)
        device_used()=(mi=CUDA.MemoryInfo();mi.total_bytes-mi.free_bytes)
        mi=CUDA.MemoryInfo();meta["device_memory_after_warmup_bytes"]=mi.total_bytes-mi.free_bytes
        meta["device_memory_total_bytes"]=mi.total_bytes
        mi.pool_reserved_bytes===nothing || (meta["cuda_pool_reserved_bytes"]=mi.pool_reserved_bytes;meta["cuda_pool_used_bytes"]=mi.pool_used_bytes)
        peak_device=meta["device_memory_after_warmup_bytes"]
        meta["startup_including_warmup_seconds"]=time()-startup
        steps=0;t=0.0;step_seconds=0.0;cfl_seconds=0.0;transfer_seconds=0.0;diag_seconds=0.0;output_seconds=0.0
        first_idx=1;resumed=nothing
        if ckdir!==nothing
            resumed=load_checkpoint!(s,ckdir)
            steps=resumed["steps"];t=resumed["t"];first_idx=resumed["next_idx"];s.ch0=Float32(resumed["ch0"])
            step_seconds,cfl_seconds,transfer_seconds,diag_seconds,output_seconds=(resumed[k] for k in
                ("step_seconds","cfl_seconds","transfer_seconds","diag_seconds","output_seconds"))
            println("RESUME $(config["name"]) $method N=$n repeat=$rep at t=$t step=$steps (output $(first_idx-1))");flush(stdout)
        end
        # Optional sudden timestep change (Tomida et al. 2026 sec. III.3): [t_start, factor, nsteps].
        jump=get(config,"dt_jump",nothing);jumped=resumed===nothing ? 0 : resumed["jumped"]
        # glm_ch_at_outputs = "cfl" (movie runs): a step shortened to land on an output keeps the c_h of
        # its CFL step. By default (as in the original driver) c_h = courant*dx/(3dt) jumps on such steps;
        # with hundreds of outputs, and a Float64 clock stepped in Float32 dt, the leftover steps can be
        # tiny and the inflated c_h made GLM runs fail.
        ch_cfl=get(config,"glm_ch_at_outputs","dt")=="cfl"
        # Repeat 1 logs every step from the jump until 300 steps after it ends: the psi and B
        # response lasts only a few hundred steps and falls between regular outputs.
        # A header written after the checkpoint survives truncation, so the files decide whether one exists.
        dense_file=joinpath(path,"dense_diagnostics.csv");spectra_file=joinpath(path,"spectra.csv")
        has_header(f)=resumed!==nothing && isfile(f) && filesize(f)>0
        dense_after=resumed===nothing ? 0 : resumed["dense_after"];dense_header=has_header(dense_file)
        spectra_header=has_header(spectra_file);spectra_seconds=resumed===nothing ? 0.0 : get(resumed,"spectra_seconds",0.0)
        resumed===nothing && rm(spectra_file;force=true)      # a fresh start in a directory left by a killed attempt
        dense_io=nothing;last_ckpt=time();stop_at_output=false
        elapsed_start=time();d0=resumed===nothing ? nothing : (mass=resumed["d0_mass"],energy=resumed["d0_energy"])
        open(joinpath(path,"diagnostics.csv"),resumed===nothing ? "w" : "a") do io
            for (idx,target) in enumerate(range(0,Float64(config["t_end"]);length=config["frames"]))
                idx<first_idx && continue
                while t<target-1e-10
                    steps<max_steps || error("Step limit reached ($max_steps)")
                    time()-elapsed_start<get(config,"max_run_seconds",1800) || error("Per-run wall limit reached")
                    started=time_ns();dt=timestep(s,p,pfloor)
                    if jump!==nothing && t>=jump[1] && jumped<jump[3]
                        dt*=Float32(jump[2]);jumped+=1
                    end
                    dt_step=dt;dt=Float32(min(dt,target-t));cfl_seconds+=(time_ns()-started)*1e-9
                    isfinite(dt) && dt>0 || error("Invalid timestep")
                    pe=(ch_cfl && dt<dt_step && is_glm(s.method) && s.method!==:glm_constch && !(s.method in FIX_METHODS)) ? with_ch_scale(p,p.glm_ch_scale*dt/dt_step) : p
                    started=time_ns();advance!(s,pe,dt,t;dt_cfl=dt_step);step_seconds+=(time_ns()-started)*1e-9
                    t+=Float64(dt);steps+=1
                    if rep==1 && jump!==nothing && jumped>0 && dense_after<300
                        jumped>=jump[3] && (dense_after+=1)
                        dense_io===nothing && (dense_io=open(dense_file,resumed===nothing ? "w" : "a"))
                        dd=Lean.diagnostics(device_fields(s,n),device_faces(s);gamma,problem)
                        drow=(step=steps,time=t,dt=Float64(dt),in_jump=jumped<jump[3] || dense_after==1,dd...)
                        dense_header || (println(dense_io,join(string.(keys(drow)),','));dense_header=true)
                        println(dense_io,join(values(drow),','))
                    end
                    if ckpt_every>0 && d0!==nothing && time()-last_ckpt>=ckpt_every
                        flush(io);dense_io===nothing || flush(dense_io)
                        save_checkpoint(s,joinpath(path,"checkpoint"),Dict{String,Any}("t"=>t,"steps"=>steps,"next_idx"=>idx,
                            "jumped"=>jumped,"dense_after"=>dense_after,"dense_header"=>dense_header,"ch0"=>Float64(s.ch0),
                            "d0_mass"=>d0.mass,"d0_energy"=>d0.energy,"step_seconds"=>step_seconds,"cfl_seconds"=>cfl_seconds,
                            "transfer_seconds"=>transfer_seconds,"diag_seconds"=>diag_seconds,"output_seconds"=>output_seconds,
                            "spectra_seconds"=>spectra_seconds,
                            "written_utc"=>string(now(UTC))))
                        last_ckpt=time()
                        # Resume check: stop each fresh run once, right after its first checkpoint ("1") or at the
                        # first output after it ("output": the resumed run must not record that output twice).
                        stop_mode=resumed===nothing ? get(ENV,"MHD_STOP_AFTER_CHECKPOINT","") : ""
                        stop_mode=="1" && throw(StopAfterCheckpoint())
                        stop_mode=="output" && (stop_at_output=true)
                    end
                end
                frame=idx-1
                started=time_ns();q=device_fields(s,n)
                d=Lean.diagnostics(q,device_faces(s);gamma,problem);d0===nothing && (d0=d)
                ref,label=reference(t);err=Lean.errors(errfields(q),ref);CUDA.synchronize();diag_seconds+=(time_ns()-started)*1e-9
                if rep==1 && spectra_due(frame)
                    started=time_ns();sp=Spectra.spectra(q[1],q[2:4],q[6:8])
                    open(spectra_file,"a") do sio
                        cols=[k for k in keys(sp) if k!==:k]
                        spectra_header || (println(sio,join(["frame","time","k",string.(cols)...],','));spectra_header=true)
                        for i in eachindex(sp.k);println(sio,join((frame,t,sp.k[i],(getfield(sp,c)[i] for c in cols)...),','));end
                    end
                    GC.gc(false)              # release the spectra temporaries (~15 GB at 512^3) before the next step
                    spectra_seconds+=(time_ns()-started)*1e-9
                end
                peak_device=max(peak_device,device_used());meta["device_memory_peak_bytes"]=peak_device
                row=(frame=idx-1,time=t,steps=steps,d...,mass_drift=abs(d.mass/d0.mass-1),energy_drift=abs(d.energy/d0.energy-1))
                idx==1 && println(io,join(string.(keys(row)),','))
                println(io,join(values(row),','));flush(io)
                stamp=Dict{String,Any}("frame"=>frame,"time"=>t,"steps"=>steps,"errors"=>err,"reference"=>label)
                rep==1 && spectra_due(frame) && (stamp["spectra"]=true)
                if rep==1 && frame in snapframes
                    started=time_ns();u,face=download(s,n;z=zsave);transfer_seconds+=(time_ns()-started)*1e-9
                    started=time_ns();stamp["file"]=write_snapshot(path,frame,u,face)
                    ref===nothing || (stamp["reference_file"]=write_snapshot(path,frame,ref[:,:,zsave,:],nothing;stem=@sprintf("ref_%03d",frame)))
                    isempty(restrict_to) || (stamp["restricted"]=Dict(string(m)=>write_snapshot(path,frame,restricted(s,n,m),nothing;
                        stem=@sprintf("rstate_N%d_%03d",m,frame)) for m in restrict_to))
                    output_seconds+=(time_ns()-started)*1e-9
                end
                if rep==1 && frame in slice_frames && !slice
                    started=time_ns();u,_=download(s,n;z=n÷2+1:n÷2+1)
                    stamp["slice"]=write_snapshot(path,frame,u,nothing;stem=@sprintf("slice_%03d",frame));output_seconds+=(time_ns()-started)*1e-9
                end
                push!(meta["frames"],stamp);meta["time_reached"]=t;meta["steps"]=steps
                meta["stepping_seconds"]=step_seconds;meta["cfl_seconds"]=cfl_seconds
                meta["solve_seconds"]=step_seconds+cfl_seconds;meta["transfer_seconds"]=transfer_seconds
                meta["diagnostics_seconds"]=diag_seconds;meta["binary_output_seconds"]=output_seconds;meta["spectra_seconds"]=spectra_seconds
                meta["evolution_wall_seconds"]=time()-elapsed_start;meta["dt_jump_steps_applied"]=jumped
                method===:glm_constch && (meta["glm_constant_ch"]=Float64(s.ch0))
                record(manifest,meta)
                d.finite && d.rho_min>0 && (d.pressure_min>0 || policy=="floor") || error("Invalid state at output $idx")
                if policy=="floor" && d.nonpositive_pressure_cells>0 && !haskey(meta,"first_nonpositive_pressure_time")
                    meta["first_nonpositive_pressure_time"]=t;record(manifest,meta)
                end
                problem=="uniform" && maximum(v for (k,v) in err if endswith(k,"absolute_Linf"))>1e-5 && error("Uniform state preservation failed")
                if planar
                    # Preserve evidence, but refuse to silently present a non-planar
                    # volume as a valid 2D reference comparison.
                    umax=maximum(x->Lean.fmax(y->abs(Float64(y)),x),filter(!isnothing,q))
                    d.z_variation<=1e-4*max(1.0,umax) || error("Out-of-plane variation exceeds tolerance")
                end
                stop_at_output && throw(StopAfterCheckpoint())
            end
        end
        dense_io===nothing || close(dense_io)
        rm(joinpath(path,"checkpoint");force=true,recursive=true);rm(joinpath(path,"checkpoint.tmp");force=true,recursive=true)
        meta["status"]="complete"
        meta["million_cell_steps_per_second"]=n^3*meta["steps"]/meta["stepping_seconds"]/1e6
        meta["finished_utc"]=string(now(UTC));record(manifest,meta)
    catch e
        e isa StopAfterCheckpoint && rethrow()     # test hook: leave the run resumable, as a preemption would
        meta["status"]="failed";meta["error"]=sprint(showerror,e,catch_backtrace());record(manifest,meta)
        println(stderr,"FAILED $(config["name"]) $method N=$n repeat=$rep: ",meta["error"])
    end
    meta
end

"Validate a parsed config; returns (grids, seeds)."
function check_config(config)
    problem_of(config) in Problems.NAMES || error("Unknown problem $(problem_of(config))")
    # cfl is the Courant number of the solver's own rule (the sum over the three directions), stable up to 1; the
    # sweep ran 0.25, the solver's default is 0.7
    0<config["cfl"]<=1 && config["t_end"]>0 && config["frames"]>=2 && config["repeats"]>=1 || error("Invalid CFL/time/frame/repetition settings")
    get(config,"pressure_policy","stop") in ("stop","floor") || error("pressure_policy must be \"stop\" or \"floor\"")
    get(config,"glm_ch_scale",1.0)>0 && get(config,"glm_cp_coef",.18)>=0 || error("Invalid cleaning parameters")
    all(f->0<=f<config["frames"],get(config,"snapshot_frames",[0])) || error("snapshot_frames outside 0:frames-1")
    all(fs->all(f->0<=f<config["frames"],fs),values(get(config,"snapshot_frames_by_grid",Dict()))) || error("snapshot_frames_by_grid outside 0:frames-1")
    get(config,"max_steps_per_n",0)>=0 || error("max_steps_per_n must be >= 0")
    all(m->m in METHODS,get(config,"methods",DEFAULT_METHODS)) || error("methods must be drawn from $(METHODS)")
    ct_module(Problems.gamma_of(problem_of(config),config))
    grids=config["grids"];all(n->n>=8 && n%4==0,grids) || error("Grids must be >=8 and multiples of 4")
    get(config,"checkpoint_seconds",0)>=0 || error("checkpoint_seconds must be >= 0")
    all(m->all(n->n<=m || n%m==0,grids),get(config,"restrict_to",Int[])) || error("every grid above a restrict_to size must be a multiple of it")
    sf=get(config,"slice_frames",Int[]);sf=="all" || all(f->0<=f<config["frames"],sf) || error("slice_frames must be \"all\" or within 0:frames-1")
    get(config,"spectra_every",0)>=0 || error("spectra_every must be >= 0")
    grids,get(config,"seeds",[42])
end

"Legacy array-index decoding (resume gate, smoke): with split_methods, method fastest, then grid, then seed."
function decode_index(config,index)
    grids,seeds=config["grids"],get(config,"seeds",[42])
    all_methods=get(config,"methods",DEFAULT_METHODS);split=get(config,"split_methods",false)===true
    nm=split ? length(all_methods) : 1
    0<=index<nm*length(grids)*length(seeds) || error("Array index out of range")
    task_methods=split ? [all_methods[index%nm+1]] : all_methods;index÷=nm
    grids[index%length(grids)+1],seeds[index÷length(grids)+1],task_methods
end

"""
    run_task(config_path, out, n, seed, methods; smoke=false) -> Bool

One grid and seed of one config with the listed methods and every repeat, into `out`. Resumable: with
checkpoint_seconds > 0 a non-empty `out` holding the same config continues (finished runs are kept).
Returns whether every run completed.
"""
function run_task(config_path,out,n,seed,task_methods;smoke=false)
    config=TOML.parsefile(config_path);grids,seeds=check_config(config)
    smoke || n in grids || error("Grid $n is not in the config's grids $grids")
    seed in seeds || error("Seed $seed is not in the config's seeds $seeds")
    all(m->m in METHODS,task_methods) || error("Unknown method in $task_methods")
    # The tiled production kernels use multi-cell halos and are not supported on
    # the old synthetic 8^3 smoke grid. 32^3 is the smallest campaign grid and
    # still keeps this execution check inexpensive.
    if smoke;n=32;config["t_end"]=min(config["t_end"],.01);config["frames"]=3;config["repeats"]=1;delete!(config,"snapshot_frames");delete!(config,"repeats_by_grid");end
    # Smoke runs must exercise the timestep-jump path (dense diagnostics) inside their short window.
    smoke && haskey(config,"dt_jump") && (config["dt_jump"]=[0.0,config["dt_jump"][2],5])
    out=abspath(out);resuming=isdir(out) && !isempty(readdir(out))
    if resuming
        get(config,"checkpoint_seconds",0)>0 || error("Output directory must be new or empty")
        isfile(joinpath(out,"input.toml")) && read(joinpath(out,"input.toml"),String)==read(config_path,String) ||
            error("Output directory holds a different config; refusing to resume")
        println("RESUMING in $out");flush(stdout)
    end
    mkpath(out);resuming || cp(config_path,joinpath(out,"input.toml");force=true)
    CUDA.functional() || error("No functional NVIDIA CUDA device. Run this inside a GPU allocation.")
    CUDA.allowscalar(false)
    open(joinpath(out,"cuda_versioninfo.txt"),"w") do io;CUDA.versioninfo(io);end
    files=vcat(readdir(joinpath(ROOT,"vendor");join=true),readdir(joinpath(ROOT,"src");join=true),[joinpath(ROOT,"env","Project.toml"),joinpath(ROOT,"env","Manifest.toml")])
    record(joinpath(out,resuming ? "environment_resumed_$(get(ENV,"SLURM_RESTART_COUNT","x"))_$(get(ENV,"SLURM_JOB_ID","x")).toml" : "environment.toml"),Dict("created_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
        "device"=>CUDA.name(CUDA.device()),"smoke"=>smoke,"actual_N"=>n,"actual_seed"=>seed,
        "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","interactive"),"slurm_array_task_id"=>get(ENV,"SLURM_ARRAY_TASK_ID","none"),
        "visible_devices"=>get(ENV,"CUDA_VISIBLE_DEVICES","unset"),
        "source_sha256"=>Dict(relpath(f,ROOT)=>bytes2hex(sha256(read(f))) for f in files if isfile(f))))
    cp(joinpath(ROOT,"env","Project.toml"),joinpath(out,"Project.toml");force=true)
    cp(joinpath(ROOT,"env","Manifest.toml"),joinpath(out,"Manifest.toml");force=true)
    results=Any[]
    # repeats_by_grid = {"256" = 3, "512" = 1}: fewer timing repeats on large grids.
    repeats=get(get(config,"repeats_by_grid",Dict()),string(n),config["repeats"])
    for rep in 1:repeats
        # Rotate the method order each repeat so no method always runs first on a warm GPU.
        methods=circshift(Symbol.(task_methods),-(rep-1))
        for method in methods
            path=joinpath(out,"$(method)_r$(rep)");manifest=joinpath(path,"run.toml")
            if resuming && isfile(manifest) && TOML.parsefile(manifest)["status"] in ("complete","failed")
                push!(results,TOML.parsefile(manifest));continue      # done in an earlier attempt
            end
            println("START $(config["name"]) N=$n seed=$seed $method repeat=$rep");flush(stdout)
            result=run_one(config,n,seed,method,rep,path);push!(results,result)
            record(joinpath(out,"suite.toml"),Dict("test"=>config["name"],"N"=>n,"seed"=>seed,"smoke"=>smoke,
                "status"=>"running","runs"=>[Dict("method"=>r["method"],"repeat"=>r["repeat"],"status"=>r["status"]) for r in results]))
            GC.gc();CUDA.reclaim()
        end
    end
    complete=all(r->r["status"]=="complete",results)
    record(joinpath(out,"suite.toml"),Dict("test"=>config["name"],"N"=>n,"seed"=>seed,"smoke"=>smoke,
        "status"=>complete ? "complete" : "failed","runs"=>[Dict("method"=>r["method"],"repeat"=>r["repeat"],"status"=>r["status"]) for r in results]))
    println(complete ? "DONE: $out" : "FAILED RUNS (kept): $out");flush(stdout)
    complete
end

"""
Parse a job's run list (tools/sweep.py writes it): one run per line, `CONFIG OUTDIR N SEED METHODS`
(METHODS comma-separated). Blank lines and lines starting with # are ignored.
"""
function read_runs(file)
    runs=NamedTuple[]
    for (i,l) in enumerate(readlines(file))
        l=strip(l);(isempty(l) || startswith(l,"#")) && continue
        f=split(l);length(f)==5 || error("$file line $i: expected CONFIG OUTDIR N SEED METHODS, got $(repr(l))")
        push!(runs,(config=String(f[1]),out=String(f[2]),n=parse(Int,f[3]),seed=parse(Int,f[4]),methods=String.(split(f[5],","))))
    end
    runs
end

# One Slurm job at a time in a run directory: OUT.lock beside OUT names the job working in it. Two jobs in one
# directory would resume from, overwrite and delete each other's files. The guard never stops a run through its
# own failure: outside Slurm, or if anything here goes wrong, the run goes ahead.
const WAITING_STATES=("PENDING","REQUEUED","REQUEUE_HOLD","REQUEUE_FED","RESV_DEL_HOLD","SPECIAL_EXIT","CONFIGURING")
"Whether Slurm reports this job as executing now. Anything else (waiting, gone, squeue unavailable) is no."
function job_running(job)
    for attempt in 1:3
        out=IOBuffer();err=IOBuffer()
        ok=try success(pipeline(`squeue -h -j $job -o %T`;stdout=out,stderr=err)) catch;return false end
        states=split(String(take!(out)))
        if !ok
            (occursin("Invalid job id",String(take!(err))) || attempt==3) && return false
            sleep(10);continue
        end
        (isempty(states) || all(in(WAITING_STATES),states)) && return false
        "COMPLETING" in states && attempt<3 && (sleep(20);continue)      # being stopped: look again
        return !("COMPLETING" in states)
    end
    false
end
lock_owner(lock)=(f=isfile(lock) ? split(read(lock,String)) : SubString{String}[];isempty(f) ? "" : String(f[1]))
"Mark OUT as worked on by this job; returns the id of another running job that holds it, else nothing."
function claim_run(out)
    me=get(ENV,"SLURM_JOB_ID","");isempty(me) && return nothing
    lock=out*".lock"
    try
        owner=lock_owner(lock)
        !isempty(owner) && owner!=me && job_running(owner) && return owner
        tmp="$lock.$me.$(getpid())";write(tmp,"$me $(gethostname()) $(now(UTC))\n");mv(tmp,lock;force=true)
        sleep(1)                                        # a job starting at the same moment may have written its own
        owner=lock_owner(lock)
        !isempty(owner) && owner!=me && job_running(owner) && return owner
    catch e
        println("run guard skipped (",sprint(showerror,e),")");flush(stdout)
    end
    nothing
end
function release_run(out)
    try
        lock=out*".lock";lock_owner(lock)==get(ENV,"SLURM_JOB_ID","-") && rm(lock;force=true)
    catch
    end
end

function main(args=ARGS)
    if !isempty(args) && args[1]=="--runs"
        # A packed job: every run in one Julia process (start-up and compilation once), each packed for download
        # as soon as it finishes. A failed run is recorded and the job goes on to the next.
        length(args)==2 || error("Usage: run_suite.jl --runs RUN_LIST")
        failed=String[]
        for r in read_runs(args[2])
            holder=claim_run(abspath(r.out))
            if holder!==nothing
                println("SKIPPED $(r.out): in use by running job $holder; nothing was touched");flush(stdout);continue
            end
            ok=try
                run_task(r.config,r.out,r.n,r.seed,r.methods)
            catch e
                println(stderr,"RUN FAILED $(r.out): ",sprint(showerror,e,catch_backtrace()));false
            finally
                release_run(abspath(r.out))
            end
            ok || push!(failed,r.out)
            dir=dirname(abspath(r.out))
            write(joinpath(dir,"exit_code.txt"),ok ? "0\n" : "1\n")
            run(ignorestatus(`bash $(joinpath(ROOT,"tools","pack_results.sh")) $dir`))
            GC.gc();CUDA.reclaim()
        end
        isempty(failed) || error("$(length(failed)) run(s) failed (results kept): "*join(failed,", "))
        return
    end
    length(args)>=2 || error("Usage: run_suite.jl CONFIG OUTPUT [array-index] [--smoke]  |  run_suite.jl --runs RUN_LIST")
    config=TOML.parsefile(args[1]);check_config(config)
    smoke="--smoke" in args;index=length(args)>=3 && !startswith(args[3],"--") ? parse(Int,args[3]) : 0
    n,seed,task_methods=decode_index(config,index)
    run_task(args[1],args[2],n,seed,task_methods;smoke) || error("One or more runs failed; partial artifacts retained in $(args[2])")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    try
        Suite.main()
    catch e
        e isa Suite.StopAfterCheckpoint ? exit(75) : rethrow()   # resume-check hook: stopped like a preemption
    end
end
