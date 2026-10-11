# Original GPU kernels plus test/measurement adapters. No solver equations edited.
module Suite
using CUDA, TOML, SHA, Dates, Printf, Statistics
include("problems.jl")
include("errors.jl")
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
const METHODS=("glm","glm_hll16","glm_hlld","glm_ppm","glm_constch","ct")
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
    :ct=>"original CTMHD.step!, 7 kernels, transverse-free Hancock, HLL, arithmetic edge EMF")
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
    if is_glm(method)
        u=CuArray(h)
        DeviceState(method,C,0f0,u,similar(u),(empty3,empty3,empty3),(),
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
    s.method===:ct && @cuda(threads=256,blocks=cld(n^3,256), s.ct.bcc_kernel!(s.scratch[1:3]...,s.faces...,n))
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
    b=s.method===:ct ? s.scratch[1:3] : (s.signals,s.signals,s.signals)
    @cuda threads=256 blocks=cld(n^3,256) signals_kernel!(s.signals,s.u,b...,s.method===:ct,n,p.gamma,pfloor)
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
function advance!(s,p,dt,t)
    m=s.method
    if m===:glm
        G.step_tiled!(s.u,s.other,s.force,p,dt,Float32(t);use_hlld=false,do_turb=false,lean=false)
    elseif m===:glm_hll16
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll)
    elseif m===:glm_hlld
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);tb=4,half=false,riemann=:hlld)
    elseif m===:glm_ppm
        G.step_plm!(s.u,s.other,s.force,p,dt,Float32(t);half=true,riemann=:hll,recon=:ppm)
    elseif m===:glm_constch
        # step_plm! sets c_h = courant*dx/(3dt)*scale; this scale makes c_h == ch0.
        pc=with_ch_scale(p,3f0*s.ch0*dt*p.N/p.courant)
        G.step_plm!(s.u,s.other,s.force,pc,dt,Float32(t);half=true,riemann=:hll)
    end
    if is_glm(m)
        s.u,s.other=s.other,s.u
    else
        s.ct.step!(s.u,s.faces...,s.scratch...,dt,1f0/p.N,p.N)
    end
    CUDA.synchronize()
end

function download(s,n)
    if is_glm(s.method);return Array(s.u),nothing;end
    centers!(s,n);CUDA.synchronize()
    u=zeros(Float32,n,n,n,9);u[:,:,:,1:5].=Array(s.u)
    for d in 1:3;u[:,:,:,5+d].=Array(s.scratch[d]);end
    u,Tuple(Array.(s.faces))
end

function write_snapshot(path,frame,u,b,full;stem=@sprintf("state_%03d",frame))
    state=full ? u : copy(@view u[:,:,1,:])
    open(joinpath(path,stem*".bin"),"w") do io;write(io,state);end
    if b!==nothing
        for d in 1:3
            face=full ? b[d] : copy(@view b[d][:,:,1])
            open(joinpath(path,@sprintf("face%d_%03d.bin",d,frame)),"w") do io;write(io,face);end
        end
    end
    stem*".bin"
end

problem_of(config)=get(config,"problem",config["name"])

function run_one(config,n,seed,method,rep,path)
    problem=problem_of(config);gamma=Problems.gamma_of(problem,config);planar=Problems.is_planar(problem,config)
    mach=get(config,"mach",2.0);snapframes=get(config,"snapshot_frames",collect(0:config["frames"]-1))
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
        "glm_ch_scale"=>Float64(p.glm_ch_scale),"glm_cp_coef"=>Float64(p.glm_cp_coef),"glm_cleaning_policy"=>"Original driver infers ch from the actual (possibly output-clipped) dt",
        "t_end_requested"=>config["t_end"],"field_order"=>["rho","mx","my","mz","E","Bx","By","Bz","psi"],
        "binary"=>"little-endian Float32, Julia/Fortran column major; x varies fastest",
        "planar"=>planar,"saved_shape"=>planar ? [n,n,9] : [n,n,n,9],"periodic"=>true,
        "snapshots_written"=>rep==1,"snapshot_frames"=>snapframes,
        "snapshot_scope"=>planar ? "z_index_1_plane; full-volume z variation recorded" : "full_3d",
        "exact_reference"=>get(config,"exact_reference","none"),"frames"=>Any[],
        "timing_scope"=>"Synchronized elapsed host wall time. stepping excludes CFL; solve=stepping+CFL. evolution includes diagnostics/transfers/output. Startup and warmup separate.",
        "floor_counts"=>"Observed cells at/below floors at saved times; internal floor activations are NOT instrumented",
        "pressure_policy"=>policy=="floor" ? "floor: nonpositive unfloored pressure tolerated; CFL uses p=max(p,1e-6); counts in diagnostics.csv" : "stop at first nonpositive unfloored pressure",
        "rejected_steps"=>0,"retry_policy"=>"No retries; failures retained")
    manifest=joinpath(path,"run.toml");record(manifest,meta)
    try
        startup=time();h,b=Problems.initial(problem,n;seed,mach,par=config)
        init=Problems.diagnostics(h,b;gamma)
        isfinite(init.div_face_max) && init.div_face_max<=Problems.div_tolerance(b,n) || error("Initial face divergence too large")
        reference(t)=Problems.reference(problem,n,t,h;seed,mach,par=config)
        s=allocate(method,h,b,gamma);CUDA.synchronize()
        arrs=is_glm(method) ? (s.u,s.other,s.signals,s.force) : (s.u,s.faces...,s.scratch...,s.signals)
        meta["allocated_array_bytes"]=sum(sizeof,arrs)
        meta["memory_scope"]="Explicit live solver + adapter arrays; not peak device memory or CUDA context/pool usage"
        warm=time();dt=timestep(s,p,pfloor)
        for _ in 1:2;advance!(s,p,dt,0.0);end
        s.ch0=0f0      # constant-c_h methods re-measure c_h from the reset initial state
        warm_u,warm_faces=download(s,n)
        Problems.diagnostics(warm_u,warm_faces;gamma)
        solution_errors(h,reference(0.0)[1])
        meta["warmup_seconds"]=time()-warm;reset!(s,h,b)
        meta["startup_including_warmup_seconds"]=time()-startup
        steps=0;t=0.0;step_seconds=0.0;cfl_seconds=0.0;transfer_seconds=0.0;diag_seconds=0.0;output_seconds=0.0
        # Optional sudden timestep change (Tomida et al. 2026 sec. III.3): [t_start, factor, nsteps].
        jump=get(config,"dt_jump",nothing);jumped=0
        elapsed_start=time();d0=nothing
        open(joinpath(path,"diagnostics.csv"),"w") do io
            for (idx,target) in enumerate(range(0,Float64(config["t_end"]);length=config["frames"]))
                while t<target-1e-10
                    steps<get(config,"max_steps",100000) || error("Step limit reached")
                    time()-elapsed_start<get(config,"max_run_seconds",1800) || error("Per-run wall limit reached")
                    started=time_ns();dt=timestep(s,p,pfloor)
                    if jump!==nothing && t>=jump[1] && jumped<jump[3]
                        dt*=Float32(jump[2]);jumped+=1
                    end
                    dt=Float32(min(dt,target-t));cfl_seconds+=(time_ns()-started)*1e-9
                    isfinite(dt) && dt>0 || error("Invalid timestep")
                    started=time_ns();advance!(s,p,dt,t);step_seconds+=(time_ns()-started)*1e-9
                    t+=Float64(dt);steps+=1
                end
                started=time_ns();u,face=download(s,n);transfer_seconds+=(time_ns()-started)*1e-9
                started=time_ns();d=Problems.diagnostics(u,face;gamma);d0===nothing && (d0=d)
                ref,label=reference(t);err=solution_errors(u,ref);diag_seconds+=(time_ns()-started)*1e-9
                row=(frame=idx-1,time=t,steps=steps,d...,mass_drift=abs(d.mass/d0.mass-1),energy_drift=abs(d.energy/d0.energy-1))
                idx==1 && println(io,join(string.(keys(row)),','))
                println(io,join(values(row),','));flush(io)
                stamp=Dict{String,Any}("frame"=>idx-1,"time"=>t,"steps"=>steps,"errors"=>err,"reference"=>label)
                if rep==1 && (idx-1) in snapframes
                    started=time_ns();stamp["file"]=write_snapshot(path,idx-1,u,face,!planar)
                    ref===nothing || (stamp["reference_file"]=write_snapshot(path,idx-1,ref,nothing,!planar;stem=@sprintf("ref_%03d",idx-1)))
                    output_seconds+=(time_ns()-started)*1e-9
                end
                push!(meta["frames"],stamp);meta["time_reached"]=t;meta["steps"]=steps
                meta["stepping_seconds"]=step_seconds;meta["cfl_seconds"]=cfl_seconds
                meta["solve_seconds"]=step_seconds+cfl_seconds;meta["transfer_seconds"]=transfer_seconds
                meta["diagnostics_seconds"]=diag_seconds;meta["binary_output_seconds"]=output_seconds
                meta["evolution_wall_seconds"]=time()-elapsed_start;meta["dt_jump_steps_applied"]=jumped
                method===:glm_constch && (meta["glm_constant_ch"]=Float64(s.ch0))
                record(manifest,meta)
                d.finite && d.rho_min>0 && (d.pressure_min>0 || policy=="floor") || error("Invalid state at output $idx")
                if policy=="floor" && d.nonpositive_pressure_cells>0 && !haskey(meta,"first_nonpositive_pressure_time")
                    meta["first_nonpositive_pressure_time"]=t;record(manifest,meta)
                end
                problem=="uniform" && maximum(abs,u.-h)>1e-5 && error("Uniform state preservation failed")
                if planar
                    # Preserve evidence, but refuse to silently present a non-planar
                    # volume as a valid 2D reference comparison.
                    d.z_variation<=1e-4*max(1.0,maximum(abs,u)) || error("Out-of-plane variation exceeds tolerance")
                end
            end
        end
        meta["status"]="complete"
        meta["million_cell_steps_per_second"]=n^3*meta["steps"]/meta["stepping_seconds"]/1e6
        meta["finished_utc"]=string(now(UTC));record(manifest,meta)
    catch e
        meta["status"]="failed";meta["error"]=sprint(showerror,e,catch_backtrace());record(manifest,meta)
        println(stderr,"FAILED $(config["name"]) $method N=$n repeat=$rep: ",meta["error"])
    end
    meta
end

"Validate a parsed config; returns (grids, seeds)."
function check_config(config)
    problem_of(config) in Problems.NAMES || error("Unknown problem $(problem_of(config))")
    0<config["cfl"]<=.5 && config["t_end"]>0 && config["frames"]>=2 && config["repeats"]>=1 || error("Invalid CFL/time/frame/repetition settings")
    get(config,"pressure_policy","stop") in ("stop","floor") || error("pressure_policy must be \"stop\" or \"floor\"")
    get(config,"glm_ch_scale",1.0)>0 && get(config,"glm_cp_coef",.18)>=0 || error("Invalid cleaning parameters")
    all(f->0<=f<config["frames"],get(config,"snapshot_frames",[0])) || error("snapshot_frames outside 0:frames-1")
    all(m->m in METHODS,get(config,"methods",DEFAULT_METHODS)) || error("methods must be drawn from $(METHODS)")
    ct_module(Problems.gamma_of(problem_of(config),config))
    grids=config["grids"];all(n->n>=8 && n%4==0,grids) || error("Grids must be >=8 and multiples of 4")
    grids,get(config,"seeds",[42])
end

function main(args=ARGS)
    length(args)>=2 || error("Usage: run_suite.jl CONFIG OUTPUT [array-index] [--smoke]")
    config=TOML.parsefile(args[1]);grids,seeds=check_config(config)
    smoke="--smoke" in args;index=length(args)>=3 && !startswith(args[3],"--") ? parse(Int,args[3]) : 0
    0<=index<length(grids)*length(seeds) || error("Array index out of range")
    n=grids[index%length(grids)+1];seed=seeds[index÷length(grids)+1]
    # The tiled production kernels use multi-cell halos and are not supported on
    # the old synthetic 8^3 smoke grid. 32^3 is the smallest campaign grid and
    # still keeps this execution check inexpensive.
    if smoke;n=32;config["t_end"]=min(config["t_end"],.01);config["frames"]=3;config["repeats"]=1;delete!(config,"snapshot_frames");end
    out=abspath(args[2]);isdir(out) && !isempty(readdir(out)) && error("Output directory must be new or empty")
    mkpath(out);cp(args[1],joinpath(out,"input.toml");force=true)
    CUDA.functional() || error("No functional NVIDIA CUDA device. Run this inside a GPU allocation.")
    CUDA.allowscalar(false)
    open(joinpath(out,"cuda_versioninfo.txt"),"w") do io;CUDA.versioninfo(io);end
    files=vcat(readdir(joinpath(ROOT,"vendor");join=true),readdir(joinpath(ROOT,"src");join=true),[joinpath(ROOT,"env","Project.toml"),joinpath(ROOT,"env","Manifest.toml")])
    record(joinpath(out,"environment.toml"),Dict("created_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
        "device"=>CUDA.name(CUDA.device()),"smoke"=>smoke,"actual_N"=>n,"actual_seed"=>seed,
        "slurm_job_id"=>get(ENV,"SLURM_JOB_ID","interactive"),"slurm_array_task_id"=>get(ENV,"SLURM_ARRAY_TASK_ID","none"),
        "visible_devices"=>get(ENV,"CUDA_VISIBLE_DEVICES","unset"),
        "source_sha256"=>Dict(relpath(f,ROOT)=>bytes2hex(sha256(read(f))) for f in files if isfile(f))))
    cp(joinpath(ROOT,"env","Project.toml"),joinpath(out,"Project.toml");force=true)
    cp(joinpath(ROOT,"env","Manifest.toml"),joinpath(out,"Manifest.toml");force=true)
    results=Any[]
    for rep in 1:config["repeats"]
        # Rotate the method order each repeat so no method always runs first on a warm GPU.
        methods=Symbol.(get(config,"methods",DEFAULT_METHODS));methods=circshift(methods,-(rep-1))
        for method in methods
            println("START $(config["name"]) N=$n seed=$seed $method repeat=$rep");flush(stdout)
            result=run_one(config,n,seed,method,rep,joinpath(out,"$(method)_r$(rep)"));push!(results,result)
            record(joinpath(out,"suite.toml"),Dict("test"=>config["name"],"N"=>n,"seed"=>seed,"smoke"=>smoke,
                "status"=>"running","runs"=>[Dict("method"=>r["method"],"repeat"=>r["repeat"],"status"=>r["status"]) for r in results]))
            GC.gc();CUDA.reclaim()
        end
    end
    complete=all(r->r["status"]=="complete",results)
    record(joinpath(out,"suite.toml"),Dict("test"=>config["name"],"N"=>n,"seed"=>seed,"smoke"=>smoke,
        "status"=>complete ? "complete" : "failed","runs"=>[Dict("method"=>r["method"],"repeat"=>r["repeat"],"status"=>r["status"]) for r in results]))
    complete || error("One or more runs failed; partial artifacts retained in $out")
    println("DONE: $out")
end
end
if abspath(PROGRAM_FILE)==@__FILE__;Suite.main();end
