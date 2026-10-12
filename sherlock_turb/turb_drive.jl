# Driven turbulence with Tom Abel's kernels (the sweep bundle's vendor/glmmhd_turb.jl, unchanged): his
# Ornstein-Uhlenbeck forcing and his step functions with do_turb=true, in the set-up README.md describes.
#   julia --project=$SWEEP_BUNDLE/env turb_drive.jl CONFIG RESULTS [--only REGEX] [--redo-failed] [--backend cpu] [--no-smoke]
# RESULTS/forcing.toml (written by ramses_turb.py calibrate) gives the driving strength of each target Mach number.
# Finished runs are kept, also those in which the solver stopped (--redo-failed runs these again); an interrupted run
# starts again. --backend cpu runs his portable CPU path (step_multi!) instead of the GPU kernels: for testing this
# driver without a GPU, not for results.
using TOML, Printf, Dates, Random, SHA
using KernelAbstractions, CUDA
const KA=KernelAbstractions
const SWEEP=get(ENV,"SWEEP_BUNDLE",normpath(joinpath(@__DIR__,"..","sherlock_sweep")))
for f in ("problems.jl","errors.jl","lean.jl","spectra.jl");include(joinpath(SWEEP,"src",f));end
const VENDOR=joinpath(SWEEP,"vendor","glmmhd_turb.jl")
include(VENDOR)
const G=GLMMHDTurb
include(joinpath(@__DIR__,"measure.jl"))
const M=TurbMeasure

const METHOD_TEXT=Dict(
    "hydro"=>"step_hydro! tb=4 half=false riemann=:hll: 5-variable Euler kernel, Float32 tile, minmod PLM, Hancock, HLL",
    "hydro16"=>"step_hydro! tb=4 half=true riemann=:hll (the kernel's defaults): Float16 shared tile, minmod PLM, Hancock, HLL",
    "glm"=>"step_tiled! use_hlld=false with zero field: the original MHD kernel as the sweep runs it (Float32 tile TB=4, MonCen PLM, Hancock, LLF, c_h proportional to 1/dt, which enters its LLF wave speed)",
    "hydro_ramsesproj"=>"step_hydro! as hydro, with the forcing's projection as RAMSES has it at the pinned commit (direction of k from the absolute values of its components): a check of what that projection does, not a result",
    "cpu_multi"=>"step_multi! use_hlld=false on the CPU: test of this driver only")
# The forcing's Helmholtz projection: "exact" is Tom's (signed wave numbers); "ramses_released" what RAMSES's find_unitk
# does, where a fifth of the force's power is compressive at comp_frac = 0 (README.md).
projection_of(method)=method=="hydro_ramsesproj" ? "ramses_released" : "exact"

"A Mach number as it appears in names and in forcing.toml: 3 -> \"3\", 0.5 -> \"0p5\"."
mkey(mach)=replace(string(mach),"."=>"p")
"The gamma every code uses: the Float32 nearest to 1 + gamma_minus_one."
gamma32(config)=Float32(1+config["gamma_minus_one"])

function params(config,method,N,mach,a_rms,seed)
    g=gamma32(config)
    G.Params(N=N,gamma=g,courant=Float32(config["gpu"]["cfl"]),smallr=1f-6,smallc=sqrt(g),glm_ch_scale=1f0,glm_cp_coef=0.18f0,
        rho0=1f0,cs0=1f0,b0=0f0,
        turb_rms=Float32(a_rms),turb_T=Float32(1/(2mach)),turb_Ndt=Int(config["turb_ndt"]),
        comp_frac=Float32(config["comp_frac"]),turb_min_rho=0f0,turb_seed=UInt32(seed))
end

# The kernels take the time only to ramp the force up over the first turb_T. RAMSES has no ramp, so they are given
# turb_T: full force from the start.
function stepper(method,be,N)
    if method=="hydro" || method=="hydro_ramsesproj"
        (u,v,a,p,dt)->G.step_hydro!(u,v,a,p,dt,p.turb_T;do_turb=true,tb=4,half=false,riemann=:hll)
    elseif method=="hydro16"
        (u,v,a,p,dt)->G.step_hydro!(u,v,a,p,dt,p.turb_T;do_turb=true,tb=4,half=true,riemann=:hll)
    elseif method=="glm"
        (u,v,a,p,dt)->G.step_tiled!(u,v,a,p,dt,p.turb_T;use_hlld=false,do_turb=true,lean=false)
    elseif method=="cpu_multi"
        rec=KA.zeros(be,Float32,N,N,N,36);flx=KA.zeros(be,Float32,N,N,N,27);ct=KA.zeros(be,Float32,N,N,N)
        (u,v,a,p,dt)->G.step_multi!(u,v,rec,flx,ct,a,p,dt,p.turb_T,be;use_hlld=false,do_turb=true)
    else
        error("Unknown method $method")
    end
end
# The GLM kernels set their cleaning speed c_h from courant dx/dt, and c_h is a lower bound of the LLF wave speed.
# A step cut short at a change of the forcing field would get a larger c_h and so more diffusion; as the sweep's
# glm_ch_at_outputs="cfl" does, such a step keeps the c_h of the full step. It has no meaning for the hydro kernels.
is_glm(method)=startswith(method,"glm") || method=="cpu_multi"
with_ch_scale(p,scale)=G.Params(;Dict(f=>getfield(p,f) for f in fieldnames(G.Params))...,glm_ch_scale=Float32(scale))

# Signal speed of the kernels' timestep rule (sum over directions of |v| + fast speed), as the sweep's
# signals_kernel! computes it with the kernels' pressure floor; Inf where the state is not physical.
@inline function signal(r,mx,my,mz,e,b1,b2,b3,gamma::Float32,pfl::Float32)
    p=(gamma-1f0)*(e-(mx*mx+my*my+mz*mz)/(2f0*r)-(b1*b1+b2*b2+b3*b3)/2f0)
    isfinite(p) && (p=max(p,pfl))
    (isfinite(r) && r>0 && isfinite(p) && p>0 && isfinite(b1+b2+b3)) || return Inf32
    q=(r,mx/r,my/r,mz/r,p,b1,b2,b3,0f0)
    abs(q[2])+G.fast_speed(q,gamma,b1)+abs(q[3])+G.fast_speed(q,gamma,b2)+abs(q[4])+G.fast_speed(q,gamma,b3)
end
function signal_max(u,p)
    g=p.gamma;pfl=G.pfloor(p);q=M.fields(u)
    Lean.fmax((r,mx,my,mz,e,b1,b2,b3)->Float64(signal(r,mx,my,mz,e,b1,b2,b3,g,pfl)),q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8])
end

# Tom's forcing, with two things set from outside (nothing in his file is changed): the amplitude of each Fourier
# mode (his default is 1/k up to k = 32; here the spectrum the config names, as RAMSES defines it), and the strength:
# after every step of the process the field is scaled to an rms of exactly sqrt(3) a_rms on the forcing grid, which is
# what RAMSES does with turb_exact_rms. Both codes then push equally hard at every moment; only the pattern differs.
mutable struct Drive
    f::G.Forcing
    p::G.Params
    be
    a_rms::Float64
    dt::Float64          # time between two fields: turb_T/turb_Ndt
    k::Int               # fields used since t = 0; the current one holds until (k+1) dt
end
kfreq(i,n)=i-1<=n÷2 ? i-1 : i-1-n
function amplitude(spectrum,k)
    spectrum=="parabolic" && return 1<k<3 ? 1-(k-2)^2 : 0.0
    spectrum=="power_law" && return 1<=k<=G.TURB_GS÷2 ? 1/k : 0.0      # Tom's default
    error("Unknown spectrum $spectrum")
end
seed!(be,seed)=be isa KA.CPU ? Random.seed!(seed) : CUDA.seed!(seed)
function normalise!(d::Drive)
    a=d.f.afield;rms=sqrt(Float64(sum(abs2,a))/G.TURB_GS^3)
    (isfinite(rms) && rms>0) || error("The forcing field has rms $rms")
    a.*=Float32(sqrt(3)*d.a_rms/rms);KA.synchronize(d.be)
end
function Drive(p,be,a_rms,seed,spectrum;burn=5,projection="exact")
    seed!(be,seed);f=G.Forcing(p,be;amp=1f0);n=G.TURB_GS
    copyto!(f.sqp,Float32[amplitude(spectrum,sqrt(kfreq(i,n)^2+kfreq(j,n)^2+kfreq(l,n)^2)) for i in 1:n,j in 1:n,l in 1:n])
    if projection=="ramses_released"       # his unit wave vectors replaced by RAMSES's: absolute values of the components
        k=[abs(kfreq(i,n)) for i in 1:n];unit=zeros(Float32,3,n,n,n)
        for l in 1:n,j in 1:n,i in 1:n
            kk=sqrt(k[i]^2+k[j]^2+k[l]^2);1<=kk<=n÷2 && (unit[:,i,j,l].=(k[i],k[j],k[l])./kk)
        end
        copyto!(f.unitk,unit)
    else
        projection=="exact" || error("Unknown projection $projection")
    end
    for _ in 1:burn*p.turb_Ndt;G.ou_advance!(f,p,be);end        # a settled process at t = 0, not one that starts from zero
    d=Drive(f,p,be,a_rms,Float64(p.turb_T)/p.turb_Ndt,0);normalise!(d);d
end
advance!(d::Drive)=(G.ou_advance!(d.f,d.p,d.be);d.k+=1;normalise!(d))

function initial!(u,p,be)
    G.ic_uniform_kernel!(be)(u,p.rho0,p.rho0*p.cs0^2/p.gamma,p.b0,p.gamma;ndrange=(p.N,p.N,p.N));KA.synchronize(be)
end
record(path,data)=(open(io->TOML.print(io,data),path*".tmp","w");mv(path*".tmp",path;force=true))
device_name(be)=be isa KA.CPU ? "CPU $(gethostname()) x $(Threads.nthreads()) threads" : CUDA.name(CUDA.device())

"""
One run. Writes run.toml, series.csv, spectra.csv, pdf.csv, slices and the final state to `dir`; `light`: series.csv
only. Returns the run's record; a solver failure is recorded (status "failed"), not thrown.
"""
function run_one(be,config,method,N,mach,a_rms,seed,dir;tend_T=Float64(config["t_end_T"]),light=false,max_seconds=6*3600.0)
    p=params(config,method,N,mach,a_rms,seed);T=Float64(p.turb_T);gamma=Float64(p.gamma)
    sper=Int(config["series_per_T"]);fper=Int(config["frames_per_T"]);sper%fper==0 || error("series_per_T must be a multiple of frames_per_T")
    every=sper÷fper;dts=T/sper;nser=round(Int,tend_T*sper);slices_every=Int(get(config,"slices_every",4))
    # movie_per_T > 0: the density in the midplane this often, as rho_####.bin (Float32, n x n), for a movie
    mper=Int(get(config,"movie_per_T",0));(mper==0 || sper%mper==0) || error("series_per_T must be a multiple of movie_per_T")
    movie=mper==0 || light ? 0 : sper÷mper
    isdir(dir) && rm(dir;recursive=true);mkpath(dir)
    meta=Dict{String,Any}("code"=>"gpu","method"=>method,"solver"=>METHOD_TEXT[method],"N"=>N,"mach_target"=>mach,"a_rms"=>a_rms,
        "turb_T"=>T,"turb_ndt"=>p.turb_Ndt,"gamma"=>gamma,"comp_frac"=>Float64(p.comp_frac),"spectrum"=>config["spectrum"],
        "seed"=>seed,"forcing_projection"=>projection_of(method),"cfl"=>Float64(p.courant),"t_end"=>tend_T*T,"t_end_T"=>tend_T,"device"=>device_name(be),
        "precision"=>"Float32 state","vendor_sha256"=>bytes2hex(open(sha256,VENDOR)),"started_utc"=>string(now(UTC)),
        "status"=>"running","frames"=>Any[])
    movie>0 && (meta["movie_per_T"]=mper;meta["movie_frames"]=0)
    record(joinpath(dir,"run.toml"),meta)
    u=KA.zeros(be,Float32,N,N,N,9);other=KA.zeros(be,Float32,N,N,N,9)
    step! =stepper(method,be,N);w=M.Writer(dir);t=0.0;steps=0;step_seconds=0.0;started=time()
    try
        initial!(u,p,be);d=Drive(p,be,a_rms,seed,config["spectrum"];projection=projection_of(method))
        # Compile everything before the clock starts: two steps, the measurements, then the initial state again.
        dt=Float32(min(Float64(p.courant)/(N*signal_max(u,p)),d.dt))
        for _ in 1:2;step!(u,other,d.f.afield,p,dt);KA.synchronize(be);u,other=other,u;end
        M.stats(M.fields(u);gamma);light || Spectra.spectra(M.fields(u)[1],M.fields(u)[2:4],M.fields(u)[6:8])
        fill!(other,0f0);initial!(u,p,be);meta["warmup_seconds"]=time()-started
        # A time of series.csv is the first step at or past it: only a change of the forcing field cuts a step short
        # (as in Tom's run_turb), so the times are late by less than one step.
        max_steps=ceil(Int,100*N*tend_T);tol=1e-6*d.dt;j=0;glm=is_glm(method)
        while true
            while t>=(d.k+1)*d.dt-tol;advance!(d);end
            if t>=j*dts-tol
                row=M.stats(M.fields(u);gamma);isframe=j%every==0;frame=j÷every
                M.series!(w,(index=j,time=t,time_over_T=t/T,steps=steps,frame=isframe ? frame : -1),row)
                row.finite || error("The state is not finite")
                if isframe && !light
                    last=j==nser
                    push!(meta["frames"],M.frame!(w,frame,t,u,row.mass;slice=frame%slices_every==0 || last,state=last ? "state_final.bin" : nothing))
                end
                if movie>0 && j%movie==0
                    write(joinpath(dir,@sprintf("rho_%04d.bin",j÷movie)),Array(u[:,:,N÷2+1,1]));meta["movie_frames"]=j÷movie+1
                end
                j+=1;j>nser && break
            end
            steps<max_steps || error("Step limit reached ($max_steps)")
            time()-started<max_seconds || error("Per-run wall limit reached ($(max_seconds) s)")
            tic=time_ns();vmax=signal_max(u,p)
            (isfinite(vmax) && vmax>0) || error("Nonphysical or nonfinite state in the timestep scan")
            full=Float64(p.courant)/(N*vmax);dt=Float32(min(full,(d.k+1)*d.dt-t))
            (isfinite(dt) && dt>0) || error("Invalid timestep $dt")
            pe=glm && dt<0.999999f0*Float32(full) ? with_ch_scale(p,p.glm_ch_scale*dt/Float32(full)) : p
            step!(u,other,d.f.afield,pe,dt);KA.synchronize(be);u,other=other,u
            step_seconds+=(time_ns()-tic)*1e-9;t+=Float64(dt);steps+=1
        end
        meta["status"]="complete"
    catch e
        e isa InterruptException && rethrow()
        meta["status"]="failed";meta["reason"]=sprint(showerror,e);meta["failed_at_time"]=t;meta["failed_at_time_over_T"]=t/T
        @printf("   FAILED at t = %.4g (%.2f T), step %d: %s\n",t,t/T,steps,first(meta["reason"],300))
    finally
        close(w)
    end
    meta["steps"]=steps;meta["step_seconds"]=step_seconds;meta["wall_seconds"]=time()-started
    meta["cell_updates_per_second"]=step_seconds>0 ? Float64(N)^3*steps/step_seconds : 0.0
    meta["timing_scope"]="step_seconds: timestep scan + kernel with the force, synchronised; measurements, output and the forcing's own update excluded"
    meta["finished_utc"]=string(now(UTC));record(joinpath(dir,"run.toml"),meta)
    u=nothing;other=nothing;be isa KA.CPU || (GC.gc();CUDA.reclaim())
    meta
end

"Mean and half-to-half change of a series.csv column over time_over_T >= from."
function window(dir,column,from)
    lines=readlines(joinpath(dir,"series.csv"));length(lines)>1 || return (NaN,NaN)
    names=split(lines[1],',');ic=findfirst(==(column),names);it=findfirst(==("time_over_T"),names)
    rows=[split(l,',') for l in lines[2:end]];x=[parse(Float64,r[ic]) for r in rows if parse(Float64,r[it])>=from-1e-9]
    isempty(x) && return (NaN,NaN)
    h=length(x)÷2;m=sum(x)/length(x)
    m,h>0 ? (sum(x[h+1:end])/(length(x)-h))/(sum(x[1:h])/h)-1 : NaN
end

function main(args)
    option(name)=(i=findfirst(==(name),args);i===nothing ? nothing : args[i+1])
    taken=[i+1 for i in eachindex(args) if args[i] in ("--only","--backend")]
    positional=[args[i] for i in eachindex(args) if !startswith(args[i],"--") && !(i in taken)]
    length(positional)==2 || error("usage: turb_drive.jl CONFIG RESULTS [--only REGEX] [--backend cpu] [--no-smoke]")
    config=TOML.parsefile(positional[1]);results=positional[2];only=option("--only");cpu=option("--backend")=="cpu"
    # the trials of one configuration file in a directory of their own (two GPU jobs may run at once): smoke, smoke-highmach, ...
    stem=splitext(basename(positional[1]))[1];smoke="smoke"*(stem=="config" ? "" : "-"*replace(replace(stem,r"^config_"=>""),"_"=>"-"))
    be=cpu ? KA.CPU() : CUDABackend()
    cpu || (CUDA.functional() || error("No usable GPU: CUDA.functional() is false"))
    forcing=TOML.parsefile(joinpath(results,"forcing.toml"))
    methods=cpu ? ["cpu_multi"] : String.(config["gpu"]["methods"]);machs=config["machs"];from=Float64(config["steady_from_T"])
    println("== driven turbulence, ",device_name(be),", ",now(UTC)," UTC")
    amp(mach)=Float64(forcing["mach_$(mkey(mach))"]["a_rms"])
    # Every kernel once on a small box first: one that does not compile or run here is left out, not fatal.
    working=String[]
    if "--no-smoke" in args
        working=methods
    else
        for method in methods
            n=cpu ? 16 : 32;mach=machs[1];dir=joinpath(results,smoke,"gpu-$method")
            r=try
                run_one(be,config,method,n,mach,amp(mach),1,dir;tend_T=0.5,light=true)
            catch e
                e isa InterruptException && rethrow()
                Dict{String,Any}("status"=>"failed","reason"=>sprint(showerror,e))
            end
            if r["status"]=="complete"
                m,_=window(dir,"mach_vol",0.25);mass,_=window(dir,"mass",0.0)
                @printf("smoke %-10s OK    %d^3, %d steps to 0.5 T: Mach %.2f, mean density %.6f, %.3g cell updates/s\n",method,n,r["steps"],m,mass,r["cell_updates_per_second"])
                push!(working,method)
            else
                @printf("smoke %-10s FAILED: %s\n",method,first(get(r,"reason","?"),600))
            end
        end
    end
    isempty(working) && error("No kernel ran its smoke test: nothing to do")
    failed=0;ran=0
    matrix=[(mach,N,method,seed) for mach in machs for N in config["grids"] for method in working for seed in (cpu ? [1] : config["gpu"]["seeds"])]
    # The checks with RAMSES's projection run his Euler kernel, and only if that kernel passed its trial.
    cpu || "hydro" in working && append!(matrix,[(mach,N,method,seed) for mach in machs for N in get(config["gpu"],"check_grids",Int[])
        for method in get(config["gpu"],"check_methods",String[]) for seed in config["gpu"]["seeds"]])
    for (mach,N,method,seed) in matrix
        name="gpu-$method-M$(mkey(mach))-N$N-s$seed";dir=joinpath(results,name)
        only===nothing || occursin(Regex(only),name) || continue
        file=joinpath(dir,"run.toml")
        if isfile(file) && (st=get(TOML.parsefile(file),"status","");st=="complete" || (st=="failed" && !("--redo-failed" in args)))
            println("kept      $name ($st)");continue
        end
        r=run_one(be,config,method,N,mach,amp(mach),seed,dir);ran+=1
        if r["status"]=="complete"
            m,drift=window(dir,"mach_vol",from)
            @printf("complete  %-32s Mach %.2f from %g T (second half %+.1f%% against the first), %d steps, %.3g cell updates/s, %.0f s\n",
                name,m,from,100drift,r["steps"],r["cell_updates_per_second"],r["wall_seconds"])
        else
            failed+=1;println("failed    $name")
        end
        flush(stdout)
    end
    println("== ",ran," runs done here, ",failed," of them failed (a failed run is a result: see its run.toml), ",now(UTC)," UTC")
end
if abspath(PROGRAM_FILE)==@__FILE__
    main(ARGS)
end
