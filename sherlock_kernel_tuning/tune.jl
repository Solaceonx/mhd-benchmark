# Launch-geometry pilot. The original solver is loaded unchanged; no numerical
# approximation, precision change, or Courant-number change is counted as tuning.
using CUDA, SHA, Statistics, TOML, Dates

const BUNDLE = get(ENV, "SWEEP_BUNDLE", normpath(joinpath(@__DIR__, "..", "sherlock_sweep")))
const SOURCE = joinpath(BUNDLE, "vendor", "glmmhd_turb.jl")
const EXPECTED = "36a8431c0bbbb48f6d6a9f7b340d6a270221a439007d05084fdcfe5824b1b35e"
bytes2hex(sha256(read(SOURCE))) == EXPECTED || error("Solver checksum changed; review the pilot before running.")
include(SOURCE)
const G = GLMMHDTurb

struct Candidate
    family::Symbol
    tile::Int
    threads::Int
end

function candidates(n)
    out = Candidate[]
    for family in (:tiled_llf32, :plm_hll16, :plm_hlld32)
        tiles = family === :plm_hll16 && n % 6 == 0 ? (4, 6) : (4,)
        for tile in tiles, threads in (96, 128, 192, 256)
            push!(out, Candidate(family, tile, threads))
        end
    end
    out
end

function baseline(c, n)
    Candidate(c.family, c.family === :plm_hll16 && n % 6 == 0 ? 6 : 4, 192)
end

function step!(c, a, b, force, p, dt)
    if c.family === :tiled_llf32
        G.step_tiled!(a, b, force, p, dt, 0f0; use_hlld=false, do_turb=false, lean=false, nthreads=c.threads)
    else
        G.step_plm!(a, b, force, p, dt, 0f0; tb=c.tile, half=c.family === :plm_hll16,
                    riemann=c.family === :plm_hll16 ? :hll : :hlld, do_turb=false, nthreads=c.threads)
    end
    nothing
end

# A nonuniform periodic 3D input, defined for this performance pilot, rather than
# the uniform-state throughput helper. B is divergence-free analytically; the z
# velocity perturbation makes the evolution genuinely three-dimensional.
function initial_kernel!(u, n, gamma)
    l = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if l <= n^3
        i=(l-1)%n+1; j=((l-1)÷n)%n+1; k=(l-1)÷(n*n)+1
        x=(Float32(i)-0.5f0)/n; y=(Float32(j)-0.5f0)/n; z=(Float32(k)-0.5f0)/n
        twopi=2f0*Float32(pi); bscale=inv(sqrt(4f0*Float32(pi)))
        q=(25f0/(36f0*Float32(pi)), -sin(twopi*y), sin(twopi*x), 0.2f0*sin(twopi*z),
           5f0/(12f0*Float32(pi)), -bscale*sin(twopi*y), bscale*sin(2f0*twopi*x), 0f0, 0f0)
        state=G.prim2cons(q,gamma)
        @inbounds for v in 1:9; u[i,j,k,v]=state[v]; end
    end
    return
end

function evolve!(c, a, b, force, p, dt, steps)
    for _ in 1:steps
        step!(c, a, b, force, p, dt)
        a,b=b,a
    end
    a
end

function checked_result(c, seed, a, b, force, p, dt, steps)
    copyto!(a,seed); CUDA.synchronize()
    result=evolve!(c,a,b,force,p,dt,steps); CUDA.synchronize()
    all(isfinite,result) || error("Nonfinite output for $c")
    minimum(view(result,:,:,:,1)) > 0 || error("Nonpositive density for $c")
    # Check the actual stored energy, without primitive pressure flooring.
    rho=view(result,:,:,:,1); energy=view(result,:,:,:,5)
    kin=(view(result,:,:,:,2).^2 .+ view(result,:,:,:,3).^2 .+ view(result,:,:,:,4).^2)./(2f0.*rho)
    mag=(view(result,:,:,:,6).^2 .+ view(result,:,:,:,7).^2 .+ view(result,:,:,:,8).^2)./2f0
    minimum(energy .- kin .- mag) > 0 || error("Nonpositive internal energy for $c")
    result
end

function measure(c, seed, a, b, force, p, dt, steps, samples)
    gpu=Float64[]; wall=Float64[]
    for _ in 1:samples
        # Every sample evolves the same input; transfers and resets are untimed.
        copyto!(a,seed); CUDA.synchronize()
        start=time_ns()
        elapsed=CUDA.@elapsed evolve!(c,a,b,force,p,dt,steps)
        CUDA.synchronize()
        push!(wall,(time_ns()-start)*1e-9/steps); push!(gpu,elapsed/steps)
    end
    gpu,wall
end

function main()
    if "--check" in ARGS
        @assert length(candidates(192))==16
        @assert length(candidates(256))==12
        for n in (48,192,256), c in candidates(n)
            @assert n % c.tile == 0 && c.threads % 32 == 0
            @assert baseline(c,n).family == c.family
        end
        println("Pilot configuration checks pass; pinned source verified. No GPU work performed.")
        return
    end
    CUDA.functional() || error("This pilot needs an NVIDIA GPU. Use --check on the Mac.")
    grids=parse.(Int,split(get(ENV,"TUNE_GRIDS","96,192,384"),','))
    samples=parse(Int,get(ENV,"TUNE_SAMPLES","5")); steps=parse(Int,get(ENV,"TUNE_STEPS","10"))
    all(n->n>0 && n%4==0,grids) || error("Grids must be positive multiples of 4")
    samples>=3 && steps>=2 || error("Need at least three timing samples and two evolution steps")
    out=abspath(get(ENV,"TUNE_OUTPUT",joinpath(@__DIR__,"results",Dates.format(now(UTC),"yyyymmddTHHMMSS"))))
    mkpath(out); metadata=joinpath(out,"environment.toml")
    isfile(metadata) && error("Output already exists: $out; choose a fresh TUNE_OUTPUT")
    open(metadata,"w") do io
        TOML.print(io,Dict("source_sha256"=>EXPECTED,"julia"=>string(VERSION),"cuda_jl"=>string(pkgversion(CUDA)),
            "device"=>CUDA.name(CUDA.device()),"grids"=>grids,"samples"=>samples,"steps"=>steps,
            "created_utc"=>string(now(UTC)),"timing_scope"=>"integrator only; fixed dt; no CFL scan, forcing, diagnostics, transfers, or output",
            "input"=>"periodic sine fields with z-velocity perturbation; early-time trajectory; not a robustness suite"))
    end
    rejected=0
    open(joinpath(out,"timings.csv"),"w") do io
        println(io,"N,family,tile,threads,status,max_abs_difference,gpu_median_seconds,gpu_min_seconds,gpu_max_seconds,wall_median_seconds,Mcell_per_second")
        for n in grids
            println("GRID $n"); flush(stdout)
            p=G.Params(N=n,gamma=5f0/3f0,courant=0.25f0,glm_ch_scale=1f0)
            seed=CUDA.zeros(Float32,n,n,n,9); a=similar(seed); b=similar(seed)
            force=CUDA.zeros(Float32,1,1,1,1) # do_turb=false: never dereferenced
            @cuda threads=256 blocks=cld(n^3,256) initial_kernel!(seed,n,p.gamma)
            CUDA.synchronize(); dt=0.025f0/n
            for family in (:tiled_llf32,:plm_hll16,:plm_hlld32)
                group=filter(c->c.family===family,candidates(n))
                control=baseline(first(group),n)
                reference=copy(checked_result(control,seed,a,b,force,p,dt,steps))
                reference_seconds=NaN
                # Interleave the baseline between candidates to expose clock drift.
                for c in group
                    actual=checked_result(c,seed,a,b,force,p,dt,steps)
                    difference=Float64(maximum(abs.(actual .- reference)))
                    if difference != 0
                        rejected+=1
                        println(io,join((n,c.family,c.tile,c.threads,"rejected",difference,"","","","",""),','));flush(io)
                        println("REJECT $c: differs from same-scheme baseline by $difference");continue
                    end
                    for (status,trial) in (("control",control),("candidate",c))
                        # Separate warmup excludes compilation and first-launch setup.
                        copyto!(a,seed); evolve!(trial,a,b,force,p,dt,steps); CUDA.synchronize()
                        gpu,wall=measure(trial,seed,a,b,force,p,dt,steps,samples)
                        println(io,join((n,trial.family,trial.tile,trial.threads,status,0.0,median(gpu),minimum(gpu),maximum(gpu),median(wall),n^3/median(gpu)/1e6),','));flush(io)
                        status=="control" && (reference_seconds=median(gpu))
                        status=="candidate" && println("$c: $(round(n^3/median(gpu)/1e6;digits=1)) Mcell/s; $(round(reference_seconds/median(gpu);digits=3)) × adjacent baseline")
                    end
                end
                CUDA.unsafe_free!(reference)
            end
            for x in (seed,a,b,force); CUDA.unsafe_free!(x);end
            GC.gc();CUDA.reclaim()
        end
    end
    println("Results: $out")
    get(ENV,"TUNE_REQUIRE_ALL","0")=="1" && rejected>0 && error("$rejected candidates failed the exact-output smoke check; larger runs are blocked.")
    println("Launch tuning only. A candidate needs longer accuracy/robustness tests before use in scientific runs.")
end

abspath(PROGRAM_FILE)==(@__FILE__) && main()
