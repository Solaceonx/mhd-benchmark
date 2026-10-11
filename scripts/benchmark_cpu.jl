# Repeated compute-only OT measurements, with a common precomputed CFL schedule.
# Run without another simulation or renderer; results are local machine samples.
include(joinpath(@__DIR__,"mhd_2d_cpu.jl"))
using .MHD2DCPU, TOML, SHA, Dates, Statistics
const M=MHD2DCPU
function schedule(p)
    g=M.initialize(p;method=:glm);c=M.initialize(p;method=:ct)
    wg=M.Workspace(p.n);wc=M.Workspace(p.n);steps=Tuple{Float32,Float32}[];t=0.0
    for target in range(0,.5;length=21)
        while t<target-1e-10
            ch=max(M.speeds(g,p)...,M.speeds(c,p)...)
            dt=Float32(min(p.cfl/(p.n*2ch),target-t))
            M.step!(g,wg,p,dt,ch,:glm);M.step!(c,wc,p,dt,ch,:ct)
            push!(steps,(dt,ch));t+=dt
        end
    end
    steps,Dict(:glm=>g.u,:ct=>c.u)
end
function replay!(s,w,p,method,steps)
    for (dt,ch) in steps;M.step!(s,w,p,dt,ch,method);end
end
function benchmark(root;repeats=5)
    isdir(root) && !isempty(readdir(root)) && error("Output directory must be empty")
    mkpath(root)
    open(joinpath(root,"samples.csv"),"w") do io
        println(io,"N,threads,repeat,order,method,steps,seconds,cell_updates_per_second")
        for n in (64,128,256)
            p=M.Params(;n);steps,expected=schedule(p)
            workspace=M.Workspace(n)
            # Full replay warms the exact measured call path and validates it.
            for method in (:glm,:ct)
                s=M.initialize(p;method);replay!(s,workspace,p,method,steps)
                @assert s.u==expected[method]
            end
            for rep in 1:repeats
                for (order,method) in enumerate(isodd(rep) ? (:glm,:ct) : (:ct,:glm))
                    s=M.initialize(p;method);GC.gc()
                    start=time_ns();replay!(s,workspace,p,method,steps);seconds=(time_ns()-start)*1e-9
                    @assert s.u==expected[method] # identical work and result for every sample
                    println(io,join((n,Threads.nthreads(),rep,order,method,length(steps),seconds,n^2*length(steps)/seconds),','));flush(io)
                end
            end
            println("Benchmark $n² complete, $(Threads.nthreads()) threads");flush(stdout)
        end
    end
    metadata=Dict("date"=>string(now()),"julia_version"=>string(VERSION),"cpu"=>Sys.CPU_NAME,
        "threads"=>Threads.nthreads(),"logical_cpus"=>Sys.CPU_THREADS,"repeats"=>repeats,
        "scope"=>"RK steps, fluxes, stage validity checks, GLM damping / CT face updates; GC enabled",
        "excluded"=>"JIT, state allocation, CFL schedule generation, diagnostics, disk output",
        "policy"=>"Full warm replay, fresh identical initial state, common CFL schedule, alternating method order, GC before samples, exact final-state equality",
        "limits"=>"Compute-only local timing, not end-to-end or equal-error cost. CPU frequency and background OS work uncontrolled; 1- and 4-thread sessions occur at different times.",
        "solver_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"mhd_2d_cpu.jl")))))
    open(joinpath(root,"metadata.toml"),"w") do io;TOML.print(io,metadata);end
end
benchmark(length(ARGS)>0 ? abspath(ARGS[1]) : joinpath(@__DIR__,"out","cpu_benchmark_t$(Threads.nthreads())"))
