# First GPU runs of Tom Abel's NestedMHD (github.com/yipihey/NestedTurbulence, core/), on one H100:
#   julia --project=env gpu_check.jl NESTED_DIR OUT_DIR
# 1. agreement: the same nested Orszag–Tang run on the CPU (PoCL) and the GPU, level by level;
# 2. speed: cell updates per second, uniform grids and a 3-level nest, CT and GLM, Float32 and Float64;
# 3. driven Mach-10 turbulence in 3-D (Tom's run_mach10.jl set-up, 3-D driving) for one crossing time, nested and
#    uniform at the finest level's resolution: wall time, Mach number, div B.
# Writes OUT_DIR/gpu_check.toml and prints as it goes.
using CUDA, KernelAbstractions, Printf, TOML
const KA = KernelAbstractions
const NT = ARGS[1]; const OUT = ARGS[2]; mkpath(OUT)
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
include(joinpath(@__DIR__, "setups.jl"))
const GPU = CUDABackend()
out = Dict{String,Any}("gpu" => CUDA.name(CUDA.device()), "cuda" => string(pkgversion(CUDA)),
                       "kernelabstractions" => string(pkgversion(KA)), "julia" => string(VERSION),
                       "nested_commit" => readchomp(joinpath(NT, "COMMIT")), "threads" => Threads.nthreads())
save() = open(io -> TOML.print(io, out), joinpath(OUT, "gpu_check.toml"), "w")
say(x...) = (println(x...); flush(stdout))
say("== ", out["gpu"], ", CUDA.jl ", out["cuda"], ", KernelAbstractions ", out["kernelabstractions"], ", NestedMHD ", out["nested_commit"][1:7])

# ---- 1. agreement --------------------------------------------------------------------------------------------
levels_U(H) = [Array(P.U) for P in H.patches]
agree = Dict{String,Any}[]
for (mode, riemann, T) in ((:ct, :hll, Float64), (:glm, :hll, Float64), (:ct, :hlld, Float32), (:glm, :hll, Float32))
    a = ot_run(KA.CPU(); N=32, nlev=3, mode, riemann, T, tend=0.05)
    b = ot_run(GPU; N=32, nlev=3, mode, riemann, T, tend=0.05)
    ua, ub = levels_U(a), levels_U(b)
    rel = [maximum(abs.(x .- y)) / maximum(abs.(x)) for (x, y) in zip(ua, ub)]
    r = Dict("mode" => string(mode), "riemann" => string(riemann), "T" => string(T), "steps_cpu" => a.nsteps, "steps_gpu" => b.nsteps,
             "max_rel_diff_by_level" => rel, "divb_gpu" => max_divb(b), "finite" => all(u -> all(isfinite, u), ub))
    push!(agree, r)
    say(@sprintf("agree %-3s %-4s %-7s steps cpu %s gpu %s  max |ΔU|/max|U| by level %s  divB %s", mode, riemann, T,
                 a.nsteps, b.nsteps, join((@sprintf("%.1e", x) for x in rel), " "), join((@sprintf("%.0e", x) for x in r["divb_gpu"]), " ")))
end
out["agreement"] = agree; save()

# ---- 2. speed ---------------------------------------------------------------------------------------------------
function timed(H, tend)
    init!(H, ot3; vecpot=aot3)
    evolve!(H, tend / 10); KA.synchronize(H.backend)               # compile and warm up
    init!(H, ot3; vecpot=aot3)
    wall = @elapsed (evolve!(H, tend); KA.synchronize(H.backend))
    upd = sum(prod(P.n) * s for (P, s) in zip(H.patches, H.nsteps))
    upd / wall, wall, copy(H.nsteps)
end
speed = Dict{String,Any}[]
for (label, N, nlev, T) in (("uniform 128³", 128, 1, Float32), ("uniform 256³", 256, 1, Float32), ("uniform 512³", 512, 1, Float32),
                            ("uniform 256³", 256, 1, Float64), ("nest 3 × 128³", 128, 3, Float32), ("nest 3 × 128³", 128, 3, Float64)),
    mode in (:ct, :glm)
    GC.gc(); CUDA.reclaim()
    H = nested_central(N, nlev; n=N, mode=mode, riemann=:hll, T=T, backend=GPU)
    rate, wall, steps = timed(H, N == 512 ? 0.01 : 0.02)
    push!(speed, Dict("setup" => label, "mode" => string(mode), "T" => string(T), "cell_updates_per_s" => rate, "wall" => wall, "steps" => steps))
    say(@sprintf("speed %-14s %-3s %-7s %7.0f Mcell/s  (%d steps on the finest level, %.1f s)", label, mode, T, rate / 1e6, steps[end], wall))
    H = nothing
end
out["speed"] = speed; save()

# ---- 3. driven Mach-10 turbulence, one crossing time -------------------------------------------------------------
driven = Dict{String,Any}[]
for (label, N, nlev) in (("nest 3 × 64³ (finest 256³ in the central 1/4)", 64, 3), ("uniform 256³", 256, 1))
    GC.gc(); CUDA.reclaim()
    probe = joinpath(OUT, "probe_" * replace(first(label, 7), r"\W" => "") * ".bin")
    wall = @elapsed (H = driven_run(GPU; N, nlev, mode=:ct, T=Float64, mach=10.0, tend=0.1, probe); KA.synchronize(GPU))
    d = only(s for s in H.sources if s isa Driving)
    finite = all(P -> all(isfinite, Array(P.U)), H.patches)
    upd = sum(prod(P.n) * s for (P, s) in zip(H.patches, H.nsteps))
    r = Dict("setup" => label, "wall" => wall, "steps" => H.nsteps, "cell_updates" => upd, "mach_mass" => d.log_mach_m[end],
             "mach_volume" => d.log_mach[end], "divb" => max_divb(H), "finite" => finite)
    push!(driven, r)
    say(@sprintf("driven %-46s %6.0f s  steps %s  %.2e updates  M_m %.2f  M_v %.2f  finite %s", label, wall, H.nsteps, upd,
                 r["mach_mass"], r["mach_volume"], finite))
    H = nothing
end
out["driven"] = driven; save()
say("== done")
