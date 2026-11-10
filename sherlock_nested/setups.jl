# The set-ups the GPU job runs, shared with test_gpu_compile.jl (which runs each `tiny` on the CPU to record its
# kernel launches). Each takes a KernelAbstractions backend and returns the hierarchy it evolved.
const b0 = 1/sqrt(4π)
# Tom Abel's 3-D perturbed Orszag–Tang (core/bench/run_experiments.jl)
ot3(x, y, z) = (25/(36π)*(1+0.1*sin(2π*z)), -sin(2π*y), sin(2π*x), 0.3*sin(2π*x), 5/(12π), -b0*sin(2π*y), b0*sin(4π*x), 0.2*b0)
aot3(x, y, z) = (0.0, 0.2*b0*x, b0*(cos(4π*x)/(4π) + cos(2π*y)/(2π)))

function ot_run(be; N=8, nlev=3, mode=:ct, riemann=:hll, T=Float32, tend=0.004)
    H = nested_central(N, nlev; n=N, mode=mode, riemann=riemann, T=T, backend=be)
    init!(H, ot3; vecpot=aot3); evolve!(H, tend)
    totals(H); max_divb(H)
    H
end

# driven isothermal 3-D turbulence as in Tom Abel's core/bench/run_mach10.jl, in 3-D: uniform field, beta = 200,
# solenoidal driving on 1 <= |k| <= 3, a probe in the central cell of the finest level
# Courant 0.25 (sum rule): 0.4, Tom's 2-D value, is unstable in 3-D even on a uniform grid (driven_debug, 2026-11-09)
function driven_run(be; N=8, nlev=3, mode=:ct, T=Float32, mach=10.0, tend=0.002, probe=tempname(), courant=0.25)
    cs = 1.0; B0 = sqrt(2cs^2/200)
    H = nested_central(N, nlev; n=N, mode=mode, riemann=:hll, T=T, backend=be, gamma=1.01, courant=courant)
    init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, cs^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0*y))
    push!(H.sources, Driving(H; mach=mach, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=1))
    push!(H.sources, Probe(H, probe; every=2, cs=cs))
    evolve!(H, tend)
    H
end

const SETUPS = [
    "OT CT HLL Float32"    => (be, tiny) -> ot_run(be; mode=:ct),
    "OT CT HLLD Float32"   => (be, tiny) -> ot_run(be; mode=:ct, riemann=:hlld),
    "OT GLM HLL Float32"   => (be, tiny) -> ot_run(be; mode=:glm),
    "OT GLM HLLD Float32"  => (be, tiny) -> ot_run(be; mode=:glm, riemann=:hlld),
    "OT CT LLF Float64"    => (be, tiny) -> ot_run(be; mode=:ct, riemann=:llf, T=Float64),
    "OT CT HLL Float64"    => (be, tiny) -> ot_run(be; mode=:ct, T=Float64),
    "OT GLM HLL Float64"   => (be, tiny) -> ot_run(be; mode=:glm, T=Float64),
    "driven CT Float32"    => (be, tiny) -> driven_run(be; mode=:ct),
    "driven GLM Float32"   => (be, tiny) -> driven_run(be; mode=:glm),
    "driven CT Float64"    => (be, tiny) -> driven_run(be; mode=:ct, T=Float64),
]
