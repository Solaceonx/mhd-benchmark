# Runs the CPU SIMD Orszag-Tang test from GLMMHDTurb and plots density slices
# (initial vs final) so we have something visual to look at, not just diagnostics.
repo = joinpath(@__DIR__, "..", "GLMMHDTurb")
include(joinpath(repo, "cpu_simd.jl"))
using Plots

function run_and_snapshot(; N=128, t_end=0.2f0)
    W_ = W
    @assert N % W_ == 0
    NP = N + 2XG
    p = G.Params(N=N, gamma=5f0/3f0, cs0=1f0)
    be = G.KA.CPU()

    u0 = zeros(Float32, N, N, N, 9)
    G.ic_ot_kernel!(be)(u0, N, p.gamma; ndrange=(N, N, N))
    G.KA.synchronize(be)
    rho0 = copy(u0[:, :, N ÷ 2, 1])

    pad1 = zeros(Float32, NP, N, N, 9)
    pad2 = zeros(Float32, NP, N, N, 9)
    @views pad1[XG+1:XG+N, :, :, :] .= u0
    ct = zeros(Float32, N, N, N)
    t = 0.0f0
    refresh_xghost!(pad1, N)
    simd_step!(pad2, pad1, p, 1.0f-3) # warmup
    while t < t_end
        @views ct .= 0.0f0
        uflat = Array(@view pad1[XG+1:XG+N, :, :, :])
        G.ctot_kernel!(be)(ct, uflat, p.gamma, p.smallr, G.pfloor(p); ndrange=(N, N, N))
        G.KA.synchronize(be)
        dt = p.courant * G.dxof(p) / maximum(ct)
        dt = min(dt, t_end - t)
        simd_step!(pad2, pad1, p, dt)
        refresh_xghost!(pad2, N)
        pad1, pad2 = pad2, pad1
        t += dt
    end
    uint = Array(@view pad1[XG+1:XG+N, :, :, :])
    rhof = uint[:, :, N ÷ 2, 1]
    return rho0, rhof
end

rho0, rhof = run_and_snapshot(N=128, t_end=0.5f0)

p1 = heatmap(rho0'; aspect_ratio=1, c=:viridis, title="density, t=0", xlabel="x", ylabel="y")
p2 = heatmap(rhof'; aspect_ratio=1, c=:viridis, title="density, t=0.5", xlabel="x", ylabel="y")
plt = plot(p1, p2; layout=(1, 2), size=(1000, 450))

outdir = joinpath(@__DIR__, "out")
mkpath(outdir)
outpath = joinpath(outdir, "ot_density.png")
savefig(plt, outpath)
println("Saved: ", outpath)
