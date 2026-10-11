# Runs the CPU SIMD Orszag-Tang test and dumps density mid-plane snapshots at even
# time intervals to raw Float32 binaries, for animating outside Julia.
repo = joinpath(@__DIR__, "..", "GLMMHDTurb")
include(joinpath(repo, "cpu_simd.jl"))

function render_frames(; N=128, t_end=0.5f0, nframes=60, outdir=joinpath(@__DIR__, "out", "frames"))
    @assert N % W == 0
    NP = N + 2XG
    p = G.Params(N=N, gamma=5.0f0 / 3.0f0, cs0=1.0f0)
    be = G.KA.CPU()

    mkpath(outdir)
    for f in readdir(outdir; join=true)
        endswith(f, ".bin") && rm(f)
    end

    u0 = zeros(Float32, N, N, N, 9)
    G.ic_ot_kernel!(be)(u0, N, p.gamma; ndrange=(N, N, N))
    G.KA.synchronize(be)

    pad1 = zeros(Float32, NP, N, N, 9)
    pad2 = zeros(Float32, NP, N, N, 9)
    @views pad1[XG+1:XG+N, :, :, :] .= u0
    ct = zeros(Float32, N, N, N)
    t = 0.0f0
    refresh_xghost!(pad1, N)
    simd_step!(pad2, pad1, p, 1.0f-3) # warmup

    dt_frame = t_end / nframes
    next_frame_t = 0.0f0
    frame_idx = 0

    function dump_frame!(idx, tval)
        slice = pad1[XG+1:XG+N, :, N ÷ 2, 1]
        open(joinpath(outdir, @sprintf("frame_%03d.bin", idx)), "w") do io
            write(io, slice)
        end
        @printf("frame %03d  t=%.4f\n", idx, tval)
    end

    dump_frame!(frame_idx, t)
    frame_idx += 1
    next_frame_t += dt_frame

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
        if t >= next_frame_t && frame_idx < nframes + 1
            dump_frame!(frame_idx, t)
            frame_idx += 1
            next_frame_t += dt_frame
        end
    end

    open(joinpath(outdir, "manifest.txt"), "w") do io
        println(io, "N=$N")
        println(io, "nframes=$(frame_idx)")
        println(io, "t_end=$t_end")
    end
    println("Wrote $(frame_idx) frames to $outdir")
end

render_frames(N=128, t_end=0.5f0, nframes=60)
