# Movie frames of the driven nest (the set-up of compare.jl: isothermal Mach 10, beta 200, solenoidal driving on
# 1 <= |k| <= 3, CT, HLL, Courant 0.25, our 3-D fixes), many more outputs than compare.jl's 3-D snapshots: every DT_FRAME
# the mid-plane z = 1/2 of every level (mean of the two cell layers either side of it), as ln ρ, |v|/c_s, log10 |B|.
#   julia --project=env movie.jl NESTED_DIR OUT_DIR case ...      cases: below; NESTED_BACKEND=cpu for the Mac check
# Per case, OUT_DIR/<case>/:
#   slices.bin   per frame: Float64 (t, M_m, M_v), then per level (coarsest first) Float32 (n, n, 3) in Fortran order:
#                ln ρ, |v|, log10 |B| over the level's own patch (level l spans the central 1/2^(l-1) of the box)
#   movie.toml   levels, n, frames written, wall time, why it stopped (empty = finished)
using CUDA, KernelAbstractions, Printf, TOML
const KA = KernelAbstractions
const CPU_RUN = get(ENV, "NESTED_BACKEND", "gpu") == "cpu"
const NT = ARGS[1]; const OUT = ARGS[2]; mkpath(OUT)
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
const NM = NestedMHD; const G = NM.G
const BE = CPU_RUN ? KA.CPU() : CUDABackend()
const CAP = parse(Float64, get(ENV, "NESTED_CASE_WALL", "10800"))
say(x...) = (println(x...); flush(stdout))

# (root cells per side, levels, precision, t_end, dt_frame)
const CASES = Dict(
    "nest64"  => (64, 3, Float64, 1.0, 0.002),       # 501 frames, ~5 min on an H100
    "nest128" => (128, 3, Float32, 1.0, 0.002),      # 501 frames, ~2.2 h (compare.jl: t = 0.5 in 63 min)
    "tiny"    => (8, 3, Float64, 0.006, 0.002))

function midslice!(out, h)              # h: one level copied to the host, ghosts included
    n = size(h, 1) - 2G; k = G + n ÷ 2
    @inbounds for j in 1:n, i in 1:n
        q(v) = 0.5 * (Float64(h[G+i, G+j, k, v]) + Float64(h[G+i, G+j, k+1, v]))
        ρ = q(1)
        out[i, j, 1] = log(ρ)
        out[i, j, 2] = sqrt(q(2)^2 + q(3)^2 + q(4)^2) / ρ
        out[i, j, 3] = log10(sqrt(q(6)^2 + q(7)^2 + q(8)^2) + 1e-30)
    end
    out
end

for name in ARGS[3:end]
    N, nlev, T, tend, dtf = CASES[name]
    dir = joinpath(OUT, name); mkpath(dir)
    say(@sprintf("== %s: nest %d × %d³ (finest %d³ in the central 1/%d), %s, Mach 10, CT, HLL, Courant 0.25, frames every %g to t = %g",
                 name, nlev, N, N * 2^(nlev-1), 2^(nlev-1), T, dtf, tend))
    GC.gc(); CPU_RUN || CUDA.reclaim()
    cs = 1.0; B0 = sqrt(2cs^2/200)
    H = nested_central(N, nlev; n=N, mode=:ct, riemann=:hll, T=T, backend=BE, gamma=1.01, courant=0.25)
    init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, cs^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0*y))
    d = Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=1)
    push!(H.sources, d)
    host = [Array{T}(undef, size(P.U)) for P in H.patches]          # copied whole (as compare.jl), sliced on the host
    sl = [Array{Float32}(undef, P.n[1], P.n[2], 3) for P in H.patches]
    io = open(joinpath(dir, "slices.bin"), "w"); why = ""; t0 = time(); t = 0.0; nf = 0
    frame() = begin
        mm = isempty(d.log_mach_m) ? 0.0 : d.log_mach_m[end]; mv = isempty(d.log_mach) ? 0.0 : d.log_mach[end]
        write(io, Float64[t, mm, mv])
        for (l, P) in enumerate(H.patches)
            copyto!(host[l], P.U); write(io, midslice!(sl[l], host[l]))
        end
        flush(io); nf += 1
    end
    try
        frame()
        k = 0
        while t < tend - 1e-12
            k += 1; t = min(round(k * dtf; digits=10), tend)
            evolve!(H, t); KA.synchronize(BE)
            frame()
            k % 50 == 0 && say(@sprintf("   t %.3f  steps %s  M_m %.2f  %.0f s", t, H.nsteps, d.log_mach_m[end], time() - t0))
            time() - t0 > CAP && (why = @sprintf("wall cap %.0f s at t = %.3f", CAP, t); break)
        end
    catch e
        why = sprint(showerror, e); say("   STOPPED: ", why)
    end
    close(io)
    wall = time() - t0
    open(io -> TOML.print(io, Dict("case" => name, "root" => N, "levels" => nlev, "n" => N, "precision" => string(T),
                                   "t_end" => tend, "dt_frame" => dtf, "frames" => nf, "t_last" => t, "finished" => isempty(why),
                                   "why" => why, "wall" => wall, "steps" => H.nsteps, "divb" => max_divb(H),
                                   "fields" => ["ln rho", "|v|/c_s", "log10 |B|"], "slice" => "mean of the two layers either side of z = 1/2",
                                   "nested_commit" => (f = joinpath(NT, "COMMIT"); isfile(f) ? readchomp(f) : "unknown"),
                                   "device" => CPU_RUN ? "cpu" : CUDA.name(CUDA.device()))),
         joinpath(dir, "movie.toml"), "w")
    say(@sprintf("   %s: %d frames to t = %.3f, %.0f s, steps %s", isempty(why) ? "finished" : "stopped", nf, t, wall, H.nsteps))
    H = nothing
end
say("== done")
