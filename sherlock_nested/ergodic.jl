# Tom Abel's ergodicity test (NestedTurbulence report, chapter 7) in 3-D on a GPU: does the time average at one point equal
# the volume average? Driven isothermal Mach-10 MHD as compare.jl (nest 3 × 64³, finest 256³-equivalent in the central 1/4,
# beta 200, solenoidal driving on 1 <= |k| <= 3, CT, HLL, Courant 0.25, Float64, t = 1 = 10 crossing times), one run per
# random seed of the driving: an ensemble.
#   julia --project=env ergodic.jl NESTED_DIR OUT_DIR seed ...        NESTED_BACKEND=cpu ERGODIC_TINY=1 for the Mac check
# Per seed, OUT_DIR/seed<k>/ (a seed whose run.toml says finished is kept):
#   probe.bin    Float64 records (t, ρ, vx, vy, vz, Bx, By, Bz) of the finest-level cell whose lower corner is the box
#                centre, every second step of that level (as Tom's Probe; read cell by cell, his Array(view) of one cell
#                over the components is not a contiguous GPU copy)
#   volume.csv   every 0.01: per level (coarsest first) the volume mean and variance of s = ln ρ and of v_x, and M_m
#   pdf.bin      Float64 counts of s in 180 bins on [-12, 6] per level, summed over the outputs with t >= 0.3 (3 crossing times)
#   run.toml     steps, wall, div B, probe position
using CUDA, KernelAbstractions, Printf, TOML
const KA = KernelAbstractions
const CPU_RUN = get(ENV, "NESTED_BACKEND", "gpu") == "cpu"
const TINY = haskey(ENV, "ERGODIC_TINY")
const NT = ARGS[1]; const OUT = ARGS[2]; mkpath(OUT)
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
const NM = NestedMHD; const G = NM.G
const BE = CPU_RUN ? KA.CPU() : CUDABackend()
const VA_MAX = parse(Float64, get(ENV, "NESTED_VA_MAX", "Inf"))
say(x...) = (println(x...); flush(stdout))
N, nlev, tend, dtout, tstat = TINY ? (8, 3, 0.006, 0.002, 0.002) : (64, 3, 1.0, 0.01, 0.3)
const SEDGES = range(-12.0, 6.0; length=181)

mutable struct PointProbe            # a source of the hierarchy: after_step! is called after every step of every level
    level::Int; cell::NTuple{3,Int}; every::Int; counter::Int; buf::Vector{Float64}
end
function NM.after_step!(pr::PointProbe, H, l)
    l == pr.level || return
    pr.counter += 1
    pr.counter % pr.every == 0 || return
    P = H.patches[l]; i, j, k = pr.cell
    u = CPU_RUN ? [Float64(P.U[i, j, k, v]) for v in 1:8] : CUDA.@allowscalar [Float64(P.U[i, j, k, v]) for v in 1:8]
    ρ = u[1]
    append!(pr.buf, (H.t[l], ρ, u[2]/ρ, u[3]/ρ, u[4]/ρ, u[6], u[7], u[8]))
end

function moments(U, n)               # host array with ghosts -> mean and variance of s and v_x over the interior, and the s counts
    r = G+1:G+n; ms = 0.0; ss = 0.0; mv = 0.0; vv = 0.0; h = zeros(180); a = first(SEDGES); w = step(SEDGES)
    @inbounds for k in r, j in r, i in r
        ρ = Float64(U[i,j,k,1]); s = log(ρ); vx = U[i,j,k,2]/ρ
        ms += s; ss += s*s; mv += vx; vv += vx*vx
        b = floor(Int, (s - a)/w) + 1; 1 <= b <= 180 && (h[b] += 1)
    end
    m = n^3; ms /= m; mv /= m
    (ms, ss/m - ms^2, mv, vv/m - mv^2, h)
end

for seed in parse.(Int, ARGS[3:end])
    dir = joinpath(OUT, "seed$seed"); mkpath(dir)
    if isfile(joinpath(dir, "run.toml")) && get(TOML.parsefile(joinpath(dir, "run.toml")), "finished", false)
        say("== seed $seed: already finished here, kept"); continue
    end
    GC.gc(); CPU_RUN || CUDA.reclaim()
    cs = 1.0; B0 = sqrt(2cs^2/200)
    H = nested_central(N, nlev; n=N, mode=:ct, riemann=:hll, T=Float64, backend=BE, gamma=1.01, courant=0.25)
    init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, cs^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0*y))
    d = isfinite(VA_MAX) ? Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=seed, va_max=VA_MAX) :
                           Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=seed)
    push!(H.sources, d)
    P = H.patches[end]; g = ntuple(q -> P.Nlev[q] ÷ 2 + 1, 3); cell = ntuple(q -> g[q] - P.lo[q] + G + 1, 3)
    pr = PointProbe(nlev, cell, 2, 0, Float64[]); push!(H.sources, pr)
    xp = ntuple(q -> (P.lo[q] + cell[q] - G - 1 - 0.5) * P.dx, 3)
    say(@sprintf("== seed %d: nest %d × %d³, probe at (%.5f, %.5f, %.5f) on level %d, to t = %g", seed, nlev, N, xp..., nlev, tend))
    host = [Array{Float64}(undef, size(Q.U)) for Q in H.patches]; pdf = [zeros(180) for _ in H.patches]
    io = open(joinpath(dir, "volume.csv"), "w")
    println(io, "t," * join(("mean_s_L$l,var_s_L$l,mean_vx_L$l,var_vx_L$l" for l in 1:nlev), ",") * ",M_m,M_v")
    t0 = time(); t = 0.0; k = 0; why = ""
    try
        while t < tend - 1e-12
            k += 1; t = min(round(k*dtout; digits=10), tend)
            evolve!(H, t); KA.synchronize(BE)
            row = String[@sprintf("%.4f", t)]
            for (l, Q) in enumerate(H.patches)
                copyto!(host[l], Q.U); ms, vs, mv, vv, h = moments(host[l], Q.n[1])
                t >= tstat - 1e-9 && (pdf[l] .+= h)
                append!(row, (@sprintf("%.6f", ms), @sprintf("%.6f", vs), @sprintf("%.6f", mv), @sprintf("%.6f", vv)))
            end
            println(io, join(row, ",") * @sprintf(",%.4f,%.4f", d.log_mach_m[end], d.log_mach[end])); flush(io)
            k % 10 == 0 && say(@sprintf("   t %.2f  steps %s  M_m %.2f  %d probe records  %.0f s", t, H.nsteps, d.log_mach_m[end], length(pr.buf) ÷ 8, time() - t0))
        end
    catch e
        why = sprint(showerror, e); say("   STOPPED: ", why)
    end
    close(io)
    write(joinpath(dir, "probe.bin"), pr.buf); open(f -> foreach(h -> write(f, h), pdf), joinpath(dir, "pdf.bin"), "w")
    open(f -> TOML.print(f, Dict("seed" => seed, "root" => N, "levels" => nlev, "t_end" => tend, "t" => t, "finished" => isempty(why), "why" => why,
        "wall" => time() - t0, "steps" => H.nsteps, "divb" => max_divb(H), "probe_x" => collect(xp), "probe_dx" => P.dx, "probe_records" => length(pr.buf) ÷ 8,
        "probe_every" => 2, "va_max" => VA_MAX, "t_stat" => tstat, "nested_commit" => (f = joinpath(NT, "COMMIT"); isfile(f) ? readchomp(f) : "unknown"),
        "device" => CPU_RUN ? "cpu" : CUDA.name(CUDA.device()))), joinpath(dir, "run.toml"), "w")
    say(@sprintf("   seed %d: %s, %.0f s, steps %s, %d probe records", seed, isempty(why) ? "finished" : "stopped", time() - t0, H.nsteps, length(pr.buf) ÷ 8))
end
say("== done")
