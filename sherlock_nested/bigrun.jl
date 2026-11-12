# A long nested run that can stop and resume (owners pre-emption, the 2-day limit): driven isothermal Mach-10 MHD as in
# compare.jl (beta 200, solenoidal driving on 1 <= |k| <= 3, gamma 1.01, CT, HLL, Courant 0.25, our 3-D fixes), with an
# optional cap on the Alfvén speed in nearly empty cells (BIG_VA_MAX, default 30 c_s; Inf = none, as compare.jl).
#   julia --project=env bigrun.jl NESTED_DIR OUT_DIR case          cases: below; NESTED_BACKEND=cpu for the Mac check
# OUT_DIR/<case>/ (every record carries its time; a resumed run cuts the files back to its checkpoint first):
#   hist.bin     every DT_OUT: per level (coarsest first) counts of ln ρ, v_x, |v|, log10 |B| over the level's own patch
#                (edges as compare.jl), then t: Float64, L*690 + 1 per record
#   series.csv   every DT_OUT: t, wall, steps per level, M_m, M_v, mass, momentum, driving amplitude
#   slices.bin   every DT_FRAME: as movie.jl (Float64 t, M_m, M_v; per level Float32 (n, n, 3): ln ρ, |v|, log10 |B| at z = 1/2)
#   snap_L<l>_<t>.bin  every DT_SNAP: level l, interior, Float32 (n, n, n, 7): ρ, v, B (cell-centred)
#   checkpoint.jls     state to resume from (Serialization); written every BIG_CKPT_WALL s, at the wall cap and at the end
#   run.toml     written at the end (and at each stop): steps, wall, div B, mass added by the floors
using CUDA, KernelAbstractions, Printf, TOML, Serialization, Random
const KA = KernelAbstractions
const CPU_RUN = get(ENV, "NESTED_BACKEND", "gpu") == "cpu"
const NT = ARGS[1]; const OUT = ARGS[2]; const NAME = ARGS[3]
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
const NM = NestedMHD; const G = NM.G
const BE = CPU_RUN ? KA.CPU() : CUDABackend()
const WALL = parse(Float64, get(ENV, "BIG_WALL", "169200"))           # stop and checkpoint after this (47 h)
const CKPT = parse(Float64, get(ENV, "BIG_CKPT_WALL", "1800"))       # checkpoint every 30 min of wall time
const STOP_AT = parse(Float64, get(ENV, "BIG_STOP_AT", "Inf"))       # checkpoint and stop at this t (restart test)
const VA_MAX = parse(Float64, get(ENV, "BIG_VA_MAX", "30"))
say(x...) = (println(x...); flush(stdout))

# root cells, levels, cells per refined level, precision, t_end, dt_out, dt_frame, dt_snap
const CASES = Dict(
    "deep192" => (128, 4, 192, Float32, 0.5, 0.01, 0.002, 0.05),   # finest 1024³-eq over the central 3/16; level 3 512³-eq over 3/8
    "wide192" => (128, 3, 192, Float32, 0.5, 0.01, 0.002, 0.05),   # finest 512³-eq over the central 3/8
    "tiny"    => (12, 3, 16, Float64, 0.006, 0.002, 0.002, 0.004))
N, nlev, nc, T, tend, dtout, dtf, dtsnap = CASES[NAME]
dir = joinpath(OUT, NAME); mkpath(dir)
const EDGES = (lnrho = range(-12.0, 6.0; length=181), vx = range(-60.0, 60.0; length=241),
               v = range(0.0, 80.0; length=161), logb = range(-4.0, 1.5; length=111))
const REC = nlev * 690 + 1

function hist!(h, x, e)
    a = first(e); w = step(e); m = length(h)
    @inbounds for v in x
        b = floor(Int, (v - a) / w) + 1
        1 <= b <= m && (h[b] += 1)
    end
    h
end
function level_hists(U, n)                 # U on the host with ghosts; counts as Float64, 690 values
    out = zeros(690); r = G+1:G+n
    hr = view(out, 1:180); hx = view(out, 181:420); hv = view(out, 421:580); hb = view(out, 581:690)
    @inbounds for k in r, j in r, i in r
        ρ = Float64(U[i,j,k,1]); vx = U[i,j,k,2]/ρ; vy = U[i,j,k,3]/ρ; vz = U[i,j,k,4]/ρ
        b = sqrt(Float64(U[i,j,k,6])^2 + Float64(U[i,j,k,7])^2 + Float64(U[i,j,k,8])^2)
        hist!(hr, (log(ρ),), EDGES.lnrho); hist!(hx, (vx,), EDGES.vx); hist!(hv, (sqrt(vx^2 + vy^2 + vz^2),), EDGES.v)
        hist!(hb, (log10(b + 1e-30),), EDGES.logb)
    end
    out
end
function midslice!(out, h, n)
    k = G + n ÷ 2
    @inbounds for j in 1:n, i in 1:n
        q(v) = 0.5 * (Float64(h[G+i, G+j, k, v]) + Float64(h[G+i, G+j, k+1, v]))
        ρ = q(1)
        out[i, j, 1] = log(ρ); out[i, j, 2] = sqrt(q(2)^2 + q(3)^2 + q(4)^2) / ρ
        out[i, j, 3] = log10(sqrt(q(6)^2 + q(7)^2 + q(8)^2) + 1e-30)
    end
    out
end
function snapfields(h, n)
    r = G+1:G+n; out = Array{Float32}(undef, n, n, n, 7)
    @inbounds for k in r, j in r, i in r
        ρ = h[i,j,k,1]; a, b, c = i-G, j-G, k-G
        out[a,b,c,1] = ρ; out[a,b,c,2] = h[i,j,k,2]/ρ; out[a,b,c,3] = h[i,j,k,3]/ρ; out[a,b,c,4] = h[i,j,k,4]/ρ
        out[a,b,c,5] = h[i,j,k,6]; out[a,b,c,6] = h[i,j,k,7]; out[a,b,c,7] = h[i,j,k,8]
    end
    out
end
truncate_to(path, bytes) = isfile(path) && filesize(path) > bytes && open(io -> truncate(io, bytes), path, "r+")

# ---------------------------------------------------------------- set up, or resume
cs = 1.0; B0 = sqrt(2cs^2/200)
H = nested_central(N, nlev; n=nc, mode=:ct, riemann=:hll, T=T, backend=BE, gamma=1.01, courant=0.25)
init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, cs^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0*y))
d = Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=1, va_max=VA_MAX)
push!(H.sources, d)
mass0 = NM.totals(H).mass
ck = joinpath(dir, "checkpoint.jls")
S = Dict{Symbol,Any}(:t => 0.0, :k => 0, :nf => 0, :nh => 0, :nextsnap => dtsnap, :wall => 0.0, :mass0 => mass0, :segments => 0)
if isfile(ck)
    C = deserialize(ck)
    for (l, P) in enumerate(H.patches)
        copyto!(P.U, C[:U][l]); copyto!(P.bx, C[:bx][l]); copyto!(P.by, C[:by][l]); copyto!(P.bz, C[:bz][l])
    end
    H.t .= C[:t_lev]; H.tp0 .= C[:tp0]; H.tp1 .= C[:tp1]; H.nsteps .= C[:nsteps]; H.wall .= C[:level_wall]
    D = C[:driving]
    empty!(d.nodes); append!(d.nodes, D[:nodes]); copy!(d.rng, D[:rng]); d.amp = D[:amp]
    for f in (:log_t, :log_mach, :log_mach_m, :log_amp); v = getfield(d, f); empty!(v); append!(v, D[f]); end
    S = C[:S]; S[:segments] += 1
    truncate_to(joinpath(dir, "hist.bin"), S[:nh] * REC * 8)
    fb = 24 + sum(P.n[1]^2 for P in H.patches) * 12
    truncate_to(joinpath(dir, "slices.bin"), S[:nf] * fb)
    lines = readlines(joinpath(dir, "series.csv"))
    open(joinpath(dir, "series.csv"), "w") do io
        for ln in lines; (startswith(ln, "t,") || parse(Float64, split(ln, ",")[1]) <= S[:t] + 1e-9) && println(io, ln); end
    end
    say(@sprintf("== %s: resumed at t = %.4f (segment %d), steps %s", NAME, S[:t], S[:segments], H.nsteps))
else
    open(io -> println(io, "t,wall," * join(("steps_L$l" for l in 1:nlev), ",") * ",M_m,M_v,mass,momentum,amp"),
         joinpath(dir, "series.csv"), "w")
    foreach(f -> rm(joinpath(dir, f); force=true), ("hist.bin", "slices.bin"))
    say(@sprintf("== %s: root %d³ + %d levels of %d³ (finest %d³-equivalent over the central %d/%d), %s, Mach 10, CT, HLL, Courant 0.25, v_A ≤ %g, to t = %g",
                 NAME, N, nlev - 1, nc, N * 2^(nlev-1), nc, N * 2^(nlev-1), T, VA_MAX, tend))
end
say("   ", CPU_RUN ? "cpu" : CUDA.name(CUDA.device()), ", nested commit ", (f = joinpath(NT, "COMMIT"); isfile(f) ? readchomp(f) : "unknown"))

host = [Array{T}(undef, size(P.U)) for P in H.patches]
sl = [Array{Float32}(undef, P.n[1], P.n[2], 3) for P in H.patches]
tohost!() = for (l, P) in enumerate(H.patches); copyto!(host[l], P.U); end

function checkpoint!(S, wall)
    S[:wall] = wall
    C = Dict(:U => [Array(P.U) for P in H.patches], :bx => [Array(P.bx) for P in H.patches], :by => [Array(P.by) for P in H.patches],
             :bz => [Array(P.bz) for P in H.patches], :t_lev => copy(H.t), :tp0 => copy(H.tp0), :tp1 => copy(H.tp1),
             :nsteps => copy(H.nsteps), :level_wall => copy(H.wall), :S => copy(S),
             :driving => Dict(:nodes => d.nodes, :rng => copy(d.rng), :amp => d.amp, :log_t => d.log_t, :log_mach => d.log_mach,
                              :log_mach_m => d.log_mach_m, :log_amp => d.log_amp))
    serialize(ck * ".tmp", C); mv(ck * ".tmp", ck; force=true)
end
function write_meta(S, why, wall)
    tt = NM.totals(H)
    open(io -> TOML.print(io, Dict("case" => NAME, "root" => N, "levels" => nlev, "n" => nc, "precision" => string(T), "va_max" => VA_MAX,
        "t_end" => tend, "t" => S[:t], "finished" => S[:t] >= tend - 1e-9, "why" => why, "wall" => wall, "segments" => S[:segments] + 1,
        "steps" => H.nsteps, "level_wall" => H.wall, "divb" => max_divb(H), "mass0" => S[:mass0], "mass" => tt.mass,
        "mass_added_fraction" => tt.mass / S[:mass0] - 1, "frames" => S[:nf], "hist_records" => S[:nh],
        "nested_commit" => (f = joinpath(NT, "COMMIT"); isfile(f) ? readchomp(f) : "unknown"),
        "device" => CPU_RUN ? "cpu" : CUDA.name(CUDA.device()))), joinpath(dir, "run.toml"), "w")
end

# ---------------------------------------------------------------- run
t0 = time() - S[:wall]; lastck = time(); why = ""
hio = open(joinpath(dir, "hist.bin"), "a"); sio = open(joinpath(dir, "slices.bin"), "a"); cio = open(joinpath(dir, "series.csv"), "a")
try
    global why, lastck
    if S[:nf] == 0                                   # frame at t = 0
        tohost!(); write(sio, Float64[0.0, 0.0, 0.0]); for l in eachindex(host); write(sio, midslice!(sl[l], host[l], H.patches[l].n[1])); end
        S[:nf] = 1; flush(sio)
    end
    while S[:t] < tend - 1e-12
        S[:k] += 1; t = min(round(S[:k] * dtf; digits=10), tend)
        evolve!(H, t); KA.synchronize(BE); S[:t] = t
        tohost!()
        write(sio, Float64[t, d.log_mach_m[end], d.log_mach[end]])
        for l in eachindex(host); write(sio, midslice!(sl[l], host[l], H.patches[l].n[1])); end
        S[:nf] += 1; flush(sio)
        if abs(t / dtout - round(t / dtout)) < 1e-6
            for (l, P) in enumerate(H.patches); write(hio, level_hists(host[l], P.n[1])); end
            write(hio, t); S[:nh] += 1; flush(hio)
            tt = NM.totals(H)
            println(cio, join((@sprintf("%.4f", t), @sprintf("%.1f", time() - t0), H.nsteps..., @sprintf("%.4f", d.log_mach_m[end]),
                               @sprintf("%.4f", d.log_mach[end]), @sprintf("%.10g", tt.mass), @sprintf("%.4g", hypot(tt.momx, tt.momy, tt.momz)),
                               @sprintf("%.5g", d.amp)), ",")); flush(cio)
            say(@sprintf("   t %.3f  steps %s  M_m %.2f  mass %+.2e  %.0f s", t, H.nsteps, d.log_mach_m[end], tt.mass / S[:mass0] - 1, time() - t0))
        end
        if t >= S[:nextsnap] - 1e-9
            for (l, P) in enumerate(H.patches); write(joinpath(dir, @sprintf("snap_L%d_%.3f.bin", l, t)), snapfields(host[l], P.n[1])); end
            S[:nextsnap] = round(S[:nextsnap] + dtsnap; digits=10)
        end
        t >= STOP_AT - 1e-12 && (why = @sprintf("stop requested at t = %.4f", t); break)
        time() - t0 > WALL && (why = @sprintf("wall cap at t = %.4f", t); break)
        time() - lastck > CKPT && (checkpoint!(S, time() - t0); lastck = time(); say(@sprintf("   checkpoint at t = %.4f", t)))
    end
catch e
    global why = sprint(showerror, e); say("   STOPPED: ", why)
end
close(hio); close(sio); close(cio)
wall = time() - t0
isempty(why) || startswith(why, "stop") || startswith(why, "wall") ? checkpoint!(S, wall) : say("   no checkpoint after an error; the last one stands")
write_meta(S, why, wall)
say(@sprintf("== %s: %s at t = %.4f, %.0f s, steps %s", NAME, isempty(why) ? "finished" : why, S[:t], wall, H.nsteps))
