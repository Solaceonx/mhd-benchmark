# Tom Abel's question for NestedMHD: does a nest reproduce the volume statistics and spectra of a uniform grid at its
# finest resolution, for much less compute? Driven isothermal Mach-10 MHD as driven_run in setups.jl (beta 200, solenoidal
# driving on 1 <= |k| <= 3, gamma 1.01), CT, HLL, Courant 0.25, with our 3-D fixes (branch gpu-3d-fixes).
#   julia --project=env compare.jl NESTED_DIR OUT_DIR case ...      cases: below; NESTED_BACKEND=cpu for the Mac check
# Per case, OUT_DIR/<case>/:
#   hist.bin + hist.toml   every DT_OUT from T_STATS: counts of ln ρ, v_x, |v|, log10 |B| over the central cube
#                          [3/8, 5/8]^3 at the finest resolution (nest: its finest level; uniform: those cells), over
#                          the whole box at the run's own resolution, and (uniform) block-averaged to the nest's root
#   snap_c_<t>.bin, snap_r_<t>.bin   every DT_SNAP: the central cube (finest resolution) and the whole box at the
#                          nest's root resolution (nest: its root; uniform: block averages), ρ, v, B (cell-centred),
#                          Float32, Fortran order (n, n, n, 7)
#   run.toml               Mach history, steps and wall time per level, cell updates, div B
using CUDA, KernelAbstractions, Printf, TOML
const KA = KernelAbstractions
const CPU_RUN = get(ENV, "NESTED_BACKEND", "gpu") == "cpu"
const NT = ARGS[1]; const OUT = ARGS[2]; mkpath(OUT)
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
const NM = NestedMHD; const G = NM.G
const BE = CPU_RUN ? KA.CPU() : CUDABackend()
const CAP = parse(Float64, get(ENV, "NESTED_CASE_WALL", "10800"))       # seconds per case, then it stops and saves
# The Alfvén-speed cap of our branch (Driving's va_max; NestedTurbulence 1164c73), the same in the nest and the uniform grid:
# without it nearly empty cells make the uniform grids' time step collapse (2026-11-10). Inf: none, as the first comparison.
const VA_MAX = parse(Float64, get(ENV, "NESTED_VA_MAX", "Inf"))
say(x...) = (println(x...); flush(stdout))

# (root cells per side, levels, precision, t_end, t_stats, dt_out, dt_snap); the nest's finest = the uniform grid
const CASES = Dict(
    "nest64"  => (64, 3, Float64, 1.0, 0.2, 0.01, 0.05),  "uni256" => (256, 1, Float64, 1.0, 0.2, 0.01, 0.05),
    "nest128" => (128, 3, Float32, 0.5, 0.2, 0.01, 0.05), "uni512" => (512, 1, Float32, 0.5, 0.2, 0.01, 0.05),
    "tiny_nest" => (8, 3, Float64, 0.006, 0.002, 0.002, 0.004), "tiny_uni" => (32, 1, Float64, 0.006, 0.002, 0.002, 0.004))
const NEST_ROOT = Dict("nest64" => 64, "uni256" => 64, "nest128" => 128, "uni512" => 128, "tiny_nest" => 8, "tiny_uni" => 8)
const EDGES = (lnrho = range(-12.0, 6.0; length=181), vx = range(-60.0, 60.0; length=241),
               v = range(0.0, 80.0; length=161), logb = range(-4.0, 1.5; length=111))

function hist(x, e)
    h = zeros(Int, length(e) - 1); a = first(e); w = step(e)
    for v in x
        b = floor(Int, (v - a) / w) + 1
        1 <= b <= length(h) && (h[b] += 1)
    end
    h
end
fields(U) = (ρ = U[:,:,:,1]; (ρ, U[:,:,:,2] ./ ρ, U[:,:,:,3] ./ ρ, U[:,:,:,4] ./ ρ, U[:,:,:,6], U[:,:,:,7], U[:,:,:,8]))
function hists(U)
    ρ, vx, vy, vz, bx, by, bz = fields(U)
    (hist(log.(ρ), EDGES.lnrho), hist(vx, EDGES.vx), hist(sqrt.(vx.^2 .+ vy.^2 .+ vz.^2), EDGES.v),
     hist(log10.(sqrt.(bx.^2 .+ by.^2 .+ bz.^2) .+ 1e-30), EDGES.logb))
end
interior(P) = Array(P.U)[G+1:G+P.n[1], G+1:G+P.n[2], G+1:G+P.n[3], :]     # the run's precision (uniform 512³ Float32: 4.3 GB)
function blockavg(U, m)                  # conserved state averaged over m^3 blocks (ρv and B too, as a restriction)
    n = size(U, 1) ÷ m
    dropdims(sum(reshape(U, m, n, m, n, m, n, size(U, 4)); dims=(1, 3, 5)); dims=(1, 3, 5)) ./ m^3
end
snapfields(U) = Float32.(cat(fields(U)...; dims=4))

for name in ARGS[3:end]
    N, nlev, T, tend, tstats, dtout, dtsnap = CASES[name]
    dir = joinpath(OUT, name); mkpath(dir); nr = NEST_ROOT[name]
    if isfile(joinpath(dir, "run.toml")) && get(TOML.parsefile(joinpath(dir, "run.toml")), "finished", false)
        say("== $name: already finished here, kept"); continue          # a requeued job does not redo its finished cases
    end
    say(@sprintf("== %s: %s, %s, Mach 10, CT, HLL, Courant 0.25, to t = %g", name, nlev == 1 ? "uniform $(N)³" :
                 "nest $nlev × $(N)³ (finest $(N*2^(nlev-1))³ in the central 1/$(2^(nlev-1)))", T, tend))
    GC.gc(); CPU_RUN || CUDA.reclaim()
    cs = 1.0; B0 = sqrt(2cs^2/200)
    H = nested_central(N, nlev; n=N, mode=:ct, riemann=:hll, T=T, backend=BE, gamma=1.01, courant=0.25)
    init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, cs^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0*y))
    d = isfinite(VA_MAX) ? Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=1, va_max=VA_MAX) :
                           Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=false, seed=1)
    push!(H.sources, d)
    fine = N * 2^(nlev-1)                                  # the finest resolution, cells across the box
    c0 = 3fine ÷ 8 + 1; c1 = 5fine ÷ 8                     # the central cube at that resolution
    hio = open(joinpath(dir, "hist.bin"), "w"); times = Float64[]; why = ""; t0 = time(); t = 0.0
    nextsnap = tstats
    try
        while t < tend - 1e-12
            t = min(round(t + dtout; digits=10), tend)
            evolve!(H, t); KA.synchronize(BE)
            t < tstats - 1e-12 && continue
            Ufine = interior(H.patches[end]); Uroot = interior(H.patches[1])
            Uc = nlev == 1 ? Ufine[c0:c1, c0:c1, c0:c1, :] : Ufine          # the nest's finest level is the central cube
            hc = hists(Uc); hw = hists(Uroot)
            Ur = nlev == 1 ? blockavg(Uroot, N ÷ nr) : Uroot                # whole box at the nest's root resolution
            hr = nlev == 1 ? hists(Ur) : hw
            for h in (hc..., hw..., hr...); write(hio, Float64.(h)); end
            write(hio, Float64(t)); flush(hio); push!(times, t)
            if t >= nextsnap - 1e-9
                write(joinpath(dir, @sprintf("snap_c_%.3f.bin", t)), snapfields(Uc))
                write(joinpath(dir, @sprintf("snap_r_%.3f.bin", t)), snapfields(Ur))
                nextsnap += dtsnap
            end
            say(@sprintf("   t %.3f  steps %s  M_m %.2f  M_v %.2f  %.0f s", t, H.nsteps, d.log_mach_m[end], d.log_mach[end], time() - t0))
            time() - t0 > CAP && (why = @sprintf("wall cap %.0f s at t = %.3f", CAP, t); break)
        end
    catch e
        why = sprint(showerror, e); say("   STOPPED: ", why)
    end
    close(hio)
    wall = time() - t0; upd = sum(prod(P.n) * s for (P, s) in zip(H.patches, H.nsteps))
    meta = Dict("case" => name, "root" => N, "levels" => nlev, "precision" => string(T), "t_end" => tend, "finished" => isempty(why),
                "why" => why, "wall" => wall, "level_wall" => H.wall, "steps" => H.nsteps, "cell_updates" => upd,
                "finest" => fine, "central_cells" => [c0, c1], "nest_root" => nr, "times" => times,
                "va_max" => VA_MAX, "mass" => NM.totals(H).mass, "mach_t" => d.log_t, "mach_m" => d.log_mach_m, "mach_v" => d.log_mach, "amp" => d.log_amp,
                "divb" => max_divb(H), "nested_commit" => (f = joinpath(NT, "COMMIT"); isfile(f) ? readchomp(f) : "unknown"),
                "device" => CPU_RUN ? "cpu" : CUDA.name(CUDA.device()))
    open(io -> TOML.print(io, meta), joinpath(dir, "run.toml"), "w")
    open(joinpath(dir, "hist.toml"), "w") do io
        TOML.print(io, Dict("order" => ["central lnrho", "central vx", "central v", "central log10B", "box lnrho", "box vx",
                                        "box v", "box log10B", "rootres lnrho", "rootres vx", "rootres v", "rootres log10B", "t"],
                            "edges" => Dict(String(k) => collect(v) for (k, v) in pairs(EDGES)),
                            "note" => "per output: 12 Float64 count arrays in this order (bins = edges - 1), then t"))
    end
    say(@sprintf("   %s: %.0f s, steps %s, %.2e cell updates (%.0f Mcell/s)", isempty(why) ? "finished" : "stopped", wall, H.nsteps, upd, upd/wall/1e6))
    H = nothing
end
say("== done")
