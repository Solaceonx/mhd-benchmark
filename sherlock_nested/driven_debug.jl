# Why the driven Mach-10 3-D nest stopped in gpu_check (job 51686178: after ~55 min level 2's CFL step had collapsed
# to ~1e-21 and ceil(Int, remaining/dt) overflowed) and whether Courant 0.25 cures it. The same physics in the sweep's
# kernels was unsafe at Courant 0.4 in 3-D (sherlock_checks, 2026-10-12: OT3D and Mach 9 stopped) and stable at 0.25;
# Tom Abel's run_mach10.jl uses 0.4, in 2-D.
#   julia --project=env driven_debug.jl NESTED_DIR OUT_DIR [case ...]
# Every root step it records, per level: the CFL step, min/max density, max speed, max |B|, where the largest signal
# speed is, the driving amplitude and Mach number (OUT_DIR/<case>.csv). A watchdog stops a case when a level's CFL
# step falls below DT_FLOOR, when a level takes more than MAX_SUB steps in one root step, or at the case's wall cap;
# then (and on any error) every level's state is written to OUT_DIR/<case>_L<l>.bin (Float32, interior + 2 ghosts,
# Fortran order (nx, ny, nz, nvar)) with OUT_DIR/<case>.toml describing it.
# NESTED_BACKEND=cpu runs it on the CPU (the Mac check of this script, tiny cases).
using CUDA, KernelAbstractions, Printf, TOML
const KA = KernelAbstractions
const CPU_RUN = get(ENV, "NESTED_BACKEND", "gpu") == "cpu"
const NT = ARGS[1]; const OUT = ARGS[2]; mkpath(OUT)
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
const NM = NestedMHD; const G = NM.G
const BE = CPU_RUN ? KA.CPU() : CUDABackend()
const CAP = parse(Float64, get(ENV, "NESTED_CASE_WALL", "900"))        # seconds per case
const DT_FLOOR = parse(Float64, get(ENV, "NESTED_DT_FLOOR", "2e-7"))
const MAX_SUB = parse(Int, get(ENV, "NESTED_MAX_SUB", "20000"))
say(x...) = (println(x...); flush(stdout))

# (name, root cells, levels, Courant, t_end, options): Mach 10, HLL, Float64, as gpu_check's driven run; options: mode
# (:ct), dims (3; 2 = Tom Abel's run_mach10.jl nest), and Hierarchy's adaptive, time_interp, skip_covered (all true)
const CASES = Dict(
    "nest32_c040" => (32, 3, 0.40, 0.1, (;)), "uni128_c040" => (128, 1, 0.40, 0.1, (;)),
    "nest32_c025" => (32, 3, 0.25, 0.1, (;)), "uni128_c025" => (128, 1, 0.25, 0.1, (;)),
    "nest64_c025" => (64, 3, 0.25, 0.1, (;)),
    # job 51708826: the 3-D nests fail at level 2's outer faces (|B| first), at either Courant number; one change each
    "nest32_glm"     => (32, 3, 0.25, 0.1, (mode=:glm,)),          # no CT: face-field prolongation, EMF correction
    "nest32_2lev"    => (32, 2, 0.25, 0.1, (;)),                   # no level that is both child and parent
    "nest32_fixed"   => (32, 3, 0.25, 0.1, (adaptive=false,)),     # ratio-2 subcycling instead of per-level CFL steps
    "nest32_stale"   => (32, 3, 0.25, 0.1, (time_interp=false,)),  # ghosts from the parent's last state, not blended
    "nest32_noskip"  => (32, 3, 0.25, 0.1, (skip_covered=false,)), # the parent also updates deep inside its child
    "tom2d_c040"     => (64, 3, 0.40, 0.1, (dims=2,)),             # Tom's 2-D nest, which runs on the CPU
    "tiny_c040"   => (8, 3, 0.40, 0.004, (;)), "tiny_2d" => (16, 3, 0.40, 0.004, (dims=2,)),   # the Mac checks
    "tiny_glm_fixed" => (8, 3, 0.25, 0.004, (mode=:glm, adaptive=false, time_interp=false, skip_covered=false)))
const ORDER = length(ARGS) > 2 ? ARGS[3:end] : ["nest32_glm", "nest32_2lev", "nest32_fixed", "nest32_stale", "nest32_noskip", "tom2d_c040"]

struct Abort <: Exception; why::String; end

mutable struct Watch
    t0::Float64; sub::Vector{Int}; io::IO; drv; nroot::Int; dtlow::Float64
end

function level_stats(H, l)
    P = H.patches[l]
    dtc = NM.cfl_dt(H, P)                        # also fills P.sig (interior cells outside the child's core)
    smax, I = findmax(P.sig)
    pos = ntuple(d -> (P.lo[d] + I[d] - G - 1 - 0.5) * P.dx, 3)
    r = ntuple(d -> G+1:G+P.n[d], 3)
    ρ = @view P.U[r..., 1]
    m2 = @view(P.U[r..., 2]).^2 .+ @view(P.U[r..., 3]).^2 .+ @view(P.U[r..., 4]).^2
    b2 = @view(P.U[r..., 6]).^2 .+ @view(P.U[r..., 7]).^2 .+ @view(P.U[r..., 8]).^2
    (dtc=dtc, rmin=Float64(minimum(ρ)), rmax=Float64(maximum(ρ)), vmax=sqrt(Float64(maximum(m2 ./ ρ.^2))),
     bmax=sqrt(Float64(maximum(b2))), smax=Float64(smax), pos=pos, cell=Tuple(I) .- G)
end

function NM.after_step!(w::Watch, H, l)
    time() - w.t0 > CAP && throw(Abort(@sprintf("wall cap %.0f s at t = %.5g", CAP, H.t[1])))
    if l > 1
        (w.sub[l] += 1) > MAX_SUB && throw(Abort("level $l took more than $MAX_SUB steps in one root step at t = $(H.t[l])"))
        return
    end
    w.nroot += 1
    amp = isempty(w.drv.log_amp) ? NaN : w.drv.log_amp[end]; mm = isempty(w.drv.log_mach_m) ? NaN : w.drv.log_mach_m[end]
    low = Inf
    for k in eachindex(H.patches)
        s = level_stats(H, k); low = min(low, s.dtc)
        @printf(w.io, "%.8g,%d,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.5f,%.5f,%.5f,%d,%d,%d,%.6g,%.6g,%.3f\n", H.t[1], k, H.nsteps[k],
                w.sub[k], s.dtc, s.rmin, s.rmax, s.vmax, s.bmax, s.smax, s.pos..., s.cell..., amp, mm, time() - w.t0)
        if s.dtc < DT_FLOOR
            flush(w.io)
            throw(Abort(@sprintf("level %d CFL step %.3g at t = %.6g: max signal %.3g at (%.4f, %.4f, %.4f), cell %s; rho %.3g..%.3g, vmax %.3g, |B|max %.3g",
                                 k, s.dtc, H.t[1], s.smax, s.pos..., s.cell, s.rmin, s.rmax, s.vmax, s.bmax)))
        end
    end
    if w.nroot % 50 == 0 || low < w.dtlow / 3
        say(@sprintf("   t %.5f  root steps %d  steps %s  smallest CFL step %.3g  M_m %.2f  amp %.3g  %.0f s", H.t[1], w.nroot,
                     H.nsteps, low, mm, amp, time() - w.t0))
    end
    w.dtlow = min(w.dtlow, low)
    fill!(w.sub, 0)
    flush(w.io)
end

function dump(H, name, why)
    for (l, P) in enumerate(H.patches)
        write(joinpath(OUT, "$(name)_L$(l).bin"), Float32.(Array(P.U)))
    end
    meta = Dict("why" => why, "t" => H.t, "nsteps" => H.nsteps, "nvar" => size(H.patches[1].U, 4), "ghosts" => G,
                "levels" => [Dict("lo" => collect(P.lo), "n" => collect(P.n), "dx" => P.dx) for P in H.patches])
    open(io -> TOML.print(io, meta), joinpath(OUT, "$(name).toml"), "w")
end

results = Dict{String,Any}()
for name in ORDER
    N, nlev, courant, tend, opt = CASES[name]
    mode = get(opt, :mode, :ct); dims = get(opt, :dims, 3); hk = Base.structdiff(opt, NamedTuple{(:mode, :dims)})
    say(@sprintf("== %s: %s, Courant %.2f, Mach 10, %s, HLL, Float64, to t = %g%s", name, dims == 2 ? "2-D nest of Tom's run_mach10.jl, root $(N)²" :
                 nlev == 1 ? "uniform $(N)³" : "nest $nlev × $(N)³ (finest $(N*2^(nlev-1))³ in the central 1/$(2^(nlev-1)))",
                 courant, uppercase(string(mode)), tend, isempty(hk) ? "" : ", " * join(("$k = $v" for (k, v) in pairs(hk)), ", ")))
    GC.gc(); CPU_RUN || CUDA.reclaim()
    cs = 1.0; B0 = sqrt(2cs^2/200)
    H = dims == 2 ?
        Hierarchy(; N0=(N, N, 1), levels=[((N÷2+1, N÷2+1, 1), (N, N, 2)), ((3N÷2+1, 3N÷2+1, 1), (N, N, 4))], mode=mode,
                  riemann=:hll, T=Float64, backend=BE, gamma=1.01, courant=courant, hk...) :
        nested_central(N, nlev; n=N, mode=mode, riemann=:hll, T=Float64, backend=BE, gamma=1.01, courant=courant, hk...)
    init!(H, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, cs^2, B0, 0.0, 0.0); vecpot=(x, y, z) -> (0.0, 0.0, B0*y))
    d = Driving(H; mach=10.0, cs=cs, kmin=1, kmax=3, zeta=1.0, two_d=dims == 2, seed=1)
    push!(H.sources, d)
    io = open(joinpath(OUT, "$name.csv"), "w")
    println(io, "t,level,nsteps,substeps,dt_cfl,rho_min,rho_max,v_max,b_max,sig_max,x,y,z,i,j,k,amp,mach_m,wall")
    w = Watch(time(), zeros(Int, nlev), io, d, 0, Inf); push!(H.sources, w)
    why = ""; t0 = time()
    try
        evolve!(H, tend); KA.synchronize(BE)
    catch e
        why = e isa Abort ? e.why : sprint(showerror, e)
        dump(H, name, why)
    end
    close(io)
    wall = time() - t0; upd = sum(prod(P.n) * s for (P, s) in zip(H.patches, H.nsteps))
    ok = isempty(why)
    say(ok ? @sprintf("   finished: %.0f s, steps %s, %.2e updates (%.0f Mcell/s), M_m %.2f, M_v %.2f, div B %s", wall, H.nsteps, upd,
                      upd/wall/1e6, d.log_mach_m[end], d.log_mach[end], join((@sprintf("%.0e", x) for x in max_divb(H)), " "))
            : "   STOPPED after $(round(Int, wall)) s, steps $(H.nsteps): $why")
    results[name] = Dict("finished" => ok, "why" => why, "wall" => wall, "steps" => H.nsteps, "updates" => upd,
                         "t" => H.t, "mach_m" => isempty(d.log_mach_m) ? NaN : d.log_mach_m[end])
    open(io -> TOML.print(io, results), joinpath(OUT, "driven_debug.toml"), "w")
    H = nothing
end
say("== done")
