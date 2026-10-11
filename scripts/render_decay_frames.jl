# Small 3D hydro decay run using the same CPU SIMD integrator as render_ot_frames.jl.
include(joinpath(@__DIR__, "..", "GLMMHDTurb", "cpu_simd.jl"))
using FFTW, Random

function initial_velocity(N, seed, mach, cs)
    rng = MersenneTwister(seed)
    vk = [fft(randn(rng, Float64, N, N, N)) for _ in 1:3]
    wave(i) = i - 1 <= N ÷ 2 ? i - 1 : i - 1 - N
    for k in 1:N, j in 1:N, i in 1:N
        q = (wave(i), wave(j), wave(k))
        q2 = sum(abs2, q)
        if 1 <= q2 <= 9
            longitudinal = sum(q[d] * vk[d][i,j,k] for d in 1:3) / q2
            envelope = exp(-0.5 * ((sqrt(q2) - 2) / 0.65)^2)
            for d in 1:3
                vk[d][i,j,k] = envelope * (vk[d][i,j,k] - q[d] * longitudinal)
            end
        else
            for d in 1:3
                vk[d][i,j,k] = 0
            end
        end
    end
    v = [real.(ifft(a)) for a in vk]
    for a in v
        a .-= sum(a) / length(a)
    end
    rms = sqrt(sum(sum(abs2, a) for a in v) / N^3)
    return [Float32.(a .* (mach * cs / rms)) for a in v]
end

# Inspect raw conserved variables: do not hide negative pressure behind a floor.
function inspect_state(u, p)
    N = p.N
    mass = energy = kinetic = sound2mass = localmach2mass = 0.0
    mxsum = mysum = mzsum = 0.0
    rhomin = pmin = Inf
    rhomax = signal = bmax = 0.0
    @inbounds for k in 1:N, j in 1:N, i in XG+1:XG+N
        rho = Float64(u[i,j,k,1])
        mx, my, mz = Float64(u[i,j,k,2]), Float64(u[i,j,k,3]), Float64(u[i,j,k,4])
        E = Float64(u[i,j,k,5])
        ke = (mx^2 + my^2 + mz^2) / (2rho)
        pressure = (p.gamma - 1) * (E - ke)
        isfinite(rho + E + ke + pressure) && rho > p.smallr && pressure > G.pfloor(p) ||
            error("Invalid raw state at ($i,$j,$k): rho=$rho, pressure=$pressure")
        cs2 = p.gamma * pressure / rho
        signal = max(signal, (abs(mx) + abs(my) + abs(mz)) / rho + 3sqrt(cs2))
        mass += rho; energy += E; kinetic += ke
        sound2mass += rho * cs2; localmach2mass += 2ke / cs2
        mxsum += mx; mysum += my; mzsum += mz
        rhomin = min(rhomin, rho); rhomax = max(rhomax, rho); pmin = min(pmin, pressure)
        for d in 6:9
            bmax = max(bmax, abs(u[i,j,k,d]))
        end
    end
    bmax == 0 || error("Magnetic field or GLM variable became nonzero")
    bulk2 = (mxsum^2 + mysum^2 + mzsum^2) / mass^2
    vrms = sqrt(max(0, 2kinetic / mass - bulk2))
    return (; mass=mass/N^3, energy=energy/N^3, kinetic=kinetic/N^3,
            mach=vrms/sqrt(sound2mass/mass), mach_local=sqrt(localmach2mass/mass),
            mach_initial_cs=vrms/p.cs0, cs_rms=sqrt(sound2mass/mass),
            rhomin, rhomax, pmin, bulk_speed=sqrt(bulk2), signal)
end

function render_decay(; N=64, mach=10.0, crossings=5.0, nframes=121, seed=42,
                      outdir=joinpath(@__DIR__, "out", "decay_64"))
    N >= 16 && N % W == 0 || error("N must be >=16 and divisible by $W")
    nframes >= 2 && crossings > 0 && mach > 0 || error("Invalid run parameters")
    isdir(outdir) && !isempty(readdir(outdir)) && error("Output directory is not empty: $outdir")
    mkpath(outdir)
    # Keep GLM algebra defined, but its artificial wave speed negligible for B=0.
    p = G.Params(N=N, gamma=1.01f0, cs0=1f0, b0=0f0, courant=0.3f0,
                 glm_ch_scale=1f-6)
    pad1 = zeros(Float32, N+2XG, N, N, 9)
    pad2 = similar(pad1)
    v = initial_velocity(N, seed, mach, p.cs0)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        x = i+XG
        pad1[x,j,k,1] = p.rho0
        for d in 1:3
            pad1[x,j,k,d+1] = p.rho0*v[d][i,j,k]
        end
        pad1[x,j,k,5] = (p.rho0*p.cs0^2/p.gamma)/(p.gamma-1f0) +
                        0.5f0*p.rho0*sum(v[d][i,j,k]^2 for d in 1:3)
    end
    refresh_xghost!(pad1, N)
    d0 = inspect_state(pad1, p)
    abs(d0.mach_initial_cs-mach) < 1e-5 || error("Mach normalization failed")
    tcross = Float64(p.boxlen)/(mach*p.cs0)
    targets = collect(range(0.0, crossings*tcross; length=nframes))
    @printf("Hydro decay: %d^3, %d threads, Mach %.2f, gamma %.3f, %.1f crossing times\n",
            N, Threads.nthreads(), d0.mach, p.gamma, crossings)
    simd_step!(pad2, pad1, p, Float32(p.courant*G.dxof(p)/d0.signal)) # compile only
    t = 0.0; step = 0; rejected_steps = 0; d = d0
    started = time()
    max_mass_error = max_energy_error = 0.0
    open(joinpath(outdir, "diagnostics.csv"), "w") do io
        println(io, "frame,time,t_cross,steps,mass,total_energy,kinetic_energy,mach,mach_local,mach_initial_cs,cs_rms,rho_min,rho_max,p_min,bulk_speed")
        for (frame, target) in enumerate(targets)
            while t < target
                dt = min(Float64(p.courant*G.dxof(p)/d.signal), target-t)
                # Round down so the integrator never advances past an output time.
                dt32 = Float32(dt)
                Float64(dt32) > dt && (dt32 = prevfloat(dt32))
                dt32 > 0 || error("Time step underflow")
                candidate = nothing
                for attempt in 1:13
                    simd_step!(pad2, pad1, p, dt32)
                    try
                        candidate = inspect_state(pad2, p)
                        break
                    catch err
                        attempt == 13 && rethrow()
                        rejected_steps += 1
                        if rejected_steps <= 5 || rejected_steps % 50 == 0
                            @printf("Rejected step %d at t=%.5f; retry with dt=%.3g\n", rejected_steps, t, dt32/2)
                        end
                        dt32 /= 2f0
                    end
                end
                refresh_xghost!(pad2, N)
                pad1, pad2 = pad2, pad1
                t += Float64(dt32); step += 1
                target-t < 1e-8*tcross && (t=target)
                d = candidate
                max_mass_error = max(max_mass_error, abs(d.mass/d0.mass-1))
                max_energy_error = max(max_energy_error, abs(d.energy/d0.energy-1))
                max_mass_error < 1e-4 && max_energy_error < 1e-4 || error("Conservation check failed")
                step < 100000 || error("Step limit exceeded")
            end
            slice = Array(@view pad1[XG+1:XG+N,:,N÷2,1])
            column = dropdims(sum(@view(pad1[XG+1:XG+N,:,:,1]); dims=3); dims=3) ./ N
            open(joinpath(outdir, @sprintf("slice_%03d.bin", frame-1)), "w") do f
                write(f, slice)
            end
            open(joinpath(outdir, @sprintf("column_%03d.bin", frame-1)), "w") do f
                write(f, column)
            end
            println(io, join((frame-1,t,t/tcross,step,d.mass,d.energy,d.kinetic,d.mach,
                             d.mach_local,d.mach_initial_cs,d.cs_rms,d.rhomin,d.rhomax,d.pmin,d.bulk_speed), ','))
            flush(io)
            if frame == 1 || frame % 10 == 0 || frame == nframes
                @printf("frame %3d/%d  t/tcross=%.2f  Mach=%.2f  K/K0=%.3f  rho=[%.3g,%.3g] elapsed=%.1fs\n",
                        frame,nframes,t/tcross,d.mach,d.kinetic/d0.kinetic,d.rhomin,d.rhomax,time()-started)
                flush(stdout)
            end
        end
    end
    open(joinpath(outdir, "manifest.txt"), "w") do io
        for (key,val) in ("N"=>N, "nframes"=>nframes, "mach0"=>mach, "gamma"=>p.gamma,
                          "seed"=>seed, "t_cross"=>tcross, "t_end"=>targets[end],
                          "threads"=>Threads.nthreads(), "steps"=>step, "rejected_steps"=>rejected_steps,
                          "runtime_seconds"=>time()-started,
                          "max_mass_relative_error"=>max_mass_error, "max_energy_relative_error"=>max_energy_error,
                          "initial_condition"=>"3D solenoidal random modes 1<=|k|<=3; amplitude envelope centered at 2",
                          "solver"=>"Orszag-Tang CPU SIMD PLM/LLF; B=psi=0; no forcing")
            println(io, key, "=", val)
        end
    end
    @printf("Done: %d steps, %.1fs; maximum relative mass/energy errors %.3e / %.3e\n",
            step,time()-started,max_mass_error,max_energy_error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    N = length(ARGS)>=1 ? parse(Int,ARGS[1]) : 64
    outdir = length(ARGS)>=2 ? abspath(ARGS[2]) : joinpath(@__DIR__,"out","decay_$N")
    render_decay(; N, outdir)
end
