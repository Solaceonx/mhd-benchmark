include(joinpath(@__DIR__,"isothermal_hydro.jl"))
using FFTW, Random, Statistics, TOML
wave(i,n)=i-1<=n÷2 ? i-1 : i-1-n
function mode_geometry(n)
    kmag=[sqrt(wave(i,n)^2+wave(j,n)^2+wave(k,n)^2) for i in 1:n,j in 1:n,k in 1:n]
    kmag, floor.(Int,kmag).+1
end
function gaussian_solenoidal(rng,n,kmag)
    f=[fft(randn(rng,Float64,n,n,n)) for _ in 1:3]
    expected=0.0
    for k in 1:n,j in 1:n,i in 1:n
        mag=kmag[i,j,k]
        if 1<=mag<=4
            q=(wave(i,n),wave(j,n),wave(k,n))
            dot=sum(q[d]*f[d][i,j,k] for d in 1:3)/mag^2
            env=exp(-0.5*((mag-2)/0.65)^2)
            expected+=2env^2/n^3
            for d in 1:3
                f[d][i,j,k]=env*(f[d][i,j,k]-q[d]*dot)
            end
        else
            for d in 1:3
                f[d][i,j,k]=0
            end
        end
    end
    [real.(ifft(x))./sqrt(expected) for x in f]
end
function kick!(a,p,acc,amp,dt)
    r=@view a[IG+1:IG+p.N,:,:,1]; mass=sum(Float64,r)
    meanacc=[sum(r.*acc[d])/mass for d in 1:3]
    for d in 1:3
        @views a[IG+1:IG+p.N,:,:,d+1] .+= Float32.(dt*amp .* r .* (acc[d].-meanacc[d]))
    end
    iso_ghost!(a,p)
end
function fields(a,p)
    r=Float64.(a[IG+1:IG+p.N,:,:,1])
    v=[Float64.(a[IG+1:IG+p.N,:,:,d+1])./r for d in 1:3]
    r,v
end
function spectra(a,p,shell)
    r,v=fields(a,p); d=iso_stats(a,p); ns=maximum(shell)
    ev=zeros(ns); ek=zeros(ns)
    for c in 1:3
        fv=fft(v[c].-d.bulk[c]); fw=fft(sqrt.(r).*(v[c].-d.bulk[c]))
        @inbounds for idx in eachindex(shell)
            s=shell[idx]
            ev[s]+=0.5abs2(fv[idx])/p.N^6; ek[s]+=0.5abs2(fw[idx])/p.N^6
        end
    end
    abs(sum(ek)-d.kinetic) < 1e-9*max(d.kinetic,1) || error("Parseval check failed")
    ev,ek
end
function write_snapshot(root,label,frame,a,p)
    r=@view a[IG+1:IG+p.N,:,:,1]
    open(joinpath(root,@sprintf("%s_density_%03d.bin",label,frame)),"w") do io
        write(io,Array(@view r[:,:,p.N÷2]))
    end
end
function save_state(root,name,a,p)
    open(joinpath(root,name*".bin"),"w") do io
        write(io,Array(@view a[IG+1:IG+p.N,:,:,:]))
    end
end
function remove_gap(a,p,kmag,lo,hi)
    r,v=fields(a,p); orig=iso_stats(a,p)
    for c in 1:3
        f=fft(v[c]); f[(kmag.>=lo).&(kmag.<hi)].=0
        v[c]=real.(ifft(f))
        # Restore common total momentum by adding only the k=0 velocity mode.
        v[c].+=orig.bulk[c]-sum(r.*v[c])/sum(r)
    end
    b=copy(a)
    for c in 1:3
        @views b[IG+1:IG+p.N,:,:,c+1].=Float32.(r.*v[c])
    end
    iso_ghost!(b,p)
    b
end
function stationarity(history,tcross,t)
    window=[h for h in history if h.t>t-4tcross-1e-8]
    length(window)>=16 || return (false,Dict{String,Any}())
    left=[h for h in window if h.t<=t-2tcross]; right=[h for h in window if h.t>t-2tcross]
    isempty(left) && return (false,Dict{String,Any}())
    rel(x,y)=abs(x-y)/max((x+y)/2,1e-10)
    kdrift=rel(mean(h.kinetic for h in left),mean(h.kinetic for h in right))
    sdrift=rel(mean(h.logstd for h in left),mean(h.logstd for h in right))
    el=reduce(+, [h.spectrum for h in left])/length(left); er=reduce(+, [h.spectrum for h in right])/length(right)
    specdrift=sum(abs.(el/sum(el)-er/sum(er)))
    mach=mean(h.mach for h in window)
    result=Dict{String,Any}("mean_mach"=>mach,"kinetic_mean_drift"=>kdrift,
        "log_density_std_drift"=>sdrift,"normalized_spectrum_l1_drift"=>specdrift,
        "window_crossings"=>4,"criteria"=>"8<=mean Mach<=12; K drift<0.25; log-density std drift<0.15; normalized velocity spectrum L1 drift<0.25")
    (8<=mach<=12 && kdrift<0.25 && sdrift<0.15 && specdrift<0.25),result
end
function run_experiment(;n=64,root=joinpath(@__DIR__,"out","spectral_gap_$n"),frames=101,decay_crossings=3.0)
    n>=32 && n%IW==0 || error("N must be >=32 and divisible by $IW")
    isdir(root) && !isempty(readdir(root)) && error("Output directory not empty: $root")
    mkpath(root); p=IsoParams(N=n); kmag,shell=mode_geometry(n); rng=MersenneTwister(20260923)
    a=zeros(Float32,n+2IG,n,n,4); a[:,:,:,1].=1; b=similar(a)
    iso_ghost!(a,p); iso_step!(b,a,p,0.001f0) # compile; leave initial state intact
    acc=gaussian_solenoidal(rng,n,kmag); amp=200.0
    tcross=0.1; force_interval=0.005; correlation_time=0.05
    alpha=exp(-force_interval/correlation_time); beta=sqrt(1-alpha^2)
    t=0.0; nextforce=force_interval; nextsample=0.0; nextreport=tcross
    sample_interval=0.025; history=NamedTuple[]; prep_rejects=prep_steps=0
    frozen_since=4tcross; settled=false; stability=Dict{String,Any}(); start=time()
    max_mass_error=0.0; max_momentum_error=0.0
    check!(d)=begin
        max_mass_error=max(max_mass_error,abs(d.mass-1))
        max_momentum_error=max(max_momentum_error,maximum(abs,d.momentum))
        max_mass_error<1e-5 && max_momentum_error<1e-5 || error("Mass/momentum conservation check failed")
    end
    open(joinpath(root,"preparation.csv"),"w") do io
        println(io,"time,t_cross,mach,kinetic,log_density_std,rho_min,rho_max,forcing_amplitude")
        while t<=20tcross+1e-8
            d=iso_stats(a,p); d===nothing && error("Invalid driven state"); check!(d)
            if t>=nextsample-1e-8
                ev,_=spectra(a,p,shell)
                push!(history,(;t,mach=d.mach,kinetic=d.kinetic,logstd=d.logstd,spectrum=ev))
                println(io,join((t,t/tcross,d.mach,d.kinetic,d.logstd,d.rmin,d.rmax,amp),',')); flush(io)
                nextsample+=sample_interval
            end
            if t>=nextreport-1e-8
                @printf("PREP %.1f crossings: Mach %.2f, sigma(log rho) %.2f, amp %.1f, elapsed %.1fs\n",t/tcross,d.mach,d.logstd,amp,time()-start); flush(stdout)
                if t<=4tcross+1e-8
                    amp*=clamp((10/max(d.mach,1))^1.5,0.5,2.0)
                elseif t-frozen_since>=4tcross-1e-8
                    settled,stability=stationarity(history,tcross,t)
                    println("  stability: ",stability); flush(stdout)
                    settled && break
                    if !(8<=stability["mean_mach"]<=12)
                        amp*=clamp((10/stability["mean_mach"])^1.5,0.6,1.6)
                        frozen_since=t
                    end
                end
                nextreport+=tcross
            end
            if t>=nextforce-1e-8
                noise=gaussian_solenoidal(rng,n,kmag)
                for c in 1:3
                    acc[c].=alpha.*acc[c].+beta.*noise[c]
                end
                nextforce+=force_interval
            end
            dt=min(p.courant/n/d.signal,nextforce-t,nextsample-t,nextreport-t)
            dt>0 || error("Nonpositive preparation dt")
            h,d,reject=iso_advance!(b,a,p,dt); a,b=b,a
            kick!(a,p,acc,amp,h)
            t+=h; prep_steps+=1; prep_rejects+=reject
        end
    end
    settled || error("Preparation did not pass stationarity checks; see preparation.csv")
    prep_seconds=time()-start; save_state(root,"prepared_state",a,p)
    original=iso_stats(a,p); decay_tcross=1/original.mach
    gaplo=4; gaphi=7; g=remove_gap(a,p,kmag,gaplo,gaphi)
    gapstats=iso_stats(g,p); original_ev,original_ek=spectra(a,p,shell); gev,gek=spectra(g,p,shell)
    gapbins=gaplo+1:gaphi
    sum(gev[gapbins])<1e-10*sum(original_ev[gapbins]) || error("Gap was not removed")
    maximum(abs.(gapstats.momentum-original.momentum))<1e-6 || error("Gap changed total momentum")
    a[:,:,:,1]==g[:,:,:,1] || error("Gap changed density")
    velocity_fraction=sum(original_ev[gapbins])/sum(original_ev[2:end])
    @printf("PREPARED Mach %.2f. Gap removes %.2f%% of nonzero-mode velocity variance; kinetic energy changes %.2f%%.\n",original.mach,100velocity_fraction,100(gapstats.kinetic/original.kinetic-1))
    save_state(root,"gap_initial_state",g,p)
    results=Dict{String,Any}()
    for (label,initial) in (("control",a),("gap",g))
        u=copy(initial); unew=similar(u); tdec=0.0; steps=rejects=0
        d=iso_stats(u,p); branch_start=time()
        open(joinpath(root,label*"_diagnostics.csv"),"w") do io
          open(joinpath(root,label*"_spectra.csv"),"w") do sp
            println(io,"frame,time,t_cross,mach,kinetic,log_density_std,rho_min,rho_max,mass,momentum_x,momentum_y,momentum_z,velocity_gap_power,weighted_gap_power")
            println(sp,"frame,k_lower,k_upper,velocity_power,kinetic_power")
            for (idx,target) in enumerate(range(0,decay_crossings*decay_tcross;length=frames))
                while tdec<target-1e-9*decay_tcross
                    dt=min(p.courant/n/d.signal,target-tdec)
                    h,d,r=iso_advance!(unew,u,p,dt); u,unew=unew,u
                    tdec+=h; steps+=1; rejects+=r; check!(d)
                end
                ev,ek=spectra(u,p,shell)
                write_snapshot(root,label,idx-1,u,p)
                println(io,join((idx-1,tdec,tdec/decay_tcross,d.mach,d.kinetic,d.logstd,d.rmin,d.rmax,d.mass,d.momentum...,sum(ev[gapbins]),sum(ek[gapbins])),',')); flush(io)
                for s in eachindex(ev)
                    println(sp,join((idx-1,s-1,s,ev[s],ek[s]),','))
                end
                if idx==1 || idx%20==0 || idx==frames
                    @printf("%s frame %d/%d t/tcross=%.2f Mach=%.2f gap velocity power=%.4g\n",label,idx,frames,tdec/decay_tcross,d.mach,sum(ev[gapbins])); flush(stdout)
                end
            end
          end
        end
        save_state(root,label*"_final_state",u,p)
        results[label]=Dict("steps"=>steps,"rejected_steps"=>rejects,"seconds"=>time()-branch_start,"final_mach"=>d.mach,"final_kinetic"=>d.kinetic)
    end
    meta=Dict{String,Any}("N"=>n,"frames"=>frames,"seed"=>20260923,"cs"=>1.0,"threads"=>Threads.nthreads(),
        "eos"=>"isothermal P=rho*cs^2; four conserved fields; no total-energy equation",
        "solver"=>"CPU SIMD MUSCL-Hancock PLM/LLF adapted from Orszag-Tang workflow",
        "gap_lower"=>gaplo,"gap_upper_exclusive"=>gaphi,"preparation_crossings"=>t/tcross,"preparation_seconds"=>prep_seconds,
        "nominal_preparation_tcross"=>tcross,"preparation_steps"=>prep_steps,"preparation_rejected_steps"=>prep_rejects,
        "forcing_amplitude_final"=>amp,"forcing_kmin"=>1,"forcing_kmax"=>4,"forcing_peak_k"=>2,"forcing_width"=>0.65,
        "forcing_correlation_time"=>correlation_time,"forcing_update_interval"=>force_interval,
        "prepared_mach"=>original.mach,"gap_initial_mach"=>gapstats.mach,"control_initial_kinetic"=>original.kinetic,
        "gap_initial_kinetic"=>gapstats.kinetic,"fraction_velocity_variance_removed"=>velocity_fraction,
        "fraction_kinetic_removed"=>1-gapstats.kinetic/original.kinetic,"decay_tcross"=>decay_tcross,
        "decay_crossings"=>decay_crossings,"max_mass_error"=>max_mass_error,"max_momentum_error"=>max_momentum_error,
        "stationarity"=>stability,"branches"=>results,"total_seconds"=>time()-start,
        "state_format"=>"little-endian Float32, Julia column-major (N,N,N,4): rho,mx,my,mz",
        "spectral_normalization"=>"shell sums: 0.5*abs(fft(v-v_bulk))^2/N^6; weighted uses sqrt(rho)*(v-v_bulk); shells [k,k+1)")
    open(joinpath(root,"metadata.toml"),"w") do io
        TOML.print(io,meta)
    end
    println("DONE ",root," in ",round(time()-start,digits=1)," seconds")
end
if abspath(PROGRAM_FILE)==@__FILE__
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 64
    root=length(ARGS)>1 ? abspath(ARGS[2]) : joinpath(@__DIR__,"out","spectral_gap_$n")
    run_experiment(;n,root)
end
