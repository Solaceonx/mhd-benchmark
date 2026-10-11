# Resource estimate for the full comparison study (GPU-hours, CPU core-hours, storage).
#   julia --project=env tools/estimate_cost.jl [--csv OUT.csv]
# Step counts come from each config's initial signal speed with the driver's CFL rule,
# calibrated by step counts measured in CPU runs of the same physics (N=32 planar, N=16 3D).
# Throughputs remain assumptions until enough completed H100 production runs replace them.
# Fixed launch/compile overhead is calibrated from the completed Sherlock H100 gates.
using TOML, Printf
include(joinpath(@__DIR__,"..","src","problems.jl"))
const P=Problems
const BUNDLE=normpath(joinpath(@__DIR__,".."))

# Every tuple is (central, optimistic, pessimistic). Cell-updates per second on one A100-class GPU. GLM figures follow the
# advisor's own notes in glmmhd_turb.jl (f16/HLL TB=6 ~2.1e9 at 512^3, ~1.8x the f32 TB=4 path,
# PPM 2.86x slower than PLM); CT and AthenaK are guesses to be replaced by measurements.
const RATES=Dict(
    "glm"=>(1.0e9,1.5e9,0.5e9), "glm_hll16"=>(1.6e9,2.1e9,0.8e9), "glm_hlld"=>(1.1e9,1.5e9,0.5e9),
    "glm_ppm"=>(0.6e9,0.9e9,0.3e9), "glm_constch"=>(1.6e9,2.1e9,0.8e9), "ct"=>(0.3e9,0.6e9,0.15e9),
    "athenak_matched"=>(0.4e9,0.8e9,0.2e9), "athenak_best"=>(0.2e9,0.4e9,0.1e9), "athenak_reference"=>(0.2e9,0.4e9,0.1e9))
const STEP_OVERHEAD=(1.5e-4,1.0e-4,4.0e-4)     # s per step: CFL scan, reduction, launches, syncs
# Steps relative to the GPU driver's rule (sum over directions, C=0.25), measured on Orszag-Tang:
# AthenaK and Athena++ use a per-direction rule with CFL 0.3; RAMSES courant 0.5 / 0.8.
const STEP_RATIO=Dict("athenak_matched"=>0.39,"athenak_best"=>0.39,"athenak_reference"=>0.39,
    "athenapp_reference"=>0.39,"ramses_matched"=>0.52,"ramses_best"=>0.34)
const RAMSES_CORE_RATE=(5.0e5,8.0e5,3.0e5)       # cell-updates/s/core, measured 4.8-5.5e5 on this Mac
const ATHENAPP_CORE_RATE=(8.0e5,1.5e6,4.0e5)
const MPI_EFFICIENCY=0.6
const RANKS=Dict("ramses_run"=>32,"athenapp_reference"=>64)
const DIAG_PER_MCELL=0.12       # s per output per Mcell: device->host, Float64 diagnostics, norms
const EXACT_REF_PER_MCELL=0.45  # s per output per Mcell: rebuilding the exact solution
const DRIVER_FIXED=(60.0,45.0,120.0)     # s per GPU-driver task: Julia start, CUDA init, compiling GLM + CT
const EXTERNAL_FIXED=(45.0,30.0,120.0)   # s per external run: launch, Kokkos/MPI init, Julia post-processing
const CONVERT_PER_MCELL=0.4              # s per output per Mcell: reading code output, writing ours, post-processing

# Steps measured by the CPU GLM runs (same CFL rule as the GPU driver): test => (N, steps).
const MEASURED=Dict("alfven"=>(32,425),"alfven_oblique"=>(32,192),"blast_2d_b0"=>(32,135),"blast_2d_b50"=>(32,147),
    "blast_2d_b100"=>(32,188),"blast_3d_b0"=>(16,51),"blast_3d_b50"=>(16,53),"blast_3d_b100"=>(16,71),"brio_wu"=>(32,72),
    "brio_wu_rotated"=>(32,51),"current_sheet"=>(32,1031),"decay"=>(16,276),"entropy"=>(32,694),"field_loop"=>(32,2332),
    "field_loop_novz"=>(32,1760),"field_loop_3d"=>(16,504),"orszag_tang"=>(32,365),"orszag_tang_3d"=>(16,364),
    "rotor"=>(32,132),"uniform"=>(32,56))

function fast(q,g,bn)
    r,vx,vy,vz,p,bx,by,bz=q;c2=g*p/r;b2=(bx^2+by^2+bz^2)/r;d=(b2+c2)/2
    sqrt(d+sqrt(max(d^2-c2*bn^2/r,0.0)))
end
"Initial maximum of sum_d(|v_d| + c_f,d), as in the driver's CFL scan."
function vmax0(name,config,n)
    u,_=P.initial(name,n;par=config,mach=get(config,"mach",2.0));g=P.gamma_of(name,config);v=0.0
    for k in 1:size(u,3),j in 1:n,i in 1:n
        r=u[i,j,k,1];m=u[i,j,k,2:4];b=u[i,j,k,6:8];p=(g-1)*(u[i,j,k,5]-sum(abs2,m)/(2r)-sum(abs2,b)/2)
        q=(r,(m./r)...,max(p,1e-6),b...);v=max(v,sum(abs(q[1+d])+fast(q,g,b[d]) for d in 1:3))
    end
    v
end
"Estimated GPU-driver steps at grid n (calibrated by measured steps when available)."
function steps_at(test,name,config,n,cache)
    base=get!(cache,test) do
        nb=P.is_planar(name,config) ? 32 : 16
        est=config["t_end"]*nb*vmax0(name,config,nb)/config["cfl"]
        haskey(MEASURED,test) ? MEASURED[test][2]/MEASURED[test][1] : 1.25*est/nb   # steps per unit N
    end
    base*n
end

function main(args)
    configs=[TOML.parsefile(p) for p in sort(filter(f->endswith(f,".toml"),readdir(joinpath(BUNDLE,"configs");join=true)))]
    rows=[];cache=Dict{String,Float64}()
    methods_default=["glm","ct"]
    ext=Dict("athenak"=>["athenak_matched","athenak_best"],"ramses"=>["ramses_matched","ramses_best"])
    for c in configs
        test=c["name"];name=get(c,"problem",test);planar=P.is_planar(name,c);R=c["repeats"];F=c["frames"]
        snaps=length(get(c,"snapshot_frames",collect(0:F-1)));seeds=length(get(c,"seeds",[42]))
        exact=name in ("uniform","entropy","alfven","alfven_oblique")
        for n in c["grids"]
            cells=n^3;mc=cells/1e6;s=steps_at(test,name,c,n,cache)
            # GPU driver: one task runs every method x repeats; diagnostics at every output.
            for (i,level) in enumerate((:central,:optimistic,:pessimistic))
                t=DRIVER_FIXED[i]
                for m in get(c,"methods",methods_default)
                    rate=RATES[m][i];tb=m=="glm_hll16"||m=="glm_constch" ? (n%6==0 ? 1.0 : 0.7) : 1.0
                    per=s*max(STEP_OVERHEAD[i],cells/(rate*tb))
                    t+=R*(per+F*mc*(DIAG_PER_MCELL+(exact ? EXACT_REF_PER_MCELL : 0.0)))
                    level in (:central,:pessimistic) && push!(rows,(test=test,n=n,code="gpu_$(level)",method=m,run_seconds=per,seeds=seeds))
                end
                level in (:central,:pessimistic) && push!(rows,(test=test,n=n,code="gpu_task_$(level)",method="all",run_seconds=t,seeds=seeds))
                push!(rows,(test=test,n=n,code="gpu_total_$(level)",method="all",run_seconds=t*seeds,seeds=seeds))
            end
            # Storage returned by the GPU driver (repeat 1 snapshots; CT adds faces; exact refs).
            vol=planar ? n^2 : n^3;per_snap=vol*9*4
            bytes=length(get(c,"methods",methods_default))*snaps*per_snap+snaps*3*vol*4+(exact ? snaps*per_snap : 0)
            push!(rows,(test=test,n=n,code="bytes_gpu",method="all",run_seconds=float(bytes*seeds),seeds=seeds))
            test=="kh_localized_dtjump" && continue
            for (code,ms) in ext, m in ms
                code=="ramses" && (n&(n-1)!=0 || !(n in get(c,"ramses_grids",c["grids"]))) && continue
                se=s*STEP_RATIO[m]
                for (i,level) in enumerate((:central,:optimistic,:pessimistic))
                    if code=="athenak"
                        per=se*max(STEP_OVERHEAD[i],cells/RATES[m][i])
                        t=EXTERNAL_FIXED[i]+R*per+(F+R-1)*mc*CONVERT_PER_MCELL
                        push!(rows,(test=test,n=n,code="athenak_total_$(level)",method=m,run_seconds=t*seeds,seeds=seeds))
                    else
                        rate=RAMSES_CORE_RATE[i]*RANKS["ramses_run"]*MPI_EFFICIENCY
                        t=EXTERNAL_FIXED[i]+R*se*cells/rate+(F+R-1)*mc*CONVERT_PER_MCELL
                        push!(rows,(test=test,n=n,code="ramses_total_$(level)",method=m,run_seconds=t*seeds*RANKS["ramses_run"],seeds=seeds))
                        level==:central && push!(rows,(test=test,n=n,code="ramses_wall",method=m,run_seconds=t,seeds=seeds))
                    end
                end
                push!(rows,(test=test,n=n,code="bytes_ext",method=m,run_seconds=float((snaps*per_snap+(exact ? snaps*per_snap : 0))*seeds),seeds=seeds))
            end
        end
        # References: AthenaK at reference_n (2D for planar tests), Athena++ at reference_check_n on CPU.
        nref=get(c,"reference_n",0)
        if nref>0
            cells=planar ? nref^2 : nref^3;mc=cells/1e6;s=steps_at(test,name,c,nref,cache)*STEP_RATIO["athenak_reference"]
            for (i,level) in enumerate((:central,:optimistic,:pessimistic))
                t=EXTERNAL_FIXED[i]+s*max(STEP_OVERHEAD[i],cells/RATES["athenak_reference"][i])+F*mc*CONVERT_PER_MCELL
                push!(rows,(test=test,n=nref,code="ref_total_$(level)",method="athenak_reference",run_seconds=t*seeds,seeds=seeds))
                nc=get(c,"reference_check_n",nref÷2);cc=planar ? nc^2 : nc^3
                sc=steps_at(test,name,c,nc,cache)*STEP_RATIO["athenapp_reference"]
                tc=EXTERNAL_FIXED[i]+sc*cc/(ATHENAPP_CORE_RATE[i]*RANKS["athenapp_reference"]*MPI_EFFICIENCY)+F*cc/1e6*CONVERT_PER_MCELL
                push!(rows,(test=test,n=nc,code="check_total_$(level)",method="athenapp_reference",run_seconds=tc*seeds*RANKS["athenapp_reference"],seeds=seeds))
                level==:central && push!(rows,(test=test,n=nc,code="check_wall",method="athenapp_reference",run_seconds=tc,seeds=seeds))
                level==:central && push!(rows,(test=test,n=nref,code="ref_wall",method="athenak_reference",run_seconds=t,seeds=seeds))
            end
            push!(rows,(test=test,n=nref,code="bytes_ref",method="restricted",run_seconds=float(F*sum((planar ? m^2 : m^3)*36 for m in c["grids"] if nref%m==0)*2*seeds),seeds=seeds))
        end
    end
    total(prefix)=sum(r.run_seconds for r in rows if r.code==prefix;init=0.0)
    h(x)=x/3600
    println("## Estimated resources (central [optimistic–pessimistic])\n")
    @printf("| Item | Estimate |\n|---|---|\n")
    for (label,key,unit,conv) in (("GPU driver (original GLM + CT)","gpu_total","GPU-h",h),("AthenaK runs","athenak_total","GPU-h",h),
                                  ("AthenaK references","ref_total","GPU-h",h),("RAMSES runs (32 ranks)","ramses_total","core-h",h),
                                  ("Athena++ checks (64 ranks)","check_total","core-h",h))
        @printf("| %s | %.1f %s [%.1f–%.1f] |\n",label,conv(total(key*"_central")),unit,conv(total(key*"_optimistic")),conv(total(key*"_pessimistic")))
    end
    gpu=[h(total(k*"_"*l)) for l in ("central","optimistic","pessimistic") for k in ("gpu_total","athenak_total","ref_total")]
    @printf("| **All GPU work** | **%.0f GPU-h [%.0f–%.0f]** |\n",sum(gpu[1:3]),sum(gpu[4:6]),sum(gpu[7:9]))
    @printf("| **All CPU work** | **%.0f core-h [%.0f–%.0f]** |\n",h(total("ramses_total_central")+total("check_total_central")),
        h(total("ramses_total_optimistic")+total("check_total_optimistic")),h(total("ramses_total_pessimistic")+total("check_total_pessimistic")))
    gb(x)=x/2^30
    @printf("| Returned data (uncompressed Float32) | %.1f GB (GPU driver %.1f, external %.1f, references %.2f) |\n",
        gb(total("bytes_gpu")+total("bytes_ext")+total("bytes_ref")),gb(total("bytes_gpu")),gb(total("bytes_ext")),gb(total("bytes_ref")))
    worst=sort([r for r in rows if r.code=="gpu_task_central"],by=r->-r.run_seconds)[1]
    @printf("\nLongest single GPU-driver array task: %s N=%d, %.1f h (central).\n",worst.test,worst.n,h(worst.run_seconds))
    println("\n## Largest consumers (central)\n\n| Test | N | Item | Hours |\n|---|---:|---|---:|")
    big=sort([r for r in rows if r.code in ("gpu_total_central","athenak_total_central","ref_total_central","ramses_total_central","check_total_central")],by=r->-r.run_seconds)
    for r in big[1:min(12,end)]
        @printf("| %s | %d | %s | %.1f %s |\n",r.test,r.n,replace(r.code,"_total_central"=>"")*(r.method=="all" ? "" : " "*r.method),h(r.run_seconds),startswith(r.code,"ramses")||startswith(r.code,"check") ? "core-h" : "GPU-h")
    end
    println("\n## Limit checks (central and pessimistic estimates)\n")
    limits=Dict(c["name"]=>(get(c,"max_run_seconds",1800),c) for c in configs)
    flagged=false
    for r in rows
        if r.code in ("gpu_central","gpu_pessimistic") && r.run_seconds>(r.code=="gpu_central" ? 0.8 : 1.0)*limits[r.test][1]
            flagged=true;@printf("- %s N=%d %s (%s): one run ≈ %.0f s vs max_run_seconds %d\n",r.test,r.n,r.method,r.code,r.run_seconds,limits[r.test][1])
        end
        if r.code in ("gpu_task_central","gpu_task_pessimistic")
            lim=get(limits[r.test][2],"slurm_time","01:00:00");sec=sum(parse.(Int,split(lim,":")).*[3600,60,1])
            r.run_seconds>(r.code=="gpu_task_central" ? 0.8 : 1.0)*sec && (flagged=true;@printf("- %s N=%d GPU-driver task (%s): ≈ %.1f h vs Slurm limit %s\n",r.test,r.n,r.code,h(r.run_seconds),lim))
        end
        if r.code in ("ramses_wall","check_wall","ref_wall") && r.run_seconds>0.8*7200
            flagged=true;@printf("- %s N=%d %s: ≈ %.1f h wall vs 2 h job limit\n",r.test,r.n,r.method,h(r.run_seconds))
        end
        if r.code in ("ramses_total_pessimistic","check_total_pessimistic","ref_total_pessimistic","athenak_total_pessimistic")
            wall=r.run_seconds/r.seeds/(startswith(r.code,"ramses") ? 32 : startswith(r.code,"check") ? 64 : 1)
            lim=get(limits[r.test][2],"slurm_time","02:00:00");sec=sum(parse.(Int,split(lim,":")).*[3600,60,1])
            wall>sec && (flagged=true;@printf("- %s N=%d %s (pessimistic): ≈ %.1f h wall vs Slurm limit %s\n",r.test,r.n,r.method,h(wall),lim))
        end
    end
    flagged || println("- No modeled run exceeds its per-run or Slurm time limit, including the pessimistic rates.")
    i=findfirst(==("--csv"),args)
    if i!==nothing
        open(args[i+1],"w") do io
            println(io,"test,N,item,method,seconds,seeds")
            for r in rows;println(io,join((r.test,r.n,r.code,r.method,r.run_seconds,r.seeds),','));end
        end
    end
end
main(ARGS)
