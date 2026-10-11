# Diagnostics, errors, spectra and reduced outputs for an external-code result group, computed with
# the functions src/run_suite.jl uses (src/lean.jl, src/spectra.jl), so every code is measured the same way.
#   julia --project=env tools/postprocess_external.jl RESULT_GROUP_DIR
# MHD_POSTPROCESS_GPU=1 computes on the job's GPU (AthenaK jobs); otherwise on the CPU (FFTW, threads).
# Frames the config does not keep (tmp_###.bin) are measured, then deleted. Safe to rerun: groups already
# processed are only cleaned up.
using TOML, Printf, Statistics, KernelAbstractions, FFTW
const GPU=get(ENV,"MHD_POSTPROCESS_GPU","")=="1"
GPU && @eval using CUDA
include(joinpath(@__DIR__,"..","src","problems.jl"))
include(joinpath(@__DIR__,"..","src","errors.jl"))
include(joinpath(@__DIR__,"..","src","lean.jl"))
include(joinpath(@__DIR__,"..","src","spectra.jl"))
FFTW.set_num_threads(Threads.nthreads())
device(x)=GPU ? CuArray(x) : x

function load(path,shape)
    a=Array{Float32}(undef,shape...);read!(path,a)
    length(shape)==3 ? reshape(a,shape[1],shape[2],1,9) : a
end
fields(u)=ntuple(c->view(u,:,:,:,c),9)
function cleanup(dir)
    for f in readdir(dir);startswith(f,"tmp_") && endswith(f,".bin") && rm(joinpath(dir,f));end
end
function process(dir)
    meta=TOML.parsefile(joinpath(dir,"run.toml"));config=TOML.parsefile(joinpath(dirname(dir),"input.toml"))
    get(meta,"postprocessed",false) && return cleanup(dir)
    # High-resolution reference groups contain restricted fields only. Building
    # and diagnosing the full 384^3 Julia state here wastes many GiB of memory.
    any(s->haskey(s,"file") || haskey(s,"final_file"),meta["frames"]) || return
    # Midplane-only movie snapshots cannot give volume diagnostics or 3D reference errors.
    isempty(get(meta,"slice","")) || return
    name=meta["problem"];n=meta["N"];seed=meta["seed"];gamma=meta["gamma"];mach=get(config,"mach",2.0)
    planar=Problems.is_planar(name,config);shape=Tuple(meta["saved_shape"]);nf=config["frames"];rep1=meta["repeat"]==1
    sf=get(config,"slice_frames",Int[]);slice_frames=planar || !rep1 ? Int[] : sf=="all" ? collect(0:nf-1) : sf
    restrict_to=planar || !rep1 ? Int[] : [m for m in get(config,"restrict_to",Int[]) if m<n]
    spectra_every=planar || !rep1 ? 0 : Int(get(config,"spectra_every",0))
    meta["slice_frames"]=slice_frames;meta["slice_shape"]=[n,n,9];meta["slice_plane"]=n÷2+1;meta["restrict_to"]=restrict_to
    meta["spectra_every"]=spectra_every;meta["diagnostics_scope"]="tools/postprocess_external.jl with src/lean.jl ($(GPU ? "GPU" : "CPU")), full volume, Float64 accumulation"
    h,_=Problems.initial(name,n;seed,mach,par=config,plane=planar);h0=h     # planar: the z = 1 plane only
    d0=Lean.diagnostics(fields(h0);gamma);rows=String[];header=nothing;spectra_rows=String[];spectra_header=nothing
    for stamp in meta["frames"]
        file=get(stamp,"file",get(stamp,"final_file",nothing));file===nothing && continue
        frame=stamp["frame"];u=device(load(joinpath(dir,file),shape));q=fields(u)
        d=Lean.diagnostics(q;gamma,problem=name)
        ref,label=Problems.reference(name,n,stamp["time"],h;seed,mach,par=config,plane=planar)
        stamp["errors"]=Lean.errors(q,ref);stamp["reference"]=label
        if ref!==nothing && haskey(stamp,"file") && !get(stamp,"temporary",false)
            stamp["reference_file"]=@sprintf("ref_%03d.bin",frame)
            write(joinpath(dir,stamp["reference_file"]),planar ? ref[:,:,1,:] : ref)
        end
        if spectra_every>0 && (frame%spectra_every==0 || frame==nf-1)
            sp=Spectra.spectra(q[1],q[2:4],q[6:8]);cols=[k for k in keys(sp) if k!==:k];stamp["spectra"]=true
            spectra_header=join(["frame","time","k",string.(cols)...],',')
            for i in eachindex(sp.k);push!(spectra_rows,join((frame,stamp["time"],sp.k[i],(getfield(sp,c)[i] for c in cols)...),','));end
        end
        if frame in slice_frames
            stamp["slice"]=@sprintf("slice_%03d.bin",frame);write(joinpath(dir,stamp["slice"]),Array(u[:,:,n÷2+1,:]))
        end
        if !get(stamp,"temporary",false) && haskey(stamp,"file") && !isempty(restrict_to)
            stamp["restricted"]=Dict(string(m)=>@sprintf("rstate_N%d_%03d.bin",m,frame) for m in restrict_to)
            for m in restrict_to;write(joinpath(dir,stamp["restricted"][string(m)]),Lean.restrict(u,m));end
        end
        # Frames kept only for measurement lose their file entry; cleanup deletes the file.
        get(stamp,"temporary",false) && (delete!(stamp,"file");delete!(stamp,"temporary");stamp["measured_then_deleted"]=true)
        row=(frame=frame,time=stamp["time"],steps=stamp["steps"],d...,mass_drift=abs(d.mass/d0.mass-1),energy_drift=abs(d.energy/d0.energy-1))
        header===nothing && (header=join(string.(keys(row)),','));push!(rows,join(values(row),','))
        u=nothing;q=nothing;GPU && GC.gc(false)
    end
    # Reference groups save only restricted files; they have no full-resolution diagnostics.
    header===nothing || open(joinpath(dir,"diagnostics.csv"),"w") do io;println(io,header);foreach(r->println(io,r),rows);end
    spectra_header===nothing || open(joinpath(dir,"spectra.csv"),"w") do io;println(io,spectra_header);foreach(r->println(io,r),spectra_rows);end
    meta["postprocessed"]=true
    open(joinpath(dir,"run.toml.tmp"),"w") do io;TOML.print(io,meta);end;mv(joinpath(dir,"run.toml.tmp"),joinpath(dir,"run.toml");force=true)
    cleanup(dir)
end
root=abspath(ARGS[1])
for dir in sort(filter(isdir,readdir(root;join=true)))
    isfile(joinpath(dir,"run.toml")) && (process(dir);println("postprocessed ",basename(dir)))
end
