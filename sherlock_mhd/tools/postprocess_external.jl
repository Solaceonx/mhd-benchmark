# Diagnostics, errors and reference snapshots for an external-code result group, computed with
# exactly the functions src/run_suite.jl uses, so every code is measured the same way.
#   julia tools/postprocess_external.jl RESULT_GROUP_DIR
using TOML, Printf, Statistics
include(joinpath(@__DIR__,"..","src","problems.jl"))
include(joinpath(@__DIR__,"..","src","errors.jl"))

function load(path,shape)
    a=Array{Float32}(undef,shape...);read!(path,a)
    length(shape)==3 ? reshape(a,shape[1],shape[2],1,9) : a
end
function process(dir)
    meta=TOML.parsefile(joinpath(dir,"run.toml"));config=TOML.parsefile(joinpath(dirname(dir),"input.toml"))
    # High-resolution reference groups contain restricted fields only. Building
    # and diagnosing the full 384^3 Julia state here wastes many GiB of memory.
    any(s->haskey(s,"file") || haskey(s,"final_file"),meta["frames"]) || return
    name=meta["problem"];n=meta["N"];seed=meta["seed"];gamma=meta["gamma"];mach=get(config,"mach",2.0)
    planar=Problems.is_planar(name,config);shape=Tuple(meta["saved_shape"])
    h,_=Problems.initial(name,n;seed,mach,par=config);h0=planar ? h[:,:,1:1,:] : h
    d0=Problems.diagnostics(h0;gamma);rows=String[];header=nothing
    for stamp in meta["frames"]
        file=get(stamp,"file",get(stamp,"final_file",nothing));file===nothing && continue
        u=load(joinpath(dir,file),shape);d=Problems.diagnostics(u;gamma)
        ref,label=Problems.reference(name,n,stamp["time"],h;seed,mach,par=config)
        ref!==nothing && planar && (ref=ref[:,:,1:1,:])
        stamp["errors"]=solution_errors(u,ref);stamp["reference"]=label
        if ref!==nothing && haskey(stamp,"file")
            stamp["reference_file"]=@sprintf("ref_%03d.bin",stamp["frame"])
            write(joinpath(dir,stamp["reference_file"]),planar ? ref[:,:,1,:] : ref)
        end
        row=(frame=stamp["frame"],time=stamp["time"],steps=stamp["steps"],d...,mass_drift=abs(d.mass/d0.mass-1),energy_drift=abs(d.energy/d0.energy-1))
        header===nothing && (header=join(string.(keys(row)),','));push!(rows,join(values(row),','))
    end
    # Reference groups save only restricted files; they have no full-resolution diagnostics.
    header===nothing || open(joinpath(dir,"diagnostics.csv"),"w") do io;println(io,header);foreach(r->println(io,r),rows);end
    open(joinpath(dir,"run.toml.tmp"),"w") do io;TOML.print(io,meta);end;mv(joinpath(dir,"run.toml.tmp"),joinpath(dir,"run.toml");force=true)
end
root=abspath(ARGS[1])
for dir in sort(filter(isdir,readdir(root;join=true)))
    isfile(joinpath(dir,"run.toml")) && (process(dir);println("postprocessed ",basename(dir)))
end
