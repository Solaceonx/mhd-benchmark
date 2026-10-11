include("run_suite.jl")
using TOML
length(ARGS)==1 || error("Usage: smoke.jl NEW_OUTPUT_DIRECTORY")
# Execution check: every config at 32^3 for a few steps, every method it lists (one call per method for
# split_methods configs). It is not a convergence check, and a solver result is not a failure: the half-precision
# (f16) GLM kernels stop on a nonphysical state on the strongest blasts at 32^3, as they did in the first campaign.
# Such stops are reported; any other failure (a driver error, a CT or float32 GLM stop, an unexpected message) fails.
const F16=("glm_hll16","glm_ppm","glm_constch")
const PHYSICS_STOPS=("Nonphysical/nonfinite state","Invalid state at output","Invalid timestep","Step limit reached")
configs=sort(filter(f->endswith(f,".toml") && !startswith(f,"gate_"),readdir(joinpath(@__DIR__,"..","configs"))))
broken=String[];stopped=String[];count=0
firstline(s)=first(split(s,'\n'))
for file in configs
    name=replace(file,".toml"=>"");path=joinpath(@__DIR__,"..","configs",file)
    config=TOML.parsefile(path)
    nm=get(config,"split_methods",false)===true ? length(get(config,"methods",Suite.DEFAULT_METHODS)) : 1
    for m in 0:nm-1
        out=joinpath(ARGS[1],"$name-$m")
        try
            Suite.main([path,out,string(m),"--smoke"]);global count+=1
        catch e
            runs=isdir(out) ? [joinpath(out,d,"run.toml") for d in readdir(out) if isfile(joinpath(out,d,"run.toml"))] : String[]
            bad=[(r,TOML.parsefile(r)) for r in runs if TOML.parsefile(r)["status"]!="complete"]
            if isempty(bad)
                push!(broken,"$name-$m: "*firstline(sprint(showerror,e)))       # failed before or outside a run
            end
            for (r,meta) in bad
                err=get(meta,"error","");label="$name $(meta["method"]): "*firstline(err)
                push!(meta["method"] in F16 && any(p->startswith(err,p),PHYSICS_STOPS) ? stopped : broken,label)
            end
        end
    end
end
isempty(stopped) || println("Solver stops (f16 GLM kernels on a nonphysical state; results, not failures):\n  ",join(stopped,"\n  "))
isempty(broken) || error("Smoke failures:\n  "*join(broken,"\n  "))
println("Smoke passed: $count config/method calls ran cleanly at 32^3; $(length(stopped)) f16 solver stops recorded. ",
    "Execution checks, not convergence certification.")
