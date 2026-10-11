# Write the C++ problem spec (and optionally the Float32 initial state) for one config and grid.
#   julia tools/write_spec.jl CONFIG.toml N OUT_PREFIX [--state] [--seed S] [--t T]
# OUT_PREFIX.spec is always written. --state also writes OUT_PREFIX.state (N^3 x 9 cell state,
# Float32, x fastest) and OUT_PREFIX.face{1,2,3} (lower faces, Float32), used by RAMSES.
using TOML
include(joinpath(@__DIR__,"..","src","problems.jl"))
function main(args)
    length(args)>=3 || error("Usage: write_spec.jl CONFIG N OUT_PREFIX [--state] [--seed S] [--t T]")
    config=TOML.parsefile(args[1]);n=parse(Int,args[2]);prefix=args[3]
    name=get(config,"problem",config["name"]);mach=get(config,"mach",2.0)
    opt(flag,default)=(i=findfirst(==(flag),args);i===nothing ? default : args[i+1])
    seed=parse(Int,opt("--seed","42"));t=parse(Float64,opt("--t","0"))
    mkpath(dirname(abspath(prefix)))
    write(prefix*".spec",Problems.spec_text(name,n;seed,mach,par=config,t))
    if "--state" in args
        u,b=Problems.initial(name,n;seed,mach,par=config,t)
        write(prefix*".state",u);for d in 1:3;write(prefix*".face$d",b[d]);end
    end
end
main(ARGS)
