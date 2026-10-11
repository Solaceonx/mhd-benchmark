# Compare an external code's saved t=0 state with problems.jl `initial` (same N, seed, params).
#   julia tools/check_external_ic.jl RESULT_GROUP_DIR...
# Prints the maximum absolute difference per field, scaled by the field's maximum magnitude.
using TOML, Printf
include(joinpath(@__DIR__,"..","src","problems.jl"))
worst=0.0
for root in ARGS, dir in filter(isdir,readdir(abspath(root);join=true))
    isfile(joinpath(dir,"run.toml")) || continue
    meta=TOML.parsefile(joinpath(dir,"run.toml"));config=TOML.parsefile(joinpath(root,"input.toml"))
    stamp=findfirst(s->s["frame"]==0 && haskey(s,"file"),meta["frames"]);stamp===nothing && continue
    shape=Tuple(meta["saved_shape"]);u=Array{Float32}(undef,shape...);read!(joinpath(dir,meta["frames"][stamp]["file"]),u)
    h,_=Problems.initial(meta["problem"],meta["N"];seed=meta["seed"],mach=get(config,"mach",2.0),par=config)
    h=length(shape)==3 ? h[:,:,1,:] : h
    # Scale each field group by its largest magnitude (vector groups share one scale).
    groups=Dict(1=>1:1,2=>2:4,3=>2:4,4=>2:4,5=>5:5,6=>6:8,7=>6:8,8=>6:8)
    scale(c)=max(maximum(abs,selectdim(h,ndims(h),groups[c])),1e-30)
    diffs=[maximum(abs,selectdim(u,ndims(u),c).-selectdim(h,ndims(h),c))/scale(c) for c in 1:8]
    global worst=max(worst,maximum(diffs))
    @printf("%-18s %-16s N=%-3d max rel diff  rho %.1e  m %.1e  E %.1e  B %.1e\n",config["name"],meta["method"],meta["N"],diffs[1],maximum(diffs[2:4]),diffs[5],maximum(diffs[6:8]))
end
@printf("worst %.2e\n",worst)
