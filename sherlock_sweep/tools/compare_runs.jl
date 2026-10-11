# Compare two GPU-driver result directories output by output (resume gate). Exits 1 on any difference
# above round-off; prints the largest relative difference per method. Text outputs (diagnostics, dense
# diagnostics, spectra) must be identical; so must the list of recorded outputs (no output twice).
using TOML
a,b=ARGS
worst=0.0;bad=String[]
function maxdiff(fa,fb,shape)
    xa=Array{Float32}(undef,shape...);read!(fa,xa)
    xb=Array{Float32}(undef,shape...);read!(fb,xb)
    maximum(abs.(xa.-xb))/max(maximum(abs,xb),1f-30)
end
for dir in sort(filter(d->isdir(joinpath(a,d)) && endswith(d,"_r1"),readdir(a)))
    ma=TOML.parsefile(joinpath(a,dir,"run.toml"));mb=TOML.parsefile(joinpath(b,dir,"run.toml"))
    ma["status"]=="complete" && mb["status"]=="complete" || (push!(bad,"$dir status $(ma["status"])/$(mb["status"])");continue)
    [f["frame"] for f in ma["frames"]]==[f["frame"] for f in mb["frames"]] ||
        push!(bad,"$dir recorded outputs $([f["frame"] for f in ma["frames"]]) vs $([f["frame"] for f in mb["frames"]])")
    shape=Tuple(ma["saved_shape"]);d=0.0
    for (fa,fb) in zip(ma["frames"],mb["frames"])
        fa["steps"]==fb["steps"] || push!(bad,"$dir frame $(fa["frame"]) steps $(fa["steps"]) vs $(fb["steps"])")
        haskey(fa,"file") && (d=max(d,maxdiff(joinpath(a,dir,fa["file"]),joinpath(b,dir,fb["file"]),shape)))
        haskey(fa,"slice") && (d=max(d,maxdiff(joinpath(a,dir,fa["slice"]),joinpath(b,dir,fb["slice"]),Tuple(ma["slice_shape"]))))
        for (m,file) in get(fa,"restricted",Dict())
            k=parse(Int,m);d=max(d,maxdiff(joinpath(a,dir,file),joinpath(b,dir,fb["restricted"][m]),(k,k,k,9)))
        end
    end
    for f in ("diagnostics.csv","dense_diagnostics.csv","spectra.csv")
        ea,eb=isfile(joinpath(a,dir,f)),isfile(joinpath(b,dir,f))
        ea==eb || (push!(bad,"$dir $f present in only one run");continue)
        ea && (read(joinpath(a,dir,f),String)==read(joinpath(b,dir,f),String) || push!(bad,"$dir $f differs"))
    end
    get(ma,"resumed_count",0)>=1 || push!(bad,"$dir was never interrupted and resumed")
    println(rpad(dir,16)," resumed=",get(ma,"resumed_count",0)," outputs=",length(ma["frames"])," max relative state difference ",d)
    global worst=max(worst,d);d>1e-6 && push!(bad,"$dir differs by $d")
end
isempty(bad) || (foreach(x->println(stderr,x),bad);exit(1))
println("RESUME CHECK PASSED (largest difference $worst)")
