include("run_suite.jl")
length(ARGS)==1 || error("Usage: smoke.jl NEW_OUTPUT_DIRECTORY")
configs=sort(filter(endswith(".toml"),readdir(joinpath(@__DIR__,"..","configs"))))
failed=String[]
for file in configs
    name=replace(file,".toml"=>"")
    try
        Suite.main([joinpath(@__DIR__,"..","configs",file),joinpath(ARGS[1],name),"0","--smoke"])
    catch e
        push!(failed,name);showerror(stderr,e);println(stderr)
    end
end
isempty(failed) || error("Smoke failures: "*join(failed,", "))
println("All $(length(configs)) GPU smoke tests completed at 32^3. These are execution checks, not convergence certification.")
