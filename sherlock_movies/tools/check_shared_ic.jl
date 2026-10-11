# Compare the shared C++ initial-data port with Julia for every configuration and seed.
# This tests the external problem-generator math without CUDA or external-code dependencies.
using TOML, Printf, Test
include(joinpath(@__DIR__,"..","src","problems.jl"))

mktempdir() do tmp
    exe=joinpath(tmp,"check_shared_ic")
    run(`$(get(ENV,"CXX","c++")) -std=c++17 -O2 $(joinpath(@__DIR__,"check_shared_ic.cpp")) -o $exe`)
    worst=0.0
    @testset "C++/Julia initial conditions, all configs and seeds" begin
        for path in sort(filter(p->endswith(p,".toml") && !startswith(basename(p),"."),readdir(joinpath(@__DIR__,"..","configs");join=true)))
            config=TOML.parsefile(path);name=get(config,"problem",config["name"])
            for seed in get(config,"seeds",[42]), n in (8,16)
                mach=get(config,"mach",2.0)
                write(joinpath(tmp,"ic.spec"),Problems.spec_text(name,n;seed,mach,par=config))
                run(`$exe $(joinpath(tmp,"ic.spec")) $(joinpath(tmp,"state.bin"))`)
                cpp=Array{Float64}(undef,n,n,n,12);read!(joinpath(tmp,"state.bin"),cpp)
                u,b=Problems.initial(name,n;seed,mach,par=config)
                expected=cat(u,b[1],b[2],b[3];dims=4)
                # Near-zero vector components use their vector group's scale, avoiding
                # huge relative errors on analytically zero values from trig roundoff.
                err=0.0
                for cols in (1:1,2:4,5:5,6:8,9:9,10:12)
                    scale=max(maximum(abs,@view expected[:,:,:,cols]),1e-30)
                    delta=maximum(abs,cpp[:,:,:,cols].-expected[:,:,:,cols])/scale
                    err=max(err,delta)
                end
                worst=max(worst,err)
                @test err <= 2e-6
                @printf("%-24s N=%2d seed=%d max scaled difference %.3e\n",config["name"],n,seed,err)
            end
        end
    end
    @printf("Worst C++/Julia scaled difference: %.3e (tolerance 2e-6)\n",worst)
end
