# Synthetic format fixtures only; does not run either GPU solver.
include(joinpath(@__DIR__,"..","src","run_suite.jl"))
using Test, TOML
length(ARGS)==1 || error("Usage: check_roundtrip.jl NEW_FIXTURE_DIRECTORY")
root=abspath(ARGS[1]);ispath(root) && error("Fixture directory must be new")
@testset "GPU bundle export format (CPU fixture)" begin
    for name in ("entropy","alfven_oblique","orszag_tang","field_loop_3d")
        n=name=="orszag_tang" ? 32 : 16
        base=joinpath(root,name);mkpath(base)
        Suite.record(joinpath(base,"environment.toml"),Dict("device"=>"CPU FORMAT FIXTURE, NOT GPU MEASUREMENT","smoke"=>true))
        u,b=Suite.Problems.initial(name,n);full=!Suite.Problems.is_planar(name)
        ref,label=Suite.Problems.reference(name,n,0.0,u)
        for method in (:glm,:ct)
            path=joinpath(base,"$(method)_r1");mkpath(path)
            faces=method===:ct ? b : nothing
            file=Suite.write_snapshot(path,0,u,faces,full)
            reffile=ref===nothing ? nothing : Suite.write_snapshot(path,0,ref,nothing,full;stem="ref_000")
            expected=(full ? n^3 : n^2)*9*4
            @test filesize(joinpath(path,file))==expected
            norm=Suite.solution_errors(u,u)
            @test norm["density_absolute_L1"]==0
            d=Suite.Problems.diagnostics(u,faces;gamma=Suite.Problems.gamma_of(name))
            row=(frame=0,time=0.0,steps=0,d...,mass_drift=0.0,energy_drift=0.0)
            open(joinpath(path,"diagnostics.csv"),"w") do io
                println(io,join(keys(row),','));println(io,join(values(row),','))
            end
            Suite.record(joinpath(path,"run.toml"),Dict("test"=>name,"N"=>n,"method"=>string(method),"repeat"=>1,
                "seed"=>42,"problem"=>name,"gamma"=>Suite.Problems.gamma_of(name),"status"=>"complete","saved_shape"=>full ? [n,n,n,9] : [n,n,9],
                "frames"=>[reffile===nothing ? Dict("frame"=>0,"time"=>0.0,"file"=>file) : Dict("frame"=>0,"time"=>0.0,"file"=>file,"reference_file"=>reffile,"reference"=>label)],"snapshots_written"=>true,
                "stepping_seconds"=>0.0,"cfl_seconds"=>0.0,"solve_seconds"=>0.0,"evolution_wall_seconds"=>0.0))
        end
    end
end
println("CPU format fixtures written to ",root)
