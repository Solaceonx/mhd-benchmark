# Relative/absolute L1, L2, Linf differences; shared by run_suite.jl and external post-processing.
using Statistics
function solution_errors(u,ref)
    ref===nothing && return Dict{String,Float64}()
    result=Dict{String,Float64}()
    for (name,cols) in (("density",1:1),("momentum",2:4),("energy",5:5),("magnetic",6:8))
        a=Float64.(u[:,:,:,cols]);b=Float64.(ref[:,:,:,cols])
        e=sqrt.(sum(abs2,a.-b;dims=4));scale=sqrt.(sum(abs2,b;dims=4))
        for (norm,num,den) in (("L1",mean(e),mean(scale)),("L2",sqrt(mean(abs2,e)),sqrt(mean(abs2,scale))),("Linf",maximum(e),maximum(scale)))
            result[name*"_absolute_"*norm]=num
            result[name*"_relative_"*norm]=den>0 ? num/den : NaN
        end
    end
    result
end

