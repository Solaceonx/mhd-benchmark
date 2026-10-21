# The field loop (in the plane: Gardiner & Stone 2005, v = (2, 1), loop radius 0.3, A = 1e-3 (0.3 - r)) with the two
# CT schemes and each corner EMF: the plain average of the four face EMFs (the kernels as on the sweep-fixes branch)
# and the upwinded one (gs05.jl). Relative L1 error of B at t = 2, after two crossings, as the sweep measures it.
#   julia -t 4 fieldloop_emf.jl [N ...]
include("vendor_physics.jl")
module CTE end
load_physics(CTE,"ct_mhd.jl",CT_LINES;edit="ntuple(t->0.5f0*moncen(M[t]-L[t],R[t]-M[t]),8)"=>"ntuple(t->moncen(M[t]-L[t],R[t]-M[t]),8)")
Base.include(CTE,joinpath(@__DIR__,"ct_driver.jl"))
include("emu.jl");using .Emu,Printf,Random;include("problems.jl")
grids=grids_from(ARGS,(64,128))
prim(x,y)=(1.0,2.0,1.0,0.0,1.0);Aloop(x,y)=(0.0,0.0,1e-3*max(.3-hypot(x-.5,y-.5),0.0))
Berr(q,q0)=sum(sqrt(sum((Float64(q[k][v])-Float64(q0[k][v]))^2 for v in 6:8)) for k in eachindex(q))/sum(sqrt(sum(Float64(q0[k][v])^2 for v in 6:8)) for k in eachindex(q))
println("Field loop in the plane, relative L1 error of B at t = 2, N = ",join(grids,", "))
for emf in (:avg,:gs05n,:gs05,:gs05v,:gs05ct)
    e=Float64[]
    for n in grids
        U0,bx0,by0,bz0=CTE.initial(n,prim,Aloop,(0.0,0.0,0.0));q0=CTE.cellprims(U0,bx0,by0,bz0,n)
        t0=time();U,q=CTE.run(n,prim,Aloop,(0.0,0.0,0.0),2.0;full=true,emf);push!(e,Berr(q,q0))
    end
    @printf("  CT prototype (fixed), %-5s edge EMF   %s\n",emf,join((@sprintf("%.3f",x) for x in e),"  "));flush(stdout)
end
for emf in (:avg,:gs05n,:gs05,:gs05v,:gs05ct)
    e=Float64[]
    for n in grids
        u0=field_loop(n,0.0);u,t,steps,why,_=Emu.evolve(u0,Emu.Run(;n=n,gamma=5f0/3f0,riem=:hll,fct=true,emf),2.0;frames=21)
        push!(e,why=="" ? field_error(u,u0) : NaN)
    end
    @printf("  cell-centred CT option, %-5s edge EMF %s\n",emf,join((@sprintf("%.3f",x) for x in e),"  "));flush(stdout)
end
let e=Float64[]
    for n in grids
        u0=field_loop(n,0.0);u,t,steps,why,_=Emu.evolve(u0,Emu.Run(;n=n,gamma=5f0/3f0,riem=:hll,ch_from=:cfl),2.0;frames=21)
        push!(e,why=="" ? field_error(u,u0) : NaN)
    end
    @printf("  GLM, for comparison                %s\n",join((@sprintf("%.3f",x) for x in e),"  "))
end
