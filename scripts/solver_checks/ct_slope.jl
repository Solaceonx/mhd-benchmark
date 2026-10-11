# The CT prototype (vendor/ct_mhd.jl) converges at first order in the sweep. Two causes, checked here on the CPU with
# Tom Abel's functions unedited (pslope, hanc1d, hll, dflux, psub, fast_speed) and the loop structure of his kernels
# (ct_driver.jl: bcc -> flux x, y, z -> edge EMFs as the average of four face fluxes -> hydro update -> face B update):
#   1. pslope returns HALF the limited slope, and hanc1d and the interface states multiply it by 1/2 again, so the
#      reconstruction uses half the slope. CTF is the same code with pslope returning the full slope (one token).
#   2. the predictor (hanc1d) advances a cell by the flux difference of the face's own direction only, so the fluxes
#      through the other faces are not centred in time. `full = true` uses the predictor of the GLM kernels instead
#      (flux differences of all directions).
#   julia ct_slope.jl [N ...]
include("vendor_physics.jl")
module CT end
load_physics(CT,"ct_mhd.jl",CT_LINES);Base.include(CT,joinpath(@__DIR__,"ct_driver.jl"))
module CTF end
load_physics(CTF,"ct_mhd.jl",CT_LINES;edit="ntuple(t->0.5f0*moncen(M[t]-L[t],R[t]-M[t]),8)"=>"ntuple(t->moncen(M[t]-L[t],R[t]-M[t]),8)")
Base.include(CTF,joinpath(@__DIR__,"ct_driver.jl"))
using Printf
noA(x,y)=(0.0,0.0,0.0)
tests=(("entropy","density L1",(x,y)->(1+.2*sin(2pi*(x+y)),.5,.5,0.0,1.0),noA,(.3,.2,0.0),1.0,
        (U,q,U0,q0)->sum(abs(Float64(U[k][1])-Float64(U0[k][1])) for k in eachindex(U))/length(U),(6.98e-2,4.18e-2,2.30e-2,1.20e-2)),
       ("alfven","relative L1 of B",(x,y)->(1.0,0.0,-.1*cos(2pi*x),-.1*sin(2pi*x),.1),(x,y)->(0.0,-.1*cos(2pi*x)/2pi,-.1*sin(2pi*x)/2pi),(1.0,0.0,0.0),1.0,
        (U,q,U0,q0)->sum(sqrt(sum((Float64(q[k][v])-Float64(q0[k][v]))^2 for v in 6:8)) for k in eachindex(q))/sum(sqrt(sum(Float64(q0[k][v])^2 for v in 6:8)) for k in eachindex(q)),
        (1.85e-2,9.66e-3,4.94e-3,2.50e-3)))
grids=isempty(ARGS) ? (32,64,128) : Tuple(parse.(Int,ARGS))
for (name,what,prim,A,B0,tend,err,sweep) in tests
    println(name,": ",what," at N = ",join(grids,", ")," (order between grids)")
    @printf("  sweep, CT prototype (N = 32, 64, 128)    %s\n",join((@sprintf("%.2e",x) for x in sweep[1:3]),"  "))
    for (label,M,full) in (("as written                             ",CT,false),("full slope                             ",CTF,false),
                           ("full slope, predictor in all directions",CTF,true))
        e=Float64[]
        for n in grids
            U0,bx0,by0,bz0=M.initial(n,prim,A,B0);q0=M.cellprims(U0,bx0,by0,bz0,n)
            U,q=M.run(n,prim,A,B0,tend;full=full);push!(e,err(U,q,U0,q0))
        end
        @printf("  %s   %s   orders %s\n",label,join((@sprintf("%.2e",x) for x in e),"  "),join((@sprintf("%.2f",log2(e[k]/e[k+1])) for k in 1:length(e)-1),", "))
    end
end
