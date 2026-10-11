# The sweep ran the kernels at Courant number 0.25 in their own timestep rule (dt = C dx / max over cells of the SUM
# over three directions of |v_d| + c_fast,d); the solver's default is 0.7. Errors on the entropy wave and survival of
# Orszag-Tang at 0.25, 0.4 and 0.7 (the driver stops at the first nonpositive pressure).   julia -t 4 courant.jl [N ...]
include("emu.jl");using .Emu,Printf,Random;include("problems.jl")
grids=grids_from(ARGS,(32,64,128));g=5f0/3f0
for riem in (:llf,:hll,:hlld),c in (0.25f0,0.4f0,0.7f0)
    e=Float64[];steps=0
    for n in grids
        u0=entropy(n);u,t,steps,why,_=Emu.evolve(u0,Emu.Run(n=n,gamma=g,riem=riem,courant=c),1.0;frames=21);push!(e,why=="" ? density_error(u,u0) : NaN)
    end
    n=last(grids);u,t,st,why,_=Emu.evolve(orszag_tang(n),Emu.Run(n=n,gamma=g,riem=riem,courant=c),0.5;frames=21)
    @printf("%-5s Courant %.2f: entropy wave L1 %s (%d steps at N = %d) | Orszag-Tang %d: t = %.2f in %d steps %s\n",riem,c,fmt(e),steps,n,n,t,st,why)
end
