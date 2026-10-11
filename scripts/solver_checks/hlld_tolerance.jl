# hlld_x guards its star-state denominator with an absolute tolerance, `abs(denK) < 1.0f-12`, in Float32. On a face
# where the field lies along the normal and the Alfven speed exceeds the sound speed, c_fast = c_Alfven and
# denK = d(S-u)(S-SM) - bn^2 is rounding noise of order 1e-6: the guard never triggers and the star states are noise.
# The checks: one such face of the current sheet; the runs that stopped in the sweep, with the original line and with
# `abs(denK) < 1.0f-4*pts`; and that nothing changes where HLLD already worked.   julia -t 4 hlld_tolerance.jl [N ...]
include("emu.jl");using .Emu,Printf,Random;include("problems.jl")
const G=Emu.G
g=5f0/3f0
println("A face of the current sheet, 14 steps in: two nearly equal states, field along the normal, beta = 0.2")
L=(0.99997056f0,0.09820223f0,-1.2903779f-6,0f0,0.09999485f0,0.0005392039f0,0.99997056f0,0f0,1.3478038f-6)
R=(0.9999714f0,0.09821387f0,-1.3326425f-6,0f0,0.09999557f0,0.00053067453f0,0.9999715f0,0f0,1.4397164f-6)
for (label,M) in (("original HLLD    ",Emu.G),("HLLD, new guard  ",Emu.GH))
    f=M.riemann(L,R,2,g,1.087563f0,1f-6,1f-6,0f0,0f0,true)
    @printf("  %s flux of x momentum % .5f, of energy % .6f, of Bx % .5f\n",label,f[2],f[5],f[6])
end
h=G.riemann_hll(L,R,2,g,1.087563f0,1f-6,1f-6)
@printf("  HLL               flux of x momentum % .5f, of energy % .6f, of Bx % .5f   (the exact flux of Bx is %.5f)\n",h[2],h[5],h[6],L[3]*L[6]-L[2]*L[7])
sheets=grids_from(ARGS,(32,64,128))
println("\nRuns (the driver stops at the first nonpositive pressure)")
for (name,ic,tend,frames,ns) in (("current sheet",current_sheet,2.5,21,sheets),("GEM",gem,1.5625,26,isempty(ARGS) ? (64,128) : sheets))
    for n in ns,(label,rs) in (("original ",Emu.rsolve),("new guard",Emu.hlld_fixed))
        u,t,steps,why,_=Emu.evolve(ic(n),Emu.Run(n=n,gamma=g,riem=:hlld,rsolver=rs),tend;frames=frames)
        @printf("  %-13s N=%3d HLLD %s reached t = %.3f of %.3f in %6d steps  %s\n",name,n,label,t,tend,steps,why)
    end
end
println("\nWhere HLLD already worked: oblique Alfven wave (relative L1 error of B) and Orszag-Tang energies at t = 0.5")
println("  sweep, GLM HLLD, N = 32, 64, 128:   7.02e-03  2.19e-03  6.20e-04")
en(u)=(sum(0.5*(Float64(c[2])^2+Float64(c[3])^2+Float64(c[4])^2)/c[1] for c in u)/length(u),sum(0.5*(Float64(c[6])^2+Float64(c[7])^2+Float64(c[8])^2) for c in u)/length(u))
ws=isempty(ARGS) ? (32,64,128) : sheets
for (label,rs) in (("original ",Emu.rsolve),("new guard",Emu.hlld_fixed))
    e=[begin u0=alfven_oblique(n);u,t,st,why,_=Emu.evolve(u0,Emu.Run(n=n,gamma=g,riem=:hlld,rsolver=rs),1/S5;frames=21);why=="" ? field_error(u,u0) : NaN end for n in ws]
    n=last(ws);u,t,st,why,_=Emu.evolve(orszag_tang(n),Emu.Run(n=n,gamma=g,riem=:hlld,rsolver=rs),0.5;frames=21);k,m=en(u)
    @printf("  HLLD %s  Alfven %s | Orszag-Tang %d: kinetic %.6f, magnetic %.6f %s\n",label,fmt(e),n,k,m,why)
end
