# Half-precision storage in the production kernel (integrator_plm! with HALF: primitives, interface states AND fluxes
# pass through Float16).   julia -t 4 half_precision.jl [N ...]
# 1. Precision: the entropy wave's error with each storage choice, next to the sweep's records.
# 2. Range: the largest Float16 is 65504. Magnetised blasts and a Mach 9 flow with each storage choice.
include("emu.jl");using .Emu,Printf,Random;include("problems.jl")
grids=grids_from(ARGS,(32,64,128,256))
function err(n;kw...)
    u0=entropy(n);u,t,steps,why,_=Emu.evolve(u0,Emu.Run(;n=n,gamma=5f0/3f0,riem=:hll,kw...),1.0;frames=21)
    why=="" ? density_error(u,u0) : NaN
end
println("Entropy wave, L1 error of the density at t = 1, N = ",join(grids,", "))
println("  sweep, GLM f16/HLL (N = 32 ... 256)                3.21e-03  1.59e-03  1.95e-03  2.53e-03")
println("  PLM/HLL, f16 as the kernel                         ",fmt([err(n;st=:f16,fst=:f16) for n in grids]))
println("  PLM/HLL, f32                                       ",fmt([err(n) for n in grids]))
println("  PLM/HLL, f16 of differences from a tile reference  ",fmt([err(n;st=:a16,fst=:a16) for n in grids]))
println("  the same, fluxes in units of the grid's largest    ",fmt([err(n;st=:a16,fst=:a16,fscale=true,ch_from=:cfl) for n in grids]))
println("  sweep, GLM PPM (N = 32 ... 256)                    1.83e-03  9.02e-04  1.55e-03  1.43e-03")
println("  PPM/HLL, f16 as the kernel                         ",fmt([err(n;st=:f16,fst=:f16,ppm=true) for n in grids]))
println("  PPM/HLL, f32                                       ",fmt([err(n;ppm=true) for n in grids]))
println("  PPM/HLL, f16 of differences from a tile reference  ",fmt([err(n;st=:a16,fst=:a16,ppm=true) for n in grids]))

big=isempty(ARGS)
cases=(("blast B=50",n->blast(n,50.0),1.4f0,0.01,11,:floor,big ? (64,128) : (32,)),("blast B=100",n->blast(n,100.0),1.4f0,0.01,11,:floor,big ? (64,128) : (32,)),
       ("blast B=0",n->blast(n,0.0),1.4f0,0.01,11,:floor,big ? (64,) : (32,)),("2D Mach 9",mach9_2d,1.01f0,0.1,6,:stop,big ? (128,256) : (64,)))
variants=(("f16 as the kernel",(;st=:f16,fst=:f16)),("f16, c_h from the CFL step",(;st=:f16,fst=:f16,ch_from=:cfl)),
          ("f16, psi flux stored as Bn*",(;st=:f16,fst=:f16,psiflux=:bn)),
          ("f16 of differences, fluxes in grid units",(;st=:a16,fst=:a16,fscale=true,ch_from=:cfl)))
println("\nRange: largest fluxes in a Float32 run (the largest Float16 is 65504), then how far each storage choice gets")
for (label,ic,gamma,tend,frames,policy,ns) in cases,n in ns
    r=Emu.Run(n=n,gamma=gamma,riem=:hll,policy=policy,track=true,ch_from=:cfl);u,t,steps,why,_=Emu.evolve(ic(n),r,tend;frames=frames);peak=r.peak
    @printf("  %-11s N=%3d f32 (c_h from the CFL step): energy flux %.2e, momentum flux %.2e, psi flux c_h^2 Bn* %.2e\n",label,n,peak[5],maximum(peak[2:4]),peak[9])
    for (what,kw) in variants
        u,t,steps,why,_=Emu.evolve(ic(n),Emu.Run(;n=n,gamma=gamma,riem=:hll,policy=policy,kw...),tend;frames=frames)
        @printf("  %-11s N=%3d %-41s reached t = %.4f of %.3f  %s\n",label,n,what,t,tend,why)
    end
end
