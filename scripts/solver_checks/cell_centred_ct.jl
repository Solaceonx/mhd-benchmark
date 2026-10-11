# Cell-centred constrained transport (flux-CT, Toth 2000 sec. 4.3) in place of GLM, with the same reconstruction,
# predictor and Riemann solvers (emu.jl, `fct = true`): one EMF per cell corner, the average of the four face fluxes
# around it, and the cell-centred field updated with central differences of it. No psi, no c_h. The divergence at cell
# corners stays at rounding. Accuracy and survival next to the GLM kernel.   julia -t 4 cell_centred_ct.jl
include("emu.jl");using .Emu,Printf,Random;include("problems.jl")
g=5f0/3f0;s5=S5;alfo=alfven_oblique;loop=field_loop;ot=orszag_tang;sheet=current_sheet;berr=field_error
emag(u)=sum(Float64(c[6])^2+Float64(c[7])^2 for c in u)
println("Alfven wave oblique to the grid, relative L1 error of B after one period, N = 32, 64, 128   (sweep: GLM LLF 7.19e-03 2.05e-03 5.98e-04; AthenaK PLM/HLLE 1.24e-02 4.51e-03 1.33e-03)")
for riem in (:llf,:hll),(label,kw) in (("GLM ",(;)),("cell CT",(;fct=true)))
    e=Float64[]
    for n in (32,64,128)
        u0=alfo(n);r=Emu.Run(;n=n,gamma=g,riem=riem,kw...);u,t,steps,why,_=Emu.evolve(u0,r,1/s5;frames=21);push!(e,why=="" ? berr(u,u0) : NaN)
    end
    @printf("  %-4s %-8s %s   orders %s\n",riem,label,join((@sprintf("%.2e",x) for x in e),"  "),join((@sprintf("%.2f",log2(e[k]/e[k+1])) for k in 1:2),", "))
end
println("Field loop, N = 64, t = 2: magnetic energy left; out-of-plane field grown with v_z = sqrt(5) (rms B_z / B_0);  corner and centred div B rms (x dx)")
for riem in (:llf,:hll),(label,kw) in (("GLM ",(;)),("cell CT",(;fct=true)))
    u0=loop(64,0.0);r=Emu.Run(;n=64,gamma=g,riem=riem,kw...);u,t,steps,why,_=Emu.evolve(u0,r,2.0;frames=21)
    v0=loop(64,s5);v,t2,_,why2,_=Emu.evolve(v0,r,2.0;frames=21);bz=sqrt(sum(Float64(c[8])^2 for c in v)/length(v))/1e-3
    d=Emu.divs(u)
    @printf("  %-4s %-8s energy %.3f %s | B_z/B_0 %.1e %s | div: corner %.1e, centred %.1e\n",riem,label,emag(u)/emag(u0),why,bz,why2,d[1],d[2])
end
println("Survival (the driver's rule: stop at the first nonpositive pressure; blast: floor)")
for (name,ic,gm,tend,frames,policy) in (("Orszag-Tang 128",()->ot(128),g,0.5,21,:stop),("current sheet 64",()->sheet(64),g,2.5,21,:stop),("current sheet 128",()->sheet(128),g,2.5,21,:stop),
                                        ("blast B=100, 64",()->blast(64,100.0),1.4f0,0.01,11,:floor),("blast B=100, 128",()->blast(128,100.0),1.4f0,0.01,11,:floor))
    for riem in (:llf,:hll,:hlld),(label,kw) in (("GLM ",(;)),("cell CT",(;fct=true)))
        r=Emu.Run(;n=size(ic(),1),gamma=gm,riem=riem,policy=policy,kw...);u,t,steps,why,_=Emu.evolve(ic(),r,tend;frames=frames)
        @printf("  %-17s %-4s %-8s t=%.3f of %.3f in %5d steps %s\n",name,riem,label,t,tend,steps,why)
    end
end
