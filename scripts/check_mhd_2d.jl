include(joinpath(@__DIR__,"mhd_2d_cpu.jl"))
using .MHD2DCPU, Test, Random, Printf
const M=MHD2DCPU
@testset "Shared 2D GLM/CT" begin
    p=M.Params(n=32); w=M.Workspace(p.n)
    for method in (:glm,:ct)
        s=M.initialize(p;method,uniform=true); before=copy(s.u)
        M.step!(s,w,p,0.001f0,2f0,method)
        @test s.u==before
    end
    a=M.initialize(p;method=:glm); b=M.initialize(p;method=:ct)
    @test a.u==b.u
    @test maximum(abs,M.div_face(b,p))==0
    # A random periodic vector potential creates a divergence-free face field.
    rng=MersenneTwister(42); az=randn(rng,Float32,p.n,p.n)
    for j in 1:p.n,i in 1:p.n
        b.bx[i,j]=(az[i,M.wrap(j+1,p.n)]-az[i,j])/p.n
        b.by[i,j]=-(az[M.wrap(i+1,p.n),j]-az[i,j])/p.n
    end
    initialdiv=M.div_face(b,p)
    randn!(rng,w.fx);randn!(rng,w.fy)
    M.curl_rhs!(w.rhs0,w,p)
    @. b.bx+=0.01f0*w.rhs0.bx
    @. b.by+=0.01f0*w.rhs0.by
    @test maximum(abs,M.div_face(b,p)-initialdiv)<1e-5
    # Deliberately non-solenoidal sinusoidal Bx: GLM must transport/damp it.
    s=M.initialize(p;uniform=true)
    for j in 1:p.n,i in 1:p.n
        old=s.u[i,j,6]; new=old+0.02f0*sin(2f0*Float32(pi)*(i-.5f0)/p.n)
        s.u[i,j,6]=new; s.u[i,j,5]+=(new^2-old^2)/2
    end
    d0=M.diagnostics(s,p,:glm)
    for _ in 1:250; M.step!(s,w,p,0.001f0,2f0,:glm); end
    d1=M.diagnostics(s,p,:glm)
    @printf("GLM divergence RMS: %.5g -> %.5g\n",d0.div_center_rms,d1.div_center_rms)
    @test d1.div_center_rms<0.4d0.div_center_rms
    @test abs(d1.mass/d0.mass-1)<1e-5
    @test abs(d1.energy/d0.energy-1)<1e-5
    for method in (:glm,:ct)
        s=M.initialize(p;method);d0=M.diagnostics(s,p,method)
        for _ in 1:50;M.step!(s,w,p,0.0005f0,3f0,method);end
        d=M.diagnostics(s,p,method)
        @test abs(d.mass/d0.mass-1)<1e-5
        @test abs(d.energy/d0.energy-1)<1e-5
        @test d.rhomin>0 && d.pmin>0
        method===:ct && @test d.div_face_max<1e-4
    end
end
