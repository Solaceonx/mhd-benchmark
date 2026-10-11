include(joinpath(@__DIR__,"isothermal_hydro.jl"))
using Test

function acoustic_error(n)
    p=IsoParams(N=n); a=zeros(Float32,n+2IG,n,n,4); b=similar(a)
    amp=1f-3
    for k in 1:n,j in 1:n,i in 1:n
        wave=amp*sin(2pi*(i-0.5)/n)
        a[i+IG,j,k,1]=1+wave; a[i+IG,j,k,2]=(1+wave)*wave
    end
    iso_ghost!(a,p); t=0.0; d=iso_stats(a,p)
    while t<0.2-1e-9
        dt=min(p.courant/n/d.signal,0.2-t)
        h,d,_=iso_advance!(b,a,p,dt); a,b=b,a; t+=h
    end
    err=sum(abs(a[i+IG,1,1,1]-(1+amp*sin(2pi*((i-0.5)/n-t)))) for i in 1:n)/n
    @test abs(d.mass-1)<1e-6
    err
end
@testset "Isothermal hydro" begin
    p=IsoParams(N=16); a=zeros(Float32,20,16,16,4); b=similar(a)
    a[:,:,:,1].=2; a[:,:,:,2].=6; a[:,:,:,3].=-2; a[:,:,:,4].=1
    iso_ghost!(a,p); iso_step!(b,a,p,0.001f0)
    @test b[IG+1:IG+p.N,:,:,:] == a[IG+1:IG+p.N,:,:,:]
    e32=acoustic_error(32); e64=acoustic_error(64)
    @printf("Acoustic mean density errors: N32=%.3e N64=%.3e\n",e32,e64)
    @test e64<0.65e32
    @test e64<1e-5
end
