using Test, TOML, SHA, Statistics
include(joinpath(@__DIR__,"..","src","problems.jl"))
const CONFIGS=joinpath(@__DIR__,"..","configs")
@testset "GPU bundle initial data (CPU only)" begin
    for file in sort(filter(endswith(".toml"),readdir(CONFIGS)))
        config=TOML.parsefile(joinpath(CONFIGS,file));name=get(config,"problem",config["name"])
        for seed in get(config,"seeds",[42])
        @testset "$file seed=$seed" begin
            @test config["name"]*".toml"==file && config["frames"]>=2 && config["repeats"]>=1
            @test name in Problems.NAMES
            n=16;gamma=Problems.gamma_of(name,config)
            mach=get(config,"mach",2.0)
            u,b=Problems.initial(name,n;par=config,seed,mach);d=Problems.diagnostics(u,b;gamma)
            @test size(u)==(n,n,n,9)
            @test d.finite && d.rho_min>0 && d.pressure_min>0
            @test d.div_face_max<=Problems.div_tolerance(b,n)
            for dim in 1:3
                @test u[:,:,:,dim+5]==Problems.center_faces(b)[dim]
            end
            Problems.is_planar(name,config) && @test d.z_variation==0
            ref,_=Problems.reference(name,n,0.0,u;par=config,seed,mach)
            ref===nothing || @test maximum(abs,u.-ref)<2e-7
            period=Dict("uniform"=>.3,"entropy"=>1.0,"alfven"=>1.0,"alfven_oblique"=>1/sqrt(5),"field_loop"=>1.0,"field_loop_3d"=>1.0)
            if haskey(period,name)
                ref,_=Problems.reference(name,n,period[name],u;par=config,seed,mach)
                @test ref!==nothing && maximum(abs,u.-ref)<2e-7
            end
            if name=="decay"
                u2,b2=Problems.initial(name,n;par=config,seed,mach)
                @test u==u2 && b==b2
                # rho=1, p=1/gamma give initial adiabatic sound speed 1.
                @test isapprox(sqrt(2d.kinetic/d.mass),mach;rtol=1e-6)
                mean_b=get(config,"b0_vector",[.1,0.0,0.0])
                @test all(isapprox(mean(b[i]),mean_b[i];rtol=1e-6,atol=1e-8) for i in 1:3)
            end
        end
        end
    end
    # Cell averaging independently with a fine-grid quadrature.
    n=16;u,_=Problems.initial("entropy",n)
    points=((1:10000).-0.5)./10000/n
    avg=mean(sin.(2pi.*points))*mean(cos.(2pi.*points))+mean(cos.(2pi.*points))*mean(sin.(2pi.*points))
    @test isapprox(u[1,1,1,1],1+.2avg;atol=1e-7)
    # Face fields are exact face averages of curl A: square waves land on faces exactly.
    u,b=Problems.initial("current_sheet",16)
    @test all(==(1f0),b[2][[1:4;13:16],:,:]) && all(==(-1f0),b[2][5:12,:,:]) && all(iszero,b[1])
    u,b=Problems.initial("brio_wu_rotated",32)
    bn=(u[:,:,:,6].+u[:,:,:,7])./sqrt(2f0);bt=(u[:,:,:,7].-u[:,:,:,6])./sqrt(2f0)
    @test maximum(abs,bn.-.75)<1e-6 && all(isapprox.(extrema(bt),(-1,1);atol=1e-6)) && maximum(abs,bt)<=1+1e-6
    # Oblique Alfven: cell B matches the continuum field to O(dx^2), and v_perp=-B_perp.
    n=64;u,_=Problems.initial("alfven_oblique",n);c=((1:n).-.5)./n;s5=sqrt(5)
    phase=[2pi*(c[i]+2c[j]) for i in 1:n,j in 1:n]
    @test maximum(abs,u[:,:,1,8].-.1sin.(phase))<5e-4
    @test maximum(abs,u[:,:,1,6].-(1/s5 .-.2cos.(phase)./s5))<5e-4
    @test maximum(abs,u[:,:,1,4].+.1sin.(phase))<1e-6
    # 3D loop is a pure in-plane-of-rotation field: B is perpendicular to the loop axis.
    u,_=Problems.initial("field_loop_3d",32)
    @test maximum(abs,(u[:,:,:,8].-u[:,:,:,6])./sqrt(2f0))<.05*maximum(abs,u[:,:,:,6])
    # Rotor mass: disk rho=10, taper, ambient rho=1.
    u,_=Problems.initial("rotor",64);@test maximum(u[:,:,1,1])==10 && minimum(u[:,:,1,1])==1
    @test Problems.parse_gamma("7/5")==1.4 && Problems.parse_gamma(2)==2.0
end
