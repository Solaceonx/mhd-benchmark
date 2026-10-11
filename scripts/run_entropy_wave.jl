#= Exact finite-amplitude entropy advection, using the unchanged planar MHD solver.

rho = 1 + 0.2 sin(2pi(x+y-t)), v=(0.5,0.5), p=1, B=(0.3,0.2).
Constant p, v and B reduce the equations to advection of rho. Conserved
variables are affine in rho, so sinc(1/N)^2 gives exact initial cell averages.
=#
include(joinpath(@__DIR__, "mhd_2d_cpu.jl"))
using .MHD2DCPU, Printf, TOML, SHA
const M=MHD2DCPU

exact_density(n,t)=[1+0.2*sinc(1/n)^2*sin(2pi*((i-.5)/n+(j-.5)/n-t)) for i in 1:n,j in 1:n]
function initial_wave(p)
    s=M.State(p.n); rho=exact_density(p.n,0)
    for j in 1:p.n,i in 1:p.n
        q=(Float32(rho[i,j]),.5f0,.5f0,0f0,1f0,.3f0,.2f0,0f0,0f0)
        M.put!(s.u,M.G.prim2cons(q,p.gamma),i,j)
        s.bx[i,j]=.3f0;s.by[i,j]=.2f0
    end
    s
end

function run_wave(root)
    isdir(root) && !isempty(readdir(root)) && error("Output directory must be empty")
    mkpath(root)
    open(joinpath(root,"errors.csv"),"w") do io
        println(io,"N,method,time,L1,L2,Linf,relative_L1,mass_drift,energy_drift,pressure_Linf,B_Linf,div_face_max")
        for n in (16,32,64,128)
            p=M.Params(;n); g=initial_wave(p); c=deepcopy(g)
            wg=M.Workspace(n);wc=M.Workspace(n)
            @assert g.u==c.u && maximum(abs,M.div_face(c,p))==0
            d0=M.diagnostics(g,p,:glm);t=0.0
            path=joinpath(root,"N$n");mkpath(path)
            for (idx,target) in enumerate(range(0,1;length=41))
                while t<target-1e-10
                    ch=max(M.speeds(g,p)...,M.speeds(c,p)...)
                    dt=Float32(min(p.cfl/(n*2ch),target-t))
                    M.step!(g,wg,p,dt,ch,:glm);M.step!(c,wc,p,dt,ch,:ct)
                    t+=dt
                end
                ref=exact_density(n,t)
                for (method,s) in ((:glm,g),(:ct,c))
                    rho=Float64.(s.u[:,:,1]);err=abs.(rho-ref)
                    d=M.diagnostics(s,p,method)
                    prmax=maximum(abs(Float64(M.G.cons2prim(M.load(s.u,i,j),p.gamma,p.floor,p.floor)[5])-1) for i in 1:n,j in 1:n)
                    bmax=max(maximum(abs.(s.u[:,:,6].-.3f0)),maximum(abs.(s.u[:,:,7].-.2f0)))
                    mass=abs(d.mass/d0.mass-1);energy=abs(d.energy/d0.energy-1)
                    @assert mass<1e-5 && energy<1e-5 && prmax<1e-4 && bmax<1e-4
                    println(io,join((n,method,t,sum(err)/n^2,sqrt(sum(abs2,err)/n^2),maximum(err),sum(err)/sum(abs,ref),mass,energy,prmax,bmax,method===:ct ? d.div_face_max : ""),','))
                    if n==64
                        open(joinpath(path,@sprintf("%s_%03d.bin",method,idx-1)),"w") do f;write(f,s.u);end
                    end
                end
            end
            @printf("Exact entropy wave %d² complete\n",n);flush(stdout)
        end
    end
    meta=Dict("problem"=>"finite-amplitude magnetized entropy advection", "N"=>[16,32,64,128],"frames"=>41,"t_end"=>1.0,
        "density"=>"1 + 0.2 sin(2pi(x+y-t)); exact cell average includes sinc(1/N)^2",
        "velocity"=>[.5,.5,0],"magnetic_field"=>[.3,.2,0],"pressure"=>1,"threads"=>Threads.nthreads(),
        "precision"=>"Float32", "gamma"=>Float64(M.Params().gamma),"cfl"=>Float64(M.Params().cfl),
        "solver_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"mhd_2d_cpu.jl")))),
        "limitations"=>"Uniform B: tests entropy transport and multidimensional convergence, not Alfven propagation or shock capture. Not a timing benchmark.")
    open(joinpath(root,"metadata.toml"),"w") do io;TOML.print(io,meta);end
end
run_wave(length(ARGS)>0 ? abspath(ARGS[1]) : joinpath(@__DIR__,"out","entropy_wave"))
