include(joinpath(@__DIR__,"mhd_2d_cpu.jl"))
using .MHD2DCPU, Printf, TOML, Base.Threads
const M=MHD2DCPU
function run_compare(;n=64,tend=0.5,nframes=101,root=joinpath(@__DIR__,"out","glm_ct_$n"))
    n>=4 && nframes>=2 && tend>0 || error("Require N>=4, frames>=2, and positive end time")
    isdir(root) && !isempty(readdir(root)) && error("Output directory not empty")
    mkpath(root);p=M.Params(;n);g=M.initialize(p;method=:glm);c=M.initialize(p;method=:ct)
    wg=M.Workspace(n);wc=M.Workspace(n)
    # Compile both paths on disposable states.
    M.step!(deepcopy(g),wg,p,0.001f0,3f0,:glm)
    M.step!(deepcopy(c),wc,p,0.001f0,3f0,:ct)
    M.diagnostics(g,p,:glm); M.diagnostics(c,p,:ct)
    t=0.0;steps=0;glm_seconds=ct_seconds=0.0;start=time()
    mass0=sum(Float64,@view g.u[:,:,1])/n^2;energy0=sum(Float64,@view g.u[:,:,5])/n^2
    errors=Dict("glm_mass"=>0.0,"glm_energy"=>0.0,"ct_mass"=>0.0,"ct_energy"=>0.0)
    handles=Dict(name=>open(joinpath(root,name*"_diagnostics.csv"),"w") for name in ("glm","ct"))
    try
        for (idx,target) in enumerate(range(0,tend;length=nframes))
            while t<target-1e-10
                gx,gy=M.speeds(g,p);cx,cy=M.speeds(c,p)
                ch=max(gx,gy,cx,cy)
                # Common step, including both GLM cleaning directions.
                dt=Float32(min(p.cfl/(n*2ch),target-t))
                # Alternate call order to reduce a systematic timing-order bias.
                if iseven(steps)
                    ts=time_ns();M.step!(g,wg,p,dt,ch,:glm);glm_seconds+=(time_ns()-ts)*1e-9
                    ts=time_ns();M.step!(c,wc,p,dt,ch,:ct);ct_seconds+=(time_ns()-ts)*1e-9
                else
                    ts=time_ns();M.step!(c,wc,p,dt,ch,:ct);ct_seconds+=(time_ns()-ts)*1e-9
                    ts=time_ns();M.step!(g,wg,p,dt,ch,:glm);glm_seconds+=(time_ns()-ts)*1e-9
                end
                t+=dt;steps+=1
            end
            for (name,s,method) in (("glm",g,:glm),("ct",c,:ct))
                d=M.diagnostics(s,p,method)
                if idx==1;println(handles[name],"frame,time,steps,",join(string.(keys(d)),','));end
                println(handles[name],join((idx-1,t,steps,values(d)...),','));flush(handles[name])
                errors[name*"_mass"]=max(errors[name*"_mass"],abs(d.mass/mass0-1))
                errors[name*"_energy"]=max(errors[name*"_energy"],abs(d.energy/energy0-1))
                max(errors[name*"_mass"],errors[name*"_energy"])<1e-4 || error("Conservation failure")
                open(joinpath(root,@sprintf("%s_%03d.bin",name,idx-1)),"w") do io;write(io,s.u);end
                if method===:ct
                    open(joinpath(root,@sprintf("ct_bxface_%03d.bin",idx-1)),"w") do io;write(io,s.bx);end
                    open(joinpath(root,@sprintf("ct_byface_%03d.bin",idx-1)),"w") do io;write(io,s.by);end
                end
                if idx==nframes && method===:ct
                    open(joinpath(root,name*"_final_bxface.bin"),"w") do io;write(io,s.bx);end
                    open(joinpath(root,name*"_final_byface.bin"),"w") do io;write(io,s.by);end
                end
            end
            if idx==1 || idx%20==0 || idx==nframes
                dg=M.diagnostics(g,p,:glm);dc=M.diagnostics(c,p,:ct)
                @printf("frame %d/%d t=%.3f steps=%d GLM div_rms=%.3g CT face div_max=%.3g\n",idx,nframes,t,steps,dg.div_center_rms,dc.div_face_max);flush(stdout)
            end
        end
    finally
        foreach(close,values(handles))
    end
    dg=M.diagnostics(g,p,:glm);dc=M.diagnostics(c,p,:ct)
    relative_l1=sum(abs.(Float64.(g.u[:,:,1]).-c.u[:,:,1]))/sum(Float64,c.u[:,:,1])
    meta=Dict("N"=>n,"dimensions"=>2,"frames"=>nframes,"t_end"=>t,"steps"=>steps,"gamma"=>p.gamma,
        "cfl"=>p.cfl,"threads"=>Threads.nthreads(),"precision"=>"Float32","flux"=>"shared existing HLL routines",
        "reconstruction"=>"MonCen PLM","time_integrator"=>"SSP-RK2, shared dt; Strang split GLM damping",
        "ct_emf"=>"Balsara-Spicer arithmetic average of magnetic face fluxes",
        "glm_cleaning_speed"=>"max directional signal over both methods, recomputed each step",
        "glm_damping_coefficient"=>p.cleaning_damping,"glm_step_seconds"=>glm_seconds,"ct_step_seconds"=>ct_seconds,
        "wall_seconds_excluding_compile"=>time()-start,"max_saved_relative_conservation_errors"=>errors,
        "final_density_relative_l1_difference"=>relative_l1,"glm_final_center_div_rms"=>dg.div_center_rms,
        "ct_final_face_div_max"=>dc.div_face_max,"ct_final_center_div_rms"=>dc.div_center_rms,
        "binary_format"=>"little-endian Float32 (N,N,9) column-major: rho,mx,my,mz,E,Bx,By,Bz,psi",
        "ct_face_format"=>"every saved frame: ct_bxface_###.bin and ct_byface_###.bin; lower faces, Float32 (N,N) column-major",
        "timing_scope"=>"step includes fluxes, RK updates and stage positivity checks; excludes CFL, output, diagnostics, compilation")
    open(joinpath(root,"metadata.toml"),"w") do io;TOML.print(io,meta);end
    @printf("DONE %d^2: GLM %.3fs CT %.3fs; density relative L1 difference %.4f\n",n,glm_seconds,ct_seconds,relative_l1)
end
if abspath(PROGRAM_FILE)==@__FILE__
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 64
    root=length(ARGS)>1 ? abspath(ARGS[2]) : joinpath(@__DIR__,"out","glm_ct_$n")
    nframes=length(ARGS)>2 ? parse(Int,ARGS[3]) : 101
    run_compare(;n,root,nframes)
end
