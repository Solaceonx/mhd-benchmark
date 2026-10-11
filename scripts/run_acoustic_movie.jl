# Save new snapshots of the existing isothermal validation; original results remain.
include(joinpath(@__DIR__,"isothermal_hydro.jl"))
using TOML
function main(root)
    isdir(root) && !isempty(readdir(root)) && error("Output directory must be empty")
    mkpath(root)
    open(joinpath(root,"profiles.csv"),"w") do io
        println(io,"N,frame,time,x,density,linear_reference")
        for n in (32,64)
            p=IsoParams(N=n);a=zeros(Float32,n+2IG,n,n,4);b=similar(a);amp=1f-3
            for k in 1:n,j in 1:n,i in 1:n
                wave=amp*sin(2pi*(i-.5)/n)
                a[i+IG,j,k,1]=1+wave;a[i+IG,j,k,2]=(1+wave)*wave
            end
            iso_ghost!(a,p);t=0.0;d=iso_stats(a,p)
            for (frame,target) in enumerate(range(0,.2;length=21))
                while t<target-1e-9
                    dt=min(p.courant/n/d.signal,target-t)
                    h,d,_=iso_advance!(b,a,p,dt);a,b=b,a;t+=h
                end
                @assert abs(d.mass-1)<1e-6
                for i in 1:n
                    x=(i-.5)/n;println(io,join((n,frame-1,t,x,a[i+IG,1,1,1],1+amp*sin(2pi*(x-t))),','))
                end
            end
        end
    end
    open(joinpath(root,"metadata.toml"),"w") do io
        TOML.print(io,Dict("frames"=>21,"N"=>[32,64],"time_end"=>.2,"amplitude"=>.001,"threads"=>Threads.nthreads(),
            "note"=>"New output-clipped run for movies; not identical timestep sequence to original acoustic check. Linear analytical reference is approximate for nonlinear equations."))
    end
end
main(length(ARGS)>0 ? abspath(ARGS[1]) : joinpath(@__DIR__,"out","acoustic_movie"))
