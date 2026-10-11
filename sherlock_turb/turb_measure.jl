# Measures the frames of a RAMSES turbulence run with the functions the GPU driver uses on its own state (measure.jl):
#   julia --project=$SWEEP_BUNDLE/env --threads=N turb_measure.jl RUN_DIRECTORY
# ramses_turb.py has written run.toml and one tmp_###.bin per frame ((N, N, N, 9) Float32, the sweep's state layout).
# This writes series.csv (one line per frame), spectra.csv, pdf.csv and the slices, keeps the last frame as
# state_final.bin and deletes the others. Safe to run again: a run already measured is only cleaned up.
using TOML, Printf, KernelAbstractions, FFTW
const SWEEP=get(ENV,"SWEEP_BUNDLE",normpath(joinpath(@__DIR__,"..","sherlock_sweep")))
for f in ("problems.jl","errors.jl","lean.jl","spectra.jl");include(joinpath(SWEEP,"src",f));end
include(joinpath(@__DIR__,"measure.jl"))
const M=TurbMeasure
FFTW.set_num_threads(Threads.nthreads())

function cleanup(dir,meta)
    frames=meta["frames"]
    if !isempty(frames) && haskey(frames[end],"file")
        file=joinpath(dir,frames[end]["file"]);isfile(file) && mv(file,joinpath(dir,"state_final.bin");force=true)
    end
    for f in readdir(dir);startswith(f,"tmp_") && endswith(f,".bin") && rm(joinpath(dir,f));end
end
function measure(dir)
    path=joinpath(dir,"run.toml");meta=TOML.parsefile(path)
    get(meta,"measured",false) && return cleanup(dir,meta)
    n=meta["N"];gamma=meta["gamma"];T=meta["turb_T"];every=get(meta,"slices_every",4)
    frames=sort(meta["frames"];by=f->f["frame"]);w=M.Writer(dir)
    for (i,fr) in enumerate(frames)
        u=Array{Float32}(undef,n,n,n,9);read!(joinpath(dir,fr["file"]),u)
        frame=fr["frame"];t=fr["time"];last=i==length(frames)
        row=M.stats(M.fields(u);gamma)
        M.series!(w,(index=frame,time=t,time_over_T=t/T,steps=fr["steps"],frame=frame),row)
        out=M.frame!(w,frame,t,u,row.mass;slice=frame%every==0 || last)
        haskey(out,"slice") && (fr["slice"]=out["slice"])
        last && (fr["state"]="state_final.bin")
        @printf("   frame %3d, t = %.3f T: Mach %.3f, density %.3g to %.3g\n",frame,t/T,row.mach_vol,row.rho_min,row.rho_max);flush(stdout)
    end
    close(w)
    meta["frames"]=frames;meta["measured"]=true
    meta["measured_with"]="turb_measure.jl: measure.jl on the CPU ($(Threads.nthreads()) threads), frames read as Float32"
    open(io->TOML.print(io,meta),path*".tmp","w");mv(path*".tmp",path;force=true)
    cleanup(dir,meta)
end
measure(abspath(ARGS[1]))
