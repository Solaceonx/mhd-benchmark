# Compiles the kernels of the order x solver study's three new set-ups of Abel et al.'s code for the NVIDIA target, with
# CUDA.jl's own compiler set-up, on a machine without a GPU (a laptop), and checks the configurations and the run list
# against the driver (as sherlock_fixes/test_gpu_compile.jl does for the fixes):
#   bash prepare.sh && julia --project=../sherlock_sweep/env test_gpu_compile.jl
using CUDA, TOML
shadow=mktempdir();symlink(joinpath(@__DIR__,"..","sherlock_sweep","src"),joinpath(shadow,"src"));symlink(joinpath(@__DIR__,"vendor"),joinpath(shadow,"vendor"))
include(joinpath(shadow,"src","run_suite.jl"))
const G=Suite.G
Suite.FIXES || error("the driver did not load the branch: run prepare.sh first")
const GPUC=CUDA.GPUCompiler
# NVIDIA's math library (libdevice) ships only for machines that can hold a GPU: calls to it (__nv_*) stay open.
struct LocalParams <: CUDA.AbstractCUDACompilerParams end
GPUC.link_libraries!(@nospecialize(job::GPUC.CompilerJob{GPUC.PTXCompilerTarget,LocalParams}),mod::CUDA.LLVM.Module)=nothing
function ptx_check(f,tt)
    sup=CUDA.llvm_compat();cap=maximum(filter(<=(v"9.0"),sup.cap));ptx=maximum(sup.ptx)
    config=GPUC.CompilerConfig(GPUC.PTXCompilerTarget(;cap,ptx,debuginfo=false),LocalParams();kernel=true,name=nothing,always_inline=false)
    job=GPUC.CompilerJob(GPUC.methodinstance(Core.Typeof(f),tt),config)
    try
        GPUC.JuliaContext() do ctx;GPUC.compile(:llvm,job);end
        return ""
    catch e
        e isa GPUC.InvalidIRError || return first(sprint(showerror,e),900)
        real=[(kind,bt,meta) for (kind,bt,meta) in e.errors if !(meta isa AbstractString && startswith(meta,"__nv"))]
        isempty(real) && return ""
        return join(unique([string(kind," (",first(string(meta),160),") at ",join(["$(fr.func) $(basename(string(fr.file))):$(fr.line)" for fr in first(bt,4)]," < ")) for (kind,bt,meta) in real]),"\n        ")
    end
end
failures=String[]
report(name,why)=(println(isempty(why) ? "ok    " : "WRONG ",name,isempty(why) ? "" : "\n        "*why);isempty(why) || push!(failures,name))
U4=CuDeviceArray{Float32,4,1};f32=Float32
args(tb,half,riem)=Tuple{Val{tb},Val{half},Val{riem},U4,U4,U4,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,Int,f32,f32,f32,Bool,Val{false},Val{false},U4}
for (f,riem,used) in ((G.integrator_plm!,:hll,"glm_fix_hll32"),(G.integrator_plm!,:hlld,"glm_fix_hlld (the fourth pair, from the rerun)"),
                      (G.integrator_plm_ppm!,:hll,"glm_fix_ppm32"),(G.integrator_plm_ppm!,:hlld,"glm_fix_ppm_hlld"))
    report("$(nameof(f)) tile 4, Float32, $riem ($used)",ptx_check(f,args(4,false,riem)))
end
for m in (:glm_fix_hll32,:glm_fix_ppm32,:glm_fix_ppm_hlld)
    try Suite.check_vendor(m);haskey(Suite.SOLVER_TEXT,m) || error("no SOLVER_TEXT") catch e;push!(failures,string(m));println("WRONG ",m,": ",sprint(showerror,e)) end
end
# every configuration passes the driver's own checks, and every planned run is one its configuration allows
configs=sort(filter(endswith(".toml"),readdir(joinpath(@__DIR__,"configs");join=true)))
for c in configs
    try Suite.check_config(TOML.parsefile(c)) catch e;push!(failures,basename(c));println("WRONG configs/",basename(c),": ",sprint(showerror,e)) end
end
plan=vcat(readlines(`python3 $(joinpath(@__DIR__,"order.py")) runs glm /nowhere`),readlines(`python3 $(joinpath(@__DIR__,"order.py")) runs glm /nowhere --pilot`))
for l in plan
    c,out,n,seed,methods=split(l);config=TOML.parsefile(joinpath(@__DIR__,c))
    ok=parse(Int,n) in config["grids"] && parse(Int,seed) in config["seeds"] && all(m->m in Suite.METHODS && m in config["methods"],split(methods,","))
    ok || (push!(failures,l);println("WRONG run: ",l))
end
println("ok    $(length(configs)) configurations, $(length(plan)) runs")
println(isempty(failures) ? "THE NEW KERNELS COMPILE FOR THE GPU; CONFIGURATIONS AND RUN LISTS AGREE WITH THE DRIVER" : "FAILED: "*join(failures,"; "))
exit(isempty(failures) ? 0 : 1)
