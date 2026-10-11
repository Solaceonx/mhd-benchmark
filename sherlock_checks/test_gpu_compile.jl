# Compiles the kernel variants the checks add for the NVIDIA target, with CUDA.jl's own compiler set-up, on a machine
# without a GPU (a laptop), and checks every generated configuration against the driver:
#   julia --project=../sherlock_sweep/env test_gpu_compile.jl
# The triples step_plm! lists are compiled too, as controls of this test.
include(joinpath(@__DIR__,"..","sherlock_sweep","src","run_suite.jl"))
using CUDA, TOML
const G=Suite.G
const GPUC=CUDA.GPUCompiler
# CUDA.jl's compiler parameters, except that NVIDIA's math library (libdevice, which ships only for machines that can
# hold a GPU) is not linked: calls to it (__nv_*) stay open and are accepted below.
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
        return join(unique([string(kind," (",first(string(meta),160),") at ",join(["$(fr.func) $(basename(string(fr.file))):$(fr.line)" for fr in first(bt,4)]," < ")) for (kind,bt,meta) in real]),"\n        ")
    end
end
failures=String[]
report(name,why)=(println(isempty(why) ? "ok    " : "WRONG ",name,isempty(why) ? "" : "\n        "*why);isempty(why) || push!(failures,name))
U=CuDeviceArray{Float32,4,1};f32=Float32
args(tb,half,riem)=Tuple{Val{tb},Val{half},Val{riem},U,U,U,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,Int,f32,f32,f32,Bool}
for (name,f,tb,half,riem) in (
        ("listed:  integrator_plm! 4-cell tile, Float16, HLL",G.integrator_plm!,4,true,:hll),
        ("listed:  integrator_plm! 6-cell tile, Float16, HLL",G.integrator_plm!,6,true,:hll),
        ("listed:  integrator_plm! 4-cell tile, Float32, HLLD (glm_hlld and the control glm_hlld_x)",G.integrator_plm!,4,false,:hlld),
        ("new:     integrator_plm! 4-cell tile, Float32, HLL (glm_hll32)",G.integrator_plm!,4,false,:hll),
        ("new:     integrator_plm! 6-cell tile, Float32, HLL (glm_hll32 on grids divisible by 6)",G.integrator_plm!,6,false,:hll),
        ("listed:  integrator_plm_ppm! 4-cell tile, Float16, HLL",G.integrator_plm_ppm!,4,true,:hll),
        ("new:     integrator_plm_ppm! 4-cell tile, Float32, HLL (glm_ppm32)",G.integrator_plm_ppm!,4,false,:hll),
        ("new:     integrator_plm_ppm! 6-cell tile, Float32, HLL (glm_ppm32 on grids divisible by 6)",G.integrator_plm_ppm!,6,false,:hll))
    report(name,ptx_check(f,args(tb,half,riem)))
end
# every configuration passes the driver's own checks, and every planned run is one its configuration allows
configs=sort(filter(endswith(".toml"),readdir(joinpath(@__DIR__,"configs");join=true)))
for c in configs
    try Suite.check_config(TOML.parsefile(c)) catch e;push!(failures,basename(c));println("WRONG ",basename(c),": ",sprint(showerror,e)) end
end
plan=readlines(`python3 $(joinpath(@__DIR__,"checks.py")) runs /nowhere`);append!(plan,readlines(`python3 $(joinpath(@__DIR__,"checks.py")) runs /nowhere --trial`))
for l in plan
    c,out,n,seed,methods=split(l);config=TOML.parsefile(joinpath(@__DIR__,c))
    ok=parse(Int,n) in config["grids"] && parse(Int,seed) in config["seeds"] && all(m->m in Suite.METHODS && m in config["methods"],split(methods,","))
    ok || (push!(failures,l);println("WRONG run: ",l))
end
println("ok    $(length(configs)) configurations, $(length(plan)) run directories")
println(isempty(failures) ? "ALL KERNEL VARIANTS COMPILE FOR THE GPU; CONFIGURATIONS AND RUN LIST AGREE WITH THE DRIVER" : "FAILED: "*join(failures,"; "))
exit(isempty(failures) ? 0 : 1)
