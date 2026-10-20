# Compiles every kernel the sweep-fixes branch adds or changes for the NVIDIA target, with CUDA.jl's own compiler
# set-up, on a machine without a GPU (a laptop), and checks the configurations and the run list against the driver:
#   bash prepare.sh && julia --project=../sherlock_sweep/env test_gpu_compile.jl
# The driver is loaded as on Sherlock: src/ (the sweep's) beside this bundle's vendor/ (the branch).
using CUDA, TOML
shadow=mktempdir();symlink(joinpath(@__DIR__,"..","sherlock_sweep","src"),joinpath(shadow,"src"));symlink(joinpath(@__DIR__,"vendor"),joinpath(shadow,"vendor"))
include(joinpath(shadow,"src","run_suite.jl"))
const G=Suite.G
Suite.FIXES || error("the driver did not load the branch: run prepare.sh first")
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
        isempty(real) && return ""
        return join(unique([string(kind," (",first(string(meta),160),") at ",join(["$(fr.func) $(basename(string(fr.file))):$(fr.line)" for fr in first(bt,4)]," < ")) for (kind,bt,meta) in real]),"\n        ")
    end
end
failures=String[]
report(name,why)=(println(isempty(why) ? "ok    " : "WRONG ",name,isempty(why) ? "" : "\n        "*why);isempty(why) || push!(failures,name))
U4=CuDeviceArray{Float32,4,1};U3=CuDeviceArray{Float32,3,1};f32=Float32
skip_ptx=get(ENV,"SKIP_PTX","")!=""          # SKIP_PTX=1: only the configurations and run lists (kernels unchanged)
skip_ptx && println("skipping the PTX compilation (SKIP_PTX)")
args(tb,half,riem,anc,fct)=Tuple{Val{tb},Val{half},Val{riem},U4,U4,U4,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,Int,f32,f32,f32,Bool,Val{anc},Val{fct},U4}
skip_ptx || for (f,tb,half,riem,anc,fct,used) in (
        (G.integrator_plm!,6,true,:hll,false,false,"glm_hll16 (original storage)"),
        (G.integrator_plm!,4,true,:hll,false,false,"glm_hll16 on grids not divisible by 6"),
        (G.integrator_plm!,6,true,:hll,true,false,"glm_fix_hll16"),
        (G.integrator_plm!,4,true,:hll,true,false,"glm_fix_hll16, 4-cell tile"),
        (G.integrator_plm!,4,false,:hlld,false,false,"glm_fix_hlld"),
        (G.integrator_plm!,6,true,:hll,true,true,"glm_fct16"),
        (G.integrator_plm!,4,true,:hll,true,true,"glm_fct16, 4-cell tile"),
        (G.integrator_plm!,6,false,:hll,false,true,"glm_fct32"),
        (G.integrator_plm!,4,false,:hll,false,true,"glm_fct32, 4-cell tile"),
        (G.integrator_plm_ppm!,6,true,:hll,false,false,"glm_ppm (original storage)"),
        (G.integrator_plm_ppm!,6,true,:hll,true,false,"glm_fix_ppm"),
        (G.integrator_plm_ppm!,4,true,:hll,true,false,"glm_fix_ppm, 4-cell tile"))
    report("$(nameof(f)) tile $tb, $(half ? "Float16" : "Float32"), $riem, anchored=$anc, fct=$fct ($used)",ptx_check(f,args(tb,half,riem,anc,fct)))
end
skip_ptx || report("fct_emf_kernel!",ptx_check(G.fct_emf_kernel!,Tuple{U4,Int}))
skip_ptx || report("fct_update_kernel!",ptx_check(G.fct_update_kernel!,Tuple{U4,U4,U4,f32,Int}))
skip_ptx || for C in unique(values(Suite.CT_BY_GAMMA))
    report("$(nameof(C)).flux_kernel! (ct_fix)",ptx_check(C.flux_kernel!,Tuple{U4,U4,U3,U3,U3,U3,Int,f32,Int}))
end
# every configuration passes the driver's own checks, and every planned run is one its configuration allows
configs=String[];plan=String[]
for (dir,name) in (("configs","fixes"),("configs_verify","verify"))
    mine=sort(filter(endswith(".toml"),readdir(joinpath(@__DIR__,dir);join=true)));append!(configs,mine)
    for c in mine
        try Suite.check_config(TOML.parsefile(c)) catch e;push!(failures,basename(c));println("WRONG ",dir,"/",basename(c),": ",sprint(showerror,e)) end
    end
    append!(plan,readlines(`python3 $(joinpath(@__DIR__,"fixes.py")) --plan $name runs /nowhere`));append!(plan,readlines(`python3 $(joinpath(@__DIR__,"fixes.py")) --plan $name runs /nowhere --trial`))
end
for l in plan
    c,out,n,seed,methods=split(l);config=TOML.parsefile(joinpath(@__DIR__,c))
    ok=parse(Int,n) in config["grids"] && parse(Int,seed) in config["seeds"] && all(m->m in Suite.METHODS && m in config["methods"],split(methods,","))
    ok || (push!(failures,l);println("WRONG run: ",l))
end
for m in Suite.FIX_METHODS
    try Suite.check_vendor(m) catch e;push!(failures,string(m));println("WRONG ",m,": ",sprint(showerror,e)) end
end
println("ok    $(length(configs)) configurations, $(length(plan)) run directories")
println(isempty(failures) ? (skip_ptx ? "CONFIGURATIONS AND RUN LISTS AGREE WITH THE DRIVER (PTX not compiled)" : "ALL NEW KERNELS COMPILE FOR THE GPU; CONFIGURATIONS AND RUN LISTS AGREE WITH THE DRIVER") : "FAILED: "*join(failures,"; "))
exit(isempty(failures) ? 0 : 1)
