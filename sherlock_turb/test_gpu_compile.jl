# Compiles every function this bundle runs on the GPU for the NVIDIA target, with CUDA.jl's own compiler set-up, on a
# machine without a GPU (a laptop):
#   julia --project=../sherlock_sweep/env test_gpu_compile.jl
# It finds what only shows on the device otherwise: code Julia runs happily on the CPU but cannot compile for a GPU
# (a call it cannot resolve at compile time). Checked: every reduction of measure.jl and
# of the timestep scan, inside CUDA.jl's reduction kernel; the velocity-gradient kernel; Tom's three kernel launches.
include(joinpath(@__DIR__,"turb_drive.jl"))
const GPUC=CUDA.GPUCompiler
# CUDA.jl's compiler parameters, except that NVIDIA's math library (libdevice, which ships only for machines that can
# hold a GPU) is not linked: calls to it (__nv_*) stay open and are accepted below. The device runtime is linked.
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
device(::Type{<:AbstractArray{T,N}}) where {T,N}=CuDeviceArray{T,N,1}
device(T::Type)=T

# Every reduction goes through Lean.lazy: record what it is called with on a small CPU state.
const RECORD=Any[]
@eval Lean lazy(f,args...)=(push!(Main.RECORD,(f,args));Broadcast.instantiate(Broadcast.broadcasted(f,args...)))
config=TOML.parsefile(joinpath(@__DIR__,"config.toml"));p=params(config,"hydro",8,3,15.0,1)
u=zeros(Float32,8,8,8,9);u[:,:,:,1].=1;u[:,:,:,5].=1f5
Base.invokelatest(M.stats,M.fields(u);gamma=Float64(p.gamma));Base.invokelatest(signal_max,u,p)
CI=CartesianIndices{3,NTuple{3,Base.OneTo{Int}}}
style=CUDA.CuArrayStyle{3,CUDA.DeviceMemory}();seen=Set{Any}();checked=0
for (f,args) in RECORD
    argtypes=map(typeof,args);(f,argtypes) in seen && continue
    push!(seen,(f,argtypes));global checked+=1;i=checked
    shown=f isa Type ? f : nameof(typeof(f))
    f=CUDA.cudaconvert(Broadcast.Broadcasted(style,f,args,nothing)).f     # as CUDA.jl hands it to a kernel (a type used as a function is wrapped)
    RT=Base.promote_op(f,map(eltype,argtypes)...);op=RT===Bool ? (&) : (+)
    R=RT===Bool ? Bool : RT<:Integer ? Int : Float64
    bc=Base.Broadcast.Broadcasted{CUDA.CuArrayStyle{3,CUDA.DeviceMemory},NTuple{3,Base.OneTo{Int}},Core.Typeof(f),Tuple{map(device,argtypes)...}}
    name="reduction $i: $shown over ($(join(map(eltype,argtypes),", "))) -> $RT"
    isconcretetype(RT) || (report(name,"its result type is not inferred: $RT");continue)
    report(name,ptx_check(CUDA.partial_mapreduce_grid,Tuple{typeof(identity),typeof(op),R,CI,CI,Val{true},CuDeviceArray{R,3,1},bc}))
end
checked>=15 || push!(failures,"only $checked different reductions were recorded")

# The KernelAbstractions kernel of measure.jl, as CUDA.jl's backend would compile it.
K=CUDA.CUDAKernels;kernel=M.velgrad_kernel!(CUDABackend())
ndrange,workgroup,iterspace,dynamic=K.KA.launch_config(kernel,(8,8,8),nothing);ctx=K.KA.mkcontext(kernel,ndrange,iterspace)
D32=CuDeviceArray{Float32,3,1};D64=CuDeviceArray{Float64,3,1}
report("velgrad_kernel!",ptx_check(kernel.f,Tuple{typeof(ctx),D64,D64,D32,D32,D32,D32,Float64}))

# Tom's kernels exactly as step_hydro! and step_tiled! launch them for the three methods.
U=CuDeviceArray{Float32,4,1};f32=Float32
hydro(half)=Tuple{Val{4},Val{half},Val{:hll},U,U,U,f32,f32,f32,f32,f32,f32,Int,f32,f32,f32,Bool}
report("hydro:   integrator_hydro! TB=4, Float32 tile, HLL",ptx_check(G.integrator_hydro!,hydro(false)))
report("hydro16: integrator_hydro! TB=4, Float16 tile, HLL",ptx_check(G.integrator_hydro!,hydro(true)))
report("glm:     integrator_tiled! (not lean)",ptx_check(G.integrator_tiled!,Tuple{Val{false},U,U,U,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,Bool,Int,f32,f32,f32,Bool}))
println(isempty(failures) ? "ALL $checked REDUCTIONS AND 4 KERNELS COMPILE FOR THE GPU" : "FAILED: "*join(failures,"; "))
exit(isempty(failures) ? 0 : 1)
