# Compiles every kernel NestedMHD launches for the NVIDIA target, on a machine without a GPU (lyxia's Mac):
#   julia --project=env test_gpu_compile.jl [path/to/NestedTurbulence]
# A short CPU (PoCL) run of each set-up the GPU job runs records every kernel launch; each is then compiled with the
# argument types it gets on a GPU (null-pointer CuDeviceArrays of the same shape), through CUDA.jl's compiler, without
# NVIDIA's math library (libdevice does not ship for macOS; its __nv_* calls are accepted).
using CUDA, KernelAbstractions
const KA = KernelAbstractions
const NT = get(ARGS, 1, joinpath(@__DIR__, "..", "external", "NestedTurbulence"))
include(joinpath(NT, "core", "src", "NestedMHD.jl")); using .NestedMHD
include(joinpath(@__DIR__, "setups.jl"))
const CC = CUDA.CUDACore            # CUDA.jl 6 keeps the compiler in CUDACore
const GPUC = CC.GPUCompiler

# record every launch on the CPU backend, then launch it as usual
const REC = Dict{Any,Any}()
@eval KA function launch_tuple(obj::Kernel{<:POCLBackend}, args::Tuple; ndrange=nothing, workgroupsize=nothing)
    key = (obj.f, map(Core.Typeof, args), ndrange isa Tuple ? length(ndrange) : ndrange)
    haskey(Main.REC, key) || (Main.REC[key] = (obj, args, ndrange, workgroupsize))
    invoke(launch_tuple, Tuple{Kernel, Tuple}, obj, args; ndrange, workgroupsize)
end

struct LocalParams <: CC.AbstractCUDACompilerParams end
GPUC.link_libraries!(@nospecialize(job::GPUC.CompilerJob{GPUC.PTXCompilerTarget,LocalParams}), mod::CC.LLVM.Module) = nothing
function ptx_check(f, tt)
    sup = CC.llvm_compat(); cap = v"9.0"; ptx = maximum(sup.ptx)          # the H100 (sm_90)
    config = GPUC.CompilerConfig(GPUC.PTXCompilerTarget(; cap, ptx, debuginfo=false), LocalParams(); kernel=true, name=nothing,
                                 always_inline=false)
    job = GPUC.CompilerJob(GPUC.methodinstance(Core.Typeof(f), tt), config)
    try
        GPUC.JuliaContext() do ctx; GPUC.compile(:llvm, job); end
        return ""
    catch e
        e isa GPUC.InvalidIRError || return first(sprint(showerror, e), 900)
        real = [(kind, bt, meta) for (kind, bt, meta) in e.errors if !(meta isa AbstractString && startswith(meta, "__nv"))]
        return join(unique([string(kind, " (", first(string(meta), 160), ") at ",
                                   join(["$(fr.func) $(basename(string(fr.file))):$(fr.line)" for fr in first(bt, 4)], " < "))
                            for (kind, bt, meta) in real]), "\n        ")
    end
end
fake(a::Array{T,N}) where {T,N} = CuDeviceArray{T,N,CUDA.AS.Global}(reinterpret(CC.LLVMPtr{T,CUDA.AS.Global}, C_NULL), size(a))
fake(x) = x

# the device side of a recorded launch, as backend_launch.jl's launch_tuple builds it for CUDABackend
function device_job(obj, args, ndrange, workgroupsize)
    be = CUDABackend()
    g = KA.Kernel{typeof(be), KA.workgroupsize(obj), KA.ndrange(obj), typeof(obj.f)}(be, obj.f)
    setup = KA.launch_setup(KA.sizes_only(g), ndrange, workgroupsize)
    setup === nothing && return nothing
    nd, wg, iterspace = setup
    launch = KA.select_launch(KA.sizes_only(g), wg, iterspace)
    ctx = KA.mkcontext(g, nd, iterspace, launch; subgroups=nothing)
    _, tt = KA.prepare_arguments(be, g.f, ctx, map(fake, args))
    return g.f, tt
end

# the check must catch a kernel that cannot run on a GPU (a dynamic call: string formatting)
@kernel function bad_kernel!(A)
    i = @index(Global)
    @inbounds A[i] = length(string(A[i]))
end
# (not launched: KernelAbstractions 0.10 compiles CPU kernels through PoCL, which refuses it too)
isempty(ptx_check(device_job(bad_kernel!(KA.CPU()), (zeros(Float32, 4),), (4,), nothing)...)) &&
    (println("WRONG the check passed a kernel that cannot compile"); exit(2))
println("ok    the check catches a dynamic call")

for (name, f) in SETUPS
    print("run   ", name, " … "); t = @elapsed f(KA.CPU(), true); println(round(t; digits=1), " s")
end
failures = String[]
for (key, (obj, args, nd, wg)) in sort(collect(REC); by = kv -> string(kv[1][1]))
    name = string(nameof(obj.f), " ", join([a isa Array ? "Array{$(eltype(a)),$(ndims(a))}" : string(typeof(a)) for a in args
                                            if !(a isa Real)], ","))
    j = device_job(obj, args, nd, wg)
    j === nothing && continue
    why = ptx_check(j...)
    println(isempty(why) ? "ok    " : "WRONG ", first(name, 150), isempty(why) ? "" : "\n        " * why)
    isempty(why) || push!(failures, name)
end
println(length(REC), " launches compiled, ", length(failures), " failed")
exit(isempty(failures) ? 0 : 1)
