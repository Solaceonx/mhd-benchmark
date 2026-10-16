# Run the GPU kernels of a GLMMHDTurb source file on the CPU, unedited: the file is loaded with `using CUDA` replaced by
# FakeCUDA below, whose @cuda loops over the blocks (threaded) and whose thread intrinsics, barriers and shared arrays
# are ordinary functions and arrays. A tiled kernel (integrator_*) runs with ONE thread per block: each of its stages is
# a strided loop over the stage's work items (t = tid; t += nth), so that thread does all of them in order and
# sync_threads has nothing to wait for. The one-cell-per-thread kernels (ct_mhd.jl, fct_*) run every thread in turn.
# Used by fix_branch.jl to check the sweep-fixes branch against the original before anything goes to Sherlock.
module FakeCUDA
export @cuda, blockIdx, threadIdx, blockDim, sync_threads, CuStaticSharedArray, CuArray, CuDeviceArray, CUDABackend, CUDA
const BLK = [(1, 1, 1) for _ in 1:Threads.maxthreadid()]
const THR = ones(Int, Threads.maxthreadid())
const DIM = ones(Int, Threads.maxthreadid())
@inline blockIdx() = (b = @inbounds(BLK[Threads.threadid()]); (x = b[1], y = b[2], z = b[3]))
@inline threadIdx() = (x = @inbounds(THR[Threads.threadid()]), y = 1, z = 1)
@inline blockDim() = (x = @inbounds(DIM[Threads.threadid()]), y = 1, z = 1)
@inline sync_threads() = nothing
CuStaticSharedArray(::Type{T}, n::Integer) where {T} = Vector{T}(undef, n)
abstract type CuArray{T,N} <: AbstractArray{T,N} end      # for signatures only: no CPU array is one
abstract type CuDeviceArray{T,N} <: AbstractArray{T,N} end
struct CUDABackend end
module CUDA
    zeros(T::Type, dims...) = Base.zeros(T, dims...)
    synchronize() = nothing
    macro elapsed(ex)
        :(Base.@elapsed $(esc(ex)))
    end
end
macro cuda(exs...)
    call = exs[end]; threads = 1; blocks = 1
    for kw in exs[1:end-1]
        kw.args[1] === :threads && (threads = kw.args[2])
        kw.args[1] === :blocks && (blocks = kw.args[2])
    end
    esc(:($FakeCUDA.launch($(call.args[1]), $threads, $blocks, $(call.args[2:end]...))))
end
function launch(f, threads, blocks, args...)
    b = blocks isa Integer ? (blocks, 1, 1) : (blocks..., ntuple(_ -> 1, 3 - length(blocks))...)
    nth = startswith(string(nameof(f)), "integrator_") ? 1 : threads
    Threads.@threads :static for idx in 0:prod(b)-1
        id = Threads.threadid()
        BLK[id] = (idx % b[1] + 1, (idx ÷ b[1]) % b[2] + 1, idx ÷ (b[1] * b[2]) + 1); DIM[id] = nth
        for t in 1:nth
            THR[id] = t; f(args...)
        end
    end
    nothing
end
end

"Load a solver file (glmmhd_turb.jl or ct_mhd.jl) as module `name`, with its CUDA calls served by FakeCUDA."
function load_solver(path::AbstractString, name::Symbol)
    swap(x) = x
    swap(x::Expr) = (x.head === :using && x.args == [Expr(:., :CUDA)]) ? :(using Main.FakeCUDA) :
                    (x.head === :using && length(x.args) == 2 && x.args[1] == Expr(:., :CUDA)) ? Expr(:using, Expr(:., :Main, :FakeCUDA), x.args[2]) :
                    Expr(x.head, map(swap, x.args)...)
    function rename(ex)
        ex isa Expr && ex.head === :module || return ex
        Expr(:module, ex.args[1], name, swap(ex.args[3]))
    end
    Base.include(rename, Main, path)
    Base.invokelatest(getfield, Main, name)
end
