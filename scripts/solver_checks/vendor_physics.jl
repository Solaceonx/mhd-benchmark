# Load Tom Abel's device physics, unedited, from the vendored solver files (sherlock_sweep/vendor) into a module, so the
# checks in this folder run his own functions on the CPU. Only the pure functions are taken (by line range); the kernels
# around them need CUDA. The files are checked against the hashes in vendor/PROVENANCE.json, so a changed vendor file
# stops the checks instead of silently shifting the line ranges.
using SHA
const VENDOR=normpath(joinpath(@__DIR__,"..","..","sherlock_sweep","vendor"))
const VENDOR_SHA=Dict("glmmhd_turb.jl"=>"36a8431c0bbbb48f6d6a9f7b340d6a270221a439007d05084fdcfe5824b1b35e",
                      "ct_mhd.jl"=>"ccacb4756280d3ac4b46d2df57e1625c88c1ba3770cedbd2a4eeb0370faf4bc8")
# glmmhd_turb.jl: constants and device physics up to the Riemann dispatch (59-367), iflo (618), the PPM helpers (917-926).
const GLM_LINES=(59:367,618:618,917:926)
# ct_mhd.jl: constants, limiter, primitives, fluxes, HLL, the one-direction Hancock predictor (14-82).
const CT_LINES=(14:82,)

"""
    load_physics(mod, file, ranges; edit=nothing)

Evaluate the given line ranges of a vendored file in module `mod`. `edit = old => new` replaces `old`, which must occur
exactly once: the one-line changes the checks test.
"""
function load_physics(mod::Module,file,ranges;edit=nothing)
    path=joinpath(VENDOR,file)
    bytes2hex(open(sha256,path))==VENDOR_SHA[file] || error("$file is not the vendored version these line ranges were read from")
    lines=readlines(path);text=join((join(lines[r],"\n") for r in ranges),"\n")
    if edit!==nothing
        count(edit.first,text)==1 || error("expected exactly one occurrence of `$(edit.first)` in $file")
        text=replace(text,edit)
    end
    Base.include_string(mod,text,file);mod
end
