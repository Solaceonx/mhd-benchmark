using Pkg, TOML
root=normpath(joinpath(@__DIR__,".."));env=joinpath(root,"env")
Pkg.activate(env)
locked=TOML.parsefile(joinpath(env,"Manifest.toml"))["julia_version"]
version=VersionNumber(locked)
if (VERSION.major,VERSION.minor)!=(version.major,version.minor)
    error("The bundled lockfile was created with Julia $locked. Load a Julia $(version.major).$(version.minor) module or set JULIA_BIN to that version. No environment was modified.")
end
Pkg.instantiate()
Pkg.precompile()
include("check_initial_conditions.jl")
println("Environment ready. Submit jobs/01_smoke.sbatch next; CUDA is not required for this setup job.")
