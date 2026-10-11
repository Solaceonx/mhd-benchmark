# Kelvin-Helmholtz with the field confined to the lower layer (sweep test kh_localized; Tomida et al. 2026 sec. III.2):
# the field that reaches the field-free upper vortex by t = 3 (median |B| in a disc of radius 0.06 around (0, 0.75),
# as on the Papers page). Reference (AthenaK 4096^2): 4.1e-07. Sweep, GLM LLF: 4.3e-05 at 128^2, 8.8e-06 at 256^2.
#   julia -t 4 leakage.jl N variant [variant ...]        (RIEM=hll for the HLL solver)
# Variants: the original psi damping, other damping rules, and cell-centred constrained transport in place of GLM.
include("emu.jl");using .Emu,Printf,Statistics,Random;include("problems.jl")
function core_median(u)
    n=size(u,1);v=Float64[]
    for j in 1:n,i in 1:n
        x=(i-.5)/n;y=(j-.5)/n;dx=min(abs(x),1-abs(x))
        dx^2+(y-.75)^2<.06^2 && push!(v,sqrt(Float64(u[i,j][6])^2+Float64(u[i,j][7])^2+Float64(u[i,j][8])^2))
    end
    median(v)
end
VARIANTS=Dict(
  "original"=>(;),                                                      # psi damped by exp(-c_h dt/(0.18 L)), c_h from the actual dt
  "ch_cfl"=>(;ch_from=:cfl),                                            # same, c_h does not jump on steps shortened to land on outputs
  "nodamp"=>(;damp=:cell,alpha=0f0,ch_from=:cfl),
  "cell0.05"=>(;damp=:cell,alpha=0.05f0,ch_from=:cfl),
  "cell0.1"=>(;damp=:cell,alpha=0.1f0,ch_from=:cfl),                    # psi damped by exp(-alpha c_h dt/dx)
  "cell0.4"=>(;damp=:cell,alpha=0.4f0,ch_from=:cfl),
  "cell1"=>(;damp=:cell,alpha=1f0,ch_from=:cfl),
  "cell3"=>(;damp=:cell,alpha=3f0,ch_from=:cfl),
  "cell_ct"=>(;fct=true),                                               # cell-centred CT (flux-CT) instead of GLM
  "scale0.25"=>(;ch_scale=0.25f0),                                      # the default glm_ch_scale of glmmhd_turb.jl
)
n=parse(Int,ARGS[1]);riem=Symbol(get(ENV,"RIEM","llf"))
for name in ARGS[2:end]
    r=Emu.Run(;n=n,gamma=5f0/3f0,riem=riem,VARIANTS[name]...)
    t0=time();u,t,steps,why,sh=Emu.evolve(kh_confined(n),r,3.0;frames=31)
    ke=sum(Float64(c[2])^2+Float64(c[3])^2 for c in u)/(2n^2);me=sum(Float64(c[6])^2+Float64(c[7])^2 for c in u)/(2n^2)
    @printf("KH N=%d %-5s %-10s t = %.2f, %6d steps: field in the field-free vortex %.3e (%.1f x the reference) | kinetic %.4f, magnetic %.3e  (%.0f s) %s\n",
            n,riem,name,t,steps,core_median(u),core_median(u)/4.1e-7,ke,me,time()-t0,why)
    flush(stdout)
end
