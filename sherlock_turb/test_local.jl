# Checks that need no GPU (run on a laptop; the CPU backend of Tom's forcing and of the measurements):
#   julia --project=../sherlock_sweep/env test_local.jl
# 1. the forcing as this bundle sets it up: exact strength, solenoidal, power only at 1 < |k| < 3, memory time turb_T;
# 2. the measurements against the sweep's Lean.diagnostics and against direct sums on the host.
include(joinpath(@__DIR__,"turb_drive.jl"))
using FFTW, Statistics
failures=String[]
check(name,ok,detail="")=(println(ok ? "ok    " : "WRONG ",name,isempty(detail) ? "" : ": "*detail);ok || push!(failures,name))

config=TOML.parsefile(joinpath(@__DIR__,"config.toml"));be=KA.CPU();n=G.TURB_GS
for comp in (0.0,1.0,0.5)
    config["comp_frac"]=comp;p=params(config,"hydro",32,3,10.0,7);d=Drive(p,be,10.0,7,"parabolic")
    a=Array(d.f.afield);rms=sqrt(sum(abs2,Float64.(a))/n^3)
    check("comp_frac $comp: rms of the force is sqrt(3) a_rms",abs(rms/(sqrt(3)*10)-1)<1e-5,@sprintf("%.6f against %.6f",rms,sqrt(3)*10))
    ah=[fft(Float64.(a[c,:,:,:])) for c in 1:3];k=[kfreq(i,n) for i in 1:n]
    kk=[sqrt(k[i]^2+k[j]^2+k[l]^2) for i in 1:n,j in 1:n,l in 1:n]
    power=sum(abs2.(x) for x in ah);inside=sum(power[(kk.>1) .& (kk.<3)])/sum(power)
    check("comp_frac $comp: all power at 1 < |k| < 3",inside>1-1e-8,@sprintf("fraction %.10f",inside))
    kdot=[k[i]*ah[1][i,j,l]+k[j]*ah[2][i,j,l]+k[l]*ah[3][i,j,l] for i in 1:n,j in 1:n,l in 1:n]
    compressive=sum(abs2.(kdot)./max.(kk.^2,1))/sum(power)
    # weights comp and 1-comp on the one compressive and two solenoidal directions: power comp^2 : 2 (1-comp)^2
    expect=comp^2/(comp^2+2*(1-comp)^2)
    check("comp_frac $comp: compressive share of the power",abs(compressive-expect)<(comp in (0.0,1.0) ? 1e-8 : 0.08),@sprintf("%.4f, expected %.4f",compressive,expect))
end
# The projection RAMSES has at the pinned commit, reproduced in Tom's forcing for the check runs: a fifth compressive.
config["comp_frac"]=0.0;p=params(config,"hydro",32,3,10.0,5);d=Drive(p,be,10.0,5,"parabolic";projection="ramses_released");shares=Float64[]
for _ in 1:12
    a=Array(d.f.afield);ah=[fft(Float64.(a[c,:,:,:])) for c in 1:3];k=[kfreq(i,n) for i in 1:n]
    kk=[sqrt(k[i]^2+k[j]^2+k[l]^2) for i in 1:n,j in 1:n,l in 1:n]
    kdot=[k[i]*ah[1][i,j,l]+k[j]*ah[2][i,j,l]+k[l]*ah[3][i,j,l] for i in 1:n,j in 1:n,l in 1:n]
    push!(shares,sum(abs2.(kdot)./max.(kk.^2,1))/sum(sum(abs2.(x) for x in ah)));for _ in 1:100;advance!(d);end
end
check("RAMSES's released projection in Tom's forcing: compressive share of the power",abs(mean(shares)-0.20)<0.03,@sprintf("%.3f +- %.3f (RAMSES's own fields: 0.200)",mean(shares),std(shares)/sqrt(length(shares))))
# Memory of the process: correlation of the field with itself after turb_T should be exp(-1) (the scaling to a fixed
# rms changes it only slightly); averaged over many independent starts.
config["comp_frac"]=0.0;p=params(config,"hydro",32,3,10.0,3);d=Drive(p,be,10.0,3,"parabolic");corr=Float64[]
for _ in 1:15
    a0=copy(Array(d.f.afield));for _ in 1:p.turb_Ndt;advance!(d);end
    a1=Array(d.f.afield);push!(corr,sum(Float64.(a0).*a1)/sqrt(sum(abs2,Float64.(a0))*sum(abs2,Float64.(a1))))
    for _ in 1:2p.turb_Ndt;advance!(d);end
end
check("force decorrelates over turb_T",abs(mean(corr)-exp(-1))<0.08,@sprintf("correlation after one turb_T %.3f +- %.3f, exp(-1) = %.3f",mean(corr),std(corr)/sqrt(length(corr)),exp(-1)))
check("field counter",d.k==15*3*p.turb_Ndt)

# Measurements on a random but smooth state.
N=24;Random.seed!(1);g=Float64(gamma32(config))
x=[(i-0.5)/N for i in 1:N];wave(a,b,c,ph)=[sin(2pi*(a*x[i]+b*x[j]+c*x[l])+ph) for i in 1:N,j in 1:N,l in 1:N]
rho=1 .+0.5 .*wave(1,2,0,0.3);vx=2 .*wave(1,1,1,1.0);vy=1.5 .*wave(1,0,2,2.0);vz=wave(2,1,0,0.5)
pr=rho./g.*(1 .+0.01 .*wave(1,1,1,0.0));E=pr./(g-1).+0.5 .*rho.*(vx.^2 .+vy.^2 .+vz.^2)
u=zeros(Float32,N,N,N,9);u[:,:,:,1]=rho;u[:,:,:,2]=rho.*vx;u[:,:,:,3]=rho.*vy;u[:,:,:,4]=rho.*vz;u[:,:,:,5]=E
q=M.fields(u);s=M.stats(q;gamma=g);l=Lean.diagnostics(q;gamma=g)
for key in (:mass,:kinetic,:thermal,:energy)
    check("stats.$key equals the sweep's diagnostics",isapprox(getfield(s,key),getfield(l,key);rtol=1e-12),@sprintf("%.10g against %.10g",getfield(s,key),getfield(l,key)))
end
r=Float64.(u[:,:,:,1]);v=[Float64.(u[:,:,:,1+c])./r for c in 1:3];v2=v[1].^2 .+v[2].^2 .+v[3].^2
check("v_rms",isapprox(s.v_rms,sqrt(mean(v2));rtol=1e-12))
check("mach_mass",isapprox(s.mach_mass,sqrt(mean(r.*v2)/mean(r))/s.cs;rtol=1e-12))
check("sigma_rho",isapprox(s.sigma_rho,std(r;corrected=false)/mean(r);rtol=1e-9))
check("sigma_s",isapprox(s.sigma_s,std(log.(r./mean(r));corrected=false);rtol=1e-9))
check("rho_min, rho_max, v_max",s.rho_min==minimum(r) && s.rho_max==maximum(r) && isapprox(s.v_max,sqrt(maximum(v2));rtol=1e-12))
# Velocity gradients: centred differences of a single wave have a closed form, sin(2 pi m/N) N/(2 pi m) times exact.
ddx(f,dim)=(circshift(f,ntuple(a->a==dim ? -1 : 0,3)).-circshift(f,ntuple(a->a==dim ? 1 : 0,3))).*(N/2)
w=(ddx(v[3],2).-ddx(v[2],3),ddx(v[1],3).-ddx(v[3],1),ddx(v[2],1).-ddx(v[1],2));dv=ddx(v[1],1).+ddx(v[2],2).+ddx(v[3],3)
check("enstrophy",isapprox(s.enstrophy,mean(w[1].^2 .+w[2].^2 .+w[3].^2);rtol=1e-10),@sprintf("%.6g",s.enstrophy))
check("dilatation",isapprox(s.dilatation,mean(dv.^2);rtol=1e-10) && s.dilatation>1,@sprintf("%.6g",s.dilatation))
check("density-weighted enstrophy",isapprox(s.enstrophy_rho,mean(r.*(w[1].^2 .+w[2].^2 .+w[3].^2));rtol=1e-10))
pure=zeros(Float32,N,N,N,9);pure[:,:,:,1].=1;pure[:,:,:,3]=wave(1,0,0,0.0);pure[:,:,:,5].=1e5       # v_y = sin(2 pi x): curl only
sp=M.stats(M.fields(pure);gamma=g);exact=(2pi)^2/2*(sin(2pi/N)*N/(2pi))^2
check("enstrophy of one shear wave",isapprox(sp.enstrophy,exact;rtol=1e-6) && sp.dilatation<1e-10,@sprintf("%.6f against %.6f",sp.enstrophy,exact))
vol,mass=M.density_pdf(Float32.(r),mean(r))
check("density distribution sums to one",isapprox(sum(vol),1;atol=1e-12) && isapprox(sum(mass),1;atol=1e-6))
check("its width",isapprox(sqrt(sum(vol.*(M.S_LO.+((1:M.S_BINS).-0.5).*M.S_STEP).^2)-sum(vol.*(M.S_LO.+((1:M.S_BINS).-0.5).*M.S_STEP))^2),s.sigma_s;rtol=0.02))
spectrum=Spectra.spectra(q[1],q[2:4],q[6:8])
check("velocity spectrum sums to <v^2>/2",isapprox(sum(spectrum.v),mean(v2)/2;rtol=1e-6),@sprintf("%.6g against %.6g",sum(spectrum.v),mean(v2)/2))
println(isempty(failures) ? "ALL CHECKS PASSED" : "FAILED: "*join(failures,"; "))
exit(isempty(failures) ? 0 : 1)
