# Initial data of the sweep's tests (sherlock_sweep/src/problems.jl), for the planar emulation in emu.jl.
const S5=sqrt(5.0)
entropy(n)=Emu.initial(n,5/3,(x,y)->(1+.2*sinc(1/n)^2*sin(2pi*(x+y)),.5,.5,0.0,1.0);B0=(.3,.2,0.0))
function alfven_oblique(n)
    tx,ty=-2/S5,1/S5;ph(x,y)=2pi*(x+2y)
    Emu.initial(n,5/3,(x,y)->(1.0,-.1*cos(ph(x,y))*tx,-.1*cos(ph(x,y))*ty,-.1*sin(ph(x,y)),.1);
        Az=(x,y)->-.1*sin(ph(x,y))/(2pi*S5),B0=(1/S5,2/S5,0.0),Bz=(x,y)->.1*sin(ph(x,y)))
end
field_loop(n,vz)=Emu.initial(n,5/3,(x,y)->(1.0,2.0,1.0,vz,1.0);Az=(x,y)->1e-3*max(.3-hypot(x-.5,y-.5),0.0))
function orszag_tang(n)
    b=1/sqrt(4pi);Emu.initial(n,5/3,(x,y)->(25/(36pi),-sin(2pi*y),sin(2pi*x),0.0,5/(12pi));Az=(x,y)->b*(cos(2pi*y)/2pi+cos(4pi*x)/4pi))
end
square_integral(x,a,b)=(x=mod(x,1.0);x<a ? x : x<b ? 2a-x : x+2a-2b)
current_sheet(n)=Emu.initial(n,5/3,(x,y)->(1.0,0.1*sin(2pi*y),0.0,0.0,0.1);Az=(x,y)->-square_integral(x,.25,.75))
function gem(n)
    L=25.6;lam=.5/L;psi0=.1/L;nbg=.2
    sheets(y)=sech((y-.25)/lam)^2+sech((y-.75)/lam)^2
    lncosh(u)=abs(u)+log1p(exp(-2abs(u)))-log(2.0)
    Emu.initial(n,5/3,(x,y)->(sheets(y)+nbg,0.0,0.0,0.0,.5*(sheets(y)+nbg));
        Az=(x,y)->lam*lncosh((y-.25)/lam)-lam*lncosh((y-.75)/lam)-y-psi0*cos(2pi*x)*cos(2pi*(y-.25)),sub=8)
end
blast(n,bx)=Emu.initial(n,7/5,(x,y)->(1.0,0.0,0.0,0.0,hypot(x-.5,y-.5)<.1 ? 1000.0 : .1);B0=(bx/sqrt(4pi),0.0,0.0),sub=8)
"Kelvin-Helmholtz with the field confined to the lower layer (kh_localized)"
function kh_confined(n)
    a=.05;sig=.2;pr=10.0;v0=1.0;b=.01;wd=.1;mean_b=2b*wd
    F(y)=sum(b*wd*(tanh((y-.25+sh)/wd)-tanh((sh-.25)/wd)) for sh in -1:1)-mean_b*y
    Emu.initial(n,5/3,(x,y)->(1.0,v0*(tanh((y-.25)/a)-tanh((y-.75)/a)-1),.01*sin(2pi*x)*(exp(-((y-.25)/sig)^2)+exp(-((y-.75)/sig)^2)),0.0,pr);
        Az=(x,y)->F(y),B0=(mean_b,0.0,0.0))
end
"A 2D stand-in for decay_supersonic: smooth random solenoidal velocities at rms Mach 9, gamma = 1.01, B = 0.3 along z."
function mach9_2d(n;seed=42)
    rng=Random.MersenneTwister(seed);modes=[(kx,ky,randn(rng),2pi*rand(rng)) for kx in -2:2,ky in -2:2 if 1<=kx^2+ky^2<=4]
    vel(x,y)=begin vx=0.0;vy=0.0
        for (kx,ky,a,ph) in modes;s=a*sin(2pi*(kx*x+ky*y)+ph)/hypot(kx,ky);vx+=ky*s;vy-=kx*s;end;(vx,vy) end
    c=((1:n).-.5)./n;rms=sqrt(sum(sum(abs2,vel(x,y)) for x in c,y in c)/n^2);sc=9.0/rms
    Emu.initial(n,1.01,(x,y)->(1.0,(sc.*vel(x,y))...,0.0,1/1.01);B0=(0.0,0.0,0.3))
end
density_error(u,u0)=sum(abs.(Emu.comp(u,1).-Emu.comp(u0,1)))/length(u)
"relative L1 error of B as the sweep defines it (src/errors.jl)"
field_error(u,u0)=sum(sqrt(sum((Float64(u[k][v])-Float64(u0[k][v]))^2 for v in 6:8)) for k in eachindex(u))/sum(sqrt(sum(Float64(u0[k][v])^2 for v in 6:8)) for k in eachindex(u))
grids_from(args,default)=isempty(args) ? default : Tuple(parse.(Int,args))
fmt(v)=join((@sprintf("%.2e",x) for x in v),"  ")
orders(e)=join((@sprintf("%.2f",log2(e[k]/e[k+1])) for k in 1:length(e)-1),", ")
