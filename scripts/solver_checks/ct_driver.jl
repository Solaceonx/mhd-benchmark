# Loop structure of the kernels of vendor/ct_mhd.jl for z-invariant problems; included into a module that holds the
# vendored functions (see ct_slope.jl), so every call below is to that module's own pslope, hanc1d, hll, ...
const U5=NTuple{5,Float32}
wr(i,n)=mod1(i,n)

function cellprims(U,bxf,byf,bz,n)
    [begin
        rho=max(U[i,j][1],SMALLR);ir=1f0/rho;vx=U[i,j][2]*ir;vy=U[i,j][3]*ir;vz=U[i,j][4]*ir
        bx=0.5f0*(bxf[i,j]+bxf[wr(i+1,n),j]);by=0.5f0*(byf[i,j]+byf[i,wr(j+1,n)]);b3=bz[i,j]
        p=max((GAMMA-1f0)*(U[i,j][5]-0.5f0*rho*(vx*vx+vy*vy+vz*vz)-0.5f0*(bx*bx+by*by+b3*b3)),SMALLP)
        (rho,vx,vy,vz,p,bx,by,b3)
    end for i in 1:n,j in 1:n]
end

function face_flux(q,i,j,d,bn,dtdx,n)
    o1=d==1 ? 1 : 0;o2=d==2 ? 1 : 0
    m2=q[wr(i-2o1,n),wr(j-2o2,n)];m1=q[wr(i-o1,n),wr(j-o2,n)];c0=q[i,j];p1=q[wr(i+o1,n),wr(j+o2,n)]
    sL=pslope(m2,m1,c0);sR=pslope(m1,c0,p1)
    mhL=hanc1d(m1,sL,d,dtdx);mhR=hanc1d(c0,sR,d,dtdx)
    L=psub(mhL,sL,0.5f0);R=psub(mhR,sR,-0.5f0)
    L=(max(L[1],SMALLR),L[2],L[3],L[4],max(L[5],SMALLP),d==1 ? bn : L[6],d==2 ? bn : L[7],L[8])
    R=(max(R[1],SMALLR),R[2],R[3],R[4],max(R[5],SMALLP),d==1 ? bn : R[6],d==2 ? bn : R[7],R[8])
    hll(L,R,d)
end

function step!(U,bxf,byf,bz,dt,n)
    dtdx=dt*n;q=cellprims(U,bxf,byf,bz,n)
    Fx=[face_flux(q,i,j,1,bxf[i,j],dtdx,n) for i in 1:n,j in 1:n]
    Fy=[face_flux(q,i,j,2,byf[i,j],dtdx,n) for i in 1:n,j in 1:n]
    Fz=[hll(hanc1d(q[i,j],ntuple(_->0f0,8),3,dtdx),hanc1d(q[i,j],ntuple(_->0f0,8),3,dtdx),3) for i in 1:n,j in 1:n]   # identical z neighbours
    finish!(U,bxf,byf,bz,Fx,Fy,Fz,dtdx,n)
end

function finish!(U,bxf,byf,bz,Fx,Fy,Fz,dtdx,n)
    ez=[0.25f0*(Fy[i,j][6]+Fy[wr(i-1,n),j][6]-Fx[i,j][7]-Fx[i,wr(j-1,n)][7]) for i in 1:n,j in 1:n]
    ex=[0.25f0*(Fz[i,j][7]+Fz[i,wr(j-1,n)][7]-2f0*Fy[i,j][8]) for i in 1:n,j in 1:n]
    ey=[0.25f0*(2f0*Fx[i,j][8]-Fz[i,j][6]-Fz[wr(i-1,n),j][6]) for i in 1:n,j in 1:n]
    for j in 1:n,i in 1:n
        ip=wr(i+1,n);jp=wr(j+1,n)
        U[i,j]=ntuple(t->U[i,j][t]+dtdx*((Fx[i,j][t]-Fx[ip,j][t])+(Fy[i,j][t]-Fy[i,jp][t])),5)
    end
    nbx=similar(bxf);nby=similar(byf);nbz=similar(bz)
    for j in 1:n,i in 1:n
        ip=wr(i+1,n);jp=wr(j+1,n)
        nbx[i,j]=bxf[i,j]-dtdx*(ez[i,jp]-ez[i,j])
        nby[i,j]=byf[i,j]+dtdx*(ez[ip,j]-ez[i,j])
        nbz[i,j]=bz[i,j]-dtdx*((ey[ip,j]-ey[i,j])-(ex[i,jp]-ex[i,j]))
    end
    bxf.=nbx;byf.=nby;bz.=nbz
end


# The same step with the predictor of the GLM kernels: a half step of the cell centre with the flux differences of ALL
# directions (hanc1d uses the face's own direction only, so the fluxes through the other faces are not time-centred).
function prim_of(Uh)
    rho=max(Uh[1],SMALLR);ir=1f0/rho;vx=Uh[2]*ir;vy=Uh[3]*ir;vz=Uh[4]*ir;bx=Uh[6];by=Uh[7];b3=Uh[8]
    p=max((GAMMA-1f0)*(Uh[5]-0.5f0*rho*(vx*vx+vy*vy+vz*vz)-0.5f0*(bx*bx+by*by+b3*b3)),SMALLP)
    (rho,vx,vy,vz,p,bx,by,b3)
end
function step_full!(U,bxf,byf,bz,dt,n)
    dtdx=dt*n;q=cellprims(U,bxf,byf,bz,n);h=0.5f0*dtdx
    sx=[pslope(q[wr(i-1,n),j],q[i,j],q[wr(i+1,n),j]) for i in 1:n,j in 1:n]
    sy=[pslope(q[i,wr(j-1,n)],q[i,j],q[i,wr(j+1,n)]) for i in 1:n,j in 1:n]
    mh=[begin
        c=q[i,j];a=dflux(psub(c,sx[i,j],0.5f0),1);b=dflux(psub(c,sx[i,j],-0.5f0),1);e=dflux(psub(c,sy[i,j],0.5f0),2);f=dflux(psub(c,sy[i,j],-0.5f0),2)
        C=toC(c);prim_of(ntuple(t->C[t]-h*((a[t]-b[t])+(e[t]-f[t])),8))
    end for i in 1:n,j in 1:n]
    fix(L,d,bn)=(max(L[1],SMALLR),L[2],L[3],L[4],max(L[5],SMALLP),d==1 ? bn : L[6],d==2 ? bn : L[7],d==3 ? bn : L[8])
    Fx=[hll(fix(psub(mh[wr(i-1,n),j],sx[wr(i-1,n),j],0.5f0),1,bxf[i,j]),fix(psub(mh[i,j],sx[i,j],-0.5f0),1,bxf[i,j]),1) for i in 1:n,j in 1:n]
    Fy=[hll(fix(psub(mh[i,wr(j-1,n)],sy[i,wr(j-1,n)],0.5f0),2,byf[i,j]),fix(psub(mh[i,j],sy[i,j],-0.5f0),2,byf[i,j]),2) for i in 1:n,j in 1:n]
    Fz=[hll(fix(mh[i,j],3,bz[i,j]),fix(mh[i,j],3,bz[i,j]),3) for i in 1:n,j in 1:n]
    finish!(U,bxf,byf,bz,Fx,Fy,Fz,dtdx,n)
end

function signal(U,bxf,byf,bz,n)
    q=cellprims(U,bxf,byf,bz,n);v=0f0
    for c in q;v=max(v,abs(c[2])+fast_speed(c,1)+abs(c[3])+fast_speed(c,2)+abs(c[4])+fast_speed(c,3));end
    v
end

# initial data as src/problems.jl: faces from edge samples of A (periodic part) plus B0; point values of the primitives.
function initial(n,prim,A,B0)
    c(i)=(i-.5)/n;f(i)=(i-1)/n;g=5/3
    Ax=[A(c(i),f(j))[1] for i in 1:n,j in 1:n];Ay=[A(f(i),c(j))[2] for i in 1:n,j in 1:n];Az=[A(f(i),f(j))[3] for i in 1:n,j in 1:n]
    bxf=Float32[n*(Az[i,wr(j+1,n)]-Az[i,j])+B0[1] for i in 1:n,j in 1:n]
    byf=Float32[-n*(Az[wr(i+1,n),j]-Az[i,j])+B0[2] for i in 1:n,j in 1:n]
    bz=Float32[n*((Ay[wr(i+1,n),j]-Ay[i,j])-(Ax[i,wr(j+1,n)]-Ax[i,j]))+B0[3] for i in 1:n,j in 1:n]
    U=Matrix{U5}(undef,n,n)
    for j in 1:n,i in 1:n
        r,vx,vy,vz,p=prim(c(i),c(j));bx=(bxf[i,j]+bxf[wr(i+1,n),j])/2f0;by=(byf[i,j]+byf[i,wr(j+1,n)])/2f0
        U[i,j]=(Float32(r),Float32(r*vx),Float32(r*vy),Float32(r*vz),Float32(p/(g-1)+r*(vx^2+vy^2+vz^2)/2+(Float64(bx)^2+Float64(by)^2+Float64(bz[i,j])^2)/2))
    end
    U,bxf,byf,bz
end

function run(n,prim,A,B0,t_end;frames=21,courant=0.25f0,full=false)
    U,bxf,byf,bz=initial(n,prim,A,B0);t=0.0
    for target in range(0,Float64(t_end);length=frames)
        while t<target-1e-10
            dt=Float32(min(courant/(n*signal(U,bxf,byf,bz,n)),target-t));(full ? step_full!(U,bxf,byf,bz,dt,n) : step!(U,bxf,byf,bz,dt,n));t+=Float64(dt)
        end
    end
    U,cellprims(U,bxf,byf,bz,n)
end

