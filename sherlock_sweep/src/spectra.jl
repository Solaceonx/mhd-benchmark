# Shell-averaged power spectra of a periodic N^3 state, on the device that holds it (CUDA via CUFFT, or the CPU
# for tests with FFTW loaded). Shell k collects modes with round(|k|) = k, k = 1 .. N/2, in units of 2 pi / L.
# Returned energies are normalized so that sum_k E(k) = <|f|^2>/2 over the box (Parseval), for:
#   v (with its solenoidal and compressive parts), sqrt(rho) v, rho^(1/3) v (Kritsuk et al. 2007), B, and
#   rho - <rho> (reported as <|drho|^2>/2).
module Spectra
using AbstractFFTs, LinearAlgebra

"Integer wavenumbers along one axis (FFT order)."
kaxis(n)=[i<=n÷2 ? i : i-n for i in 0:n-1]
const SHELL_CACHE=Dict{Tuple{Int,DataType},Any}()
"""
Modes of shells 1:N/2 grouped by shell (counting sort on the host, once per N and array type): the
device permutation `perm` lists shell 1's linear indices, then shell 2's, ...; `first`/`count` locate
each shell in it. Each shell is then summed on its own, so faint high-k shells keep full precision.
"""
function shells(n,like)
    get!(SHELL_CACHE,(n,typeof(like))) do
        k=kaxis(n);nk=n÷2;k2=k.^2
        key(i,j,l)=round(Int,sqrt(Float64(k2[i]+k2[j]+k2[l])))
        count=zeros(Int,nk)
        for l in 1:n,j in 1:n,i in 1:n;s=key(i,j,l);1<=s<=nk && (count[s]+=1);end
        first=cumsum([1;count[1:end-1]]);next=copy(first);perm=Vector{Int32}(undef,sum(count))
        lin=LinearIndices((n,n,n))
        for l in 1:n,j in 1:n,i in 1:n
            s=key(i,j,l);1<=s<=nk || continue
            perm[next[s]]=lin[i,j,l];next[s]+=1
        end
        d=similar(like,Int32,length(perm));copyto!(d,perm)
        (perm=d,first=first,count=count)
    end
end
"Sum `p` over each shell 1:nk."
function shellsum(p,sh)
    q=p[sh.perm]
    [Float64(sum(view(q,f:f+c-1))) for (f,c) in zip(sh.first,sh.count)]
end

"""
spectra(rho, m, b): rho (N,N,N), momentum m and field b as 3-tuples of (N,N,N) arrays on one device.
Returns a NamedTuple of Vector{Float64} (length N/2) plus k.
"""
function spectra(rho,m,b)
    n=size(rho,1);sh=shells(n,rho);norm=1/Float64(n)^6
    e(p)=shellsum(p,sh).*(norm/2)
    power(xs)=sum(x->abs2.(fft(complex.(Float64.(x)))),xs)
    v=map(mi->mi./rho,m)
    vh=map(x->fft(complex.(Float64.(x))),v)
    k=kaxis(n);kx=similar(rho,Float64,n,1,1);ky=similar(rho,Float64,1,n,1);kz=similar(rho,Float64,1,1,n)
    copyto!(kx,reshape(Float64.(k),n,1,1));copyto!(ky,reshape(Float64.(k),1,n,1));copyto!(kz,reshape(Float64.(k),1,1,n))
    # compressive part: projection of v-hat on k (the k = 0 mode is not in any shell)
    pc=abs2.(kx.*vh[1].+ky.*vh[2].+kz.*vh[3])./max.(kx.^2 .+ky.^2 .+kz.^2,1.0)
    ev=e(abs2.(vh[1]).+abs2.(vh[2]).+abs2.(vh[3]));ec=e(pc);es=ev.-ec
    vh=nothing;pc=nothing
    ew=e(power(map(vi->sqrt.(rho).*vi,v)))
    e3=e(power(map(vi->cbrt.(rho).*vi,v)))
    eb=e(power(b))
    dr=Float64.(rho);dr=dr.-sum(dr)/length(dr);er=e(abs2.(fft(complex.(dr))))
    (k=collect(1:n÷2),v=ev,v_solenoidal=es,v_compressive=ec,sqrtrho_v=ew,cbrtrho_v=e3,b=eb,rho=er)
end
end
