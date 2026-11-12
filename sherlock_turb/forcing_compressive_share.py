# Compressive share (along the true k) of RAMSES's forcing power, released find_unitk vs fixed, parabolic spectrum, 3-D
import numpy as np
GS=64; h=GS//2
s=np.arange(GS); sg=np.where(s>h, s-GS, s)                  # signed wave number (build_power_spectrum)
wr=np.where(s>h, GS-s, s)                                   # find_unitk's |k| for the negative half
K=np.stack(np.meshgrid(sg,sg,sg,indexing='ij'),-1).astype(float)
W=np.stack(np.meshgrid(wr,wr,wr,indexing='ij'),-1).astype(float)
km=np.linalg.norm(K,axis=-1); P=np.where((km>1)&(km<3),1-(km-2)**2,0.)
m=P>0; w=P[m]**2; k=K[m]/km[m,None]; u=W[m]/np.linalg.norm(W[m],axis=-1)[:,None]
c2=(k*u).sum(-1)**2                                         # cos^2 between true k and the vector used
print("driven modes (both halves):", m.sum(), " mixed-sign share of power: %.3f" % (w[c2<1-1e-12].sum()/w.sum()))
for cf in (0.0, 1/3, 0.5, 2/3, 1.0):
    c,s_=cf,1-cf
    tot=(c*c+2*s_*s_)
    bug=(w*(c*c*c2+s_*s_*(1-c2))).sum()/(w.sum()*tot); ok=c*c/tot
    print(f"comp_frac {cf:.3f}: compressive share of forcing power intended {ok:.3f}, released RAMSES {bug:.3f}")
