"""Write the RAMSES II patch directory for the speed test: 3D Orszag-Tang.

RAMSES II ships Orszag-Tang (INIT=OT) for two dimensions only: in 3D its condinit writes one element past the
primitive array, and its vector potential returns the same function for all three components. At the pinned commit
INIT=OT also does not compile (pi is declared twice). This script copies
upstream hydro/condinit.f90 and hydro/vecpotentialinit.f90 with the two OT blocks replaced, and amr/update_time.f90
with the energies and the time printed to full precision (two format lines), so that NDIM=3 gives the
problem of our throughput test (sherlock_sweep, test throughput_ot3d = orszag_tang_3d, epsilon = 0.2):

    rho = 25/(36 pi), p = 5/(12 pi), v = (-(1+e sin 2 pi z) sin 2 pi y, (1+e sin 2 pi z) sin 2 pi x, e sin 2 pi z),
    A = (0, 0, (cos 4 pi x/(4 pi) + cos 2 pi y/(2 pi))/sqrt(4 pi)),      e = 0.2

usage: python3 prepare_patch.py RAMSES2_DIR OUT_PATCH_DIR
"""
from pathlib import Path
import sys

CONDINIT_OLD = """#if INIT==OT
  pi=acos(-1.0d0)
  do i=1,nn
     xc=x(i,1)
     yc=x(i,2)
     q(i,1)=25.0/(36.0*pi)
     q(i,2)=-sin(2.0*pi*yc)
     q(i,3)=+sin(2.0*pi*xc)
     q(i,4)=0.0
     q(i,5)=5.0/(12.0*pi)
     q(i,nvar+1)=0.0 ! Bz
  end do
#endif
"""
CONDINIT_NEW = """#if INIT==OT
  ! abel_mhd speed test: Orszag-Tang, modulated along z in 3D (epsilon = 0.2)
  pi=acos(-1.0d0)
  do i=1,nn
     xc=x(i,1)
     yc=x(i,2)
     q(i,1)=25.0d0/(36.0d0*pi)
#if NDIM==3
     q(i,2)=-(1.0d0+0.2d0*sin(2.0d0*pi*x(i,3)))*sin(2.0d0*pi*yc)
     q(i,3)=+(1.0d0+0.2d0*sin(2.0d0*pi*x(i,3)))*sin(2.0d0*pi*xc)
     q(i,4)=0.2d0*sin(2.0d0*pi*x(i,3))
#else
     q(i,2)=-sin(2.0d0*pi*yc)
     q(i,3)=+sin(2.0d0*pi*xc)
     q(i,4)=0.0d0
#endif
     q(i,5)=5.0d0/(12.0d0*pi)
#if NDIM==2
     q(i,nvar+1)=0.0d0 ! Bz
#endif
  end do
#endif
"""
# Upstream declares pi twice when INIT=OT (once for every problem, once in the OT branch), which does not compile.
DECL_OLD = """#elif INIT==OT
  real(kind=8)::pi,xc,yc
"""
DECL_NEW = """#elif INIT==OT
  real(kind=8)::xc,yc
"""
VECPOT_OLD = """     A(i) = B0*(cos(4.0*pi*xx)/(4.0*pi)+cos(2.0*pi*yy)/(2.0*pi))
"""
VECPOT_NEW = """     A(i) = 0.0d0
     if(idim==3)A(i) = B0*(cos(4.0d0*pi*xx)/(4.0d0*pi)+cos(2.0d0*pi*yy)/(2.0d0*pi))
"""


# The code prints its energies and the time with three and six digits; validation needs all of them.
FORMAT_OLD = (("779     format(' Main step= ',i0,' mcons=',1pe9.2,' econs=',1pe9.2,' epot=',1pe9.2,' ekin=',1pe9.2,' eint=',1pe9.2,' emag=',1pe9.2)",
               "779     format(' Main step= ',i0,' mcons=',1pe9.2,' econs=',1pe9.2,' epot=',1pe9.2,' ekin=',1pe23.15,' eint=',1pe23.15,' emag=',1pe23.15)"),
              ("888 format(' Fine step= ',i0,' t=',1pe12.5,' dt=',1pe10.3,", "888 format(' Fine step= ',i0,' t=',1pe23.15,' dt=',1pe23.15,"))


def swap(text, old, new, name):
    if text.count(old) != 1:sys.exit(f"{name}: the upstream Orszag-Tang block is not the one this patch was written for")
    return text.replace(old, new)


def main(src, out):
    src, out = Path(src), Path(out);out.mkdir(parents=True, exist_ok=True)
    cond = swap((src/"hydro/condinit.f90").read_text(), CONDINIT_OLD, CONDINIT_NEW, "condinit.f90")
    (out/"condinit.f90").write_text(swap(cond, DECL_OLD, DECL_NEW, "condinit.f90 (declarations)"))
    (out/"vecpotentialinit.f90").write_text(swap((src/"hydro/vecpotentialinit.f90").read_text(), VECPOT_OLD, VECPOT_NEW, "vecpotentialinit.f90"))
    time = (src/"amr/update_time.f90").read_text()
    for old, new in FORMAT_OLD:time = swap(time, old, new, "update_time.f90")
    (out/"update_time.f90").write_text(time)
    print("RAMSES II patch written to", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
