"""Assemble the mini-ramses patch directory for the comparison suite.

    python3 prepare_patch.py MINIRAMSES_DIR OUT_PATCH_DIR

mini-ramses takes a file in PATCH= in place of the upstream file of the same name (bin/Makefile, VPATH). Three files:

  condinit.f90               ours (patch/condinit.f90): the primitive state from the sweep's initial data
  input_hydro_condinit.f90   our module abel_ic (patch/abel_ic.f90, which reads the data), then upstream, with the face-centred field set from the sweep's face files (abel_faces) in
                             place of the curl of vecpotentialinit's vector potential
  newdt_fine.f90             upstream, with the step before an output shortened so that the output lands on its time
                             (what classic RAMSES's exact_output_time does; mini-ramses otherwise writes the output at
                             the first step past it)

Nothing in the solver is touched. Each edit checks that the upstream text it changes is there exactly once.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent

FACES = '''
!#########################################################################
! Comparison suite (sherlock_miniramses): the face-centred field from the sweep's face files.
!#########################################################################
subroutine abel_faces(r,g,m,ilevel)
  use amr_parameters, only: ndim, twotondim, nvector
  use amr_commons, only: run_t, global_t, mesh_t
  use abel_ic, only: face_value
  implicit none
  type(run_t)::r
  type(global_t)::g
  type(mesh_t)::m
  integer::ilevel
  integer::igrid,ind,idim,nstride,i
  real(kind=8),dimension(1:ndim)::xx
  real(kind=8)::dx

  if(m%noct(ilevel)==0)return
#ifdef MHD
  dx=r%boxlen/2**ilevel
  do igrid=m%head(ilevel),m%tail(ilevel)
     do ind=1,twotondim
        do idim=1,ndim
           nstride=2**(idim-1)
           xx(idim)=(2*m%grid(igrid)%ckey(idim)+MOD((ind-1)/nstride,2)+0.5)*dx-m%skip(idim)
        end do
        do idim=1,ndim
           m%bold(ind,idim,igrid)=face_value(xx,dx,ndim,idim,0)
           m%bold(ind,idim+3,igrid)=face_value(xx,dx,ndim,idim,1)
        end do
     end do
  end do
#endif
end subroutine abel_faces
'''

CLAMP = '''  ! Comparison suite (sherlock_miniramses): land exactly on the output times, as classic RAMSES's exact_output_time.
  if(ilevel==r%levelmin.and.g%iout<=r%noutput)then
     if(r%tout(g%iout)>g%t.and.g%t+g%dtnew(ilevel)>r%tout(g%iout))then
        g%dtnew(ilevel)=r%tout(g%iout)-g%t
        do while(g%t+g%dtnew(ilevel)<r%tout(g%iout))
           g%dtnew(ilevel)=nearest(g%dtnew(ilevel),1d0)
        end do
     end if
  end if
'''


def edit(text, old, new, name):
    if text.count(old) != 1:sys.exit(f"{name}: expected exactly one '{old.strip()[:60]}' upstream, found {text.count(old)}")
    return text.replace(old, new)


def main(src, out):
    src, out = Path(src), Path(out);out.mkdir(parents=True, exist_ok=True)
    (out/"condinit.f90").write_text((HERE/"patch"/"condinit.f90").read_text())
    name = "hydro/input_hydro_condinit.f90";text = (src/name).read_text()
    text = edit(text, "  call input_hydro_vecpot(r,g,m,ilevel)\n", "  call abel_faces(r,g,m,ilevel)\n", name)
    # module abel_ic first: the Makefile builds this file before condinit.f90 (bin/deps.mk), which uses the module
    (out/"input_hydro_condinit.f90").write_text((HERE/"patch"/"abel_ic.f90").read_text() + "\n" + text + FACES)
    name = "pm/newdt_fine.f90";text = (src/name).read_text()
    text = edit(text, "  if(r%fixed_dt>0.0d0)g%dtnew(ilevel)=r%fixed_dt\n", CLAMP + "  if(r%fixed_dt>0.0d0)g%dtnew(ilevel)=r%fixed_dt\n", name)
    (out/"newdt_fine.f90").write_text(text)
    print("mini-ramses patch written to", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:sys.exit(__doc__)
    main(*sys.argv[1:])
