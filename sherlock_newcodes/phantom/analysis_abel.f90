!--------------------------------------------------------------------------!
! abel: the particles of a Phantom dump for the comparison suite's deposit onto its grid (ANALYSIS=analysis_abel.f90).
! Writes <dumpfile>.abel, a stream file: npart (int64), time, particle mass (float64), then 12 arrays of npart
! float64: x y z h rho vx vy vz u Bx By Bz (rho from h, Phantom's own rhoh).
!--------------------------------------------------------------------------!
module analysis
 implicit none
 character(len=20), parameter, public :: analysistype = 'abel'
 public :: do_analysis

 private

contains

subroutine do_analysis(dumpfile,num,xyzh,vxyzu,particlemass,npart,time,iunit)
 use part, only:Bxyz,mhd,rhoh
 character(len=*), intent(in) :: dumpfile
 integer,          intent(in) :: num,npart,iunit
 real,             intent(in) :: xyzh(:,:),vxyzu(:,:)
 real,             intent(in) :: particlemass,time
 integer :: i,iu
 real(kind=8), allocatable :: a(:,:)

 allocate(a(npart,12))
 do i=1,npart
    a(i,1:4) = xyzh(1:4,i)
    a(i,5)   = rhoh(xyzh(4,i),particlemass)
    a(i,6:8) = vxyzu(1:3,i)
    a(i,9)   = vxyzu(4,i)
    if (mhd) then
       a(i,10:12) = Bxyz(1:3,i)
    else
       a(i,10:12) = 0.
    endif
 enddo
 open(newunit=iu,file=trim(dumpfile)//'.abel',access='stream',form='unformatted',status='replace')
 write(iu) int(npart,kind=8),real(time,kind=8),real(particlemass,kind=8)
 write(iu) a
 close(iu)

end subroutine do_analysis

end module analysis
