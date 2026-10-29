!--------------------------------------------------------------------------!
! abel: the comparison suite's starting state for Phantom (SETUPFILE=setup_abel.f90).
!
! Reads <fileprefix>.particles, written by newcodes_run.py from the suite's own state (tools/write_spec.jl):
! equal-mass particles on near-cubic lattices pasted together, of the spacing each cell's density needs (one particle per cell at the mean
! density), each with the velocity, specific internal energy and field of the suite's cell it lies in, and the
! periodic box (the unit square times a slab of 8 layers for planar tests, the unit cube otherwise).
! Stream file: npart (int64), particle mass, xmin xmax ymin ymax zmin zmax, gamma, tmax, dtmax (float64), then
! 11 arrays of npart float64: x y z h vx vy vz u Bx By Bz. Every dump is a full dump, one each dtmax.
!--------------------------------------------------------------------------!
module setup
 implicit none
 public :: setpart

 private

contains

subroutine setpart(id,npart,npartoftype,xyzh,massoftype,vxyzu,polyk,gamma,hfact,time,fileprefix)
 use dim,          only:maxp,maxvxyzu,mhd
 use setup_params, only:npart_total,ihavesetupB
 use io,           only:fatal
 use boundary,     only:set_boundary
 use part,         only:igas,Bxyz
 use timestep,     only:tmax,dtmax
 use io_control,   only:nfulldump
 use kernel,       only:hfact_default
 integer,          intent(in)    :: id
 integer,          intent(out)   :: npart
 integer,          intent(out)   :: npartoftype(:)
 real,             intent(out)   :: xyzh(:,:)
 real,             intent(out)   :: massoftype(:)
 real,             intent(out)   :: vxyzu(:,:)
 real,             intent(out)   :: polyk,gamma,hfact
 real,             intent(inout) :: time
 character(len=*), intent(in)    :: fileprefix
 integer         :: i,iu,ios
 integer(kind=8) :: n8
 real(kind=8)    :: pmass,box(6),gam,t_max,dt_max
 real(kind=8), allocatable :: a(:,:)

 open(newunit=iu,file=trim(fileprefix)//'.particles',access='stream',form='unformatted',status='old',iostat=ios)
 if (ios /= 0) call fatal('setup_abel','cannot open '//trim(fileprefix)//'.particles')
 read(iu) n8,pmass,box,gam,t_max,dt_max
 if (n8 > maxp) call fatal('setup_abel','more particles than maxp')
 allocate(a(n8,11))
 read(iu) a
 close(iu)

 npart = int(n8)
 npart_total = n8
 time  = 0.
 hfact = hfact_default
 gamma = real(gam)
 polyk = 0.
 call set_boundary(real(box(1)),real(box(2)),real(box(3)),real(box(4)),real(box(5)),real(box(6)))
 do i=1,npart
    xyzh(1:4,i)  = real(a(i,1:4))
    vxyzu(1:3,i) = real(a(i,5:7))
    if (maxvxyzu >= 4) vxyzu(4,i) = real(a(i,8))
    if (mhd) Bxyz(1:3,i) = real(a(i,9:11))
 enddo
 npartoftype(:) = 0
 npartoftype(igas) = npart
 massoftype(igas)  = real(pmass)
 if (mhd) ihavesetupB = .true.
 tmax      = real(t_max)
 dtmax     = real(dt_max)
 nfulldump = 1

end subroutine setpart

end module setup
