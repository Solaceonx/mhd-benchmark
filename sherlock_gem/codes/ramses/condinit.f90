!================================================================
! RAMSES initial conditions for the GLM/CT comparison suite (sherlock_mhd).
! Reads the exact Float32 initial state written by tools/write_spec.jl --state:
!   $ABEL_IC_PREFIX.state : N^3 x 9 cell state (rho, m, E, Bcc, psi), x fastest
!   $ABEL_IC_PREFIX.face1..3 : lower-face B (N^3 each), x fastest; $ABEL_IC_N = N
! Requires a uniform periodic grid on [0,1]^3 with levelmin = levelmax = log2(N). RAMSES also
! calls condinit on coarser levels while building the tree; those get conservative block
! averages of the fine data (and are replaced by restriction of the fine level anyway).
! Build: make SOLVER=mhd NDIM=3 PATCH=<this directory>.
!================================================================
module abel_ic
  use amr_parameters, only: dp
  implicit none
  integer, save :: nic = 0
  real(kind=4), allocatable, save :: st(:,:,:,:), bf(:,:,:,:)
contains
  subroutine load_ic()
    character(len=4096) :: prefix, nstr
    integer :: unit, d, status, n
    character(len=1) :: c
    if (nic > 0) return
    call get_environment_variable('ABEL_IC_N', nstr, status=status)
    if (status /= 0) then
       write(*,*) 'abel condinit: ABEL_IC_N is not set'
       stop 1
    end if
    read(nstr, *) n
    call get_environment_variable('ABEL_IC_PREFIX', prefix, status=status)
    if (status /= 0) then
       write(*,*) 'abel condinit: ABEL_IC_PREFIX is not set'
       stop 1
    end if
    allocate(st(n, n, n, 9), bf(n, n, n, 3))
    open(newunit=unit, file=trim(prefix)//'.state', access='stream', form='unformatted', status='old')
    read(unit) st
    close(unit)
    do d = 1, 3
       write(c, '(i1)') d
       open(newunit=unit, file=trim(prefix)//'.face'//c, access='stream', form='unformatted', status='old')
       read(unit) bf(:, :, :, d)
       close(unit)
    end do
    nic = n
  end subroutine load_ic
end module abel_ic

subroutine condinit(x, u, dx, nn)
  use amr_parameters
  use hydro_parameters
  use abel_ic
  implicit none
  integer :: nn
  real(dp) :: dx
  real(dp), dimension(1:nvector, 1:nvar+3) :: u
  real(dp), dimension(1:nvector, 1:ndim) :: x
  integer :: i, n, r, ix, iy, iz, i0, j0, k0, i1, j1, k1, v
#if NDIM != 3
  write(*,*) 'abel condinit requires NDIM=3'
  stop 1
#endif
  call load_ic()
  n = nint(boxlen/dx); r = nic/n
  if (r*n /= nic) then
     write(*,*) 'abel condinit: level with', n, 'cells does not divide N =', nic
     stop 1
  end if
  u(1:nn, :) = 0.0d0
  do i = 1, nn
     ix = int(x(i,1)/dx); iy = int(x(i,2)/dx); iz = int(x(i,3)/dx)
     i0 = ix*r + 1; j0 = iy*r + 1; k0 = iz*r + 1; i1 = i0 + r - 1; j1 = j0 + r - 1; k1 = k0 + r - 1
     do v = 1, 5
        u(i, v) = sum(real(st(i0:i1, j0:j1, k0:k1, v), dp))/r**3
     end do
     ! Lower faces of this (coarse) cell, and upper faces = lower faces of the next cell.
     u(i, 6) = sum(real(bf(i0, j0:j1, k0:k1, 1), dp))/r**2
     u(i, 7) = sum(real(bf(i0:i1, j0, k0:k1, 2), dp))/r**2
     u(i, 8) = sum(real(bf(i0:i1, j0:j1, k0, 3), dp))/r**2
     u(i, nvar+1) = sum(real(bf(modulo(i1, nic) + 1, j0:j1, k0:k1, 1), dp))/r**2
     u(i, nvar+2) = sum(real(bf(i0:i1, modulo(j1, nic) + 1, k0:k1, 2), dp))/r**2
     u(i, nvar+3) = sum(real(bf(i0:i1, j0:j1, modulo(k1, nic) + 1, 3), dp))/r**2
  end do
end subroutine condinit
