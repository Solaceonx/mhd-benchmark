!================================================================
! RAMSES initial conditions for the GLM/CT comparison suite (sherlock_mhd).
! Reads the exact Float32 initial state written by tools/write_spec.jl --state (NDIM=2: --state --plane):
!   $ABEL_IC_PREFIX.state : N x N x Nz x 9 cell state (rho, m, E, Bcc, psi), x fastest; Nz = N (3D) or 1 (2D)
!   $ABEL_IC_PREFIX.face1..3 : lower-face B (N x N x Nz each), x fastest; $ABEL_IC_N = N
! Requires a uniform periodic grid on [0,1]^NDIM with levelmin = levelmax = log2(N). RAMSES also
! calls condinit on coarser levels while building the tree; those get conservative block
! averages of the fine data (and are replaced by restriction of the fine level anyway).
! In 2D (z-invariant problems) Bz has no z faces: both "faces" carry the cell value, which is
! what the z-invariant lower z face holds.
! Build: make SOLVER=mhd NDIM=3 (or NDIM=2) PATCH=<this directory>.
!================================================================
module abel_ic
  use amr_parameters, only: dp
  implicit none
  integer, save :: nic = 0, nzic = 0
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
#if NDIM == 3
    nzic = n
#else
    nzic = 1
#endif
    allocate(st(n, n, nzic, 9), bf(n, n, nzic, 3))
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
  integer :: i, n, r, rz, ix, iy, iz, i0, j0, k0, i1, j1, k1, v
#if NDIM == 1
  write(*,*) 'abel condinit requires NDIM=2 or 3'
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
     ix = int(x(i,1)/dx); iy = int(x(i,2)/dx)
#if NDIM == 3
     iz = int(x(i,3)/dx); rz = r
#else
     iz = 0; rz = 1
#endif
     i0 = ix*r + 1; j0 = iy*r + 1; k0 = iz*rz + 1; i1 = i0 + r - 1; j1 = j0 + r - 1; k1 = k0 + rz - 1
     do v = 1, 5
        u(i, v) = sum(real(st(i0:i1, j0:j1, k0:k1, v), dp))/(r*r*rz)
     end do
     ! Lower faces of this (coarse) cell, and upper faces = lower faces of the next cell.
     u(i, 6) = sum(real(bf(i0, j0:j1, k0:k1, 1), dp))/(r*rz)
     u(i, 7) = sum(real(bf(i0:i1, j0, k0:k1, 2), dp))/(r*rz)
     u(i, 8) = sum(real(bf(i0:i1, j0:j1, k0, 3), dp))/(r*r)
     u(i, nvar+1) = sum(real(bf(modulo(i1, nic) + 1, j0:j1, k0:k1, 1), dp))/(r*rz)
     u(i, nvar+2) = sum(real(bf(i0:i1, modulo(j1, nic) + 1, k0:k1, 2), dp))/(r*rz)
#if NDIM == 3
     u(i, nvar+3) = sum(real(bf(i0:i1, j0:j1, modulo(k1, nic) + 1, 3), dp))/(r*r)
#else
     u(i, nvar+3) = u(i, 8)
#endif
  end do
end subroutine condinit
