!================================================================
! mini-ramses initial conditions for the comparison suite (sherlock_miniramses).
! Reads the exact Float32 initial state the sweep writes for RAMSES (tools/write_spec.jl --state; NDIM=2: --plane):
!   $ABEL_IC_PREFIX.state : N x N x Nz x 9 cell state (rho, m, E, Bcc, psi), x fastest; Nz = N (3D) or 1 (a plane,
!                           which a 3D build repeats through z: the GPU build of mini-ramses-ism is 3D only)
!   $ABEL_IC_PREFIX.face1..3 : lower-face B (N x N x Nz each), x fastest; $ABEL_IC_N = N
! This module goes at the top of input_hydro_condinit.f90 (prepare_patch.py): the Makefile compiles that file before
! condinit.f90, which uses it. condinit returns primitive variables (mini-ramses' convention); abel_faces (the end of
! input_hydro_condinit.f90) sets the face-centred field from the face files, in place of the vector potential. A cell
! coarser than the data (the tree below levelmin) gets the conservative block average of the fine cells it covers.
! In 2D, Bz is a cell value (q(:,nvar+1)), as upstream.
!================================================================
module abel_ic
  implicit none
  integer, save :: nic = 0, nzic = 0
  real(kind=4), allocatable, save :: st(:,:,:,:), bf(:,:,:,:)
contains
  subroutine load_ic()
    character(len=4096) :: prefix, nstr
    integer :: unit, d, status, n
    integer(kind=8) :: nbytes
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
    ! A plane (N x N x 1, the sweep's --plane data) is the same at every z: a 3D run of a planar test reads it once and
    ! wrap() repeats it through z. Otherwise N x N x N in 3D.
    inquire(file=trim(prefix)//'.state', size=nbytes)
    nzic = 1
#if NDIM == 3
    if (nbytes /= int(n, 8)*n*9*4) nzic = n
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

  pure integer function wrap(i, n)
    integer, intent(in) :: i, n
    wrap = modulo(i - 1, n) + 1
  end function wrap

  ! The fine cells covered by a cell of size dx centred at xc (box [0,1]): first index per direction and count r.
  subroutine block(xc, dx, ndim, i0, r)
    integer, intent(in) :: ndim
    real(kind=8), intent(in) :: xc(ndim), dx
    integer, intent(out) :: i0(3), r
    integer :: d
    call load_ic()
    r = max(1, nint(dx*nic))
    i0 = 1
    do d = 1, ndim
       i0(d) = wrap(nint((xc(d) - 0.5d0*dx)*nic) + 1, nic)
    end do
  end subroutine block

  ! Conservative average of the state over a cell of size dx centred at xc: (rho, m, E, Bcc).
  subroutine cell_state(xc, dx, ndim, u)
    integer, intent(in) :: ndim
    real(kind=8), intent(in) :: xc(ndim), dx
    real(kind=8), intent(out) :: u(8)
    integer :: i0(3), r, a, b, c, rz
    call block(xc, dx, ndim, i0, r)
    rz = 1
    if (ndim == 3) rz = r
    u = 0d0
    do c = 0, rz - 1
       do b = 0, r - 1
          do a = 0, r - 1
             u = u + st(wrap(i0(1) + a, nic), wrap(i0(2) + b, nic), wrap(i0(3) + c, nzic), 1:8)
          end do
       end do
    end do
    u = u/dble(r*r*rz)
  end subroutine cell_state

  ! Field on the lower (side = 0) or upper (side = 1) face along idim of a cell of size dx centred at xc: the average
  ! of the fine faces that make it up.
  real(kind=8) function face_value(xc, dx, ndim, idim, side)
    integer, intent(in) :: ndim, idim, side
    real(kind=8), intent(in) :: xc(ndim), dx
    integer :: i0(3), r, a, b, c, rz, idx(3), na(3)
    call block(xc, dx, ndim, i0, r)
    rz = 1
    if (ndim == 3) rz = r
    na = (/ r, r, rz /)
    na(idim) = 1
    face_value = 0d0
    do c = 0, na(3) - 1
       do b = 0, na(2) - 1
          do a = 0, na(1) - 1
             idx = i0 + (/ a, b, c /)
             idx(idim) = i0(idim) + side*r
             face_value = face_value + bf(wrap(idx(1), nic), wrap(idx(2), nic), wrap(idx(3), nzic), idim)
          end do
       end do
    end do
    face_value = face_value/dble(na(1)*na(2)*na(3))
  end function face_value
end module abel_ic
