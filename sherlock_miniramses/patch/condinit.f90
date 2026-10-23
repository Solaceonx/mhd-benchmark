!================================================================
! mini-ramses initial conditions for the comparison suite: the primitive state of each cell from the sweep's
! initial data (module abel_ic, patch/abel_ic.f90). See abel_ic.f90 for the files read.
!================================================================
subroutine condinit(r,g,x,q,dx,nn)
  use amr_parameters, only: ndim, nvector
  use hydro_parameters, only: nvar
  use amr_commons, only: run_t, global_t
  use abel_ic
  implicit none
  type(run_t)::r
  type(global_t)::g
  integer ::nn                            ! Number of cells
  real(kind=8)::dx                        ! Cell size
#ifdef MHD
  real(kind=8),dimension(1:nvector,1:nvar+3-ndim)::q ! Primitive variables
#else
  real(kind=8),dimension(1:nvector,1:nvar)::q ! Primitive variables
#endif
  real(kind=8),dimension(1:nvector,1:ndim)::x ! Cell centre positions
  integer :: i
  real(kind=8) :: u(8), rho, v(3)
  do i = 1, nn
     call cell_state(x(i,:), dx, ndim, u)
     rho = u(1)
     v = u(2:4)/rho
     q(i,1) = rho
     q(i,2:4) = v
     q(i,5) = (r%gamma - 1d0)*(u(5) - 0.5d0*rho*sum(v*v) - 0.5d0*sum(u(6:8)*u(6:8)))
     if (nvar > 5) q(i,6:nvar) = 0d0
#ifdef MHD
#if NDIM == 2
     q(i,nvar+1) = u(8)
#endif
#endif
  end do
end subroutine condinit
