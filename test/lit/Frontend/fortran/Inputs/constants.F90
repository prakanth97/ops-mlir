module app_constants
  implicit none
  integer :: imax, jmax, nsteps
  real(8) :: coef(3)
  real(8), parameter :: pi = 2.0_8*asin(1.0_8)
  real(8), parameter :: w(3) = [0.25_8, 0.5_8, 0.25_8]
end module app_constants
