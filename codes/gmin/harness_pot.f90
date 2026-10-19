! OptBench callback. When registered, POTENTIAL evaluates this instead of
! the native potential and still increments NPCALL. A null pointer leaves
! the native Lennard-Jones, binary Lennard-Jones, and TIP4P paths in place.
module gmin_harness
   use iso_c_binding
   use prec, only: real64
   implicit none

   abstract interface
      integer(c_int) function harness_iface(n, x, energy, grad) bind(c)
         import :: c_int, c_double
         integer(c_int), value :: n
         real(c_double), intent(in) :: x(n)
         real(c_double), intent(out) :: energy
         real(c_double), intent(out) :: grad(n)
      end function harness_iface
   end interface

   procedure(harness_iface), pointer :: eval_ptr => null()

contains

   logical function gmin_harness_on()
      gmin_harness_on = associated(eval_ptr)
   end function gmin_harness_on

end module gmin_harness

subroutine gmin_set_harness(ptr) bind(c, name="gmin_set_harness")
   use iso_c_binding
   use gmin_harness, only: eval_ptr
   type(c_funptr), value :: ptr
   if (c_associated(ptr)) then
      call c_f_procpointer(ptr, eval_ptr)
   else
      eval_ptr => null()
   end if
end subroutine gmin_set_harness

subroutine gmin_harness_eval(x, grad, ereal, n)
   use prec, only: real64
   use gmin_harness, only: eval_ptr
   implicit none
   integer, intent(in) :: n
   real(real64), intent(in) :: x(*)
   real(real64), intent(out) :: grad(*)
   real(real64), intent(out) :: ereal
   integer :: status
   status = eval_ptr(n, x(1:n), ereal, grad(1:n))
   ! A nonzero status is a harness stop. A huge energy is not a minimum.
   if (status /= 0) then
      ereal = 1.0d30
      grad(1:n) = 0.0d0
   end if
end subroutine gmin_harness_eval

subroutine gmin_trace(x, grad, ereal, n, gradt)
   use prec, only: real64
   implicit none
   integer, intent(in) :: n
   real(real64), intent(in) :: x(*)
   real(real64), intent(in) :: grad(*)
   real(real64), intent(in) :: ereal
   logical, intent(in) :: gradt
   character(len=1024) :: path
   integer :: length, status, ios
   integer, save :: unit = -1
   logical, save :: ready = .false.
   logical, save :: enabled = .false.
   if (.not. ready) then
      call get_environment_variable("OPTBENCH_GMIN_TRACE", path, length, status)
      enabled = status == 0 .and. length > 0
      if (enabled) then
         open(newunit=unit, file=path(1:length), status="replace", access="stream", form="unformatted", iostat=ios)
         if (ios /= 0) enabled = .false.
      end if
      ready = .true.
   end if
   if (.not. enabled) return
   ! The last integer is 1 when this entry computed a gradient. An energy-only
   ! entry, such as the final SORT, still counts and still carries the energy.
   write(unit) n, ereal, x(1:n), grad(1:n), merge(1, 0, gradt)
   flush(unit)
end subroutine gmin_trace
