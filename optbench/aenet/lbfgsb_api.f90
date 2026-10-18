! C entry points for the L-BFGS-B 3.0 setulb shipped with aenet.
! The Fortran state stays in this module. Python passes f, g, and x only.
module aenet_lbfgsb_api
  use iso_c_binding
  implicit none

  integer, parameter :: dp = c_double
  integer :: n = 0
  integer :: m = 20
  real(dp) :: f, factr, pgtol
  integer :: iprint
  character(len=60) :: task, csave
  logical :: lsave(4)
  integer :: isave(44)
  real(dp) :: dsave(29)
  real(dp), allocatable :: x(:), lbound(:), ubound(:), g(:), wa(:)
  integer, allocatable :: nbd(:), iwa(:)

  interface
    subroutine setulb(n, m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, &
                      task, iprint, csave, lsave, isave, dsave)
      import :: dp
      integer :: n, m, iprint
      integer :: nbd(*), iwa(*), isave(44)
      real(dp) :: f, factr, pgtol
      real(dp) :: x(*), l(*), u(*), g(*), wa(*), dsave(29)
      character(len=60) :: task, csave
      logical :: lsave(4)
    end subroutine setulb
  end interface

contains

  function classify(text) result(code)
    character(len=*), intent(in) :: text
    integer :: code
    if (text(1:2) == 'FG') then
      code = 1
    else if (text(1:2) == 'NE') then
      code = 2
    else if (text(1:2) == 'CO') then
      code = 3
    else if (text(1:2) == 'ER' .or. text(1:2) == 'AB') then
      code = 4
    else if (text(1:2) == 'WA') then
      code = 5
    else if (text(1:2) == 'ST') then
      code = 0
    else
      code = 6
    end if
  end function classify

  subroutine call_setulb()
    call setulb(n, m, x, lbound, ubound, nbd, f, g, factr, pgtol, wa, iwa, &
                task, iprint, csave, lsave, isave, dsave)
  end subroutine call_setulb

  function lbfgsb_start(n_in, factr_in, pgtol_in, x_in) result(code) &
      bind(C, name="lbfgsb_start")
    integer(c_int), value :: n_in
    real(c_double), value :: factr_in, pgtol_in
    real(c_double), intent(in) :: x_in(*)
    integer(c_int) :: code
    integer :: nwa

    if (allocated(x)) deallocate(x, lbound, ubound, g, nbd, wa, iwa)
    n = int(n_in)
    m = 20
    factr = factr_in
    pgtol = pgtol_in
    iprint = -1
    nwa = (2*m + 5)*n + 11*m*m + 8*m
    allocate(x(n), lbound(n), ubound(n), g(n), nbd(n), wa(nwa), iwa(3*n))
    x(1:n) = x_in(1:n)
    lbound = 0.0_dp
    ubound = 0.0_dp
    g = 0.0_dp
    nbd = 0
    wa = 0.0_dp
    iwa = 0
    f = 0.0_dp
    task = 'START'
    csave = ' '
    lsave = .false.
    isave = 0
    dsave = 0.0_dp
    call call_setulb()
    code = classify(task)
  end function lbfgsb_start

  function lbfgsb_step(f_in, g_in, x_out) result(code) bind(C, name="lbfgsb_step")
    real(c_double), value :: f_in
    real(c_double), intent(in) :: g_in(*)
    real(c_double), intent(out) :: x_out(*)
    integer(c_int) :: code

    f = f_in
    g(1:n) = g_in(1:n)
    call call_setulb()
    ! aenet calls setulb again on NEW_X without a new evaluation.
    if (task(1:2) == 'NE') call call_setulb()
    code = classify(task)
    x_out(1:n) = x(1:n)
  end function lbfgsb_step

  subroutine lbfgsb_task(buf) bind(C, name="lbfgsb_task")
    character(kind=c_char), intent(out) :: buf(61)
    integer :: i

    do i = 1, 60
      if (i <= len_trim(task)) then
        buf(i) = task(i:i)
      else
        buf(i) = c_null_char
        return
      end if
    end do
    buf(61) = c_null_char
  end subroutine lbfgsb_task

end module aenet_lbfgsb_api
