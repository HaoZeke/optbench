module prec
  integer, parameter :: q = selected_real_kind(14, 80)
end module prec

module base
  type in_struct
    integer :: iu0, iu5, iu6
    logical :: lopen
  end type in_struct
end module base

module poscar
  use prec
  integer, parameter :: max_ions = 512
  type type_info
    integer :: nions
  end type type_info
  type dynamics
    real(q), pointer :: posion(:, :)
  end type dynamics
  type nh_chains
    integer :: dummy
  end type nh_chains
end module poscar

module lattice
  use prec
  type latt
    real(q) :: A(3, 3)
    real(q) :: B(3, 3)
    real(q) :: omega
  end type latt
contains
  subroutine dirkar(nions, pos, a)
    integer, intent(in) :: nions
    real(q), intent(inout) :: pos(3, nions)
    real(q), intent(in) :: a(3, 3)
    integer :: i
    real(q) :: x, y, z
    do i = 1, nions
      x = pos(1, i) * a(1, 1) + pos(2, i) * a(2, 1) + pos(3, i) * a(3, 1)
      y = pos(1, i) * a(1, 2) + pos(2, i) * a(2, 2) + pos(3, i) * a(3, 2)
      z = pos(1, i) * a(1, 3) + pos(2, i) * a(2, 3) + pos(3, i) * a(3, 3)
      pos(1, i) = x
      pos(2, i) = y
      pos(3, i) = z
    end do
  end subroutine dirkar

  subroutine kardir(nions, pos, b)
    integer, intent(in) :: nions
    real(q), intent(inout) :: pos(3, nions)
    real(q), intent(in) :: b(3, 3)
    integer :: i
    real(q) :: x, y, z
    do i = 1, nions
      x = pos(1, i) * b(1, 1) + pos(2, i) * b(2, 1) + pos(3, i) * b(3, 1)
      y = pos(1, i) * b(1, 2) + pos(2, i) * b(2, 2) + pos(3, i) * b(3, 2)
      z = pos(1, i) * b(1, 3) + pos(2, i) * b(2, 3) + pos(3, i) * b(3, 3)
      pos(1, i) = x
      pos(2, i) = y
      pos(3, i) = z
    end do
  end subroutine kardir
end module lattice

module main_mpi
  use prec
  type comm_type
    integer :: node_me
    integer :: ncpu
  end type comm_type
  type(comm_type) :: comm_chain
  integer :: images
contains
  subroutine m_sum_d(comm, vec, n)
    use mpi
    type(comm_type), intent(in) :: comm
    integer, intent(in) :: n
    real(q), intent(inout) :: vec(n)
    real(q), allocatable :: total(:)
    integer :: ierr
    allocate(total(n))
    call MPI_Allreduce(vec, total, n, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    vec = total
  end subroutine m_sum_d
end module main_mpi

module geometry_store
  use prec
  use poscar
  integer :: current_dir = -1
  integer :: stored_nions = 0
  real(q) :: box(3)
  real(q) :: reactant(3, max_ions)
  real(q) :: product(3, max_ions)
  integer :: frozen(max_ions)
  real(q), target :: endpoint(3, max_ions)
end module geometry_store

subroutine make_dir_app(idir)
  use geometry_store
  integer, intent(in) :: idir
  current_dir = idir
end subroutine make_dir_app

subroutine rd_poscar_head(latt_cur, t_i, niond, nionpd, ntypd, ntyppd, iu0, iu6)
  use lattice
  use poscar
  type(latt), intent(inout) :: latt_cur
  type(type_info), intent(inout) :: t_i
  integer, intent(inout) :: niond, nionpd, ntypd, ntyppd, iu0, iu6
  latt_cur%a = 0.0_q
  latt_cur%b = 0.0_q
  t_i%nions = 0
  niond = 0
  nionpd = 0
  ntypd = 0
  ntyppd = 0
  iu0 = iu0
  iu6 = iu6
end subroutine rd_poscar_head

subroutine rd_poscar(latt_cur, t_i, dyn, nhc, niond, nionpd, ntypd, ntyppd, iu0, iu6)
  use lattice
  use poscar
  use geometry_store
  use main_mpi, only: images
  type(latt), intent(inout) :: latt_cur
  type(type_info), intent(inout) :: t_i
  type(dynamics), intent(inout) :: dyn
  type(nh_chains), intent(inout) :: nhc
  integer, intent(inout) :: niond, nionpd, ntypd, ntyppd, iu0, iu6
  integer :: i
  latt_cur%a = 0.0_q
  latt_cur%b = 0.0_q
  latt_cur%a(1, 1) = box(1)
  latt_cur%a(2, 2) = box(2)
  latt_cur%a(3, 3) = box(3)
  latt_cur%b(1, 1) = 1.0_q / box(1)
  latt_cur%b(2, 2) = 1.0_q / box(2)
  latt_cur%b(3, 3) = 1.0_q / box(3)
  latt_cur%omega = box(1) * box(2) * box(3)
  t_i%nions = stored_nions
  endpoint(:, 1:stored_nions) = 0.0_q
  if (current_dir == 0) then
    endpoint(:, 1:stored_nions) = reactant(:, 1:stored_nions)
  else if (current_dir == images + 1) then
    endpoint(:, 1:stored_nions) = product(:, 1:stored_nions)
  end if
  dyn%posion => endpoint(:, 1:stored_nions)
  nhc%dummy = 0
  niond = stored_nions
  nionpd = stored_nions
  ntypd = 1
  ntyppd = 1
  iu0 = iu0
  iu6 = iu6
  i = stored_nions
end subroutine rd_poscar

subroutine expro(a, b, c)
  use prec
  real(q), intent(out) :: a(3)
  real(q), intent(in) :: b(3), c(3)
  a(1) = b(2) * c(3) - b(3) * c(2)
  a(2) = b(3) * c(1) - b(1) * c(3)
  a(3) = b(1) * c(2) - b(2) * c(1)
end subroutine expro

subroutine rdatab(lopen, fname, iu, key, eq, comment, semi, kind, ival, rval, cval, lval, charac, n, nmax, ierr)
  use prec
  logical, intent(in) :: lopen
  character*(*), intent(in) :: fname, key, eq, comment, semi, kind, charac
  integer, intent(in) :: iu, nmax
  integer, intent(inout) :: ival, n, ierr
  real(q), intent(inout) :: rval
  complex(q), intent(inout) :: cval
  logical, intent(inout) :: lval
  if (key == 'IMAGES') then
    ival = 8
    n = 1
    ierr = 0
  else
    ierr = 3
    n = 0
  end if
  if (.false.) then
    print *, lopen, fname, iu, eq, comment, semi, kind, rval, cval, lval, charac, nmax
  end if
end subroutine rdatab
