! vtstcode 6.6.1 neb.F and lbfgs.F are compiled unmodified.
!   mpifort -ffree-form -cpp -DMPI -O2 -I. stubs.f90 neb.F lbfgs.F driver.f90 -o band
!   mpirun -np 8 ./band geometry
! Stock INCAR defaults are kept. IMAGES is 8 because that is the problem.
! The stop is the archive Fmag 0.01, and a path stops at 10000 Morse evaluations.
program vtst_band
  use prec
  use base
  use poscar
  use lattice
  use main_mpi
  use geometry_store
  use neb, only: neb_init, neb_step
  use lbfgs, only: lbfgs_init, lbfgs_step
  use mpi
  implicit none
  integer, parameter :: images_n = 8
  integer, parameter :: cap = 10000
  real(q), parameter :: fmax = 0.01_q
  real(q), parameter :: morse_d = 0.7102_q
  real(q), parameter :: morse_a = 1.6047_q
  real(q), parameter :: morse_r0 = 2.897_q
  real(q), parameter :: morse_rc = 9.5_q
  type(in_struct) :: io
  type(type_info) :: tinfo
  real(q) :: chain_jacob
  real(q), allocatable :: pos(:, :)
  real(q), allocatable :: force(:, :)
  real(q) :: stress(3, 3)
  real(q) :: latt_a(3, 3), latt_b(3, 3)
  real(q) :: energy, fneb, fglobal
  real(q) :: cart(3, max_ions)
  logical :: optflag, ml_turnoff
  integer :: rank, nproc, ierr, node, i, iter, calls
  integer :: nions, image, count_images
  character(len=512) :: path
  real(q) :: band(3, max_ions, 0:images_n + 1)

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nproc, ierr)
  if (nproc /= images_n) then
    if (rank == 0) write(0, *) 'need 8 ranks'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if
  node = rank + 1
  images = images_n
  comm_chain%node_me = node
  comm_chain%ncpu = nproc

  call get_command_argument(1, path)
  open(unit=10, file=trim(path), status='old', action='read')
  read(10, *) nions
  read(10, *) box(1), box(2), box(3)
  read(10, *) count_images
  if (count_images /= images_n) then
    if (rank == 0) write(0, *) 'image count is not 8'
    call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
  end if
  stored_nions = nions
  do image = 0, images_n + 1
    do i = 1, nions
      if (image == 0) then
        read(10, *) band(1, i, image), band(2, i, image), band(3, i, image), frozen(i)
      else
        read(10, *) band(1, i, image), band(2, i, image), band(3, i, image)
      end if
    end do
  end do
  close(10)
  do image = 0, images_n + 1
    do i = 1, nions
      band(1, i, image) = band(1, i, image) / box(1)
      band(2, i, image) = band(2, i, image) / box(2)
      band(3, i, image) = band(3, i, image) / box(3)
    end do
  end do
  reactant(:, 1:nions) = band(:, 1:nions, 0)
  product(:, 1:nions) = band(:, 1:nions, images_n + 1)

  io%iu0 = -1
  io%iu5 = -1
  io%iu6 = -1
  io%lopen = .false.
  tinfo%nions = nions
  latt_a = 0.0_q
  latt_b = 0.0_q
  latt_a(1, 1) = box(1)
  latt_a(2, 2) = box(2)
  latt_a(3, 3) = box(3)
  latt_b(1, 1) = 1.0_q / box(1)
  latt_b(2, 2) = 1.0_q / box(2)
  latt_b(3, 3) = 1.0_q / box(3)
  stress = 0.0_q

  call neb_init(tinfo, io, chain_jacob)
  call lbfgs_init(tinfo, io)

  allocate(pos(3, nions), force(3, nions))
  pos(:, 1:nions) = band(:, 1:nions, node)
  calls = 0
  iter = 0
  fglobal = 1.0e30_q
  do while (calls + images_n <= cap)
    call direct_to_cartesian(pos, nions, cart)
    call morse(cart, nions, energy, force)
    calls = calls + images_n
    call neb_step(optflag, pos, energy, force, stress, latt_a, latt_b)
    fneb = sqrt(sum(force(:, 1:nions) ** 2))
    call MPI_Allreduce(fneb, fglobal, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
    iter = iter + 1
    if (fglobal < fmax) exit
    ml_turnoff = .true.
    call lbfgs_step(optflag, pos, energy, force, latt_a, latt_b, ml_turnoff)
    call hold_frozen(pos, nions)
  end do
  if (rank == 0) then
    write(*, '(A,1X,I8,1X,I8,1X,ES16.8,1X,L1)') 'RESULT', calls, iter, fglobal, fglobal < fmax
  end if
  call MPI_Finalize(ierr)
  stop

contains
  subroutine direct_to_cartesian(xyz, n, out)
    integer, intent(in) :: n
    real(q), intent(in) :: xyz(3, nions)
    real(q), intent(out) :: out(3, max_ions)
    integer :: j, k
    out = 0.0_q
    do j = 1, n
      do k = 1, 3
        out(k, j) = xyz(k, j) * box(k)
      end do
    end do
  end subroutine direct_to_cartesian

  subroutine hold_frozen(xyz, n)
    integer, intent(in) :: n
    real(q), intent(inout) :: xyz(3, nions)
    integer :: j, k
    do j = 1, n
      if (frozen(j) /= 0) then
        do k = 1, 3
          xyz(k, j) = reactant(k, j)
        end do
      end if
    end do
  end subroutine hold_frozen

  subroutine morse(cart, n, energy, f)
    integer, intent(in) :: n
    real(q), intent(in) :: cart(3, max_ions)
    real(q), intent(out) :: energy
    real(q), intent(out) :: f(3, nions)
    integer :: i, j, k
    real(q) :: diff(3), r, x, xc, raw, shift, dedr, unit(3)
    real(q) :: rc
    energy = 0.0_q
    f(:, 1:n) = 0.0_q
    rc = morse_rc - morse_r0
    xc = exp(-morse_a * rc)
    shift = morse_d * (xc * xc - 2.0_q * xc)
    do i = 1, n - 1
      do j = i + 1, n
        do k = 1, 3
          diff(k) = cart(k, i) - cart(k, j)
          diff(k) = diff(k) - box(k) * real(nint(diff(k) / box(k)), q)
        end do
        r = sqrt(sum(diff * diff))
        if (r <= 1.0e-12_q .or. r >= morse_rc) cycle
        x = exp(-morse_a * (r - morse_r0))
        raw = morse_d * (x * x - 2.0_q * x)
        energy = energy + (raw - shift)
        dedr = 2.0_q * morse_a * morse_d * x * (1.0_q - x)
        unit = diff / r
        f(:, i) = f(:, i) - dedr * unit
        f(:, j) = f(:, j) + dedr * unit
      end do
    end do
    do i = 1, n
      if (frozen(i) /= 0) f(:, i) = 0.0_q
    end do
  end subroutine morse
end program vtst_band
