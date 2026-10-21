! USERPOT client for the optbench potential server.
!
! OPTIM calls userpot_potentialhess with six arguments. Energy and
! gradient travel together, so a gradient-free request still asks for
! calculate. The server cache collapses the repeat at the same geometry.
! A Hessian request has nowhere to go: this signature does not return one.

module optbench_pot
  use iso_c_binding, only: c_char, c_int, c_int16_t, c_loc, c_long, c_null_char, c_ptr, c_size_t
  implicit none
  integer, parameter :: AF_INET = 2
  integer, parameter :: SOCK_STREAM = 1
  integer, save :: port = -1
  integer, save :: atomic_number = 1
  integer, save :: known_atoms = 0
  double precision, save :: box(9) = 0.0d0
  logical, save :: ready = .false.
  character(len=64), save :: host = "127.0.0.1"

  interface
    function c_socket(domain, typ, protocol) bind(c, name="socket")
      import :: c_int
      integer(c_int) :: c_socket
      integer(c_int), value :: domain, typ, protocol
    end function c_socket

    function c_connect(fd, addr, addrlen) bind(c, name="connect")
      import :: c_int, c_ptr
      integer(c_int) :: c_connect
      integer(c_int), value :: fd
      type(c_ptr), value :: addr
      integer(c_int), value :: addrlen
    end function c_connect

    function c_send(fd, buf, n, flags) bind(c, name="send")
      import :: c_char, c_int, c_long, c_size_t
      integer(c_long) :: c_send
      integer(c_int), value :: fd, flags
      character(kind=c_char), intent(in) :: buf(*)
      integer(c_size_t), value :: n
    end function c_send

    function c_recv(fd, buf, n, flags) bind(c, name="recv")
      import :: c_char, c_int, c_long, c_size_t
      integer(c_long) :: c_recv
      integer(c_int), value :: fd, flags
      character(kind=c_char), intent(out) :: buf(*)
      integer(c_size_t), value :: n
    end function c_recv

    function c_close(fd) bind(c, name="close")
      import :: c_int
      integer(c_int) :: c_close
      integer(c_int), value :: fd
    end function c_close

    subroutine c_exit(code) bind(c, name="exit")
      import :: c_int
      integer(c_int), value :: code
    end subroutine c_exit
  end interface

contains

  subroutine die(code, message)
    integer, intent(in) :: code
    character(len=*), intent(in) :: message
    write(*, "(A)") "userpot> " // trim(message)
    flush(6)
    call c_exit(int(code, c_int))
  end subroutine die

  subroutine load_env
    character(len=256) :: spec, box_text, z_text
    integer :: length, status, colon, ios
    if (ready) return
    call get_environment_variable("OPTBENCH_POTSERVER", spec, length, status)
    if (status /= 0 .or. length <= 0) call die(4, "OPTBENCH_POTSERVER is unset")
    spec = spec(1:length)
    colon = index(spec, ":")
    if (colon <= 1) call die(4, "OPTBENCH_POTSERVER needs host:port")
    host = spec(1:colon - 1)
    read(spec(colon + 1:), *, iostat=ios) port
    if (ios /= 0 .or. port <= 0) call die(4, "OPTBENCH_POTSERVER has no port")
    call get_environment_variable("OPTBENCH_BOX", box_text, length, status)
    if (status == 0 .and. length > 0) then
      read(box_text(1:length), *, iostat=ios) box
      if (ios /= 0) call die(4, "OPTBENCH_BOX needs nine floats")
    end if
    call get_environment_variable("OPTBENCH_Z", z_text, length, status)
    if (status == 0 .and. length > 0) then
      read(z_text(1:length), *, iostat=ios) atomic_number
      if (ios /= 0) call die(4, "OPTBENCH_Z is not an integer")
    end if
    ready = .true.
  end subroutine load_env

  subroutine parse_ipv4(text, octets, ok)
    character(len=*), intent(in) :: text
    integer, intent(out) :: octets(4)
    logical, intent(out) :: ok
    integer :: count, value, i
    character(len=1) :: piece
    count = 0
    value = -1
    ok = .false.
    octets = 0
    do i = 1, len_trim(text)
      piece = text(i:i)
      if (piece >= "0" .and. piece <= "9") then
        if (value < 0) value = 0
        value = value * 10 + iachar(piece) - iachar("0")
        if (value > 255) return
      else if (piece == ".") then
        if (value < 0 .or. count == 4) return
        count = count + 1
        octets(count) = value
        value = -1
      else
        return
      end if
    end do
    if (value < 0 .or. count /= 3) return
    octets(4) = value
    ok = .true.
  end subroutine parse_ipv4

  subroutine exchange(request, reply, nreply)
    character(len=*), intent(in) :: request
    character(kind=c_char), intent(out) :: reply(*)
    integer, intent(out) :: nreply
    character(kind=c_char), allocatable, target :: raw(:)
    character(kind=c_char), allocatable :: outbound(:)
    integer(c_int16_t) :: family
    character(kind=c_char) :: family_bytes(2)
    integer :: octets(4), fd, i, offset
    integer(c_long) :: moved
    logical :: ok
    call parse_ipv4(trim(host), octets, ok)
    if (.not. ok) call die(4, "OPTBENCH_POTSERVER host is not an IPv4 address")
    allocate(raw(16))
    raw = c_null_char
    family = 2_c_int16_t
    family_bytes = transfer(family, family_bytes)
    raw(1:2) = family_bytes
    raw(3) = achar(port / 256)
    raw(4) = achar(mod(port, 256))
    raw(5) = achar(octets(1))
    raw(6) = achar(octets(2))
    raw(7) = achar(octets(3))
    raw(8) = achar(octets(4))
    fd = c_socket(AF_INET, SOCK_STREAM, 0)
    if (fd < 0) call die(4, "socket failed")
    if (c_connect(fd, c_loc(raw), 16) /= 0) call die(4, "connect failed")
    allocate(outbound(len(request)))
    do i = 1, len(request)
      outbound(i) = request(i:i)
    end do
    offset = 0
    do while (offset < len(request))
      moved = c_send(fd, outbound(offset + 1), int(len(request) - offset, c_size_t), 0)
      if (moved <= 0) call die(4, "send failed")
      offset = offset + int(moved)
    end do
    nreply = 0
    do
      moved = c_recv(fd, reply(nreply + 1), int(4000000 - nreply, c_size_t), 0)
      if (moved == 0) exit
      if (moved < 0) call die(4, "recv failed")
      nreply = nreply + int(moved)
      if (nreply >= 4000000) call die(4, "potential reply is too long")
    end do
    i = c_close(fd)
  end subroutine exchange

  subroutine take_line(text, cursor, line)
    character(len=*), intent(in) :: text
    integer, intent(inout) :: cursor
    character(len=*), intent(out) :: line
    integer :: start, limit, mark
    line = ""
    limit = len(text)
    if (cursor > limit) return
    start = cursor
    mark = cursor
    do while (mark <= limit)
      if (text(mark:mark) == achar(10)) exit
      mark = mark + 1
    end do
    if (mark > start .and. text(mark - 1:mark - 1) == achar(13)) then
      line = text(start:mark - 2)
    else if (mark > start) then
      line = text(start:mark - 1)
    end if
    cursor = mark + 1
  end subroutine take_line

  subroutine read_reply(text, natoms, energy, gradient)
    character(len=*), intent(in) :: text
    integer, intent(in) :: natoms
    double precision, intent(out) :: energy
    double precision, intent(out) :: gradient(*)
    character(len=512) :: line
    integer :: cursor, ios, atom, count
    double precision :: fx, fy, fz
    cursor = 1
    call take_line(text, cursor, line)
    if (trim(line) == "error") then
      call take_line(text, cursor, line)
      if (index(line, "ForceCallCap") > 0) call die(3, trim(line))
      if (index(line, "WindowMet") > 0) call die(5, trim(line))
      call die(4, trim(line))
    end if
    if (trim(line) /= "ok") call die(4, "potential reply is not ok")
    call take_line(text, cursor, line)
    call take_line(text, cursor, line)
    call take_line(text, cursor, line)
    read(line, *, iostat=ios) energy
    if (ios /= 0) call die(4, "potential energy did not parse")
    do atom = 1, natoms
      call take_line(text, cursor, line)
      read(line, *, iostat=ios) fx, fy, fz
      if (ios /= 0) call die(4, "potential gradient did not parse")
      count = 3 * (atom - 1)
      gradient(count + 1) = fx
      gradient(count + 2) = fy
      gradient(count + 3) = fz
    end do
  end subroutine read_reply

  subroutine calculate(dof, x, gradient, energy)
    integer, intent(in) :: dof
    double precision, intent(in) :: x(*)
    double precision, intent(out) :: gradient(*)
    double precision, intent(out) :: energy
    character(len=:), allocatable :: request
    character(len=256) :: row
    character(kind=c_char), allocatable :: reply(:)
    character(len=:), allocatable :: text
    integer :: natoms, atom, nreply, i
    natoms = dof / 3
    if (natoms < 1 .or. mod(dof, 3) /= 0) call die(4, "degrees of freedom are not 3N")
    write(row, "(I0)") natoms
    request = "calculate" // achar(10) // trim(row) // achar(10)
    write(row, "(9(ES24.16E3,1X))") box
    request = request // trim(adjustl(row)) // achar(10)
    do atom = 1, natoms
      write(row, "(I0,1X,3(ES24.16E3,1X))") atomic_number, x(3 * atom - 2), x(3 * atom - 1), x(3 * atom)
      request = request // trim(adjustl(row)) // achar(10)
    end do
    allocate(reply(4000000))
    call exchange(request, reply, nreply)
    allocate(character(len=nreply) :: text)
    do i = 1, nreply
      text(i:i) = reply(i)
    end do
    call read_reply(text, natoms, energy, gradient)
  end subroutine calculate

end module optbench_pot

subroutine userpot_init
  use optbench_pot, only: load_env
  implicit none
  call load_env
end subroutine userpot_init

subroutine userpot_get_natoms(num_atoms)
  use optbench_pot, only: known_atoms
  implicit none
  integer, intent(out) :: num_atoms
  character(len=512) :: line, label
  double precision :: x, y, z
  integer :: ios, parsed
  num_atoms = 0
  do
    read(5, "(A)", iostat=ios) line
    if (ios /= 0) exit
    if (len_trim(line) == 0) cycle
    parsed = 0
    read(line, *, iostat=parsed) label, x, y, z
    if (parsed /= 0) then
      write(*, "(A)") "userpot> coordinate line did not parse: " // trim(line)
      stop 4
    end if
    num_atoms = num_atoms + 1
  end do
  if (num_atoms < 1) then
    write(*, "(A)") "userpot> odata has no coordinates after USERPOT"
    stop 4
  end if
  known_atoms = num_atoms
end subroutine userpot_get_natoms

subroutine userpot_potentialhess(dof, x, gradient, energy, gradt, sect)
  use optbench_pot, only: calculate, load_env, ready
  implicit none
  integer, intent(in) :: dof
  double precision, intent(in) :: x(dof)
  double precision, intent(out) :: gradient(dof)
  double precision, intent(out) :: energy
  logical, intent(in) :: gradt, sect
  if (.not. ready) call load_env
  if (sect) then
    write(*, "(A)") "userpot> Hessian requested; USERPOT does not return one"
    flush(6)
    stop 4
  end if
  ! gradt false is still one geometry. The harness cache collapses the
  ! following gradient request when the coordinates are unchanged.
  if (.not. gradt) gradient = 0.0d0
  call calculate(dof, x, gradient, energy)
end subroutine userpot_potentialhess

subroutine userpot_distance(dof, x, y, dist)
  implicit none
  integer, intent(in) :: dof
  double precision, intent(in) :: x(dof), y(dof)
  double precision, intent(out) :: dist
  integer :: i
  dist = 0.0d0
  do i = 1, dof
    dist = dist + (x(i) - y(i))**2
  end do
  dist = sqrt(dist)
end subroutine userpot_distance

subroutine userpot_potential(dof, x, gradient, energy, gradt)
  implicit none
  integer, intent(in) :: dof
  double precision, intent(in) :: x(dof)
  double precision, intent(out) :: gradient(dof)
  double precision, intent(out) :: energy
  logical, intent(in) :: gradt
  write(*, "(A)") "userpot> userpot_potential is not the OPTIM entry"
  stop 4
end subroutine userpot_potential

subroutine userpot_initialize_gmin(dof, x)
  implicit none
  integer, intent(in) :: dof
  double precision, intent(out) :: x(dof)
  x = 0.0d0
  write(*, "(A)") "userpot> userpot_initialize_gmin is not used"
  stop 4
end subroutine userpot_initialize_gmin

subroutine userpot_dump_configuration(filename, coords)
  implicit none
  character(len=*) :: filename
  double precision :: coords(*)
  write(*, "(A)") "userpot> dump ignored for " // trim(filename)
end subroutine userpot_dump_configuration

subroutine userpot_dump_lowest
  implicit none
end subroutine userpot_dump_lowest
