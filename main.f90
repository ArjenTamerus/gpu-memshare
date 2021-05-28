program shared_gpu
  use,intrinsic :: iso_c_binding
  use openacc
  use mpi_f08

  implicit none

  interface
    integer function mapACCData(hostmem, devmem, n) bind(C,name="mapACCData")
        use openacc
        use,intrinsic :: iso_c_binding
        type(c_ptr),value,intent(in) :: hostmem
        type(c_devptr),value,intent(in) :: devmem
        integer :: n
    end function
  end interface

  interface
    integer function getMemHandleSize() bind(C,name="getMemHandleSize")
    end function
  end interface

  interface
    type(c_ptr) function getMemHandle(mem) bind(C,name="getMemHandle")
        use openacc
        use iso_c_binding
        type(c_devptr),value,intent(in) :: mem
    end function
  end interface

  interface
    integer function openMemHandle(mem, handle_str) bind(C,name="openMemHandle")
        use iso_c_binding
        use openacc
        type(c_devptr),intent(inout) :: mem
        type(c_ptr),value,intent(in) :: handle_str
    end function
  end interface

  interface
    integer function closeMemHandle(mem) bind(C,name="closeMemHandle")
        use iso_c_binding
        use openacc
        type(c_devptr),value,intent(in) :: mem
    end function
  end interface

  real(8),dimension(:),allocatable,target :: mem
  integer,parameter :: NELEM = 32
  type(c_ptr),target :: mem_handle
  type(c_ptr) :: mem_cptr
  type(c_devptr),target :: mem_cdevptr
  character(len=128),pointer :: handle_str
  character(len=128),target :: handle_str_s
  integer :: i, err, handle_size

  integer :: mpi_rank
  logical :: master

  call MPI_Init(err)
  call MPI_Comm_rank(mpi_comm_world,mpi_rank,err)

  if (mpi_rank .eq. 0) then
    master = .true.
  else
    master = .false.
  endif

  handle_size = getMemHandleSize()

  if(master) then
    allocate(mem(NELEM))

    !$acc data copy(mem)

    !$acc parallel loop
    do i=1,NELEM
      mem(i) = 1.0
    end do
    !$acc end parallel loop

    write(*,*) "Memory handle size is: ", handle_size

    mem_cdevptr = acc_deviceptr(mem)

    if (mem_cdevptr .eq. C_NULL_DEVPTR) then
      write(*,*) "DEVICE POINTER IS NULLL"
    else
      write(*,*) "DEVICE POINTER IS NOT NULL"
    endif

    mem_handle = getMemHandle(mem_cdevptr)

    call C_F_POINTER(mem_handle, handle_str)

    call mpi_bcast(handle_str, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD, err)

    call mpi_barrier(MPI_COMM_WORLD)
    !$acc end data
  else
    allocate(mem(0))
    call mpi_bcast(handle_str_s, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD, err)

    mem_cptr = c_loc(mem)

    err = openMemHandle(mem_cdevptr, c_loc(handle_str_s))

    !err = acc_map_data(mem_cptr, mem_cdevptr, NELEM*8)
    err = mapACCData(mem_cptr, mem_cdevptr, NELEM)

    !$acc parallel loop present(mem(:NELEM))
    do i=1,NELEM
      mem(i) = 2.0
    end do
    !$acc end parallel loop

    err = closeMemHandle(mem_cdevptr)
    write(*,*) "CLOSE_ERR: ", err

    call mpi_barrier(MPI_COMM_WORLD)
  endif

  if(master) then
    do i=1,NELEM
      write(*,*) mem(i)
    end do
  endif

  deallocate(mem)
end program
