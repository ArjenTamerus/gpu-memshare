program shared_gpu
  use,intrinsic :: iso_c_binding
  use openacc
  use mpi_f08

  implicit none

  interface
    integer function getMemHandleSize() bind(C,name="getMemHandleSize")
    end function
  end interface

  interface
    type(c_ptr) function getMemHandle(mem) bind(C,name="getMemHandle")
        use openacc
        use iso_c_binding
        type(c_devptr),intent(in) :: mem
    end function
  end interface

  interface
    integer function openMemHandle(mem, handle_str) bind(C,name="openMemHandle")
        use,intrinsic :: iso_c_binding
        type(c_ptr),intent(in) :: mem
        type(c_ptr),intent(in) :: handle_str
    end function
  end interface

  real(8),dimension(:),allocatable,target :: mem
  integer,parameter :: NELEM = 32
  type(c_ptr),target :: mem_handle
  type(c_ptr) :: mem_cptr, mem_cptrptr
  type(c_devptr) :: mem_cdevptr
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

  allocate(mem(NELEM))

  handle_size = getMemHandleSize()

  if(master) then
    !$acc data copy(mem)

    !$acc parallel loop
    do i=1,NELEM
      mem(i) = i
    end do
    !$acc end parallel loop

    write(*,*) "Memory handle size is: ", handle_size

    !mem_cptr = c_loc(mem)
    ! $acc host_data use_device(mem)
    mem_cdevptr = acc_deviceptr(mem)
    ! $acc end host_data

    mem_handle = getMemHandle(mem_cdevptr);
    if (mem_cdevptr .eq. C_NULL_DEVPTR) then
      write(*,*) " IS NULLL"
    else
      write(*,*) "NO IS NULL"
    endif

    call mpi_bcast(mem_handle, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD, err)

    !$acc end data
  else
    !$acc data copy(mem)

    !$acc parallel loop
    do i=1,NELEM
      mem(i) = NELEM-i
    end do
    !$acc end parallel loop

    !$acc end data
    call mpi_bcast(mem_handle, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD, err)

    mem_cptr = c_loc(mem)
    mem_cptrptr = c_loc(mem_cptr)

    err = openMemHandle(mem_cptr, mem_handle)

    write(*,*) "Open status: ", err

  endif

  deallocate(mem)
end program
