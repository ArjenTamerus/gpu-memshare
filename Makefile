CU_INCLUDE=-I/opt/cuda/targets/x86_64-linux/include/
all: fortran

c:
	nvcc -ccbin g++-9 -c -g -o cuda_ipcmemhandle.o cuda_ipcmemhandle.cu
	$(CC) $(CU_INCLUDE) -c -g -o shared_iface.o shared_iface.c
	mpicxx -acc -ta:tesla -Minfo:accel -g -traceback -o cuda-ipc-mem main.cxx shared_iface.o cuda_ipcmemhandle.o -lcudart -L/opt/cuda/lib64 -I/opt/cuda/include

fortran:
	$(CC) $(CU_INCLUDE) -c -g -o shared_iface.o shared_iface.c
	mpifort -acc -ta:tesla -Minfo:accel -g -traceback -o fortran-ipc-wrap main.f90 shared_iface.o -lcudart -L/opt/cuda/lib64 -I/opt/cuda/include

run: run-fortran

run-all: run-c run-fortran

run-c:
	mpirun -np 2 ./cuda-ipc-mem

run-fortran:
	mpirun -np 2 ./fortran-ipc-wrap
