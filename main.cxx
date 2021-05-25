#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <openacc.h>
#include "shared_iface.h"

#define N_MEMELEM 32

void do_cuda_init(double *devmem);
void do_cuda_double(double *devmem);
cudaIpcMemHandle_t get_memhandle(void *devmem);
void *cuda_open_handle(cudaIpcMemHandle_t dev_mem_handle);
void cuda_close_handle(void *devmem);

int main(int argc, char **argv)
{
	int mpi_rank, mpi_size;
	bool master;


	double *dev_mem = NULL;
	double *host_mem = NULL;
	cudaIpcMemHandle_t dev_mem_handle;
	cudaError_t err_dev;

	std::cout << "size: " << sizeof(cudaIpcMemHandle_t) << " vs " << sizeof(dev_mem_handle.reserved) << " vs " << CUDA_IPC_HANDLE_SIZE << std::endl;

	MPI_Init(&argc, &argv);
	//cudaSetDevice(0);
	//cudaFree(0);

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	master = !mpi_rank;

	//if (mpi_size != 2) {
	//	std::cout << "Pls only run with 2 ranks for now, thanks!" << std::endl;
	//	MPI_Abort(MPI_COMM_WORLD, 1);
	//}
	int handle_size = getMemHandleSize();
	char *handle_str;

	if (master) {
		err_dev = cudaMalloc(&dev_mem, N_MEMELEM*sizeof(double));

		do_cuda_init(dev_mem);

		//dev_mem_handle = get_memhandle(dev_mem);
		err_dev = cudaIpcGetMemHandle(&dev_mem_handle, dev_mem);
		handle_str = getMemHandle(dev_mem);
	}
	else {
		handle_str = (char*) malloc(handle_size);
	}

	MPI_Bcast(&dev_mem_handle, sizeof(dev_mem_handle), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(handle_str, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD);

	if (!master) {
		host_mem = new double[N_MEMELEM];

		//err_dev = cudaIpcOpenMemHandle((void**)&dev_mem, dev_mem_handle, cudaIpcMemLazyEnablePeerAccess);
		//err_dev = cudaIpcOpenMemHandle((void**)&dev_mem, *(cudaIpcMemHandle_t*)handle_str, cudaIpcMemLazyEnablePeerAccess);
		std::cout << "Status: " << openMemHandle((void**)&dev_mem, handle_str) << std::endl;

		err_dev = cudaMemcpy(host_mem, dev_mem, N_MEMELEM*sizeof(double), cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < N_MEMELEM; i++)
			std::cout << host_mem[i] << " ";
		std::cout << std::endl;

		//err_dev = cudaIpcCloseMemHandle(dev_mem);
		closeMemHandle(dev_mem);

		delete[] host_mem;
	}

	/* Barrier here - otherwise rank0 may race to cudaFree before rank1 completes 
	 * its shared memory access. */
	MPI_Barrier(MPI_COMM_WORLD);

//	MPI_Win dev_window;
//
//	if (master) {
//		MPI_Win_create(NULL, 0, sizeof(double), MPI_INFO_NULL , MPI_COMM_WORLD, &dev_window);
//		MPI_Win_fence(0, win);
//		MPI_Accumulate();
//	}
//	else {
//		MPI_Win_create(dev_mem, N_MEMELEM*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &dev_window);
//		MPI_Win_fence(0, win);
//		MPI_Win_fence(0, win);
//	}

	if (!master) {
		host_mem = new double[N_MEMELEM];

		err_dev = cudaIpcOpenMemHandle((void**)&dev_mem, dev_mem_handle, cudaIpcMemLazyEnablePeerAccess);
		//openMemHandle(dev_mem, handle_str);

		acc_map_data(host_mem, dev_mem, N_MEMELEM*sizeof(double));


/* present clause doesn't _seem_ to be necessary, so acc_map properly keeps
 * track of GPU mem mapping. */
//#pragma acc parallel loop present(host_mem[N_MEMELEM])
#pragma acc parallel loop
		for (size_t i = 0; i < N_MEMELEM; i++) {
			host_mem[i] = 2*host_mem[i];
		}

		acc_unmap_data(host_mem);

		err_dev = cudaIpcCloseMemHandle(dev_mem);
		//closeMemHandle(dev_mem);

		delete[] host_mem;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(master) {
		host_mem = new double[N_MEMELEM];

		err_dev = cudaMemcpy(host_mem, dev_mem, N_MEMELEM*sizeof(double), cudaMemcpyDeviceToHost);

		for (size_t i = 0; i < N_MEMELEM; i++)
			std::cout << host_mem[i] << " ";
		std::cout << std::endl;

		delete[] host_mem;
	}

	/* Barrier here - otherwise rank0 may race to cudaFree before rank1 completes 
	 * its shared memory access. */
	MPI_Barrier(MPI_COMM_WORLD);

	if (master) {
		err_dev = cudaFree(dev_mem);
	}


	if(master){
		host_mem = new double[N_MEMELEM];
#pragma acc data copy(host_mem[N_MEMELEM])
		{
#pragma acc parallel loop
			for(int i=0;i<N_MEMELEM;i++) host_mem[i]=3*i;

			dev_mem = (double*)acc_deviceptr(host_mem);
			handle_str = getMemHandle(dev_mem);
			MPI_Bcast(handle_str, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);

		}


		for (size_t i = 0; i < N_MEMELEM; i++)
			std::cout << host_mem[i] << " ";
		std::cout << std::endl;
	}
	else {
		MPI_Bcast(handle_str, handle_size, MPI_BYTE, 0, MPI_COMM_WORLD);

		openMemHandle((void**)&dev_mem, handle_str);
		acc_map_data(host_mem, dev_mem, N_MEMELEM*sizeof(double));

#pragma acc parallel loop present(host_mem)
			for(int i=0;i<N_MEMELEM;i++) host_mem[i]=4*i;

		closeMemHandle(dev_mem);
		MPI_Barrier(MPI_COMM_WORLD);
	}



	MPI_Finalize();

	return 0;
}
