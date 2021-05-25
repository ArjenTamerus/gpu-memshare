#include <iostream>

#define N_MEMELEM 32

__global__ void init_devmem(double *mem)
{
	mem[blockIdx.x*blockDim.x + threadIdx.x] = blockIdx.x*blockDim.x + threadIdx.x;
}

__global__ void double_devmem(double *mem)
{
	mem[blockIdx.x*blockDim.x + threadIdx.x] *= 2;
}

void do_cuda_init(double *dev_mem)
{
	init_devmem<<<N_MEMELEM/32,32>>>(dev_mem);
}

void do_cuda_double(double *dev_mem)
{
	double_devmem<<<N_MEMELEM/32,32>>>(dev_mem);
}

cudaIpcMemHandle_t get_memhandle(void *devmem)
{
	cudaError_t err_dev;
	cudaIpcMemHandle_t dev_mem_handle;

	err_dev = cudaIpcGetMemHandle(&dev_mem_handle, devmem);
	std::cout << "getHandle: " << cudaGetErrorName(err_dev) << ": " << cudaGetErrorString(err_dev) << std::endl;

	return dev_mem_handle;
}

void *cuda_open_handle(cudaIpcMemHandle_t dev_mem_handle)
{
	cudaError_t err_dev;
	void *mem_ptr=NULL;

	err_dev = cudaIpcOpenMemHandle(&mem_ptr, dev_mem_handle, cudaIpcMemLazyEnablePeerAccess);
	std::cout << "openHandle: " << cudaGetErrorName(err_dev) << ": " << cudaGetErrorString(err_dev) << std::endl;

	return mem_ptr;
}

void cuda_close_handle(void *devmem)
{
	cudaError_t err_dev;

	err_dev = cudaIpcCloseMemHandle(devmem);
	std::cout << "closeHandle: " << cudaGetErrorName(err_dev) << ": " << cudaGetErrorString(err_dev) << std::endl;
}

