#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <openacc.h>

int getMemHandleSize()
{
	return CUDA_IPC_HANDLE_SIZE;
}

// Filthy filthy haxx
char* getMemHandle(void *mem)
{
	cudaError_t err;
	cudaIpcMemHandle_t *handle;

	handle = calloc(1, sizeof(cudaIpcMemHandle_t));

	err = cudaIpcGetMemHandle(handle, mem);

	fprintf(stderr, "gethandle Status: %d - %p -%s\n", err, mem, cudaGetErrorString(err));
	if (err) return NULL;

	return (char*)handle; 
}

int openMemHandle(void **mem, char *handle_str)
{
	cudaError_t err;
	//cudaIpcMemHandle_t *handle = (cudaIpcMemHandle_t*) handle_str;

	//memcpy(&(handle.reserved), handle_str, CUDA_IPC_HANDLE_SIZE);
	
	err = cudaIpcOpenMemHandle(mem, *(cudaIpcMemHandle_t*)handle_str, cudaIpcMemLazyEnablePeerAccess);

	fprintf(stderr, "Status: %d - %s\n", err, cudaGetErrorString(err));

	if(err)
		return -1;

	return 0;
}

int closeMemHandle(void *mem)
{
	cudaError_t err;

	err = cudaIpcCloseMemHandle(mem);

	if(err)
		return -1;

	return 0;
}

int mapACCData(void *host, void *device, int n)
{
	acc_map_data(host, device, n*sizeof(double));
	return 0;
}
