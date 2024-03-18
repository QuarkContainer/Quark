#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
   
    // CUdevice device;
    // // cuInit(0);
    // cuDeviceGet(&device, 0);
    // printf("device name is %d",device);

    unsigned flags;
    int is_active;
    cudaSetDevice(0);
    CUresult status = cuDevicePrimaryCtxGetState(0, &flags, &is_active);
    if (status != CUDA_SUCCESS) { 
        printf("got error cuevicePrimaryCTX \n");
     }
     printf("status is %d",status);
  
}