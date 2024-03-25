#include <cuda_runtime.h>
#include <stdio.h>

/*
To test, go to repo's root dir, then execute following script:

(cd test/rust/cuda_hook; cargo build)
nvcc -cudart shared -o target/cuda_test test/cu/cuda_test.cu
LD_PRELOAD=test/rust/cuda_hook/target/debug/libcuda_hook.so ./target/cuda_test
*/

__global__ void hello_kernel(void) {
  printf("Hello from the GPU!\n");
}

int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);

  printf("Hello from the CPU!\n");
  hello_kernel<<<2, 3>>>();
  cudaDeviceReset();
  cudaDeviceSynchronize();

  return 0;
}
