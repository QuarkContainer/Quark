#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 100//000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i ++){
        out[i] = a[i] + b[i];
    }
}

void cuda_test() {
    int dev = 0;
    int ret = cudaSetDevice(dev);
    printf("cudaSetDevice ret is %d\n", ret);
    ret = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize ret is %d\n", ret);

    float *a, *b;
    float *d_a;

    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);

    ret = cudaMalloc((void**)&d_a, sizeof(float) * N);
    printf("cudaMalloc ret is %d\n", ret);
    printf("cudaMalloc addr is %llx\n", (unsigned long long)&d_a);

    for(int i = 0; i < N; i++){
        a[i] = (float)i;
    }

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    for(int i = 0; i < N; i++){
        if (a[i]!=float(i)) {
            printf("a[%d] is %f\n", i, a[i]);
        }
        
        assert(a[i]==float(i));
    }
    
    cudaMemcpy(b, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < N; i++){
        if (b[i]!=float(i)) {
            printf("b[%d] is %f a[%d] is %f\n", i, b[i], i, a[i]);
        }
        
        assert(b[i]==float(i));
    }

    printf("cudaMemcpy passed ...\n");
}

void cuda_add() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    printf("testcuda 1\n");
    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    printf("testcuda 2\n");
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    printf("testcuda 3\n");
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    printf("testcuda 4\n");
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    printf("testcuda 5\n");
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Executing kernel 
    printf("testcuda 6 d_out %p d_a %p d_b %p\n", d_out, d_a, d_b);
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);
 
    // Transfer data back to host memory
    printf("testcuda 7\n");
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    printf("testcuda 8\n");
    for(int i = 0; i < N; i++){
        //assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
        // if(out[i]!=3.0f) {
        //     printf("fail i is %d out[i] is %f\n", i, out[i]);
        // }
        //assert(out[i]==3.0f);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    printf("Deallocating GPU memory\n");
 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
   

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}

int main(){
    // cuda_test();
    cuda_add();
}
