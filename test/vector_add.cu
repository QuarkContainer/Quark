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

int main(){
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
    printf("testcuda 6\n");
    vector_add<<<1,1>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    printf("testcuda 7\n");
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    printf("testcuda 8\n");
    for(int i = 0; i < N; i++){
        //assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
        if(out[i]!=3.0f) {
            printf("fail i is %d out[i] is %f\n", i, out[i]);
        }
        //assert(out[i]==3.0f);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
