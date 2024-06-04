#include <iostream>
#include <vector>
#include <nccl.h>
// #include <mpi.h>
#include <cuda_runtime.h>

#define CHECK_NCCL(cmd) do { \
    ncclResult_t result = cmd; \
    if (result != ncclSuccess) { \
        std::cerr << "Failed, NCCL error: " << ncclGetErrorString(result) << std::endl; \
        return false; \
    } else { \
        std::cout << "Success: " << #cmd << std::endl; \
    } \
} while(0)

#define CHECK_CUDA(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        std::cerr << "Failed, CUDA error: " << cudaGetErrorString(result) << std::endl; \
        return false; \
    } else { \
        std::cout << "Success: " << #cmd << std::endl; \
    } \
} while(0)

bool testNcclGetUniqueId() {
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        std::cout << std::hex << static_cast<int>(id.internal[i]) << " ";
    }
    return true;
}

bool testNcclCommInitRank() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRank." << std::endl;
        return false;
    }

    ncclUniqueId id;
    std::vector<ncclComm_t> comms(deviceCount);
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclGroupStart());
    for (int i=0; i<deviceCount; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclCommInitRank(&comms[i], deviceCount, id, i));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (auto comm : comms) {
        CHECK_NCCL(ncclCommAbort(comm));
    }
    return true;
    
}
bool testNcclCommInitRankConfig() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRankConfig." << std::endl;
        return false;
    }

    ncclUniqueId id;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    std::vector<ncclComm_t> comms(deviceCount);
    CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_NCCL(ncclGroupStart());
    for (int i=0; i<deviceCount; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclCommInitRankConfig(&comms[i], deviceCount, id, i, &config));
    }
    CHECK_NCCL(ncclGroupEnd());
    for (auto comm : comms) {
        CHECK_NCCL(ncclCommAbort(comm));
    }
    return true;
}


    

bool testNcclAllReduce() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclAllreduce." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    int nDev = deviceCount;
    int size = 32*1024*1024;
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;


    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        CHECK_CUDA(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CHECK_CUDA(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(s+i));
    }


    //initializing NCCL
    CHECK_NCCL(ncclCommInitAll(comms.data(), nDev, devs.data()));


    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
            comms[i], s[i]));
    CHECK_NCCL(ncclGroupEnd());


    //synchronizing on CUDA streams to wait for completion of NCCL operation
    float* input = (float*)malloc(size * sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(s[i]));
        CHECK_CUDA(cudaMemcpy(output, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(input, sendbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < size; ++j) {
            if (input[j]*nDev != output[j]) {
                std::cerr << "Error in Allreduce result " << i << std::endl;
                std::cerr << "input[" << j << "] = " << input[j]  << ", output[" << j << "] = " << output[j] << std::endl;
                return false;
            }
        }   

    }


    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
    }
    //free host buffers
    free (input);
    free (output);

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclCommDestroy(comms[i]));

    return true;
}
bool testNcclAllGather() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclAllGather." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    int nDev = deviceCount;
    int size = 32*1024*1024;
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * size * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)recvbuff + i, nDev * size * sizeof(float)));
        CHECK_CUDA(cudaMemset(sendbuff[i], i+1, size * sizeof(float)));
        CHECK_CUDA(cudaMemset(recvbuff[i], 0, 2 * size * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(s+i));
    }

    //initializing NCCL
    CHECK_NCCL(ncclCommInitAll(comms.data(), nDev, devs.data()));

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclAllGather((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat,
            comms[i], s[i]));
    CHECK_NCCL(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    float** input = (float**)malloc(nDev * sizeof(float*));
    for (int i = 0; i < nDev; ++i) {
        input[i] = (float*)malloc(size * sizeof(float));
    }
    float* output = (float*)malloc(nDev * size * sizeof(float));
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(s[i]));
        CHECK_CUDA(cudaMemcpy(output, recvbuff[i], nDev * size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(input[i], sendbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < size; ++j) {
            if (input[i][j] != output[j+i*size]) {
                std::cerr << "Error in Allgather result " << i << std::endl;
                std::cerr << "input[" << j << "] = " << input[i][j]  << ", output[" << j << "] = " << output[j+i*size] << std::endl;
                return false;
            }
        }   
    }
    //free host buffers
    for (int i = 0; i < nDev; ++i) {
        free(input[i]);
    }
    free(output);    
    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclCommDestroy(comms[i]));

    return true;
}

bool testNcclReduceScatter() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclReduceScatter." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    int nDev = deviceCount;
    int size = 32*1024*1024;
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc((void**)sendbuff + i, nDev * size * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        CHECK_CUDA(cudaMemset(sendbuff[i], 1, nDev * size * sizeof(float)));
        CHECK_CUDA(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(s+i));
    }

    //initializing NCCL
    CHECK_NCCL(ncclCommInitAll(comms.data(), nDev, devs.data()));

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclReduceScatter((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat,
            ncclSum, comms[i], s[i]));
    CHECK_NCCL(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    // float** input = (float**)malloc(nDev * sizeof(float*));
    // for (int i = 0; i < nDev; ++i) {
    //     input[i] = (float*)malloc( size * sizeof(float));
    // }
    float* input = (float*)malloc(nDev * size * sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(s[i]));
        CHECK_CUDA(cudaMemcpy(output, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(input, sendbuff[i], nDev * size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < size; ++j) {
            if (input[j] * nDev != output[j]) {
                std::cerr << "Error in ReduceScatter result " << i << std::endl;
                std::cerr << "input[" << j << "] = " << input[j] << ", output[" << j << "] = " << output[j] << std::endl;
                return false;
            }
        }   
    }
    //free host buffers
   
    free(output); 
    free(input);   
    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclCommDestroy(comms[i]));

    return true;
}

bool testNcclSendRecv() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclSendRecv." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    int nDev = deviceCount;
    int size = 32*1024*1024;
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;

    //allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
        CHECK_CUDA(cudaMemset(sendbuff[i], i+1, size * sizeof(float)));
        CHECK_CUDA(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(s+i));
    }

    //initializing NCCL
    CHECK_NCCL(ncclCommInitAll(comms.data(), nDev, devs.data()));

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nDev; ++i){
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclSend((const void*)sendbuff[i], size, ncclFloat, (i/2)*2 + (i+1)%nDev, comms[i], s[i]));
        CHECK_NCCL(ncclRecv((void*)recvbuff[i], size, ncclFloat, (i/2)*2 +(i+1)%nDev, comms[i], s[i]));
    }
    CHECK_NCCL(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    float* input = (float*)malloc(size * sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(s[i]));
        CHECK_CUDA(cudaMemcpy(output, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(input, sendbuff[(i/2)*2 + (i+1)%nDev], size * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < size; ++j) {
            if (input[j] != output[j]) {
                std::cerr << "Error in SendRecv result " << i << std::endl;
                std::cerr << "input[" << j << "] = " << input[j]  << ", output[" << j << "] = " << output[j] << std::endl;
                return false;
            }
        }   
    }

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(sendbuff[i]));
        CHECK_CUDA(cudaFree(recvbuff[i]));
    }

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        CHECK_NCCL(ncclCommDestroy(comms[i]));

    return true;
}


bool testNcclCommInitAll() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitAll." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;

    CHECK_NCCL(ncclCommInitAll(comms.data(), deviceCount, devs.data()));

    for (auto comm : comms) {
        CHECK_NCCL(ncclCommAbort(comm));
    }
    return true;
}
bool testNcclCommCount() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommCount." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;

    CHECK_NCCL(ncclCommInitAll(comms.data(), deviceCount, devs.data()));

    int count;
    for (auto comm : comms) {
        CHECK_NCCL(ncclCommCount(comm, &count));
        if (count != deviceCount) {
            std::cerr << "ncclCommCount returned incorrect count." << std::endl;
            return false;
        }
        CHECK_NCCL(ncclCommAbort(comm));
    }
    return true;
}

bool testNcclUserRank() {
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclUserRank." << std::endl;
        return false;
    }

    std::vector<ncclComm_t> comms(deviceCount);
    std::vector<int> devs(deviceCount);
    for (int i = 0; i < deviceCount; ++i) devs[i] = i;

    CHECK_NCCL(ncclCommInitAll(comms.data(), deviceCount, devs.data()));

    int rank;
    for (int i = 0; i < deviceCount; ++i) {
        CHECK_NCCL(ncclCommUserRank(comms[i], &rank));
        if (rank != i) {
            std::cerr << "ncclCommUserRank returned incorrect rank." << std::endl;
            return false;
        }
        CHECK_NCCL(ncclCommAbort(comms[i]));
    }
    return true;
}
bool testNcclGetErrorString() {
    // Deliberately create an error by passing an invalid argument to an NCCL function
    ncclComm_t comm;
    ncclUniqueId id;
    ncclResult_t result = ncclCommInitRank(&comm, 1, id, 1);

    // Check the error string for the invalid result
    // std::cout << result << std::endl;
    const char* errorString = ncclGetErrorString(result);
    if (result != ncclSuccess) {
        std::cout << "ncclGetErrorString: " << errorString << std::endl;
    } else {
        std::cerr << "Expected failure, but ncclCommInitAll succeeded unexpectedly." << std::endl;
        return false;
    }

    return true;
}



int main() {

    std::cout << "Testing ncclCommInitRankConfig: " << (testNcclCommInitRankConfig() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclAllReduce: " << (testNcclAllReduce() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclAllGather: " << (testNcclAllGather() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclreduceScatter: " << (testNcclReduceScatter() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclsendRecv: " << (testNcclSendRecv() ? "Passed" : "Failed") << std::endl;


    return 0;
}
