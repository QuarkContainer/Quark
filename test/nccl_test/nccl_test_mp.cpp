#include <iostream>
#include <vector>
#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
// bool testNcclGetUniqueId() {
//     ncclUniqueId id;
//     CHECK_NCCL(ncclGetUniqueId(&id));
//     for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
//         std::cout << std::hex << static_cast<int>(id.internal[i]) << " ";
//     }
//     return true;
// }

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

// bool testNcclCommInitRank() {
//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 2) {
//         std::cerr << "Need at least 2 devices to test ncclCommInitRank." << std::endl;
//         return false;
//     }

//     ncclUniqueId id;
//     std::vector<ncclComm_t> comms(deviceCount);
//     CHECK_NCCL(ncclGetUniqueId(&id));
//     CHECK_NCCL(ncclGroupStart());
//     for (int i=0; i<deviceCount; i++) {
//         CHECK_CUDA(cudaSetDevice(i));
//         CHECK_NCCL(ncclCommInitRank(&comms[i], deviceCount, id, i));
//     }
//     CHECK_NCCL(ncclGroupEnd());
//     for (auto comm : comms) {
//         CHECK_NCCL(ncclCommAbort(comm));
//     }
//     return true;
    
// }
// bool testNcclCommInitRank(myRank,nRanks) {
//     int deviceCount;
//     CUDACHECK(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 2) {
//         std::cerr << "Need at least 2 devices to test ncclCommInitRankConfig." << std::endl;
//         return false;
//     }
//     if (myRank == 0) {
//         ncclGetUniqueId(&id);
//     }
//     MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
//                        MPI_COMM_WORLD));

//     // picking a GPU based on localRank, allocate device buffers
//     CUDACHECK(cudaSetDevice(localRank));
//     CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
//     CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
//     CUDACHECK(cudaStreamCreate(&s));
    

//     ncclUniqueId id;
//     ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
//     std::vector<ncclComm_t> comms(deviceCount);
//     CHECK_NCCL(ncclGetUniqueId(&id));
//     CHECK_NCCL(ncclGroupStart());
//     for (int i=0; i<deviceCount; i++) {
//         CHECK_CUDA(cudaSetDevice(i));
//         CHECK_NCCL(ncclCommInitRankConfig(&comms[i], deviceCount, id, i, &config));
//     }
//     CHECK_NCCL(ncclGroupEnd());
//     for (auto comm : comms) {
//         CHECK_NCCL(ncclCommAbort(comm));
//     }
//     return true;
// }


    

bool testNcclAllReduce(int myRank,int nRanks, int localRank) {
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRankConfig." << std::endl;
        return false;
    }
    ncclUniqueId id;
    if (myRank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));

    float *sendbuff, *recvbuff;
    int size = 32*1024*1024;
    cudaStream_t s;
    ncclComm_t comm;
    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));


    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    float* input = (float*)malloc(size * sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    CUDACHECK(cudaMemcpy(output, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(input, sendbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < size; ++j) {
        if (input[j]*deviceCount != output[j]) {
            std::cerr << "Error in Allreduce result " << j << std::endl;
            return false;
        }
    }
    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    //free host buffers
    free(input);
    free(output);


    //finalizing NCCL
    NCCLCHECK(ncclCommDestroy(comm));
    // printf("[MPI Rank %d] Success \n", myRank);
    return true;
}

bool testNcclAllGather(int myRank,int nRanks, int localRank) {
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRankConfig." << std::endl;
        return false;
    }
    ncclUniqueId id;
    if (myRank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));

    float *sendbuff, *recvbuff;
    int size = 32*1024*1024;
    cudaStream_t s;
    ncclComm_t comm;
    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, nRanks * size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    NCCLCHECK(ncclAllGather((const void*)sendbuff, (void*)recvbuff, size, ncclFloat,
        comm, s));


    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    float* input = (float*)malloc(sizeof(float));
    float* output = (float*)malloc(nRanks*size * sizeof(float));
    CUDACHECK(cudaMemcpy(output, recvbuff, nRanks*size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(input, sendbuff, sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < nRanks*size; ++j) {
        if (*input != output[j]) {
            std::cerr << "Error in Allgather result " << j << std::endl;
            return false;
        }
    }
    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    //free host buffers
    free(input);
    free(output);


    //finalizing NCCL
    NCCLCHECK(ncclCommDestroy(comm));
    // printf("[MPI Rank %d] Success \n", myRank);
    return true;
}

bool testNcclReduceScatter(int myRank,int nRanks, int localRank) {
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRankConfig." << std::endl;
        return false;
    }
    ncclUniqueId id;
    if (myRank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));

    float *sendbuff, *recvbuff;
    int size = 32*1024*1024;
    cudaStream_t s;
    ncclComm_t comm;
    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, nRanks * size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, 1, nRanks * size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    NCCLCHECK(ncclReduceScatter((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum, comm, s));


    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    float* input = (float*)malloc(sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    CUDACHECK(cudaMemcpy(output, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(input, sendbuff, sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < size; ++j) {
        if (*input * nRanks != output[j]) {
            std::cerr << "Error in Allreducescatter result " << j << std::endl;
            std::cerr << *input << " " << nRanks << " " << output[j] << std::endl;
            return false;
        }
    }
    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    //free host buffers
    free(input);
    free(output);


    //finalizing NCCL
    NCCLCHECK(ncclCommDestroy(comm));
    // printf("[MPI Rank %d] Success \n", myRank);
    return true;
}

bool testNcclSendRecv(int myRank,int nRanks, int localRank) {
    int deviceCount;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRankConfig." << std::endl;
        return false;
    }
    ncclUniqueId id;
    if (myRank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0,
                       MPI_COMM_WORLD));

    float *sendbuff, *recvbuff;
    int size = 32*1024*1024;
    cudaStream_t s;
    ncclComm_t comm;
    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff, myRank, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff, 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclSend(sendbuff, size, ncclFloat, (myRank/2)*myRank +(myRank+1)%nRanks, comm, s));
    NCCLCHECK(ncclRecv(recvbuff, size, ncclFloat, (myRank/2)*myRank +(myRank+1)%nRanks, comm, s));
    NCCLCHECK(ncclGroupEnd());
    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    float* input = (float*)malloc(sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    CUDACHECK(cudaMemset(sendbuff, (myRank/2)*myRank +(myRank+1)%nRanks, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(output, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(input, sendbuff, sizeof(float), cudaMemcpyDeviceToHost));
    for (int j = 0; j < size; ++j) {
        if (*input  != output[j]) {
            std::cerr << "Error in sendrecv result " << j << std::endl;
            return false;
        }
    }
    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));
    //free host buffers
    free(input);
    free(output);


    //finalizing NCCL
    NCCLCHECK(ncclCommDestroy(comm));
    // printf("[MPI Rank %d] Success \n", myRank);
    return true;
}

// bool testNcclSendRecv() {
//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 2) {
//         std::cerr << "Need at least 2 devices to test ncclSendRecv." << std::endl;
//         return false;
//     }

//     std::vector<ncclComm_t> comms(deviceCount);
//     int nDev = deviceCount;
//     int size = 32*1024*1024;
//     std::vector<int> devs(deviceCount);
//     for (int i = 0; i < deviceCount; ++i) devs[i] = i;

//     //allocating and initializing device buffers
//     float** sendbuff = (float**)malloc(nDev * sizeof(float*));
//     float** recvbuff = (float**)malloc(nDev * sizeof(float*));
//     cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

//     for (int i = 0; i < nDev; ++i) {
//         CHECK_CUDA(cudaSetDevice(i));
//         CHECK_CUDA(cudaMalloc((void**)sendbuff + i, size * sizeof(float)));
//         CHECK_CUDA(cudaMalloc((void**)recvbuff + i, size * sizeof(float)));
//         CHECK_CUDA(cudaMemset(sendbuff[i], i+1, size * sizeof(float)));
//         CHECK_CUDA(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
//         CHECK_CUDA(cudaStreamCreate(s+i));
//     }

//     //initializing NCCL
//     CHECK_NCCL(ncclCommInitAll(comms.data(), nDev, devs.data()));

//     //calling NCCL communication API. Group API is required when using
//     //multiple devices per thread
//     CHECK_NCCL(ncclGroupStart());
//     for (int i = 0; i < nDev; ++i){
//         CHECK_CUDA(cudaSetDevice(i));
//         CHECK_NCCL(ncclSend((const void*)sendbuff[i], size, ncclFloat, i/2 + (i+1)%nDev, comms[i], s[i]));
//         CHECK_NCCL(ncclRecv((void*)recvbuff[i], size, ncclFloat, i/2 +(i+1)%nDev, comms[i], s[i]));
//     }
//     CHECK_NCCL(ncclGroupEnd());

//     //synchronizing on CUDA streams to wait for completion of NCCL operation
//     float* input = (float*)malloc(size * sizeof(float));
//     float* output = (float*)malloc(size * sizeof(float));
//     for (int i = 0; i < nDev; ++i) {
//         CHECK_CUDA(cudaSetDevice(i));
//         CHECK_CUDA(cudaStreamSynchronize(s[i]));
//         CHECK_CUDA(cudaMemcpy(output, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
//         CHECK_CUDA(cudaMemcpy(input, sendbuff[i/2 + (i+1)%nDev], size * sizeof(float), cudaMemcpyDeviceToHost));
//         for (int j = 0; j < size; ++j) {
//             if (input[j] != output[j]) {
//                 std::cerr << "Error in SendRecv result " << i << std::endl;
//                 std::cerr << "input[" << j << "] = " << input[j]  << ", output[" << j << "] = " << output[j] << std::endl;
//                 return false;
//             }
//         }   
//     }

//     //free device buffers
//     for (int i = 0; i < nDev; ++i) {
//         CHECK_CUDA(cudaSetDevice(i));
//         CHECK_CUDA(cudaFree(sendbuff[i]));
//         CHECK_CUDA(cudaFree(recvbuff[i]));
//     }

//     //finalizing NCCL
//     for(int i = 0; i < nDev; ++i)
//         CHECK_NCCL(ncclCommDestroy(comms[i]));

//     return true;
// }


// bool testNcclCommInitAll() {
//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 2) {
//         std::cerr << "Need at least 2 devices to test ncclCommInitAll." << std::endl;
//         return false;
//     }

//     std::vector<ncclComm_t> comms(deviceCount);
//     std::vector<int> devs(deviceCount);
//     for (int i = 0; i < deviceCount; ++i) devs[i] = i;

//     CHECK_NCCL(ncclCommInitAll(comms.data(), deviceCount, devs.data()));

//     for (auto comm : comms) {
//         CHECK_NCCL(ncclCommAbort(comm));
//     }
//     return true;
// }
// bool testNcclCommCount() {
//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 2) {
//         std::cerr << "Need at least 2 devices to test ncclCommCount." << std::endl;
//         return false;
//     }

//     std::vector<ncclComm_t> comms(deviceCount);
//     std::vector<int> devs(deviceCount);
//     for (int i = 0; i < deviceCount; ++i) devs[i] = i;

//     CHECK_NCCL(ncclCommInitAll(comms.data(), deviceCount, devs.data()));

//     int count;
//     for (auto comm : comms) {
//         CHECK_NCCL(ncclCommCount(comm, &count));
//         if (count != deviceCount) {
//             std::cerr << "ncclCommCount returned incorrect count." << std::endl;
//             return false;
//         }
//         CHECK_NCCL(ncclCommAbort(comm));
//     }
//     return true;
// }

// bool testNcclUserRank() {
//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 2) {
//         std::cerr << "Need at least 2 devices to test ncclUserRank." << std::endl;
//         return false;
//     }

//     std::vector<ncclComm_t> comms(deviceCount);
//     std::vector<int> devs(deviceCount);
//     for (int i = 0; i < deviceCount; ++i) devs[i] = i;

//     CHECK_NCCL(ncclCommInitAll(comms.data(), deviceCount, devs.data()));

//     int rank;
//     for (int i = 0; i < deviceCount; ++i) {
//         CHECK_NCCL(ncclCommUserRank(comms[i], &rank));
//         if (rank != i) {
//             std::cerr << "ncclCommUserRank returned incorrect rank." << std::endl;
//             return false;
//         }
//         CHECK_NCCL(ncclCommAbort(comms[i]));
//     }
//     return true;
// }
// bool testNcclGetErrorString() {
//     // Deliberately create an error by passing an invalid argument to an NCCL function
//     ncclComm_t comm;
//     ncclUniqueId id;
//     ncclResult_t result = ncclCommInitRank(&comm, 1, id, 1);

//     // Check the error string for the invalid result
//     // std::cout << result << std::endl;
//     const char* errorString = ncclGetErrorString(result);
//     if (result != ncclSuccess) {
//         std::cout << "ncclGetErrorString: " << errorString << std::endl;
//     } else {
//         std::cerr << "Expected failure, but ncclCommInitAll succeeded unexpectedly." << std::endl;
//         return false;
//     }

//     return true;
// }



int main(int argc, char* argv[]) {
    int myRank, nRanks, localRank = 0;
    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    // std::cout << "Testing ncclCommInitRankConfig: " << (testNcclCommInitRankConfig(myRank,nRanks) ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclAllReduce: " << (testNcclAllReduce(myRank,nRanks,localRank) ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclAllGather: " << (testNcclAllGather(myRank,nRanks,localRank) ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclreduceScatter: " << (testNcclReduceScatter(myRank,nRanks,localRank) ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclsendRecv: " << (testNcclSendRecv(myRank,nRanks,localRank) ? "Passed" : "Failed") << std::endl;

    //finalizing MPI
    MPICHECK(MPI_Finalize());
    return 0;
}
