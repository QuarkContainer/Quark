#include <iostream>
#include <vector>
#include <nccl.h>
#include <mpi.h>
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
    // Initialize MPI
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI does not provide needed threading level" << std::endl;
        return false;
    }

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        std::cerr << "Need at least 2 devices to test ncclCommInitRank." << std::endl;
        MPI_Finalize();
        return false;
    }

    ncclUniqueId id;
    ncclComm_t comm;

    if (world_rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&id));
    }

    // Broadcast the unique ID to all ranks
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Assign devices to MPI ranks
    int local_rank = world_rank % deviceCount;
    CHECK_CUDA(cudaSetDevice(local_rank));

    CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

    // Destroy the NCCL communicator
    CHECK_NCCL(ncclCommDestroy(comm));

    // Finalize MPI
    MPI_Finalize();

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

// bool testNcclCommAbort() {
//     int deviceCount;
//     CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
//     if (deviceCount < 1) {
//         std::cerr << "Need at least 1 device to test ncclCommAbort." << std::endl;
//         return false;
//     }

//     ncclComm_t comm;
//     ncclUniqueId id;
//     CHECK_NCCL(ncclGetUniqueId(&id));
//     CHECK_NCCL(ncclCommInitRank(&comm, deviceCount, id, 0));

//     // Aborting the communicator
//     ncclResult_t abortResult = ncclCommAbort(comm);
//     if (abortResult != ncclSuccess) {
//         std::cerr << "ncclCommAbort failed: " << ncclGetErrorString(abortResult) << std::endl;
//         return false;
//     } else {
//         std::cout << "Success: ncclCommAbort" << std::endl;
//     }

//     return true;
// }

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


// Define other test functions similarly

int main() {
    std::cout << "Testing ncclCommInitRank: " << (testNcclCommInitRank() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclGetUniqueId: " << (testNcclGetUniqueId() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclCommCount: " << (testNcclCommCount() ? "Passed" : "Failed") << std::endl;
    std::cout << "Testing ncclUserRank: " << (testNcclUserRank() ? "Passed" : "Failed") << std::endl;
    // std::cout << "Testing ncclCommInitAll: " << (testNcclCommInitAll() ? "Passed" : "Failed") << std::endl;
    // std::cout << "Testing ncclCommAbort: " << (testNcclCommAbort() ? "Passed" : "Failed") << std::endl;
    // std::cout << "Testing ncclGetErrorString: " << (testNcclGetErrorString() ? "Passed" : "Failed") << std::endl;
    // std::cout << "Testing ncclCommAbort: " << (testNcclCommAbort() ? "Passed" : "Failed") << std::endl;


    return 0;
}
