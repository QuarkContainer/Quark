#include <iostream>
#include <chrono>
#include <random>
#include <sys/mman.h> // for mlock()
#include <fstream> // for file I/O
#include <string> // for std::string
#include <vector> // for std::vector

// CUDA header
#include <cuda_runtime.h>

#define MB (1ULL << 20)

void checkCUDAError(const cudaError_t error, const char* file, const int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCUDAError(call, __FILE__, __LINE__)

// void writeCSVHeader(std::ofstream& file) {
//     file << "GPU_Model,Transfer_Bus,GPU_Driver,GPU_Memory_Size (MB),GPU_MEMBUS_WIDTH (Bits),Throughput_mlock_HostToDevice_GB/s,Throughput_mlock_DeviceToHost_GB/s,Throughput_hostAlloc_HostToDevice_GB/s,Throughput_hostAlloc_DeviceToHost_GB/s,Throughput_managedMalloc_HostToDevice_GB/s,Throughput_managedMalloc_DeviceToHost_GB/s,Throughput_multiGPU_mlock_HostToDevice_GB/s,Throughput_multiGPU_mlock_DeviceToHost_GB/s,Throughput_multiGPU_hostAlloc_HostToDevice_GB/s,Throughput_multiGPU_hostAlloc_DeviceToHost_GB/s,Throughput_multiGPU_managedMalloc_HostToDevice_GB/s,Throughput_multiGPU_managedMalloc_DeviceToHost_GB/s,dataSize,totalSize\n";
// }

// void writeCSVRow(std::ofstream& file, const std::string& gpuModel, const std::string& transferBus, const std::string& gpuDriver, const std::string& gpuMemSize, const std::string& gpuMemoryBusWidth, double mlockHostToDevice, double mlockDeviceToHost, double hostAllocHostToDevice, double hostAllocDeviceToHost, double managedMallocHostToDevice, double managedMallocDeviceToHost, double multiGPU_mlockHostToDevice, double multiGPU_mlockDeviceToHost, double multiGPU_hostAllocHostToDevice, double multiGPU_hostAllocDeviceToHost, double multiGPU_managedMallocHostToDevice, double multiGPU_managedMallocDeviceToHost, size_t dataSize, size_t totalSize) {
//     file << gpuModel << "," << transferBus << "," << gpuDriver << "," << gpuMemSize << "," << gpuMemoryBusWidth << ",";
//     file << mlockHostToDevice << "," << mlockDeviceToHost << "," << hostAllocHostToDevice << "," << hostAllocDeviceToHost << "," << managedMallocHostToDevice << "," << managedMallocDeviceToHost << "," << multiGPU_mlockHostToDevice << "," << multiGPU_mlockDeviceToHost << "," << multiGPU_hostAllocHostToDevice << "," << multiGPU_hostAllocDeviceToHost << "," << multiGPU_managedMallocHostToDevice << "," << multiGPU_managedMallocDeviceToHost << "," << dataSize << "," << totalSize << "\n";
// }

// std::pair<double, double> allocateMemoryAndTransfer_mlock(size_t dataSize, int numTests) {

//     double totalThroughputHostToDevice = 0.0;
//     double totalThroughputDeviceToHost = 0.0;

//     // Set the GPU device
//     CUDA_CHECK(cudaSetDevice(0));

//     // Allocate memory on device (GPU)
//     char* deviceData;
//     CUDA_CHECK(cudaMalloc(&deviceData, dataSize));

//     for (int i = 0; i < numTests; ++i) {
//         // Allocate memory on host (RAM) and lock it
//         char* hostData = static_cast<char*>(mmap(NULL, dataSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
//         if (hostData == MAP_FAILED) {
//             std::cerr << "Failed to allocate memory with mmap.\n";
//             return {0.0, 0.0};
//         }
//         if (mlock(hostData, dataSize) == -1) {
//             std::cerr << "Failed to lock memory.\n";
//             return {0.0, 0.0};
//         }

//         // Transfer data from host to device
//         auto startHostToDevice = std::chrono::high_resolution_clock::now();
//         CUDA_CHECK(cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice));
//         auto endHostToDevice = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedHostToDevice = endHostToDevice - startHostToDevice;
//         double throughputHostToDevice = static_cast<double>(dataSize) / elapsedHostToDevice.count() / (1ULL << 30); // GB/s
//         totalThroughputHostToDevice += throughputHostToDevice;

//         // Transfer data from device to host
//         auto startDeviceToHost = std::chrono::high_resolution_clock::now();
//         CUDA_CHECK(cudaMemcpy(hostData, deviceData, dataSize, cudaMemcpyDeviceToHost));
//         auto endDeviceToHost = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedDeviceToHost = endDeviceToHost - startDeviceToHost;
//         double throughputDeviceToHost = static_cast<double>(dataSize) / elapsedDeviceToHost.count() / (1ULL << 30); // GB/s
//         totalThroughputDeviceToHost += throughputDeviceToHost;

//         // Unlock and release memory
//         munlock(hostData, dataSize);
//         munmap(hostData, dataSize);
//     }

//     // Calculate average throughput
//     double averageThroughputHostToDevice = totalThroughputHostToDevice / numTests;
//     double averageThroughputDeviceToHost = totalThroughputDeviceToHost / numTests;

//     std::cout << "Average throughput from host to device (mlock): " << averageThroughputHostToDevice << " GB/s" << std::endl;
//     std::cout << "Average throughput from device to host (mlock): " << averageThroughputDeviceToHost << " GB/s" << std::endl;

//     // Free device memory
//     CUDA_CHECK(cudaFree(deviceData));
//     return std::make_pair(averageThroughputHostToDevice, averageThroughputDeviceToHost);
// }

std::pair<double, double> allocateMemoryAndTransfer_cudaHostAlloc(size_t dataSize, int numTests) {

    double totalThroughputHostToDevice = 0.0;
    double totalThroughputDeviceToHost = 0.0;

    // Set the GPU device
    CUDA_CHECK(cudaSetDevice(0));

    for (int i = 0; i < numTests; ++i) {
        // Allocate pinned memory using cudaHostAlloc
        char* hostData;
        CUDA_CHECK(cudaHostAlloc((void**)&hostData, dataSize, cudaHostAllocDefault));

        // Random data generation can be added here if needed

        // Allocate memory on device (GPU)
        char* deviceData;
        CUDA_CHECK(cudaMalloc(&deviceData, dataSize));

        // Transfer data from host to device
        auto startHostToDevice = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice));
        auto endHostToDevice = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedHostToDevice = endHostToDevice - startHostToDevice;
        double throughputHostToDevice = static_cast<double>(dataSize) / elapsedHostToDevice.count() / (1ULL << 30); // GB/s
        totalThroughputHostToDevice += throughputHostToDevice;

        // Transfer data from device to host
        auto startDeviceToHost = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(hostData, deviceData, dataSize, cudaMemcpyDeviceToHost));
        auto endDeviceToHost = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedDeviceToHost = endDeviceToHost - startDeviceToHost;
        double throughputDeviceToHost = static_cast<double>(dataSize) / elapsedDeviceToHost.count() / (1ULL << 30); // GB/s
        totalThroughputDeviceToHost += throughputDeviceToHost;

        // Free host and device memory
        CUDA_CHECK(cudaFreeHost(hostData));
        CUDA_CHECK(cudaFree(deviceData));
    }

    // Calculate average throughput
    double averageThroughputHostToDevice = totalThroughputHostToDevice / numTests;
    double averageThroughputDeviceToHost = totalThroughputDeviceToHost / numTests;

    std::cout << "Average throughput from host to device (cudaHostAlloc): " << averageThroughputHostToDevice << " GB/s" << std::endl;
    std::cout << "Average throughput from device to host (cudaHostAlloc): " << averageThroughputDeviceToHost << " GB/s" << std::endl;
    return std::make_pair(averageThroughputHostToDevice, averageThroughputDeviceToHost);
}

// std::pair<double, double> allocateMemoryAndTransfer_cudaMallocManaged(size_t dataSize, int numTests) {

//     double totalThroughputHostToDevice = 0.0;
//     double totalThroughputDeviceToHost = 0.0;

//     for (int i = 0; i < numTests; ++i) {
//         // Allocate managed memory
//         char* deviceData;
//         CUDA_CHECK(cudaMallocManaged(&deviceData, dataSize));

//         // Simulate data generation
//         std::fill_n(deviceData, dataSize, 'x');

//         // Transfer data from host to device (implicit with cudaMallocManaged)
//         auto startHostToDevice = std::chrono::high_resolution_clock::now();
//         CUDA_CHECK(cudaMemPrefetchAsync(deviceData, dataSize, 0));
//         CUDA_CHECK(cudaDeviceSynchronize());
//         auto endHostToDevice = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedHostToDevice = endHostToDevice - startHostToDevice;
//         double throughputHostToDevice = static_cast<double>(dataSize) / elapsedHostToDevice.count() / (1ULL << 30); // GB/s
//         totalThroughputHostToDevice += throughputHostToDevice;

//         // Transfer data from device to host (implicit with cudaMallocManaged)
//         auto startDeviceToHost = std::chrono::high_resolution_clock::now();
//         CUDA_CHECK(cudaMemPrefetchAsync(deviceData, dataSize, cudaCpuDeviceId));
//         CUDA_CHECK(cudaDeviceSynchronize());
//         auto endDeviceToHost = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedDeviceToHost = endDeviceToHost - startDeviceToHost;
//         double throughputDeviceToHost = static_cast<double>(dataSize) / elapsedDeviceToHost.count() / (1ULL << 30); // GB/s
//         totalThroughputDeviceToHost += throughputDeviceToHost;

//         // Free managed memory
//         CUDA_CHECK(cudaFree(deviceData));
//     }

//     // Calculate average throughput
//     double averageThroughputHostToDevice = totalThroughputHostToDevice / numTests;
//     double averageThroughputDeviceToHost = totalThroughputDeviceToHost / numTests;

//     std::cout << "Average throughput from host to device (cudaMallocManaged): " << averageThroughputHostToDevice << " GB/s" << std::endl;
//     std::cout << "Average throughput from device to host (cudaMallocManaged): " << averageThroughputDeviceToHost << " GB/s" << std::endl;
//     return std::make_pair(averageThroughputHostToDevice, averageThroughputDeviceToHost);
// }

// std::pair<double, double> multiGPUTransfer_mlock(size_t dataSize, int numTests) {

//     int deviceCount;
//     CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

//     double totalThroughputHostToDevice = 0.0;
//     double totalThroughputDeviceToHost = 0.0;

//     std::vector<char*> deviceData(deviceCount);
//     std::vector<char*> hostData(deviceCount);

//     for (int i = 0; i < deviceCount; ++i) {
//         CUDA_CHECK(cudaSetDevice(i));
//         CUDA_CHECK(cudaMalloc(&deviceData[i], dataSize));
//         hostData[i] = static_cast<char*>(mmap(NULL, dataSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
//         if (hostData[i] == MAP_FAILED) {
//             std::cerr << "Failed to allocate memory with mmap.\n";
//             return {0.0, 0.0};
//         }
//         if (mlock(hostData[i], dataSize) == -1) {
//             std::cerr << "Failed to lock memory.\n";
//             return {0.0, 0.0};
//         }
//     }

//     for (int test = 0; test < numTests; ++test) {
//         auto startHostToDevice = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaMemcpyAsync(deviceData[i], hostData[i], dataSize, cudaMemcpyHostToDevice, 0));
//         }

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaDeviceSynchronize());
//         }

//         auto endHostToDevice = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedHostToDevice = endHostToDevice - startHostToDevice;
//         double throughputHostToDevice = (static_cast<double>(dataSize) * deviceCount) / elapsedHostToDevice.count() / (1ULL << 30); // GB/s
//         totalThroughputHostToDevice += throughputHostToDevice;

//         auto startDeviceToHost = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaMemcpyAsync(hostData[i], deviceData[i], dataSize, cudaMemcpyDeviceToHost, 0));
//         }

//         // for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaDeviceSynchronize());
//         // }

//         auto endDeviceToHost = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedDeviceToHost = endDeviceToHost - startDeviceToHost;
//         double throughputDeviceToHost = (static_cast<double>(dataSize) * deviceCount) / elapsedDeviceToHost.count() / (1ULL << 30); // GB/s
//         totalThroughputDeviceToHost += throughputDeviceToHost;
//     }

//     for (int i = 0; i < deviceCount; ++i) {
//         munlock(hostData[i], dataSize);
//         munmap(hostData[i], dataSize);
//         CUDA_CHECK(cudaFree(deviceData[i]));
//     }

//     double averageThroughputHostToDevice = totalThroughputHostToDevice / numTests;
//     double averageThroughputDeviceToHost = totalThroughputDeviceToHost / numTests;

//     std::cout << "Average multi-GPU throughput from host to devices (mlock): " << averageThroughputHostToDevice << " GB/s" << std::endl;
//     std::cout << "Average multi-GPU throughput from devices to host (mlock): " << averageThroughputDeviceToHost << " GB/s" << std::endl;

//     return std::make_pair(averageThroughputHostToDevice, averageThroughputDeviceToHost);
// }

// std::pair<double, double> multiGPUTransfer_cudaHostAlloc(size_t dataSize, int numTests) {

//     int deviceCount;
//     CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

//     double totalThroughputHostToDevice = 0.0;
//     double totalThroughputDeviceToHost = 0.0;

//     std::vector<char*> deviceData(deviceCount);
//     std::vector<char*> hostData(deviceCount);

//     for (int i = 0; i < deviceCount; ++i) {
//         CUDA_CHECK(cudaSetDevice(i));
//         CUDA_CHECK(cudaMalloc(&deviceData[i], dataSize));
//         CUDA_CHECK(cudaHostAlloc(&hostData[i], dataSize, cudaHostAllocDefault));
//     }

//     for (int test = 0; test < numTests; ++test) {
//         auto startHostToDevice = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaMemcpyAsync(deviceData[i], hostData[i], dataSize, cudaMemcpyHostToDevice, 0));
//         }

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaDeviceSynchronize());
//         }

//         auto endHostToDevice = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedHostToDevice = endHostToDevice - startHostToDevice;
//         double throughputHostToDevice = (static_cast<double>(dataSize) * deviceCount) / elapsedHostToDevice.count() / (1ULL << 30); // GB/s
//         totalThroughputHostToDevice += throughputHostToDevice;

//         auto startDeviceToHost = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaMemcpyAsync(hostData[i], deviceData[i], dataSize, cudaMemcpyDeviceToHost, 0));
//         }

//         // for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaDeviceSynchronize());
//         // }

//         auto endDeviceToHost = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedDeviceToHost = endDeviceToHost - startDeviceToHost;
//         double throughputDeviceToHost = (static_cast<double>(dataSize) * deviceCount) / elapsedDeviceToHost.count() / (1ULL << 30); // GB/s
//         totalThroughputDeviceToHost += throughputDeviceToHost;
//     }

//     for (int i = 0; i < deviceCount; ++i) {
//         CUDA_CHECK(cudaFree(deviceData[i]));
//         CUDA_CHECK(cudaFreeHost(hostData[i]));
//     }

//     double averageThroughputHostToDevice = totalThroughputHostToDevice / numTests;
//     double averageThroughputDeviceToHost = totalThroughputDeviceToHost / numTests;

//     std::cout << "Average multi-GPU throughput from host to devices (cudaHostAlloc): " << averageThroughputHostToDevice << " GB/s" << std::endl;
//     std::cout << "Average multi-GPU throughput from devices to host (cudaHostAlloc): " << averageThroughputDeviceToHost << " GB/s" << std::endl;

//     return std::make_pair(averageThroughputHostToDevice, averageThroughputDeviceToHost);
// }

// std::pair<double, double> multiGPUTransfer_cudaMallocManaged(size_t dataSize, int numTests) {

//     int deviceCount;
//     CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

//     double totalThroughputHostToDevice = 0.0;
//     double totalThroughputDeviceToHost = 0.0;

//     std::vector<char*> deviceData(deviceCount);

//     for (int i = 0; i < deviceCount; ++i) {
//         CUDA_CHECK(cudaSetDevice(i));
//         CUDA_CHECK(cudaMallocManaged(&deviceData[i], dataSize));
//         std::fill_n(deviceData[i], dataSize, 'x');
//     }

//     for (int test = 0; test < numTests; ++test) {
//         auto startHostToDevice = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaMemPrefetchAsync(deviceData[i], dataSize, i));
//         }

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaDeviceSynchronize());
//         }

//         auto endHostToDevice = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedHostToDevice = endHostToDevice - startHostToDevice;
//         double throughputHostToDevice = (static_cast<double>(dataSize) * deviceCount) / elapsedHostToDevice.count() / (1ULL << 30); // GB/s
//         totalThroughputHostToDevice += throughputHostToDevice;

//         auto startDeviceToHost = std::chrono::high_resolution_clock::now();

//         for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaMemPrefetchAsync(deviceData[i], dataSize, cudaCpuDeviceId));
//         }

//         // for (int i = 0; i < deviceCount; ++i) {
//             CUDA_CHECK(cudaDeviceSynchronize());
//         // }

//         auto endDeviceToHost = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsedDeviceToHost = endDeviceToHost - startDeviceToHost;
//         double throughputDeviceToHost = (static_cast<double>(dataSize) * deviceCount) / elapsedDeviceToHost.count() / (1ULL << 30); // GB/s
//         totalThroughputDeviceToHost += throughputDeviceToHost;
//     }

//     for (int i = 0; i < deviceCount; ++i) {
//         CUDA_CHECK(cudaFree(deviceData[i]));
//     }

//     double averageThroughputHostToDevice = totalThroughputHostToDevice / numTests;
//     double averageThroughputDeviceToHost = totalThroughputDeviceToHost / numTests;

//     std::cout << "Average multi-GPU throughput from host to devices (cudaMallocManaged): " << averageThroughputHostToDevice << " GB/s" << std::endl;
//     std::cout << "Average multi-GPU throughput from devices to host (cudaMallocManaged): " << averageThroughputDeviceToHost << " GB/s" << std::endl;

//     return std::make_pair(averageThroughputHostToDevice, averageThroughputDeviceToHost);
// }

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataSize_MB> <totalSize_MB>" << std::endl;
        return 1;
    }

    size_t dataSize_MB = std::stoull(argv[1]);
    size_t totalSize_MB = std::stoull(argv[2]);

    size_t dataSize = dataSize_MB * MB;
    size_t totalSize = totalSize_MB * MB;
    int numTests = totalSize / dataSize;
    // std::ifstream fileCheck("multi_gpu_throughput.csv");
    // bool fileExists = fileCheck.good();
    // fileCheck.close();

    // std::ofstream csvFile("multi_gpu_throughput.csv", fileExists ? std::ios_base::app : std::ios_base::out); // Open for append if file exists, otherwise create new

    // if (csvFile.is_open()) {
    //     if (!fileExists) {
    //         writeCSVHeader(csvFile);
    //         std::cout << "CSV file created with headers." << std::endl;
    //     }
    // }
    
    // cudaDeviceProp deviceProp;
    // CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    // std::string gpuModel = deviceProp.name;
    // std::string transferBus = (deviceProp.pciBusID == 0 ? "PCI" : "Other"); // Assuming PCI is the transfer bus
    // std::string gpuDriver =  std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor);
    // std::string gpuMemSize = std::to_string(deviceProp.totalGlobalMem / MB);
    // std::string gpuMemoryBusWidth = std::to_string(deviceProp.memoryBusWidth);
    
    // auto mlockThroughput = allocateMemoryAndTransfer_mlock(dataSize, numTests);
    auto hostAllocThroughput = allocateMemoryAndTransfer_cudaHostAlloc(dataSize, numTests);
    // auto managedMallocThroughput = allocateMemoryAndTransfer_cudaMallocManaged(dataSize, numTests);

    // auto multiGPUMlockThroughput = multiGPUTransfer_mlock(dataSize, numTests);
    // auto multiGPUHostAllocThroughput = multiGPUTransfer_cudaHostAlloc(dataSize, numTests);
    // auto multiGPUManagedMallocThroughput = multiGPUTransfer_cudaMallocManaged(dataSize, numTests);

    // writeCSVRow(csvFile, gpuModel, transferBus, gpuDriver, gpuMemSize, gpuMemoryBusWidth,
    //             mlockThroughput.first, mlockThroughput.second,
    //             hostAllocThroughput.first, hostAllocThroughput.second,
    //             managedMallocThroughput.first, managedMallocThroughput.second,
    //             multiGPUMlockThroughput.first, multiGPUMlockThroughput.second,
    //             multiGPUHostAllocThroughput.first, multiGPUHostAllocThroughput.second,
    //             multiGPUManagedMallocThroughput.first, multiGPUManagedMallocThroughput.second,
    //             dataSize_MB, totalSize_MB);

    // csvFile.close();

    return 0;
}
