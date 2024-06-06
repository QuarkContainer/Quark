use crate::qlib::proxy::*;
use std::os::raw::*;

use cuda11_cublasLt_sys::{
    cublasLtHandle_t, cublasLtMatmulAlgo_t, cublasLtMatmulDesc_t, cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t, cublasLtMatrixLayout_t,
};
use cuda_driver_sys::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUfunction_attribute, CUmodule, CUresult,
    CUstream,
};
use cuda_runtime_sys::{
    cudaDeviceAttr, cudaDeviceP2PAttr, cudaDeviceProp, cudaError_t, cudaEvent_t, cudaFuncAttribute,
    cudaFuncAttributes, cudaFuncCache, cudaLimit, cudaMemoryAdvise, cudaSharedMemConfig,
    cudaStreamCaptureMode, cudaStreamCaptureStatus, cudaStream_t,
};
use rcublas_sys::{cublasHandle_t, cudaMemLocation};

#[link(name = "nccl")]
extern "C" {
    pub fn ncclGetVersion(version: *mut ::std::os::raw::c_int) -> NcclResultT;
    pub fn ncclGetUniqueId(handle: *mut NcclUniqueId) -> NcclResultT;
    pub fn ncclCommInitRank(
        comm: *mut NcclCommT,
        nRanks: ::std::os::raw::c_int,
        commId: NcclUniqueId,
        rank: ::std::os::raw::c_int,
    ) -> NcclResultT;
    pub fn ncclCommDestroy(comm: NcclCommT) -> NcclResultT;
    pub fn ncclCommInitAll(
        comms: *mut NcclCommT,
        ndevs: c_int,
        devs: *const c_int,
    ) -> NcclResultT;
    pub fn ncclCommAbort(comm: NcclCommT) -> NcclResultT;
    pub fn ncclCommCuDevice(comm: NcclCommT, device: *mut ::std::os::raw::c_int) -> NcclResultT;

    pub fn ncclGetErrorString(
        error: u32,
    ) -> *const c_char;
    pub fn ncclCommGetAsyncError(
        comm: NcclCommT,
        async_error: *mut NcclResultT,
    ) -> NcclResultT;
    pub fn ncclCommInitRankConfig(
        comm: *mut NcclCommT,
        n_rank: c_int,
        ncclUniqueId_: NcclUniqueId,
        rank: c_int,
        ncclConfig_t: *const NcclConfig
    ) -> NcclResultT;
    pub fn ncclCommCount(comm: NcclCommT, count: *mut c_int) -> NcclResultT;
    pub fn ncclCommUserRank(comm: NcclCommT, rank: *mut c_int) -> NcclResultT;
    pub fn ncclSend(sendbuff: *const c_void, count: usize, datatype: NcclDataTypeT, peer: c_int, comm: NcclCommT, stream: cudaStream_t) -> NcclResultT;
    pub fn ncclRecv(recvbuff: *mut c_void, count: usize, datatype: NcclDataTypeT, peer: c_int, comm: NcclCommT, stream: cudaStream_t) -> NcclResultT;
    pub fn ncclGroupStart() -> NcclResultT;
    pub fn ncclGroupEnd() -> NcclResultT;
    pub fn ncclAllReduce(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: NcclDataTypeT, op: NcclRedOpT, comm: NcclCommT, stream: cudaStream_t) -> NcclResultT;
    pub fn ncclAllGather(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: NcclDataTypeT, comm: NcclCommT, stream: cudaStream_t) -> NcclResultT;
    pub fn ncclReduceScatter(sendbuff: *const c_void, recvbuff: *mut c_void, count: usize, datatype: NcclDataTypeT, op: NcclRedOpT, comm: NcclCommT, stream: cudaStream_t) -> NcclResultT;
}

#[link(name = "cuda")]
extern "C" {
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut c_uint,
        active: *mut c_int,
    ) -> CUresult;
    pub fn cuInit(Flags: c_uint) -> CUresult;

    pub fn cuModuleGetLoadingMode(mode: *mut CumoduleLoadingModeEnum) -> u32;
    pub fn cuCtxCreate(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxPushCurrent(pctx: CUcontext) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuFuncGetAttribute(
        pi: *mut c_int,
        attrib: CUfunction_attribute,
        hfunc: CUfunction,
    ) -> CUresult;
    pub fn cuFuncSetAttribute(hfunc: CUfunction, attrib: u32, value: i32) -> CUresult;
    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut c_int,
        func: CUfunction,
        blockSize: c_int,
        dynamicSMemSize: usize,
        flags: c_uint,
    ) -> CUresult;

    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    pub fn cuMemGetAllocationGranularity(granularity: u64, prop: u64, option: usize) -> CUresult;
    pub fn cuMemAddressReserve(
        ptr: u64,
        size: usize,
        alignment: usize,
        addr: u64,
        flags: c_ulonglong,
    ) -> CUresult;
    pub fn cuMemCreate(handle: u64, size: usize, prop: u64, flags: c_ulonglong) -> CUresult;
    pub fn cuMemMap(
        ptr: u64,
        size: usize,
        offset: usize,
        handle: u64,
        flags: c_ulonglong,
    ) -> CUresult;
    pub fn cuMemSetAccess(ptr: u64, size: usize, desc: u64, count: usize) -> CUresult;
    pub fn cuMemUnmap(ptr: u64, size: usize) -> CUresult;
    pub fn cuMemRelease(handle: u64) -> CUresult;
    pub fn cuMemAddressFree(ptr: u64, size: usize) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: u64) -> CUresult;
    pub fn cuDevicePrimaryCtxReset(device: CUdevice) -> CUresult;
    pub fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut u32) -> CUresult;
}

#[link(name = "cudart")]
extern "C" {
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: u64, src: u64, count: u64, kind: u64) -> u32;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceGetStreamPriorityRange(
        leastPriority: *mut c_int,
        greatestPriority: *mut c_int,
    ) -> cudaError_t;
    pub fn cudaStreamIsCapturing(
        stream: cudaStream_t,
        pCaptureStatus: *mut cudaStreamCaptureStatus,
    ) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
    pub fn cudaMemcpyAsync(dst: u64, src: u64, count: u64, kind: u64, stream: cudaStream_t) -> u32;
    pub fn cudaHostAlloc(pHost: u64, size: usize, flags: u32) -> u32;
    pub fn cudaHostRegister(ptr: u64, size: usize, flags: u32) -> cudaError_t;
    pub fn cudaMallocManaged(devPtr: u64, size: usize, flags: u32) -> cudaError_t;
    pub fn cudaMemAdvise_v2(
        devPtr: u64,
        count: usize,
        advice: cudaMemoryAdvise,
        location: cudaMemLocation,
    ) -> cudaError_t;
    pub fn cudaMemPrefetchAsync(
        devPtr: u64,
        count: usize,
        dstDevice: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaFreeHost(ptr: u64) -> u32;
    pub fn cudaHostUnregister(ptr: u64) -> cudaError_t;

    pub fn cudaChooseDevice(device: *mut c_int, prop: *const cudaDeviceProp) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(
        value: *mut c_int,
        attr: cudaDeviceAttr,
        device: c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetByPCIBusId(device: *mut c_int, pciBusId: *const c_char) -> cudaError_t;
    pub fn cudaDeviceGetCacheConfig(pCacheConfig: u64) -> cudaError_t;
    pub fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
    pub fn cudaDeviceGetP2PAttribute(
        value: *mut c_int,
        attr: cudaDeviceP2PAttr,
        srcDevice: c_int,
        dstDevice: c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, device: c_int) -> cudaError_t;
    pub fn cudaDeviceGetSharedMemConfig(pConfig: u64) -> cudaError_t;
    pub fn cudaDeviceSetCacheConfig(cacheConfig: u32) -> cudaError_t;
    pub fn cudaSetDeviceFlags(flags: c_uint) -> cudaError_t;
    pub fn cudaDeviceReset() -> cudaError_t;

    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
    pub fn cudaStreamCreateWithPriority(
        pStream: *mut cudaStream_t,
        flags: c_uint,
        priority: c_int,
    ) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamGetFlags(hStream: cudaStream_t, flags: *mut c_uint) -> cudaError_t;
    pub fn cudaStreamGetPriority(hStream: cudaStream_t, priority: *mut c_int) -> cudaError_t;
    pub fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: c_uint,
    ) -> cudaError_t;
    pub fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t;
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void)
        -> cudaError_t;
    pub fn cudaFuncSetAttribute(
        func: *const c_void,
        attr: cudaFuncAttribute,
        value: c_int,
    ) -> cudaError_t;
    pub fn cudaFuncSetCacheConfig(func: *const c_void, cacheConfig: cudaFuncCache) -> cudaError_t;
    pub fn cudaFuncSetSharedMemConfig(
        func: *const c_void,
        config: cudaSharedMemConfig,
    ) -> cudaError_t;

    pub fn cudaGetErrorString(error: u32) -> *const c_char;
    pub fn cudaGetErrorName(error: u32) -> *const c_char;
    pub fn cudaPeekAtLastError() -> cudaError_t;
    pub fn cudaDeviceSetLimit(limit: usize, value: usize) -> cudaError_t;
    pub fn cudaDeviceSetSharedMemConfig(config: u64) -> cudaError_t;
    pub fn cudaGetDeviceFlags(flags: *mut c_uint) -> cudaError_t;
    pub fn cudaSetValidDevices(device_arr: *mut c_int, len: c_int) -> cudaError_t;

    pub fn cudaMemset(devPtr: *const c_void, value: c_int, count: usize) -> cudaError_t;
    pub fn cudaMemsetAsync(
        devPtr: *const c_void,
        value: c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut c_int,
        func: *const c_void,
        blockSize: c_int,
        dynamicSMemSize: usize,
        flags: c_uint,
    ) -> cudaError_t;
}

#[link(name = "nvidia-ml")]
extern "C" {
    pub fn nvmlInitWithFlags(flags: c_uint) -> u32;
    pub fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> u32;
    pub fn nvmlInit() -> u32;
    pub fn nvmlInit_v2() -> u32;
    pub fn nvmlShutdown() -> u32;
}
#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> u32;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> u32;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> u32;
    pub fn cublasSetWorkspace_v2(
        handle: cublasHandle_t,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
    ) -> u32;
    pub fn cublasSetMathMode(handle: cublasHandle_t, mode: u32) -> u32;
    pub fn cublasSgemmStridedBatched(
        handle: cublasHandle_t,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        strideA: c_longlong,
        B: *const f32,
        ldb: c_int,
        strideB: c_longlong,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
        strideC: c_longlong,
        batchCount: c_int,
    ) -> u32;
    pub fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut u32) -> u32;
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        B: *const f32,
        ldb: c_int,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
    ) -> u32;
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: u32,
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: u64,
        A: u64,
        Atype: u32,
        lda: c_int,
        B: u64,
        Btype: u32,
        ldb: c_int,
        beta: u64,
        C: u64,
        Ctype: u32,
        ldc: c_int,
        computeType: u32,
        algo: u32,
    ) -> u32;
    pub fn cublasGemmStridedBatchedEx(
        handle: cublasHandle_t,
        transa: u32,
        transb: u32,
        m: i32,
        n: i32,
        k: i32,
        alpha: u64,
        A: u64,
        Atype: u32,
        lda: i32,
        strideA: i64,
        B: u64,
        Btype: u32,
        ldb: i32,
        strideB: i64,
        beta: u64,
        C: u64,
        Ctype: u32,
        ldc: i32,
        strideC: i64,
        batchCount: i32,
        computeType: u32,
        algo: u32,
    ) -> u32;
    pub fn cublasGetStream(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> u32;
}

#[link(name = "cublasLt")]
extern "C" {
    pub fn cublasLtMatmul(
        lightHandle: cublasLtHandle_t,
        computeDesc: cublasLtMatmulDesc_t,
        alpha: *const f64,
        A: *const c_void,
        Adesc: cublasLtMatrixLayout_t,
        B: *const c_void,
        Bdesc: cublasLtMatrixLayout_t,
        beta: *const f64,
        C: *const c_void,
        Cdesc: cublasLtMatrixLayout_t,
        D: *mut c_void,
        Ddesc: cublasLtMatrixLayout_t,
        algo: *const cublasLtMatmulAlgo_t,
        workspace: *mut c_void,
        workspaceSizeInBytes: usize,
        stream: cudaStream_t,
    ) -> u32;
    pub fn cublasLtMatmulAlgoGetHeuristic(
        lightHandle: cublasLtHandle_t,
        operationDesc: cublasLtMatmulDesc_t,
        Adesc: cublasLtMatrixLayout_t,
        Bdesc: cublasLtMatrixLayout_t,
        Cdesc: cublasLtMatrixLayout_t,
        Ddesc: cublasLtMatrixLayout_t,
        preference: cublasLtMatmulPreference_t,
        requestedAlgoCount: c_int,
        heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
        returnAlgoCount: *mut c_int,
    ) -> u32;
}
