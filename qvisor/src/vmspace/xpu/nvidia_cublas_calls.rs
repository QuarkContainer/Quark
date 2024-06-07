use crate::qlib::common::*;
use crate::qlib::proxy::*;
use crate::qlib::config::*;
use crate::xpu::cuda_api::*;
use std::os::raw::*;

use cuda11_cublasLt_sys::{
    cublasLtHandle_t, cublasLtMatmulAlgo_t, cublasLtMatmulDesc_t, cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t, cublasLtMatrixLayout_t,
};

use cuda_runtime_sys::cudaStream_t;
use rcublas_sys::cublasHandle_t;

use super::cuda::BLASHANDLE;
use super::cuda::STREAMS;
use crate::nvidia::{MEM_MANAGER};
use crate::{QUARK_CONFIG};

pub fn CublasCreateV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasCreateV2");
    let mut handle: u64 = 0;

    let ret = unsafe { cublasCreate_v2(&mut handle) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cublasCreate_v2: {}", ret as u32);
    }

    unsafe { *(parameters.para1 as *mut u64) = handle.clone() };
    BLASHANDLE.lock().insert(handle.clone(), handle.clone());
    return Ok(ret as u32);
}
pub fn CublasDestroyV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasDestroyV2");
    let handle = match BLASHANDLE.lock().get(&parameters.para1) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let ret = unsafe { cublasDestroy_v2(handle as cublasHandle_t) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasDestroy_v2: {}",
            ret as u32
        );
    }

    BLASHANDLE.lock().remove(&parameters.para1);
    return Ok(ret as u32);
}
pub fn CublasSetStreamV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSetStreamV2");
    let stream = match STREAMS.lock().get(&parameters.para2) {
        Some(s)=> s.clone(),
        None => panic!(),
    };
    let handle = match BLASHANDLE.lock().get(&parameters.para1) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let ret = unsafe {
        cublasSetStream_v2(
            handle as cublasHandle_t,
            stream,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasSetStream_v2: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CublasSetWorkspaceV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSetWorkspaceV2");
    let handle = match BLASHANDLE.lock().get(&parameters.para1) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let ret = unsafe {
        cublasSetWorkspace_v2(
            handle as cublasHandle_t,
            parameters.para2 as u64,
            parameters.para3 as usize,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasSetWorkspace_v2: {}",
            ret as u32
        );
    }

    if QUARK_CONFIG.lock().CudaMemType == CudaMemType::MemPool {
        match MEM_MANAGER
            .lock()
            .ctxManager
            .cublasStatus
            .get_mut(&handle) {
                Some(status) => {
                    (*status).workspacePtr = parameters.para2.clone();
                    (*status).workspaceSize = parameters.para3.clone() as usize;
                },
                None => panic!(),
            }
    }
    return Ok(ret as u32);
}
pub fn CublasSetMathMode(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSetMathMode");
    let handle = match BLASHANDLE.lock().get(&parameters.para1) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let ret = unsafe {
        cublasSetMathMode(handle as cublasHandle_t, parameters.para2 as u32)
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasSetMathMode: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CublasSgemmStridedBatched(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSgemmStridedBatched");
    let info =
        unsafe { *(parameters.para1 as *const u8 as *const SgemmStridedBatchedInfo) };
    let handle = match BLASHANDLE.lock().get(&info.handle) {
        Some(handle)=> handle.clone(),
        None => 0,
    };
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasSgemmStridedBatched(
            handle as cublasHandle_t,
            info.transa,
            info.transb,
            info.m,
            info.n,
            info.k,
            &alpha,
            info.A,
            info.lda,
            info.strideA,
            info.B,
            info.ldb,
            info.strideB,
            &beta,
            info.C,
            info.ldc,
            info.strideC,
            info.batchCount,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasSgemmStridedBatched: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CublasLtMatmul(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasLtMatmul");
    let info = unsafe { *(parameters.para1 as *const u8 as *const CublasLtMatmulInfo) };

    let handle = match BLASHANDLE.lock().get(&info.lightHandle) {
        Some(handle)=> handle.clone(),
        None => 0,
    };
    let alpha = unsafe { *(parameters.para2 as *const f64) };
    let beta = unsafe { *(parameters.para3 as *const f64) };
    let ret = unsafe {
        cublasLtMatmul(
            handle as cublasLtHandle_t,
            info.computeDesc as cublasLtMatmulDesc_t,
            &alpha,
            info.A,
            info.Adesc as cublasLtMatrixLayout_t,
            info.B,
            info.Bdesc as cublasLtMatrixLayout_t,
            &beta,
            info.C,
            info.Cdesc as cublasLtMatrixLayout_t,
            info.D,
            info.Ddesc as cublasLtMatrixLayout_t,
            info.algo as *const cublasLtMatmulAlgo_t,
            info.workspace,
            info.workspaceSizeInBytes,
            info.stream as cudaStream_t,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cublasLtMatmul: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CublasLtMatmulAlgoGetHeuristic(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasLtMatmulAlgoGetHeuristic");
    let info = unsafe {
        *(parameters.para1 as *const u8 as *const CublasLtMatmulAlgoGetHeuristicInfo)
    };
    let handle = match BLASHANDLE.lock().get(&info.lightHandle) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let mut heuristicResultsArray: Vec<cublasLtMatmulHeuristicResult_t> =
        Vec::with_capacity(info.requestedAlgoCount as usize);
    unsafe {
        heuristicResultsArray.set_len(info.requestedAlgoCount as usize);
    };
    let mut returnAlgoCount: c_int = 0;

    let ret = unsafe {
        cublasLtMatmulAlgoGetHeuristic(
            handle as cublasLtHandle_t,
            info.operationDesc as cublasLtMatmulDesc_t,
            info.Adesc as cublasLtMatrixLayout_t,
            info.Bdesc as cublasLtMatrixLayout_t,
            info.Cdesc as cublasLtMatrixLayout_t,
            info.Ddesc as cublasLtMatrixLayout_t,
            info.preference as cublasLtMatmulPreference_t,
            info.requestedAlgoCount,
            &mut heuristicResultsArray[0] as *mut cublasLtMatmulHeuristicResult_t,
            &mut returnAlgoCount,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasLtMatmulAlgoGetHeuristic: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para3 as *mut _) = returnAlgoCount };
    for i in 0..returnAlgoCount as u64 {
        unsafe {
            (*((parameters.para2 + i) as *mut u8 as *mut cublasLtMatmulHeuristicResult_t)) =
                heuristicResultsArray[i as usize]
        };
    }
    return Ok(ret as u32);
}
pub fn CublasGetMathMode(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasGetMathMode");
    let mut mode: u32 = 0;
    let handle = match BLASHANDLE.lock().get(&parameters.para1) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };

    let ret = unsafe { cublasGetMathMode(handle as u64, &mut mode) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasGetMathMode: {}",
            ret as u32
        );
    }

    unsafe { *(parameters.para2 as *mut u32) = mode as u32 };
    return Ok(ret as u32);
}

pub fn CublasSgemmV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSgemm_v2");
    let info = unsafe { *(parameters.para1 as *const u8 as *const CublasSgemmV2Info) };
    let handle = match BLASHANDLE.lock().get(&info.handle) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };

    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasSgemm_v2(
            handle as cublasHandle_t,
            info.transa,
            info.transb,
            info.m,
            info.n,
            info.k,
            &alpha,
            info.A,
            info.lda,
            info.B,
            info.ldb,
            &beta,
            info.C,
            info.ldc,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cublasSgemm_v2: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CublasGemmEx(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSgemm_v2");
    let info = unsafe { *(parameters.para1 as *const u8 as *const GemmExInfo) };
    let handle = match BLASHANDLE.lock().get(&info.handle) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasGemmEx(
            handle as cublasHandle_t,
            info.transa,
            info.transb,
            info.m,
            info.n,
            info.k,
            &alpha as *const _ as u64,
            info.A,
            info.Atype,
            info.lda,
            info.B,
            info.Btype,
            info.ldb,
            &beta as *const _ as u64,
            info.C,
            info.Ctype,
            info.ldc,
            info.computeType,
            info.algo,
        )
    };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cublasGemmEx: {}", ret as u32);
    }

    return Ok(ret as u32);
}
pub fn CublasGemmStridedBatchedEx(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSgemm_v2");
    let info =
        unsafe { *(parameters.para1 as *const u8 as *const GemmStridedBatchedExInfo) };
    let handle = match BLASHANDLE.lock().get(&info.handle) {
        Some(handle)=> handle.clone(),
        None => panic!(),
    };
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasGemmStridedBatchedEx(
            handle as cublasHandle_t,
            info.transa,
            info.transb,
            info.m,
            info.n,
            info.k,
            &alpha as *const _ as u64,
            info.A,
            info.Atype,
            info.lda,
            info.strideA,
            info.B,
            info.Btype,
            info.ldb,
            info.strideB,
            &beta as *const _ as u64,
            info.C,
            info.Ctype,
            info.ldc,
            info.strideC,
            info.batchCount,
            info.computeType,
            info.algo,
        )
    };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasGemmStridedBatchedEx: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
