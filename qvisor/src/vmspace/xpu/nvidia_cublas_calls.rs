use crate::qlib::common::*;
use crate::qlib::proxy::*;
use crate::xpu::cuda_api::*;
use std::os::raw::*;

use cuda11_cublasLt_sys::{
    cublasLtHandle_t, cublasLtMatmulAlgo_t, cublasLtMatmulDesc_t, cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreference_t, cublasLtMatrixLayout_t,
};

use cuda_runtime_sys::cudaStream_t;
use rcublas_sys::cublasHandle_t;

pub fn CublasCreateV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasCreateV2");
    let mut handle: u64 = 0;

    let ret = unsafe { cublasCreate_v2(&mut handle) };
    if ret as u32 != 0 {
        error!("nvidia.rs: error caused by cublasCreate_v2: {}", ret as u32);
    }

    unsafe { *(parameters.para1 as *mut u64) = handle };
    return Ok(ret as u32);
}
pub fn CublasDestroyV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasDestroyV2");
    let ret = unsafe { cublasDestroy_v2(parameters.para1 as cublasHandle_t) };
    if ret as u32 != 0 {
        error!(
            "nvidia.rs: error caused by cublasDestroy_v2: {}",
            ret as u32
        );
    }

    return Ok(ret as u32);
}
pub fn CublasSetStreamV2(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSetStreamV2");
    let ret = unsafe {
        cublasSetStream_v2(
            parameters.para1 as cublasHandle_t,
            parameters.para2 as u64,
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
    let ret = unsafe {
        cublasSetWorkspace_v2(
            parameters.para1 as cublasHandle_t,
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

    return Ok(ret as u32);
}
pub fn CublasSetMathMode(parameters: &ProxyParameters) -> Result<u32> {
    //error!("nvidia.rs: CublasSetMathMode");
    let ret = unsafe {
        cublasSetMathMode(parameters.para1 as cublasHandle_t, parameters.para2 as u32)
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
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasSgemmStridedBatched(
            info.handle as cublasHandle_t,
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

    let alpha = unsafe { *(parameters.para2 as *const f64) };
    let beta = unsafe { *(parameters.para3 as *const f64) };
    let ret = unsafe {
        cublasLtMatmul(
            info.lightHandle as cublasLtHandle_t,
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
    let mut heuristicResultsArray: Vec<cublasLtMatmulHeuristicResult_t> =
        Vec::with_capacity(info.requestedAlgoCount as usize);
    unsafe {
        heuristicResultsArray.set_len(info.requestedAlgoCount as usize);
    };
    let mut returnAlgoCount: c_int = 0;

    let ret = unsafe {
        cublasLtMatmulAlgoGetHeuristic(
            info.lightHandle as cublasLtHandle_t,
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

    let ret = unsafe { cublasGetMathMode(parameters.para1 as u64, &mut mode) };
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
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasSgemm_v2(
            info.handle as cublasHandle_t,
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
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasGemmEx(
            info.handle as cublasHandle_t,
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
    let alpha = unsafe { *(parameters.para2 as *const f32) };
    let beta = unsafe { *(parameters.para3 as *const f32) };

    let ret = unsafe {
        cublasGemmStridedBatchedEx(
            info.handle as cublasHandle_t,
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
