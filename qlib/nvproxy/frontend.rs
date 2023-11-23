// Copyright (c) 2021 Quark Container Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use alloc::vec::Vec;

use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::task::Task;
use crate::qlib::common::*;
use crate::qlib::linux::ioctl::*;
use crate::qlib::linux_def::SysErr;
use crate::qlib::nvproxy::frontend_type::*;
use crate::qlib::nvproxy::nvgpu::*;

use super::frontendfd::*;

pub fn FrontendIoctlCmd(nr: u32, argSize: u32) -> u64 {
    return IOCBits::IOWR(NV_IOCTL_MAGIC, nr, argSize) as u64
}

// frontendIoctlState holds the state of a call to NvFrontendFileOptions.Ioctl().
pub struct FrontendIoctlState <'a> {
    pub fd: NvFrontendFileOptions,              
	pub task: &'a Task,
    pub nr: u32,
    pub ioctlParamsAddr: u64,
	pub ioctlParamsSize: u32,
}

pub fn FrontendIoctlInvokePtr(fi: &FrontendIoctlState, paramsAddr: u64) -> Result<u64> {
    let n = HostSpace::IoCtl(
        fi.fd.fd, 
        FrontendIoctlCmd(fi.nr, fi.ioctlParamsSize), 
        paramsAddr
    ); 
    if n < 0 {
        return Err(Error::SysError(n as i32));
    }

    return Ok(n as u64)
}

pub fn FrontendIoctlInvoke<Params: Sized>(
    fi: &FrontendIoctlState, 
    params: Option<&Params>
) -> Result<u64> {
    let paramsAddr = match params {
        None => 0,
        Some(p) => p as * const _ as u64
    };

    let n = HostSpace::IoCtl(
        fi.fd.fd, 
        FrontendIoctlCmd(fi.nr, fi.ioctlParamsSize), 
        paramsAddr
    ); 
    if n < 0 {
        return Err(Error::SysError(n as i32));
    }

    return Ok(n as u64)
}

pub fn RMControlInvoke(
    fi: &FrontendIoctlState, 
    ioctlParams: &NVOS54Parameters, 
    ctrlParamsAddr: P64
) -> Result<u64> {
    let mut ioctlParamsTmp = *ioctlParams;
    ioctlParamsTmp.params = ctrlParamsAddr;

    let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;
    let mut outIoctlParams = ioctlParamsTmp;
    outIoctlParams.params = ioctlParams.params;
    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    return Ok(n)
}

pub fn CtrlClientSystemGetBuildVersionInvoke(
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS54Parameters,
    ctrlParams: &Nv0000CtrlSystemGetBuildVersionParams, 
    driverVersionBuf: P64, 
    versionBuf: P64,
    titleBuf: P64
) -> Result<u64> {
    let mut ctrlParamsTmp = *ctrlParams;
    ctrlParamsTmp.driverVersionBuffer = driverVersionBuf; // as * const _ as P64;
    ctrlParamsTmp.versionBuffer = versionBuf;// as * const _ as P64;
    ctrlParamsTmp.titleBuffer = titleBuf;// as * const _ as P64;
    let n = RMControlInvoke(fi, ioctlParams, &ctrlParamsTmp as * const _ as u64)?;
    let mut outCtrlParams = ctrlParamsTmp;
    outCtrlParams.driverVersionBuffer = ctrlParams.driverVersionBuffer;
    outCtrlParams.versionBuffer = ctrlParams.versionBuffer;
    outCtrlParams.titleBuffer = ctrlParams.titleBuffer;

    fi.task.CopyOutObj(&outCtrlParams, ioctlParams.params)?;
    return Ok(n)
}

pub fn CtrlDevFIFOGetChannelList(
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS54Parameters,
) -> Result<u64> {
    if ioctlParams.paramsSize as usize != core::mem::size_of::<Nv0080CtrlFifoGetChannellistParams>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ctrlParams: Nv0080CtrlFifoGetChannellistParams = fi.task.CopyInObj(ioctlParams.params)?;
    if ctrlParams.numChannels == 0 {
        // Compare
		// src/nvidia/src/kernel/gpu/fifo/kernel_fifo_ctrl.c:deviceCtrlCmdFifoGetChannelList_IMPL().
		return Err(Error::SysError(SysErr::EINVAL));
    }

    let channelHandleList: Vec<u32> = fi.task.CopyInVec(ctrlParams.channelHandleList, ctrlParams.numChannels as usize)?;
    let channelList: Vec<u32> = fi.task.CopyInVec(ctrlParams.channelList, ctrlParams.numChannels as usize)?;

    let mut ctrlParamsTmp = ctrlParams;
    ctrlParamsTmp.channelHandleList = &channelHandleList[0] as * const _ as u64;
    ctrlParamsTmp.channelList = &channelList[0] as * const _ as u64;

    let n = RMControlInvoke(fi, ioctlParams, &ctrlParamsTmp as * const _ as u64)?;

    fi.task.CopyOutSlice(&channelHandleList, ctrlParams.channelHandleList, ctrlParams.numChannels as usize)?;
    fi.task.CopyOutSlice(&channelList, ctrlParams.channelList, ctrlParams.numChannels as usize)?;
    
    return Ok(n);
}

pub fn CtrlSubdevGRGetInfo(
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS54Parameters,
) -> Result<u64> {
    if ioctlParams.paramsSize as usize != core::mem::size_of::<Nv2080CtrlGrGetInfoParams>() {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let ctrlParams: Nv2080CtrlGrGetInfoParams  = fi.task.CopyInObj(ioctlParams.params)?;

    if ctrlParams.GRInfoListSize == 0 {
        // Compare
		// src/nvidia/src/kernel/gpu/gr/kernel_graphics.c:_kgraphicsCtrlCmdGrGetInfoV2().
        return Err(Error::SysError(SysErr::EINVAL));
    }

    let len = core::mem::size_of::<NvxxxxCtrlXxxInfo>() * ctrlParams.GRInfoListSize as usize;
    let infoList: Vec<u8> = fi.task.CopyInVec(ctrlParams.GRInfoList, len)?;

    let mut ctrlParamsTmp = ctrlParams;
    ctrlParamsTmp.GRInfoList = &infoList[0] as * const _ as u64;

    let n = RMControlInvoke(fi, ioctlParams, &ctrlParamsTmp as * const _ as u64)?;

    fi.task.CopyOutSlice(&infoList, ctrlParams.GRInfoList, len)?;

    return Ok(n)
}

pub fn RMAllocInvokeInner<AllocParams: RmAllocParamType + Copy + Clone>(
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS64ParametersV535,
    allocParams: P64
) -> Result<u64> {
    let mut ioctlParamsTmp = AllocParams::FromOS64V535(ioctlParams);
    ioctlParamsTmp.SetPAllocParms(allocParams);

    let mut rightsRequested : RsAccessMask = RsAccessMask::default();
    if ioctlParams.rightsRequested != 0 {
        rightsRequested = fi.task.CopyInObj(ioctlParams.rightsRequested)?;
        ioctlParamsTmp.SetPRightsRequested(&rightsRequested as * const _ as u64);
    }

    let n = FrontendIoctlInvokePtr(fi, &ioctlParamsTmp as * const _ as u64)?;
    if ioctlParams.rightsRequested != 0 {
        fi.task.CopyOutObj(&rightsRequested, ioctlParams.rightsRequested)?;
    }

    ioctlParamsTmp.SetPAllocParms(ioctlParams.allocParms);
    if ioctlParams.rightsRequested != 0 {
        ioctlParamsTmp.SetPRightsRequested(ioctlParams.rightsRequested);
    }

    fi.task.CopyOutObj(&ioctlParamsTmp, fi.ioctlParamsAddr)?;
    return Ok(n)
}

pub fn RMAllocInvoke(
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS64ParametersV535,
    allocParams: P64,
    isNVOS64: bool
) -> Result<u64> {
    let isV535 = fi.fd.nvp.lock().useRmAllocParamsV535;
    if isNVOS64 {
        if isV535 {
            return RMAllocInvokeInner::<NVOS64ParametersV535>(fi, ioctlParams, allocParams);
        } else {
            return RMAllocInvokeInner::<NVOS64Parameters>(fi, ioctlParams, allocParams);
        }
    } else {
        if isV535 {
            return RMAllocInvokeInner::<NVOS21ParametersV535>(fi, ioctlParams, allocParams);
        } else {
            return RMAllocInvokeInner::<NVOS21Parameters>(fi, ioctlParams, allocParams);
        }
    }
}

pub fn RMAllocInvoke1 <Params: Sized> (
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS64Parameters,
    allocParams: Option<&Params>,
    isNVOS64: bool
) -> Result<u64> {
    let allocParamsAddr = match allocParams {
        None => 0,
        Some(p) => p as * const _ as u64,
    };
    if isNVOS64 {
        let mut ioctlParamsTmp: NVOS64Parameters = *ioctlParams;
        ioctlParamsTmp.allocParms = allocParamsAddr;
        let rightsRequested = if ioctlParams.rightsRequested != 0 {
            let rightsRequested: RsAccessMask = fi.task.CopyInObj(ioctlParams.rightsRequested)?;
            ioctlParamsTmp.rightsRequested = &rightsRequested as * const _ as u64;
            rightsRequested
        } else {
            RsAccessMask::default()
        };

        let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;

        if ioctlParams.rightsRequested != 0 {
            fi.task.CopyOutObj(&rightsRequested, ioctlParams.rightsRequested)?;
        }

        let mut outIoctlParams = ioctlParamsTmp;
        outIoctlParams.allocParms = ioctlParams.allocParms;
        outIoctlParams.rightsRequested = ioctlParams.rightsRequested;
        fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
        return Ok(n);
    }

    let ioctlParamsTmp = NVOS21Parameters {
        root: ioctlParams.root,
        objectParent: ioctlParams.objectParent,
        objectNew: ioctlParams.objectNew,
        class: ioctlParams.class,
        allocParms: allocParamsAddr,
        status: ioctlParams.status
    };

    let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;

    let outIoctlParams = NVOS21Parameters {
        root: ioctlParamsTmp.root,
        objectParent: ioctlParamsTmp.objectParent,
        objectNew: ioctlParamsTmp.objectNew,
        class: ioctlParamsTmp.class,
        allocParms: ioctlParams.allocParms,
        status: ioctlParamsTmp.status,
    };

    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;

    return Ok(n)
}

pub fn RMVidHeapControlAllocSize (
    fi: &FrontendIoctlState,
    ioctlParams: &NVOS32Parameters
) -> Result<u64> {
    let allocSizeParams = unsafe {
        &*(&ioctlParams.data[0] as * const _ as u64 as * const NVOS32AllocSize)
    };

    let mut ioctlParamsTmp = *ioctlParams;
    let mut allocSizeParamsTmp = unsafe {
        &mut *(&mut ioctlParamsTmp.data[0] as * mut _ as u64 as * mut NVOS32AllocSize)
    };

    let mut addr: u64= 0;
    if allocSizeParams.address != 0 {
        addr = fi.task.CopyInObj(allocSizeParams.address)?;
        allocSizeParamsTmp.address = &addr as * const _ as u64;
    }

    let n = FrontendIoctlInvoke(fi, Some(&ioctlParamsTmp))?;

    let mut outIoctlParams = ioctlParamsTmp;
    let outAllocSizeParams = unsafe {
        &mut *(&mut outIoctlParams.data[0] as * mut _ as u64 as * mut NVOS32AllocSize)
    };

    if allocSizeParams.address != 0 {
        fi.task.CopyOutObj(&addr, allocSizeParams.address)?;
        outAllocSizeParams.address = allocSizeParams.address;
    }

    fi.task.CopyOutObj(&outIoctlParams, fi.ioctlParamsAddr)?;
    return Ok(n)
}