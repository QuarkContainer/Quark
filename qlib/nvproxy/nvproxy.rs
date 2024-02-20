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

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;
use alloc::fmt;

use crate::qlib::kernel::Kernel::HostSpace;
use crate::qlib::kernel::PAGE_MGR;
use crate::qlib::linux_def::MemoryDef;
use crate::qlib::mutex::QMutex;
use crate::qlib::nvproxy::nvgpu;
use crate::qlib::range::Range;
use crate::qlib::common::*;
use crate::qlib::kernel::task::*;

use super::classes::*;
use super::frontend::CtrlDevFIFOGetChannelList;
use super::frontend::CtrlSubdevGRGetInfo;
use super::frontend::FrontendIoctlState;
use super::frontend_type::*;
use super::frontendfd::*;
use super::nvgpu::*;
use super::uvm::*;
use super::uvm::UvmIoctlState;
use super::uvmfd::*;

type FrontendIoctlHandler = fn(fi: &FrontendIoctlState) -> Result<u64>;
type ControlCmdHandler = fn(fi: &FrontendIoctlState, ioctlParams: &NVOS54Parameters) -> Result<u64>;
type AllocationClassHandler = fn(fi: &FrontendIoctlState, ioctlParams: &NVOS64ParametersV535, isNVOS64: bool) -> Result<u64>;
type UvmIoctlHandler = fn(task: &Task, fi: &UvmIoctlState) -> Result<u64>;

#[derive(Clone)]
pub struct NVProxyInner {
    pub objs: BTreeMap<nvgpu::Handle, NVObject>,
    pub frontendIoctl: BTreeMap<u32, FrontendIoctlHandler>,
    pub uvmIoctl: BTreeMap<u32, UvmIoctlHandler>,
    pub controlCmd: BTreeMap<u32, ControlCmdHandler>,
    pub allocationClass: BTreeMap<u32, AllocationClassHandler>,
    
    pub useRmAllocParamsV535: bool,
}

impl fmt::Debug for NVProxyInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NVProxyInner")
         .field("objs", &self.objs)
         .finish()
    }
}

impl Default for NVProxyInner {
    fn default() -> Self {
        let version = RMAPIVersion {
            cmd: 0,
            reply: 0,
            versionString: [0; 64]
        };
        HostSpace::NividiaDriverVersion(&version);
        
        let mut inner = Self {
            objs: BTreeMap::default(),
            frontendIoctl: BTreeMap::default(),
            uvmIoctl: BTreeMap::default(),
            controlCmd: BTreeMap::default(),
            allocationClass: BTreeMap::default(),
            useRmAllocParamsV535: true,
        };

        inner.Setv535_104_05();
        return inner;
    }
}

impl NVProxyInner {
    pub fn Setv525_60_13(&mut self) {
        self.frontendIoctl.insert(NV_ESC_CARD_INFO, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_CHECK_VERSION_STR, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_SYS_PARAMS, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_RM_DUP_OBJECT, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_RM_SHARE, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_RM_UNMAP_MEMORY, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO, FrontendIoctlSimple);
        self.frontendIoctl.insert(NV_ESC_REGISTER_FD, FrontendRegisterFD);
        self.frontendIoctl.insert(NV_ESC_ALLOC_OS_EVENT, RMAllocOSEvent);
        self.frontendIoctl.insert(NV_ESC_FREE_OS_EVENT, RMFreeOSEvent);
        self.frontendIoctl.insert(NV_ESC_NUMA_INFO, RMNumaInfo);
        self.frontendIoctl.insert(NV_ESC_RM_ALLOC_MEMORY, RMAllocMemory);
        self.frontendIoctl.insert(NV_ESC_RM_FREE, RMFree);
        self.frontendIoctl.insert(NV_ESC_RM_CONTROL, RMControl);
        self.frontendIoctl.insert(NV_ESC_RM_ALLOC, RMAlloc);
        self.frontendIoctl.insert(NV_ESC_RM_VID_HEAP_CONTROL, RMVidHeapControl);
        self.frontendIoctl.insert(NV_ESC_RM_MAP_MEMORY, RMMapMemory);

        self.uvmIoctl.insert(UVM_INITIALIZE, UvmInitialize);
        self.uvmIoctl.insert(UVM_DEINITIALIZE, UvmIoctlNoParams);
        self.uvmIoctl.insert(UVM_CREATE_RANGE_GROUP, UvmIoctlSimple::<UvmCreateRangeGroupParams>);
        self.uvmIoctl.insert(UVM_DESTROY_RANGE_GROUP, UvmIoctlSimple::<UvmDestroyRangeGroupParams>);
        self.uvmIoctl.insert(UVM_REGISTER_GPU_VASPACE, UvmIoctlSimple::<UvmRegisterGpuVaspaceParams>);
        self.uvmIoctl.insert(UVM_UNREGISTER_GPU_VASPACE, UvmIoctlSimple::<UvmUnregisterGpuVaspaceParams>);
        self.uvmIoctl.insert(UVM_REGISTER_CHANNEL, UvmIoctlSimple::<UvmRegisterChannelParams>);
        self.uvmIoctl.insert(UVM_UNREGISTER_CHANNEL, UvmIoctlSimple::<UvmUnregisterChannelParams>);
        self.uvmIoctl.insert(UVM_MAP_EXTERNAL_ALLOCATION, UvmIoctlSimple::<UvmMapExternalAllocationParams>);
        self.uvmIoctl.insert(UVM_FREE, UvmIoctlSimple::<UvmFreeParams>);
        self.uvmIoctl.insert(UVM_REGISTER_GPU, UvmIoctlSimple::<UvmRegisterGpuParams>);
        self.uvmIoctl.insert(UVM_UNREGISTER_GPU, UvmIoctlSimple::<UvmUnregisterGpuParams>);
        self.uvmIoctl.insert(UVM_PAGEABLE_MEM_ACCESS, UvmIoctlSimple::<UvmPageableMemAccessParams>);
        self.uvmIoctl.insert(UVM_MAP_DYNAMIC_PARALLELISM_REGION, UvmIoctlSimple::<UvmMapDynamicParallelismRegionParams>);
        self.uvmIoctl.insert(UVM_ALLOC_SEMAPHORE_POOL, UvmIoctlSimple::<UvmAllocSemaphorePoolParams>);
        self.uvmIoctl.insert(UVM_VALIDATE_VA_RANGE, UvmIoctlSimple::<UvmValidateVaRangeParams>);
        self.uvmIoctl.insert(UVM_CREATE_EXTERNAL_RANGE, UvmIoctlSimple::<UvmCreateExternalRangeParams>);


        self.controlCmd.insert(NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_GET_ID_INFO, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_GET_PROBED_IDS, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_ATTACH_IDS, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_DETACH_IDS, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_GET_PCI_INFO, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX, RMControlSimple);
        self.controlCmd.insert(NV0080_CTRL_CMD_FB_GET_CAPS_V2, RMControlSimple);
        self.controlCmd.insert(NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES, RMControlSimple);
        self.controlCmd.insert(NV0080_CTRL_CMD_GPU_QUERY_SW_STATE_PERSISTENCE, RMControlSimple);
        self.controlCmd.insert(NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE, RMControlSimple);
        self.controlCmd.insert(0x80028b, RMControlSimple);
        self.controlCmd.insert(NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2, RMControlSimple);
        self.controlCmd.insert(NV0080_CTRL_CMD_HOST_GET_CAPS_V2, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_BUS_GET_PCI_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_BUS_GET_INFO_V2, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_CE_GET_ALL_CAPS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_FB_GET_INFO_V2, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_INFO_V2, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_NAME_STRING, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_ACQUIRE_COMPUTE_MODE_RESERVATION, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_RELEASE_COMPUTE_MODE_RESERVATION, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_GID_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_ENGINES_V2, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GPU_GET_COMPUTE_POLICY_CONFIG, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GET_GPU_FABRIC_PROBE_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_GET_CAPS_V2, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_GET_GPC_MASK, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_GET_TPC_MASK, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_GSP_GET_FEATURES, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_MC_GET_ARCH_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_PERF_BOOST, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_RC_GET_WATCHDOG_INFO, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG, RMControlSimple);
        self.controlCmd.insert(NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO, RMControlSimple);
        self.controlCmd.insert(NV503C_CTRL_CMD_REGISTER_VA_SPACE, RMControlSimple);
        self.controlCmd.insert(NV503C_CTRL_CMD_REGISTER_VIDMEM, RMControlSimple);
        self.controlCmd.insert(NV503C_CTRL_CMD_UNREGISTER_VIDMEM, RMControlSimple);
        self.controlCmd.insert(NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK, RMControlSimple);
        self.controlCmd.insert(NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES, RMControlSimple);
        self.controlCmd.insert(NV83DE_CTRL_CMD_DEBUG_CLEAR_ALL_SM_ERROR_STATES, RMControlSimple);
        self.controlCmd.insert(NV906F_CTRL_CMD_RESET_CHANNEL, RMControlSimple);
        self.controlCmd.insert(NV90E6_CTRL_CMD_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK, RMControlSimple);
        self.controlCmd.insert(NVC36F_CTRL_GET_CLASS_ENGINEID, RMControlSimple);
        self.controlCmd.insert(NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, RMControlSimple);
        self.controlCmd.insert(NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, RMControlSimple);
        self.controlCmd.insert(NVA06C_CTRL_CMD_SET_TIMESLICE, RMControlSimple);
        self.controlCmd.insert(NVA06C_CTRL_CMD_PREEMPT, RMControlSimple);
        self.controlCmd.insert(NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION, CtrlClientSystemGetBuildVersion);
        self.controlCmd.insert(NV0080_CTRL_CMD_FIFO_GET_CHANNELLIST, CtrlDevFIFOGetChannelList);
        self.controlCmd.insert(NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS, CtrlSubdevFIFODisableChannels);
        self.controlCmd.insert(NV2080_CTRL_CMD_GR_GET_INFO, CtrlSubdevGRGetInfo);

        self.allocationClass.insert(NV01_ROOT, RMAllocSimple::<Handle>);
        self.allocationClass.insert(NV01_ROOT_NON_PRIV, RMAllocSimple::<Handle>);
        self.allocationClass.insert(NV01_ROOT_CLIENT, RMAllocSimple::<Handle>);
        self.allocationClass.insert(NV01_EVENT_OS_EVENT, RMAllocEventOSEvent);
        self.allocationClass.insert(NV01_DEVICE_0, RMAllocSimple::<Nv0080AllocParameters>);
        self.allocationClass.insert(NV20_SUBDEVICE_0, RMAllocSimple::<Nv0080AllocParameters>);
        self.allocationClass.insert(NV50_THIRD_PARTY_P2P, RMAllocSimple::<Nv0080AllocParameters>);
        self.allocationClass.insert(GT200_DEBUGGER, RMAllocSimple::<Nv0080AllocParameters>);
        self.allocationClass.insert(FERMI_CONTEXT_SHARE_A, RMAllocSimple::<NvCtxshareAllocationParameters>);
        self.allocationClass.insert(FERMI_VASPACE_A, RMAllocSimple::<NvVaspaceAllocationParameters>);
        self.allocationClass.insert(KEPLER_CHANNEL_GROUP_A, RMAllocSimple::<NvChannelGroupAllocationParameters>);
        self.allocationClass.insert(TURING_CHANNEL_GPFIFO_A, RMAllocSimple::<NvChannelAllocParams>);
        self.allocationClass.insert(AMPERE_CHANNEL_GPFIFO_A, RMAllocSimple::<NvChannelAllocParams>);
        self.allocationClass.insert(TURING_DMA_COPY_A, RMAllocSimple::<Nvb0b5AllocationParameters>);
        self.allocationClass.insert(AMPERE_DMA_COPY_A, RMAllocSimple::<Nvb0b5AllocationParameters>);
        self.allocationClass.insert(AMPERE_DMA_COPY_B, RMAllocSimple::<Nvb0b5AllocationParameters>);
        self.allocationClass.insert(HOPPER_DMA_COPY_A, RMAllocSimple::<Nvb0b5AllocationParameters>);
        self.allocationClass.insert(TURING_COMPUTE_A, RMAllocSimple::<NvGrAllocationParameters>);
        self.allocationClass.insert(AMPERE_COMPUTE_A, RMAllocSimple::<NvGrAllocationParameters>);
        self.allocationClass.insert(AMPERE_COMPUTE_B, RMAllocSimple::<NvGrAllocationParameters>);
        self.allocationClass.insert(ADA_COMPUTE_A, RMAllocSimple::<NvGrAllocationParameters>);
        self.allocationClass.insert(HOPPER_COMPUTE_A, RMAllocSimple::<NvGrAllocationParameters>);
        self.allocationClass.insert(HOPPER_USERMODE_A, RMAllocSimple::<NvHopperUsermodeAParams>);
        self.allocationClass.insert(GF100_SUBDEVICE_MASTER, RMAllocNoParams);
        self.allocationClass.insert(TURING_USERMODE_A, RMAllocNoParams);
        self.allocationClass.insert(NV_MEMORY_FABRIC, RMAllocSimple::<Nv00f8AllocationParameters>);
    }

    pub fn Setv525_89_02(&mut self) {
        self.Setv525_60_13();
    }

    pub fn Setv525_105_17(&mut self) {
        self.Setv525_89_02();
    }

    pub fn Setv525_125_06(&mut self) {
        self.Setv525_105_17();
    }

    pub fn Setv535_43_02(&mut self) {
        self.Setv525_89_02();
        self.useRmAllocParamsV535 = true;
        self.controlCmd.insert(NV_CONF_COMPUTE_CTRL_CMD_SYSTEM_GET_CAPABILITIES, RMControlSimple);
        self.allocationClass.insert(NV_CONFIDENTIAL_COMPUTE, RMAllocSimple::<NvConfidentialComputeAllocParams>);
        self.uvmIoctl.insert(UVM_MM_INITIALIZE, UvmMMInitialize);
    }

    pub fn Setv535_54_03(&mut self) {
        self.Setv535_43_02();
    }

    pub fn Setv535_104_05(&mut self) {
        self.Setv535_54_03();
    }
}

#[derive(Clone, Debug, Default)]
pub struct NVProxy(Arc<QMutex<NVProxyInner>>);

impl Deref for NVProxy {
    type Target = Arc<QMutex<NVProxyInner>>;

    fn deref(&self) -> &Arc<QMutex<NVProxyInner>> {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct OSDescMem {
    pub pinnedRange: Vec<Range>
}

impl Drop for OSDescMem {
    fn drop(&mut self) {
        for r in &self.pinnedRange {
            let mut paddr = r.start;
            while paddr < r.End() {
                PAGE_MGR.DerefPage(paddr);
                paddr += MemoryDef::PAGE_SIZE;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum NVObject {
    OSDescMem(OSDescMem)
}

impl From<OSDescMem> for NVObject {
    fn from(o: OSDescMem) -> Self {
        NVObject::OSDescMem(o)
    }
}
