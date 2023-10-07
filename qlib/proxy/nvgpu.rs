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

pub const NV_MAJOR_DEVICE_NUMBER          : u16 = 195; // from kernel-open/common/inc/nv.h
pub const NV_CONTROL_DEVICE_MINOR         : u16 = 255; // from kernel-open/common/inc/nv-linux.h
pub const NVIDIA_UVM_PRIMARY_MINOR_NUMBER : u16 = 0;   // from kernel-open/nvidia-uvm/uvm_common.h


// Handle is NvHandle, from src/common/sdk/nvidia/inc/nvtypes.h.
#[repr(C)]
pub struct Handle {
    pub Val: u32,
}

// P64 is NvP64, from src/common/sdk/nvidia/inc/nvtypes.h.
pub type P64 = u64;

// From src/common/sdk/nvidia/inc/nvlimits.h:
pub const NV_MAX_DEVICES    : usize = 32;
pub const NV_MAX_SUBDEVICES : usize = 8;

// RsAccessMask is RS_ACCESS_MASK, from
// src/common/sdk/nvidia/inc/rs_access.h.
#[repr(C)]
pub struct RsAccessMask {
    pub limbs: [u32; SDK_RS_ACCESS_MAX_LIMBS],
}

pub const SDK_RS_ACCESS_MAX_LIMBS : usize = 1;

// RS_SHARE_POLICY is RS_SHARE_POLICY, from
// src/common/sdk/nvidia/inc/rs_access.h.
//
#[repr(C)]
pub struct RsSharePolicy {
    pub target     : u32, 
	pub accessMask : RsAccessMask,
	pub _type      : u16, 
	pub action     : u8, 
	// pub Pad        : u8
}


// From src/nvidia/interface/deprecated/rmapi_deprecated.h:
pub const RM_GSS_LEGACY_MASK : u64  = 0x00008000;

// From src/common/sdk/nvidia/inc/ctrl/ctrlxxxx.h:
#[repr(C)]
pub struct NvxxxxCtrlXxxInfo {
    pub index: u32,
    pub data: u32,
}

// From src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000client.h:
pub const NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE : u64        = 0xd01;
pub const NV0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY : u64 = 0xd04;

// From src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h:
pub const NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS  : u64 = 0x201;
pub const NV0000_CTRL_CMD_GPU_GET_ID_INFO       : u64 = 0x202;
pub const NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2    : u64 = 0x205;
pub const NV0000_CTRL_CMD_GPU_GET_PROBED_IDS    : u64 = 0x214;
pub const NV0000_CTRL_CMD_GPU_ATTACH_IDS        : u64 = 0x215;
pub const NV0000_CTRL_CMD_GPU_DETACH_IDS        : u64 = 0x216;
pub const NV0000_CTRL_CMD_GPU_GET_PCI_INFO      : u64 = 0x21b;
pub const NV0000_CTRL_CMD_GPU_QUERY_DRAIN_STATE : u64 = 0x279;
pub const NV0000_CTRL_CMD_GPU_GET_MEMOP_ENABLE  : u64 = 0x27b;


// From src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000syncgpuboost.h:
pub const NV0000_CTRL_CMD_SYNC_GPU_BOOST_GROUP_INFO  : u64 = 0xa04;

// From src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000system.h:
pub const NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION   : u64 = 0x101;
pub const NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS        : u64 = 0x127;
pub const NV0000_CTRL_CMD_SYSTEM_GET_FABRIC_STATUS   : u64 = 0x136;
pub const NV0000_CTRL_CMD_SYSTEM_GET_P2P_CAPS_MATRIX : u64 = 0x13a;

#[repr(C)]
pub struct Nv0000CtrlSystemGetBuildVersionParams {
    pub sizeOfStrings: u32,
    // pub pad: u32,
    pub driverVersionBuffer: P64,
    pub versionBuffer: P64,
    pub titleBuffer: P64,
    pub changelistNumber: u32,
    pub officialChangelistNumber: u32, 
}

// From src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080fifo.h:
pub const NV0080_CTRL_CMD_FB_GET_CAPS_V2 : u64 = 0x801307;

#[repr(C)]
pub struct Nv0080CtrlFifoGetChannellistParams {
    pub numChannels        : u32,
	// pub pad                : u32,
	pub channelHandleList : P64,
	pub channelList       : P64,
}

// From src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gpu.h:
pub const NV0080_CTRL_CMD_GPU_GET_NUM_SUBDEVICES         : u64 = 0x800280;
pub const NV0080_CTRL_CMD_GPU_QUERY_SW_STATE_PERSISTENCE : u64 = 0x800288;
pub const NV0080_CTRL_CMD_GPU_GET_VIRTUALIZATION_MODE    : u64 = 0x800289;
pub const NV0080_CTRL_CMD_GPU_GET_CLASSLIST_V2           : u64 = 0x800292;

// From src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gr.h:
// NV0080_CTRL_GR_ROUTE_INFO
#[repr(C)]
pub struct Nv0080CtrlGrRouteInfo {
    pub flags : u32,
	//pub pad   : u32,
	pub route : u64,
}

// From src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080host.h:
pub const NV0080_CTRL_CMD_HOST_GET_CAPS_V2 : u64 = 0x801402;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080bus.h:
pub const NV2080_CTRL_CMD_BUS_GET_PCI_INFO                   : u64 = 0x20801801;
pub const NV2080_CTRL_CMD_BUS_GET_PCI_BAR_INFO               : u64 = 0x20801803;
pub const NV2080_CTRL_CMD_BUS_GET_INFO_V2                    : u64 = 0x20801823;
pub const NV2080_CTRL_CMD_BUS_GET_PCIE_SUPPORTED_GPU_ATOMICS : u64 = 0x2080182a;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080ce.h:
pub const NV2080_CTRL_CMD_CE_GET_ALL_CAPS : u64 = 0x20802a0a;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fb.h:
pub const NV2080_CTRL_CMD_FB_GET_INFO_V2 : u64 = 0x20801303;


// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h:
pub const NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS : u64 = 0x2080110b;
pub const NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES : usize = 64;

// NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS
#[repr(C)]
pub struct Nv2080CtrlFifoDisableChannelsParams {
	pub BDisable               : u8,
	//pub Pad1                 : [u8; 3],
	pub numChannels            : u32,
	pub onlyDisableScheduling  : u8,
	pub rewindGpPut            : u8,
	pub Pad2                   : [u8; 6],
	pub runlistPreemptEvent    : P64,
	pub HClientList            : [Handle; NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES],
	pub HChannelList           : [Handle; NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES],
}


// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gpu.h:
pub const NV2080_CTRL_CMD_GPU_GET_INFO_V2                      : u64 = 0x20800102;
pub const NV2080_CTRL_CMD_GPU_GET_NAME_STRING                  : u64 = 0x20800110;
pub const NV2080_CTRL_CMD_GPU_GET_SHORT_NAME_STRING            : u64 = 0x20800111;
pub const NV2080_CTRL_CMD_GPU_GET_SIMULATION_INFO              : u64 = 0x20800119;
pub const NV2080_CTRL_CMD_GPU_QUERY_ECC_STATUS                 : u64 = 0x2080012f;
pub const NV2080_CTRL_CMD_GPU_QUERY_COMPUTE_MODE_RULES         : u64 = 0x20800131;
pub const NV2080_CTRL_CMD_GPU_ACQUIRE_COMPUTE_MODE_RESERVATION : u64 = 0x20800145; // undocumented; paramSize == 0
pub const NV2080_CTRL_CMD_GPU_RELEASE_COMPUTE_MODE_RESERVATION : u64 = 0x20800146; // undocumented; paramSize == 0
pub const NV2080_CTRL_CMD_GPU_GET_GID_INFO                     : u64 = 0x2080014a;
pub const NV2080_CTRL_CMD_GPU_GET_ENGINES_V2                   : u64 = 0x20800170;
pub const NV2080_CTRL_CMD_GPU_GET_ACTIVE_PARTITION_IDS         : u64 = 0x2080018b;
pub const NV2080_CTRL_CMD_GPU_GET_COMPUTE_POLICY_CONFIG        : u64 = 0x20800195;
pub const NV2080_CTRL_CMD_GET_GPU_FABRIC_PROBE_INFO            : u64 = 0x208001a3;


// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h:
pub const NV2080_CTRL_CMD_GR_GET_INFO                  : u64 = 0x20801201;
pub const NV2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE : u64 = 0x20801210;
pub const NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE       : u64 = 0x20801218;
pub const NV2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER       : u64 = 0x2080121b;
pub const NV2080_CTRL_CMD_GR_GET_CAPS_V2               : u64 = 0x20801227;
pub const NV2080_CTRL_CMD_GR_GET_GPC_MASK              : u64 = 0x2080122a;
pub const NV2080_CTRL_CMD_GR_GET_TPC_MASK              : u64 = 0x2080122b;


// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gsp.h:
pub const NV2080_CTRL_CMD_GSP_GET_FEATURES : u64 = 0x20803601;

// NV2080_CTRL_GR_GET_INFO_PARAMS
#[repr(C)]
pub struct Nv2080CtrlGrGetInfoParams {
    pub GRInfoListSize : u32, // in elements
	// pub pad            : u32,
	pub GRInfoList     : P64,
	pub GRRouteInfo    : Nv0080CtrlGrRouteInfo,
}

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080mc.h:
pub const NV2080_CTRL_CMD_MC_GET_ARCH_INFO      : u64 = 0x20801701;
pub const NV2080_CTRL_CMD_MC_SERVICE_INTERRUPTS : u64 = 0x20801702;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080nvlink.h:
pub const NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS : u64 = 0x20803002;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080perf.h:
pub const NV2080_CTRL_CMD_PERF_BOOST : u64 = 0x2080200a;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080rc.h:
pub const NV2080_CTRL_CMD_RC_GET_WATCHDOG_INFO         : u64 = 0x20802209;
pub const NV2080_CTRL_CMD_RC_RELEASE_WATCHDOG_REQUESTS : u64 = 0x2080220c;
pub const NV2080_CTRL_CMD_RC_SOFT_DISABLE_WATCHDOG     : u64 = 0x20802210;

// From src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080tmr.h:
pub const NV2080_CTRL_CMD_TIMER_GET_GPU_CPU_TIME_CORRELATION_INFO: u64 = 0x20800406;

// From src/common/sdk/nvidia/inc/ctrl/ctrl503c.h:
pub const NV503C_CTRL_CMD_REGISTER_VA_SPACE : u64 = 0x503c0102;
pub const NV503C_CTRL_CMD_REGISTER_VIDMEM   : u64 = 0x503c0104;
pub const NV503C_CTRL_CMD_UNREGISTER_VIDMEM : u64 = 0x503c0105;

// From src/common/sdk/nvidia/inc/ctrl/ctrl83de/ctrl83dedebug.h:
pub const NV83DE_CTRL_CMD_DEBUG_SET_EXCEPTION_MASK        : u64 = 0x83de0309;
pub const NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES  : u64 = 0x83de030c;
pub const NV83DE_CTRL_CMD_DEBUG_CLEAR_ALL_SM_ERROR_STATES : u64 = 0x83de0310;

// From src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h:
pub const NVC36F_CTRL_GET_CLASS_ENGINEID               : u64 = 0xc36f0101;
pub const NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN : u64 = 0xc36f0108;

// From src/common/sdk/nvidia/inc/ctrl/ctrl906f.h:
pub const NV906F_CTRL_CMD_RESET_CHANNEL : u64 = 0x906f0102;

// From src/common/sdk/nvidia/inc/ctrl/ctrl90e6.h:
pub const NV90E6_CTRL_CMD_MASTER_GET_ERROR_INTR_OFFSET_MASK                : u64 = 0x90e60101;
pub const NV90E6_CTRL_CMD_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK : u64 = 0x90e60102;

// From src/common/sdk/nvidia/inc/ctrl/ctrl90e6.h:
pub const NVA06C_CTRL_CMD_GPFIFO_SCHEDULE : u64 = 0xa06c0101;
pub const NVA06C_CTRL_CMD_SET_TIMESLICE   : u64 = 0xa06c0103;
pub const NVA06C_CTRL_CMD_PREEMPT         : u64 = 0xa06c0105;

// Status codes, from src/common/sdk/nvidia/inc/nvstatuscodes.h.
pub const NV_ERR_INVALID_ADDRESS : u64 = 0x0000001e;
pub const NV_ERR_INVALID_LIMIT   : u64 = 0x0000002e;
pub const NV_ERR_NOT_SUPPORTED   : u64 = 0x00000056;