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

use super::nvgpu::*;

// Class handles, from src/nvidia/generated/g_allclasses.h.
pub const NV01_ROOT                        : u32 = 0x00000000;
pub const NV01_ROOT_NON_PRIV               : u32 = 0x00000001;
pub const NV01_MEMORY_SYSTEM               : u32 = 0x0000003e;
pub const NV01_ROOT_CLIENT                 : u32 = 0x00000041;
pub const NV01_MEMORY_SYSTEM_OS_DESCRIPTOR : u32 = 0x00000071;
pub const NV01_EVENT_OS_EVENT              : u32 = 0x00000079;
pub const NV01_DEVICE_0                    : u32 = 0x00000080;
pub const NV_MEMORY_FABRIC                 : u32 = 0x000000f8;
pub const NV20_SUBDEVICE_0                 : u32 = 0x00002080;
pub const NV50_THIRD_PARTY_P2P             : u32 = 0x0000503c;
pub const GT200_DEBUGGER                   : u32 = 0x000083de;
pub const GF100_SUBDEVICE_MASTER           : u32 = 0x000090e6;
pub const FERMI_CONTEXT_SHARE_A            : u32 = 0x00009067;
pub const FERMI_VASPACE_A                  : u32 = 0x000090f1;
pub const KEPLER_CHANNEL_GROUP_A           : u32 = 0x0000a06c;
pub const VOLTA_USERMODE_A                 : u32 = 0x0000c361;
pub const VOLTA_CHANNEL_GPFIFO_A           : u32 = 0x0000c36f;
pub const TURING_USERMODE_A                : u32 = 0x0000c461;
pub const TURING_CHANNEL_GPFIFO_A          : u32 = 0x0000c46f;
pub const AMPERE_CHANNEL_GPFIFO_A          : u32 = 0x0000c56f;
pub const TURING_DMA_COPY_A                : u32 = 0x0000c5b5;
pub const TURING_COMPUTE_A                 : u32 = 0x0000c5c0;
pub const HOPPER_USERMODE_A                : u32 = 0x0000c661;
pub const AMPERE_DMA_COPY_A                : u32 = 0x0000c6b5;
pub const AMPERE_COMPUTE_A                 : u32 = 0x0000c6c0;
pub const AMPERE_DMA_COPY_B                : u32 = 0x0000c7b5;
pub const AMPERE_COMPUTE_B                 : u32 = 0x0000c7c0;
pub const HOPPER_DMA_COPY_A                : u32 = 0x0000c8b5;
pub const ADA_COMPUTE_A                    : u32 = 0x0000c9c0;
pub const NV_CONFIDENTIAL_COMPUTE		   : u32 = 0x0000cb33;
pub const HOPPER_COMPUTE_A                 : u32 = 0x0000cbc0;

// Class handles for older generations that are not supported by the open source
// driver. Volta was the last such generation. These are defined in files under
// src/common/sdk/nvidia/inc/class/.
pub const VOLTA_COMPUTE_A  : u32 = 0x0000c3c0;
pub const VOLTA_DMA_COPY_A : u32 = 0x0000c3b5;

// Nv0005AllocParameters NV0005_ALLOC_PARAMETERS is the alloc params type for NV01_EVENT_OS_EVENT,
// from src/common/sdk/nvidia/inc/class/cl0005.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv0005AllocParameters {
	pub parentClient : Handle,
	pub SrcResource  : Handle,
	pub hClass        : u32,
	pub notifyIndex   : u32,
	pub data          : P64, // actually FD for NV01_EVENT_OS_EVENT, see src/nvidia/src/kernel/rmapi/event.c:eventConstruct_IMPL() => src/nvidia/arch/nvalloc/unix/src/os.c:osUserHandleToKernelPtr()
}

// Nv0080AllocParameters NV0080_ALLOC_PARAMETERS is the alloc params type for NV01_DEVICE_0, from
// src/common/sdk/nvidia/inc/class/cl0080.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv0080AllocParameters {
    pub DeviceID        : u32,
	pub clientShare    : Handle,
	pub targetClient   : Handle,
	pub targetDevice   : Handle,
	pub flags           : u32,
	//pub pad0            : u32,
	pub vASpaceSize     : u64,
	pub vAStartInternal : u64,
	pub vALimitInternal : u64,
	pub vAMode          : u32,
	// pub Pad1            : u64,
}

// Nv2080AllocParameters NV2080_ALLOC_PARAMETERS is the alloc params type for NV20_SUBDEVICE_0, from
// src/common/sdk/nvidia/inc/class/cl2080.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv2080AllocParameters {
    pub subDeviceID : u32,
}

// Nv503cAllocParameters NV503C_ALLOC_PARAMETERS is the alloc params type for NV50_THIRD_PARTY_P2P,
// from src/common/sdk/nvidia/inc/class/cl503c.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv503cAllocParameters {
    pub Flags: u32,
}

// Nv83deAllocParameters NV83DE_ALLOC_PARAMETERS is the alloc params type for GT200_DEBUGGER,
// from src/common/sdk/nvidia/inc/class/cl83de.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv83deAllocParameters {
    pub debuggerClient_Obsolete : Handle,
	pub appClient               : Handle,
	pub class3DObject           : Handle,
}

// NvCtxshareAllocationParameters NV_CTXSHARE_ALLOCATION_PARAMETERS is the alloc params type for
// FERMI_CONTEXT_SHARE_A, from src/common/sdk/nvidia/inc/nvos.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvCtxshareAllocationParameters {
    pub vASpace : Handle,
	pub flags   : u32,
	pub subctxID: u32,
}

// NvVaspaceAllocationParameters NV_VASPACE_ALLOCATION_PARAMETERS is the alloc params type for
// FERMI_VASPACE_A, from src/common/sdk/nvidia/inc/nvos.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvVaspaceAllocationParameters {
    pub index           : u32,
	pub flags           : u32,
	pub vASize          : u64,
	pub vAStartInternal : u64,
	pub vALimitInternal : u64,
	pub bigPageSize     : u32,
	//pub Pad0            : u32,
	pub vABase          : u64,
}


// NvChannelGroupAllocationParameters NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS is the alloc params type for
// KEPLER_CHANNEL_GROUP_A, from src/common/sdk/nvidia/inc/nvos.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvChannelGroupAllocationParameters {
    pub objectError                : Handle, 
	pub objectECCError             : Handle,
	pub vASpace                    : Handle,
	pub engineType                 : u32, 
	pub isCallingContextVgpuPlugin : u8,
	// pub Pad0                        : [u8; 3],
}

// NvMemoryDescParams NV_MEMORY_DESC_PARAMS is from
// src/common/sdk/nvidia/inc/alloc/alloc_channel.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvMemoryDescParams {
    pub Base         : u64,
	pub Size         : u64,
	pub AddressSpace : u32,
	pub CacheAttrib  : u32,
}

// NvChannelAllocParams NV_CHANNEL_ALLOC_PARAMS is the alloc params type for TURING_CHANNEL_GPFIFO_A
// and AMPERE_CHANNEL_GPFIFO_A, from
// src/common/sdk/nvidia/inc/alloc/alloc_channel.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvChannelAllocParams {
    pub HObjectError        : Handle,
	pub HObjectBuffer       : Handle,
	pub GPFIFOOffset        : u64,
	pub GPFIFOEntries       : u32,
	pub Flags               : u32,
	pub HContextShare       : Handle,
	pub HVASpace            : Handle,
	pub HUserdMemory        : [Handle; NV_MAX_SUBDEVICES],
	pub UserdOffset         : [u64; NV_MAX_SUBDEVICES],
	pub EngineType          : u32,
	pub CID                 : u32,
	pub SubDeviceID         : u32,
	pub HObjectECCError     : Handle,
	pub InstanceMem         : NvMemoryDescParams,
	pub UserdMem            : NvMemoryDescParams,
	pub RamfcMem            : NvMemoryDescParams,
	pub MthdbufMem          : NvMemoryDescParams,
	pub HPhysChannelGroup   : Handle,
	pub InternalFlags       : u32,
	pub ErrorNotifierMem    : NvMemoryDescParams,
	pub ECCErrorNotifierMem : NvMemoryDescParams,
	pub ProcessID           : u32,
	pub SubProcessID        : u32,
}

// Nvb0b5AllocationParameters NVB0B5_ALLOCATION_PARAMETERS is the alloc param type for TURING_DMA_COPY_A,
// AMPERE_DMA_COPY_A, and AMPERE_DMA_COPY_B from
// src/common/sdk/nvidia/inc/class/clb0b5sw.h.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nvb0b5AllocationParameters {
    pub version    : u32,
	pub engineType : u32,
}

// NvGrAllocationParameters NV_GR_ALLOCATION_PARAMETERS is the alloc param type for TURING_COMPUTE_A,
// AMPERE_COMPUTE_A, and ADA_COMPUTE_A, from src/common/sdk/nvidia/inc/nvos.h.
//
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvGrAllocationParameters {
    pub version : u32,
	pub flags   : u32,
	pub size    : u32,
	pub caps    : u32,
}

// NvHopperUsermodeAParams NV_HOPPER_USERMODE_A_PARAMS is the alloc param type for HOPPER_USERMODE_A,
// from src/common/sdk/nvidia/inc/nvos.h.
//
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvHopperUsermodeAParams {
    pub Bar1Mapping : u8,
	pub Priv        : u8,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv00f8Map {
	pub offset  : u64,
	pub vidMem  : Handle,
	pub flags   : u32,
}

// NV00F8_ALLOCATION_PARAMETERS is the alloc param type for NV_MEMORY_FABRIC,
// from src/common/sdk/nvidia/inc/class/cl00f8.h
//
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Nv00f8AllocationParameters {
    pub Alignment  : u64,
	pub AllocSize  : u64,
	pub PageSize   : u32,
	pub AllocFlags : u32,
	pub Map        : Nv00f8Map,
}

// NV_CONFIDENTIAL_COMPUTE_ALLOC_PARAMS is the alloc param type for
// NV_CONFIDENTIAL_COMPUTE, from src/common/sdk/nvidia/inc/class/clcb33.h.

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NvConfidentialComputeAllocParams {
	pub handler: Handle,
	//pub pad: u32,
}

