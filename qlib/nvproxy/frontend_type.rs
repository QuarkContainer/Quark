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

// NV_IOCTL_MAGIC is the "canonical" IOC_TYPE for frontend ioctls.
// The driver ignores IOC_TYPE, allowing any value to be passed.
pub const NV_IOCTL_MAGIC : u32 = 'F' as u32;

// Frontend ioctl numbers.
// Note that these are only the IOC_NR part of the ioctl command.

// From kernel-open/common/inc/nv-ioctl-numbers.h:
pub const NV_IOCTL_BASE            : u64 = 200;
pub const NV_ESC_CARD_INFO         : u64 = NV_IOCTL_BASE + 0;
pub const NV_ESC_REGISTER_FD       : u64 = NV_IOCTL_BASE + 1;
pub const NV_ESC_ALLOC_OS_EVENT    : u64 = NV_IOCTL_BASE + 6;
pub const NV_ESC_FREE_OS_EVENT     : u64 = NV_IOCTL_BASE + 7;
pub const NV_ESC_CHECK_VERSION_STR : u64 = NV_IOCTL_BASE + 10;
pub const NV_ESC_SYS_PARAMS        : u64 = NV_IOCTL_BASE + 14;

// From kernel-open/common/inc/nv-ioctl-numa.h:
pub const NV_ESC_NUMA_INFO : u64 = NV_IOCTL_BASE + 15;

// From src/nvidia/arch/nvalloc/unix/include/nv_escape.h:
pub const NV_ESC_RM_ALLOC_MEMORY               : u64 = 0x27;
pub const NV_ESC_RM_FREE                       : u64 = 0x29;
pub const NV_ESC_RM_CONTROL                    : u64 = 0x2a;
pub const NV_ESC_RM_ALLOC                      : u64 = 0x2b;
pub const NV_ESC_RM_DUP_OBJECT                 : u64 = 0x34;
pub const NV_ESC_RM_SHARE                      : u64 = 0x35;
pub const NV_ESC_RM_VID_HEAP_CONTROL           : u64 = 0x4a;
pub const NV_ESC_RM_MAP_MEMORY                 : u64 = 0x4e;
pub const NV_ESC_RM_UNMAP_MEMORY               : u64 = 0x4f;
pub const NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO : u64 = 0x5e;

// Frontend ioctl parameter structs, from src/common/sdk/nvidia/inc/nvos.h or
// kernel-open/common/inc/nv-ioctl.h.

// IoctlRegisterFD is nv_ioctl_register_fd_t, the parameter type for
// NV_ESC_REGISTER_FD.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct IoctlRegisterFD {
    pub ctlFD: i32,
}

// IoctlAllocOSEvent is nv_ioctl_alloc_os_event_t, the parameter type for
// NV_ESC_ALLOC_OS_EVENT.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct IoctlAllocOSEvent {
    pub client : Handle,
	pub device : Handle,
	pub fd      : u32,
	pub status  : u32,
}

// IoctlFreeOSEvent is nv_ioctl_free_os_event_t, the parameter type for
// NV_ESC_FREE_OS_EVENT.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct IoctlFreeOSEvent {
    pub client : Handle,
	pub device : Handle,
	pub fd      : u32,
	pub status  : u32,
}


// RMAPIVersion is nv_rm_api_version_t, the parameter type for
// NV_ESC_CHECK_VERSION_STR.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct RMAPIVersion {
    pub cmd           : u32,
	pub reply         : u32,
	pub versionString : [u8; 64]
}

// IoctlSysParams is nv_ioctl_sys_params_t, the parameter type for
// NV_ESC_SYS_PARAMS.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct IoctlSysParams {
    pub memblockSize: u64,
}

// IoctlNVOS02ParametersWithFD is nv_ioctl_nvos2_parameters_with_fd, the
// parameter type for NV_ESC_RM_ALLOC_MEMORY.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct IoctlNVOS02ParametersWithFD {
    pub params : NVOS02Parameters,
	pub fd     : i32,
	// pub pad0   : i32,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS02Parameters {
    pub root         : Handle,
	pub objectParent : Handle,
	pub objectNew    : Handle,
	pub class        : u32,
	pub flags         : u32,
	// pub Pad0          : u32,
	pub memory       : P64, // address of application mapping, without indirection
	pub limit         : u64,
	pub status        : u32,
	// pub Pad1          : u32,
}

// NVOS00Parameters is NVOS00_PARAMETERS, the parameter type for
// NV_ESC_RM_FREE.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS00Parameters {
    pub root         : Handle,
	pub objectParent : Handle,
	pub objectOld    : Handle,
	pub status        : u32,
}

// NVOS21Parameters is NVOS21_PARAMETERS, one possible parameter type for
// NV_ESC_RM_ALLOC.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS21Parameters {
    pub root         : Handle,
	pub objectParent : Handle,
	pub objectNew    : Handle,
	pub class        : u32,
	pub allocParms   : P64,
	pub status        : u32,
	// pub Pad0          : u32,
}

// NVOS55Parameters is NVOS55_PARAMETERS, the parameter type for
// NV_ESC_RM_DUP_OBJECT.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS55Parameters {
    pub client    : Handle,
	pub parent    : Handle,
	pub object    : Handle,
	pub clientSrc : Handle,
	pub objectSrc : Handle,
	pub flags     : u32,
	pub status    : u32,
}

// NVOS57Parameters is NVOS57_PARAMETERS, the parameter type for
// NV_ESC_RM_SHARE.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS57Parameters {
    pub client     : Handle,
	pub object     : Handle,
	pub sharePolicy : RsSharePolicy,
	pub status      : u32,
}

// NVOS32Parameters is NVOS32_PARAMETERS, the parameter type for
// NV_ESC_RM_VID_HEAP_CONTROL.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS32Parameters {
    pub root         : Handle,
	pub objectParent : Handle,
	pub function      : u32,
	pub HVASpace      : Handle,
	pub IVCHeapNumber : i16,
	// pub Pad           : [u8; 2],
	pub status        : u32,
	pub total         : u64,
	pub free          : u64,
	pub data          : [u8; 144], // union
}

// Possible values for NVOS32Parameters.Function:
pub const NVOS32_FUNCTION_ALLOC_SIZE : u32 = 2;

// NVOS32AllocSize is the type of NVOS32Parameters.Data for
// NVOS32_FUNCTION_ALLOC_SIZE.
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS32AllocSize {
    pub owner           : u32,
	pub memory         : Handle,
	pub _type            : u32,
	pub flags           : u32,
	pub attr            : u32,
	pub format          : u32,
	pub comprCovg       : u32,
	pub zcullCovg       : u32,
	pub partitionStride : u32,
	pub width           : u32,
	pub height          : u32,
	// pub pad0            : u32,
	pub size            : u64,
	pub alignment       : u64,
	pub offset          : u64,
	pub limit           : u64,
	pub address         : P64,
	pub rangeBegin      : u64,
	pub rangeEnd        : u64,
	pub attr2           : u32,
	pub ctagOffset      : u32,
}

// IoctlNVOS33ParametersWithFD is nv_ioctl_nvos33_parameters_with_fd, the
// parameter type for NV_ESC_RM_MAP_MEMORY, from
// src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct IoctlNVOS33ParametersWithFD {
    pub params : NVOS33Parameters,
	pub fd     : i32,
	// pub pad0   : u32,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS33Parameters {
    pub client        : Handle,
	pub device        : Handle,
	pub memory        : Handle,
	//pub Pad0           : u32,
	pub offset         : u64,
	pub length         : u64,
	pub linearAddress : P64, // address of application mapping, without indirection
	pub status         : u32,
	pub flags          : u32,
}

// NVOS34Parameters is NVOS34_PARAMETERS, the parameter type for
// NV_ESC_RM_UNMAP_MEMORY.
//
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct NVOS34Parameters {
    pub client        : Handle,
	pub device        : Handle,
	pub memory        : Handle,
	//pub Pad0           : u32,
	pub linearAddress : P64, // address of application mapping, without indirection
	pub status         : u32,
	pub flags          : u32,
}

// NVOS54Parameters is NVOS54_PARAMETERS, the parameter type for
// NV_ESC_RM_CONTROL.
//
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NVOS54Parameters {
    pub client    : Handle,
	pub object    : Handle,
	pub cmd        : u32,
	pub flags      : u32,
	pub params     : P64,
	pub paramsSize : u32,
	pub status     : u32,
}

// NVOS56Parameters is NVOS56_PARAMETERS, the parameter type for
// NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO.
//
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NVOS56Parameters {
    pub client        : Handle,
	pub device        : Handle,
	pub memory        : Handle,
	//pub Pad0           : u32,
	pub oldCPUAddress : P64,
	pub newCPUAddress : P64,
	pub status         : u32,
	//pub Pad1           : u32,
}

// NVOS64Parameters is NVOS64_PARAMETERS, one possible parameter type for
// NV_ESC_RM_ALLOC.
//
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct NVOS64Parameters {
    pub root            : Handle,
	pub objectParent    : Handle,
	pub objectNew       : Handle,
	pub class           : u32,
	pub allocParms      : P64,
	pub rightsRequested : P64,
	pub flags            : u32,
	pub status           : u32,
}

pub const SIZEOF_IOCTL_REGISTER_FD                  : u32 = core::mem::size_of::<IoctlRegisterFD>() as u32;
pub const SIZEOF_IOCTL_ALLOC_OSEVENT                : u32 = core::mem::size_of::<IoctlAllocOSEvent>() as u32;
pub const SIZEOF_IOCTL_FREE_OSEVENT                 : u32 = core::mem::size_of::<IoctlFreeOSEvent>() as u32;
pub const SIZEOF_RMAPIVERSION                       : u32 = core::mem::size_of::<RMAPIVersion>() as u32;
pub const SIZEOF_IOCTL_SYS_PARAMS                   : u32 = core::mem::size_of::<IoctlSysParams>() as u32;
pub const SIZEOF_IOCTL_NVOS02_PARAMETERS_WITH_FD    : u32 = core::mem::size_of::<IoctlNVOS02ParametersWithFD>() as u32;
pub const SIZEOF_NVOS00_PARAMETERS                  : u32 = core::mem::size_of::<NVOS00Parameters>() as u32;
pub const SIZEOF_NVOS21_PARAMETERS                  : u32 = core::mem::size_of::<NVOS21Parameters>() as u32;
pub const SIZEOF_IOCTL_NVOS33_PARAMETERS_WITH_FD    : u32 = core::mem::size_of::<IoctlNVOS33ParametersWithFD>() as u32;
pub const SIZEOF_NVOS55_PARAMETERS                  : u32 = core::mem::size_of::<NVOS55Parameters>() as u32;
pub const SIZEOF_NVOS57_PARAMETERS                  : u32 = core::mem::size_of::<NVOS57Parameters>() as u32;
pub const SIZEOF_NVOS32_PARAMETERS                  : u32 = core::mem::size_of::<NVOS32Parameters>() as u32;
pub const SIZEOF_NVOS34_PARAMETERS                  : u32 = core::mem::size_of::<NVOS34Parameters>() as u32;
pub const SIZEOF_NVOS54_PARAMETERS                  : u32 = core::mem::size_of::<NVOS54Parameters>() as u32;
pub const SIZEOF_NVOS56_PARAMETERS                  : u32 = core::mem::size_of::<NVOS56Parameters>() as u32;
pub const SIZEOF_NVOS64_PARAMETERS                  : u32 = core::mem::size_of::<NVOS64Parameters>() as u32;
