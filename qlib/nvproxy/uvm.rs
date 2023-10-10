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

pub trait HasRMCtrlFD  {
    fn GetRMCtrlFD (&self) -> i32;
	fn SetRMCtrlFD (&mut self, fd: i32);
}

// UVM ioctl commands.

// From kernel-open/nvidia-uvm/uvm_linux_ioctl.h:
pub const UVM_INITIALIZE   : u64 = 0x30000001;
pub const UVM_DEINITIALIZE : u64 = 0x30000002;

// From kernel-open/nvidia-uvm/uvm_ioctl.h:
pub const UVM_CREATE_RANGE_GROUP             : u64 = 23;
pub const UVM_DESTROY_RANGE_GROUP            : u64 = 24;
pub const UVM_REGISTER_GPU_VASPACE           : u64 = 25;
pub const UVM_UNREGISTER_GPU_VASPACE         : u64 = 26;
pub const UVM_REGISTER_CHANNEL               : u64 = 27;
pub const UVM_UNREGISTER_CHANNEL             : u64 = 28;
pub const UVM_MAP_EXTERNAL_ALLOCATION        : u64 = 33;
pub const UVM_FREE                           : u64 = 34;
pub const UVM_REGISTER_GPU                   : u64 = 37;
pub const UVM_UNREGISTER_GPU                 : u64 = 38;
pub const UVM_PAGEABLE_MEM_ACCESS            : u64 = 39;
pub const UVM_MAP_DYNAMIC_PARALLELISM_REGION : u64 = 65;
pub const UVM_ALLOC_SEMAPHORE_POOL           : u64 = 68;
pub const UVM_VALIDATE_VA_RANGE              : u64 = 72;
pub const UVM_CREATE_EXTERNAL_RANGE          : u64 = 73;

// UVM_INITIALIZE_PARAMS
#[repr(C)]
pub struct UvmInitializeParams {
    pub flags    : u64, 
	pub rMStatus : u32, 
	// pub Pad0     : u32,
}

// UVM_INITIALIZE_PARAMS flags, from kernel-open/nvidia-uvm/uvm_types.h.
pub const UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE : i32 = 0x2;

// UVM_CREATE_RANGE_GROUP_PARAMS
#[repr(C)]
pub struct UvmCreateRangeGroupParams {
	pub rangeGroupID : u64,
	pub rMStatus     : u32,
	// pub pad0         : u32,
}


// UVM_DESTROY_RANGE_GROUP_PARAMS
#[repr(C)]
pub struct UvmDestroyRangeGroupParams {
	pub rangeGroupID : u64,
	pub rMStatus     : u32,
	// pub pad0         : u32,
}

// UVM_REGISTER_GPU_VASPACE_PARAMS
#[repr(C)]
pub struct UvmRegisterGpuVaspaceParams {
    pub GPUUUID  : [u8; 16],
	pub RMCtrlFD  : i32,
	pub client  : Handle,
	pub vASpace : Handle,
	pub rMStatus : u32,
}

impl HasRMCtrlFD  for UvmRegisterGpuVaspaceParams {
    fn GetRMCtrlFD (&self) -> i32 {
        return self.RMCtrlFD 
    }

	fn SetRMCtrlFD (&mut self, fd: i32) {
        self.RMCtrlFD  = fd;
    }
}

// UVM_REGISTER_CHANNEL_PARAMS
#[repr(C)]
pub struct UvmRegisterChannelParams {
	pub GPUUUID  : [u8; 16],
	pub RMCtrlFD  : i32,
	pub client  : Handle,
	pub channel : Handle,
	// pub Pad      : u32,
	pub base     : u64,
	pub length   : u64,
	pub rMStatus : u32,
	// pub Pad0     : u32,
}

impl HasRMCtrlFD  for UvmRegisterChannelParams {
    fn GetRMCtrlFD (&self) -> i32 {
        return self.RMCtrlFD 
    }

	fn SetRMCtrlFD (&mut self, fd: i32) {
        self.RMCtrlFD  = fd;
    }
}

// UVM_UNREGISTER_CHANNEL_PARAMS
#[repr(C)]
pub struct UvmUnregisterChannelParams {
    pub GPUUUID  : [u8; 16],
	pub client  : Handle,
	pub channel : Handle,
	pub rMStatus : u32,
}

// UVM_MAP_EXTERNAL_ALLOCATION_PARAMS
#[repr(C)]
pub struct UvmMapExternalAllocationParams {
    pub base               : u64,
	pub length             : u64,
	pub offset             : u64,
	pub perGPUAttributes   : [UvmGpuMappingAttributes; UVM_MAX_GPUS],
	pub GPUAttributesCount : u64,
	pub RMCtrlFD           : i32,
	pub client            : Handle,
	pub memory            : Handle,
	pub RMStatus           : u32,
}

impl HasRMCtrlFD  for UvmMapExternalAllocationParams {
    fn GetRMCtrlFD (&self) -> i32 {
        return self.RMCtrlFD 
    }

	fn SetRMCtrlFD (&mut self, fd: i32) {
        self.RMCtrlFD  = fd;
    }
}

// UVM_FREE_PARAMS
#[repr(C)]
pub struct UvmFreeParams {
	pub base     : u64,
	pub length   : u64,
	pub RMStatus : u32,
	// pub Pad0     : u32,
}

// UVM_REGISTER_GPU_PARAMS
#[repr(C)]
pub struct UvmRegisterGpuParams {
	pub GPUUUID     : [u8; 16],
	pub numaEnabled : u8,
	// pub pad         : [u8; 3],
	pub numaNodeID  : i32,
	pub RMCtrlFD    : i32,
	pub client      : Handle,
	pub SMCPartRef  : Handle,
	pub RMStatus    : u32,
}

impl HasRMCtrlFD  for UvmRegisterGpuParams {
    fn GetRMCtrlFD (&self) -> i32 {
        return self.RMCtrlFD 
    }

	fn SetRMCtrlFD (&mut self, fd: i32) {
        self.RMCtrlFD  = fd;
    }
}

// UVM_UNREGISTER_GPU_PARAMS
#[repr(C)]
pub struct UvmUnregisterGpuParams {
	pub GPUUUID     : [u8; 16],
    pub RMStatus    : u32,
}

// UVM_PAGEABLE_MEM_ACCESS_PARAMS
#[repr(C)]
pub struct UvmPageableMemAccessParams {
	pub pageableMemAccess : u8,
	//pub Pad               : [u8; 3],
	pub RMStatus          : u32,
}

// UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS
#[repr(C)]
pub struct UvmMapDynamicParallelismRegionParams {
    pub base     : u64,
	pub length   : u64,
	pub GPUUUID  : [u8; 16],
	pub RMStatus : u32,
	// pub Pad0     : [u8; 4],
}

// UVM_ALLOC_SEMAPHORE_POOL_PARAMS
#[repr(C)]
pub struct UvmAllocSemaphorePoolParams {
	pub base               : u64,
	pub length             : u64,
	pub perGPUAttributes   : [UvmGpuMappingAttributes; UVM_MAX_GPUS],
	pub GPUAttributesCount : u64,
	pub RMStatus           : u32,
	// pub Pad0               : [u8; 4],
}

// UVM_VALIDATE_VA_RANGE_PARAMS
#[repr(C)]
pub struct UvmValidateVaRangeParams {
	pub base     : u64,
	pub length   : u64,
	pub RMStatus : u32,
	// pub Pad0     : [u8; 4],
}

// UVM_CREATE_EXTERNAL_RANGE_PARAMS
#[repr(C)]
pub struct UvmCreateExternalRangeParams {
	pub base     : u64,
	pub length   : u64,
	pub RMStatus : u32,
	// pub Pad0     : [u8; 4],
}

// From kernel-open/nvidia-uvm/uvm_types.h:

pub const UVM_MAX_GPUS : usize = NV_MAX_DEVICES;

// UvmGpuMappingAttributes
#[repr(C)]
pub struct UvmGpuMappingAttributes {
	pub GPUUUID            : [u8; 16],
	pub GPUMappingType     : u32,
	pub GPUCachingType     : u32,
	pub GPUFormatType      : u32,
	pub GPUElementBits     : u32,
	pub GPUCompressionType : u32,
}