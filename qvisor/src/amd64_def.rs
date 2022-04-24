// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
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

use kvm_bindings::kvm_segment;

pub type SegmentDescriptorFlags = u32;

pub const SEGMENT_DESCRIPTOR_ACCESS: SegmentDescriptorFlags = 1 << 8; // Access bit (always set).
pub const SEGMENT_DESCRIPTOR_WRITE: SegmentDescriptorFlags = 1 << 9; // Write permission.
pub const SEGMENT_DESCRIPTOR_EXPAND_DOWN: SegmentDescriptorFlags = 1 << 10; // Grows down, not used.
pub const SEGMENT_DESCRIPTOR_EXECUTE: SegmentDescriptorFlags = 1 << 11; // Execute permission.
pub const SEGMENT_DESCRIPTOR_SYSTEM: SegmentDescriptorFlags = 1 << 12; // Zero => system, 1 => user code/data.
pub const SEGMENT_DESCRIPTOR_PRESENT: SegmentDescriptorFlags = 1 << 15; // Present.
pub const SEGMENT_DESCRIPTOR_AVL: SegmentDescriptorFlags = 1 << 20; // Available.
pub const SEGMENT_DESCRIPTOR_LONG: SegmentDescriptorFlags = 1 << 21; // Long mode.
pub const SEGMENT_DESCRIPTOR_DB: SegmentDescriptorFlags = 1 << 22; // 16 or 32-bit.
pub const SEGMENT_DESCRIPTOR_G: SegmentDescriptorFlags = 1 << 23; // Granularity: page or byte.

//pub const CR0_PE : u64 = 1 << 0;
//pub const CR0_ET : u64 = 1 << 4;
//pub const CR0_AM : u64 = 1 << 18;
//pub const CR0_PG : u64 = 1 << 31;

pub fn ToBool(x: u32) -> u8 {
    if x != 0 {
        return 1;
    }

    return 0;
}

#[repr(C)]
#[derive(Default)]
pub struct SegmentDescriptor {
    pub bits: [u32; 2],
}

impl SegmentDescriptor {
    pub fn New(data: u64) -> Self {
        let mut ret = Self::default();
        unsafe { *(&mut ret.bits[0] as *mut _ as *mut u64) = data }

        ret
    }

    pub fn GenKvmSegment(&self, selector: u16) -> kvm_segment {
        let flag = self.Flags();

        let mut seg = kvm_segment {
            base: self.Base() as u64,
            limit: self.Limit(),
            type_: ((flag >> 8) & 0xf) as u8 | 1,
            s: ToBool(flag & SEGMENT_DESCRIPTOR_SYSTEM),
            dpl: self.DPL() as u8,
            present: ToBool(flag & SEGMENT_DESCRIPTOR_PRESENT),
            avl: ToBool(flag & SEGMENT_DESCRIPTOR_AVL),
            l: ToBool(flag & SEGMENT_DESCRIPTOR_LONG),
            db: ToBool(flag & SEGMENT_DESCRIPTOR_DB),
            g: ToBool(flag & SEGMENT_DESCRIPTOR_G),
            unusable: 0,
            padding: 0,
            selector: selector,
        };

        if seg.l != 0 {
            seg.limit = 0xffffffff
        }

        return seg;
    }

    pub fn AsU64(&self) -> u64 {
        return unsafe { *(&self.bits[0] as *const _ as *const u64) };
    }

    // Base returns the descriptor's base linear address.
    pub fn Base(&self) -> u32 {
        return self.bits[1] & 0xFF000000 | (self.bits[1] & 0x000000FF) << 16 | self.bits[0] >> 16;
    }

    pub fn Limit(&self) -> u32 {
        let mut l = self.bits[0] & 0xFFFF | self.bits[1] & 0xF0000;
        if self.bits[1] & SEGMENT_DESCRIPTOR_G as u32 != 0 {
            l <<= 12;
            l |= 0xFFF;
        }

        return l;
    }

    pub fn Flags(&self) -> SegmentDescriptorFlags {
        return self.bits[1] & 0x00F09F00;
    }

    pub fn DPL(&self) -> i32 {
        return ((self.bits[1] >> 13) & 3) as i32;
    }

    pub fn SetNull(&mut self) {
        self.bits[0] = 0;
        self.bits[1] = 0;
    }

    pub fn Set(mut self, base: u32, limit: u32, dpl: i32, flags: SegmentDescriptorFlags) -> Self {
        let mut flags = flags | SEGMENT_DESCRIPTOR_PRESENT;
        let mut limit = limit;
        if limit >> 12 > 0 {
            limit >>= 12;
            flags |= SEGMENT_DESCRIPTOR_G;
        }

        self.bits[0] = base << 16 | limit & 0xFFFF;
        self.bits[1] = base & 0xFF000000
            | (base >> 16) & 0xFF
            | limit & 0x000F0000
            | flags as u32
            | (dpl << 13) as u32;
        self
    }

    pub fn SetCode32(self, base: u32, limit: u32, dpl: i32) -> Self {
        self.Set(
            base,
            limit,
            dpl,
            SEGMENT_DESCRIPTOR_DB | SEGMENT_DESCRIPTOR_EXECUTE | SEGMENT_DESCRIPTOR_SYSTEM,
        )
    }

    pub fn SetCode64(self, base: u32, limit: u32, dpl: i32) -> Self {
        self.Set(
            base,
            limit,
            dpl,
            SEGMENT_DESCRIPTOR_G
                | SEGMENT_DESCRIPTOR_LONG
                | SEGMENT_DESCRIPTOR_EXECUTE
                | SEGMENT_DESCRIPTOR_SYSTEM,
        )
    }

    pub fn SetData(self, base: u32, limit: u32, dpl: i32) -> Self {
        self.Set(
            base,
            limit,
            dpl,
            SEGMENT_DESCRIPTOR_WRITE | SEGMENT_DESCRIPTOR_SYSTEM,
        )
    }

    pub fn SetHi(mut self, base: u32) -> Self {
        self.bits[0] = base;
        self.bits[1] = 0;
        self
    }
}
