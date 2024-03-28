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

use kvm_bindings::{kvm_sregs, kvm_xcrs};
use core::mem::size_of;

use crate::VMS;
use crate::amd64_def::{SegmentDescriptor,
                       SEGMENT_DESCRIPTOR_PRESENT,
                       SEGMENT_DESCRIPTOR_ACCESS,
                       SEGMENT_DESCRIPTOR_WRITE,
                       SEGMENT_DESCRIPTOR_EXECUTE};
use crate::qlib::common::Error;
use crate::kvm_vcpu::AlignedAllocate;
use crate::qlib::linux_def::MemoryDef;
use crate::qlib::kernel::asm::xgetbv;
use crate::arch::__cpu_arch::x86_64vCPU;
use crate::qlib::common::{CR0_PE, CR0_AM, CR0_ET, CR0_PG, CR0_NE,
                          CR4_PSE, CR4_PAE, CR4_PGE, CR4_OSFXSR,
                          CR4_OSXMMEXCPT, CR4_FSGSBASE, CR4_OSXSAVE,
                          EFER_LME, EFER_LMA, EFER_SCE, EFER_NX,
                          KCODE, UDATA, KDATA, TSS};
use crate::qlib::cpuid::XSAVEFeature::{XSAVEFeatureBNDCSR,
                                       XSAVEFeatureBNDREGS};

impl x86_64vCPU {
   pub(in super::super::super::__cpu_arch) fn vcpu_reg_init(&mut self, id: usize)
   -> Result<(), Error> {
        self.gdtAddr = AlignedAllocate(
            MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize,
            true,
        )?;

        self.idtAddr = AlignedAllocate(
            MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize,
            true,
        )?;

        self.tssIntStackStart = AlignedAllocate(
            MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize,
            true,
        )?;

        self.tssAddr = AlignedAllocate(
            MemoryDef::PAGE_SIZE as usize,
            MemoryDef::PAGE_SIZE as usize,
            true,
        )?;

        info!("The tssIntStackStart is {:x}, tssAddr address is {:x}, idt addr is {:x}, gdt addr is {:x}",
             self.tssIntStackStart, self.tssAddr, self.idtAddr, self.gdtAddr);
       self.setup_long_mode();
       self.SetXCR0()?;

        Ok(())
    }

    fn setup_long_mode(&self) -> Result<(), Error> {
        let mut vcpu_sregs = self
            .vcpu_fd
            .unwrap()
            .get_sregs()
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;

        //vcpu_sregs.cr0 = CR0_PE | CR0_MP | CR0_AM | CR0_ET | CR0_NE | CR0_WP | CR0_PG;
        vcpu_sregs.cr0 = CR0_PE | CR0_AM | CR0_ET | CR0_PG | CR0_NE; // | CR0_WP; // | CR0_MP | CR0_NE;
        vcpu_sregs.cr3 = VMS.lock().pageTables.GetRoot();
        //vcpu_sregs.cr4 = CR4_PAE | CR4_OSFXSR | CR4_OSXMMEXCPT;
        vcpu_sregs.cr4 =
            CR4_PSE | CR4_PAE | CR4_PGE | CR4_OSFXSR | CR4_OSXMMEXCPT
            | CR4_FSGSBASE | CR4_OSXSAVE; // | CR4_UMIP ;// CR4_PSE | | CR4_SMEP | CR4_SMAP;

        vcpu_sregs.efer = EFER_LME | EFER_LMA | EFER_SCE | EFER_NX;

        vcpu_sregs.idt = kvm_bindings::kvm_dtable {
            base: 0,
            limit: 4095,
            ..Default::default()
        };

        vcpu_sregs.gdt = kvm_bindings::kvm_dtable {
            base: self.gdtAddr,
            limit: 4095,
            ..Default::default()
        };

        self.SetupGDT(&mut vcpu_sregs);
        self.vcpu_fd
            .unwrap()
            .set_sregs(&vcpu_sregs)
            .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
        Ok(())
    }

    fn SetupGDT(&self, sregs: &mut kvm_sregs) {
        let gdtTbl = unsafe {
            std::slice::from_raw_parts_mut(
                self.gdtAddr as *mut u64, 4096 / 8)
        };

        let KernelCodeSegment = SegmentDescriptor::default().SetCode64(0, 0, 0);
        let KernelDataSegment = SegmentDescriptor::default().SetData(0, 0xffffffff, 0);
        let _UserCodeSegment32 = SegmentDescriptor::default().SetCode64(0, 0, 3);
        let UserDataSegment = SegmentDescriptor::default().SetData(0, 0xffffffff, 3);
        let UserCodeSegment64 = SegmentDescriptor::default().SetCode64(0, 0, 3);

        sregs.cs = KernelCodeSegment.GenKvmSegment(KCODE);
        sregs.ds = UserDataSegment.GenKvmSegment(UDATA);
        sregs.es = UserDataSegment.GenKvmSegment(UDATA);
        sregs.ss = KernelDataSegment.GenKvmSegment(KDATA);
        sregs.fs = UserDataSegment.GenKvmSegment(UDATA);
        sregs.gs = UserDataSegment.GenKvmSegment(UDATA);

        gdtTbl[1] = KernelCodeSegment.AsU64();
        gdtTbl[2] = KernelDataSegment.AsU64();
        gdtTbl[3] = UserDataSegment.AsU64();
        gdtTbl[4] = UserCodeSegment64.AsU64();

        let stack_end = x86_64::VirtAddr::from_ptr((
            self.tssIntStackStart
            + MemoryDef::INTERRUPT_STACK_PAGES
            * MemoryDef::PAGE_SIZE)
            as *const u64);

        let tssSegment = self.tssAddr as *mut x86_64::structures::tss::TaskStateSegment;
        unsafe {
            (*tssSegment).interrupt_stack_table[0] = stack_end;
            (*tssSegment).iomap_base = -1 as i16 as u16;
            let (tssLow, tssHigh, limit) = Self::TSStoDescriptor(&(*tssSegment));

            gdtTbl[5] = tssLow;
            gdtTbl[6] = tssHigh;

            sregs.tr = SegmentDescriptor::New(tssLow).GenKvmSegment(TSS);
            sregs.tr.base = self.tssAddr;
            sregs.tr.limit = limit as u32;
        }
    }

    fn TSStoDescriptor(tss: &x86_64::structures::tss::TaskStateSegment) -> (u64, u64, u16) {
        let (tssBase, tssLimit) = Self::TSS(tss);
        let low = SegmentDescriptor::default().Set(
            tssBase as u32,
            tssLimit as u32,
            0,
            SEGMENT_DESCRIPTOR_PRESENT
                | SEGMENT_DESCRIPTOR_ACCESS
                | SEGMENT_DESCRIPTOR_WRITE
                | SEGMENT_DESCRIPTOR_EXECUTE,
        );

        let hi = SegmentDescriptor::default().SetHi((tssBase >> 32) as u32);
        return (low.AsU64(), hi.AsU64(), tssLimit);
    }

    fn TSS(tss: &x86_64::structures::tss::TaskStateSegment) -> (u64, u16) {
        let addr = tss as *const _ as u64;
        let size = (size_of::<x86_64::structures::tss::TaskStateSegment>() - 1) as u64;
        return (addr, size as u16);
    }

    fn SetXCR0(&self) -> Result<(), Error> {
        let xcr0 = xgetbv();
        // mask MPX feature as it is not fully supported in VM yet
        let maskedXCR0 = xcr0 & !(XSAVEFeatureBNDREGS as u64
                                  | XSAVEFeatureBNDCSR as u64);
        let mut xcrs_args = kvm_xcrs::default();
        xcrs_args.nr_xcrs = 1;
        xcrs_args.xcrs[0].value = maskedXCR0;
        self.vcpu_fd
            .unwrap()
            .set_xcrs(&xcrs_args)
            .map_err(|e| Error::IOError(format!("failed to set kvm xcr0, {}", e)))?;
        Ok(())
    }
}
