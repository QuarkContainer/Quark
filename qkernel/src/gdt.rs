// Copyright (c) 2021 QuarkSoft LLC
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

use lazy_static::lazy_static;
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;
use x86_64::VirtAddr;
//use crate::println;

pub const DOUBLE_FAULT_IST_INDEX: u16 = 0;

lazy_static! {
    static ref TSS: TaskStateSegment = {
        let mut tss = TaskStateSegment::new();

        let stack = {
            const STACK_SIZE: usize = 4096;
            static mut STACK: [u8; STACK_SIZE] = [0; STACK_SIZE];

            let stack_start = VirtAddr::from_ptr(unsafe { &STACK });
            let stack_end = stack_start + STACK_SIZE;
            stack_end
        };

        tss.interrupt_stack_table[DOUBLE_FAULT_IST_INDEX as usize] = stack;
        tss
    };
}

lazy_static! {
    static ref GDT: (GlobalDescriptorTable, Selectors) = {
        let mut gdt = GlobalDescriptorTable::new();
        let code_selector = gdt.add_entry(Descriptor::kernel_code_segment());
        //let data_selector = gdt.add_entry(Descriptor::kernel_code_segment());
        let tss_selector = gdt.add_entry(Descriptor::tss_segment(&TSS));

        (
            gdt,
            Selectors {
                code_selector,
                //data_selector,
                tss_selector,
            },
        )
    };
}

struct Selectors {
    code_selector: SegmentSelector,
    //data_selector: SegmentSelector,
    tss_selector: SegmentSelector,
}

pub fn init() {
    //use x86_64::instructions::segmentation::set_cs;
    //use x86_64::instructions::tables::load_tss;


    let gdtAddr: u64 = &GDT.0 as *const _ as u64;
    let limit = (64 - 1) as u16; // 8 * 8

    info!("the gdt is {:?}", GDT.0);
    info!("the gdtAddr is {:x}, the limit is {:x}", gdtAddr, limit);
    info!("the code_selector is {:?}, the ts_selector is {:?}",
    GDT.1.code_selector, GDT.1.tss_selector);

    //GDT.0.load();

    //super::Kernel::Kernel::LoadGDT(gdtAddr, limit);
    info!("after loadgdt");

    /*unsafe {
        //set_cs(GDT.1.code_selector);
        //load_tss(GDT.1.tss_selector);
        info!("start load ds");
        //x86_64::instructions::segmentation::load_ds(GDT.1.data_selector);
        //x86_64::instructions::segmentation::load_ss(GDT.1.data_selector);
        //x86_64::instructions::segmentation::load_es(GDT.1.data_selector);
        //x86_64::instructions::segmentation::load_fs(GDT.1.data_selector);
        //x86_64::instructions::segmentation::load_gs(GDT.1.data_selector);
    }*/
}
