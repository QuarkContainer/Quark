// Copyright (c) 2021 Quark Container Authors
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

//use crate::qlib::kernel::TSC;
//use crate::qlib::singleton::*;
#![allow(unused_imports)]
use core::fmt;
use core::fmt::{LowerHex, Formatter};

use crate::qlib::common::{Error, TaskRunState};
use crate::qlib::vcpu_mgr::{CPULocal, VcpuMode};
use crate::qlib::kernel::{task::Task, threadmgr::task_sched::SchedState,
                          asm::aarch64::{CurrentUserTable},
                          SignalDef::{PtRegs, SignalInfo}};
use crate::interrupt::aarch64::{EsrDefs, GetFaultAccessType, ReturnToApp};
use crate::qlib::addr::{Addr, AccessType};
use crate::qlib::linux_def::MemoryDef;
use crate::qlib::linux_def::{MmapProt, Signal};
use crate::qlib::pagetable::PageTables;
use crate::qlib::kernel::arch::__arch::mm::pagetable::{PageTableEntry, PageTableFlags};


use crate::{MainRun, SHARESPACE, panic};

bitflags! {
    #[repr(transparent)]
    pub struct PageFaultErrorCode: u64 {
        // Types (should be mutual exclusive)
        const NO_HANDLER            = 1 << 0; //Unhandled, Unforeseen, Unforgiven! Kill it.
        const PROTECTION_VIOLATION  = 1 << 1;
        const ADDRESS_SIZE          = 1 << 2;
        const TRANSLATION           = 1 << 3;
        const ACCESS_FLAG           = 1 << 4;
        const TYPE_MASK             = 0xff;
        // Attributes
        const CAUSED_BY_WRITE       = 1 << 8;
        const USER_MODE             = 1 << 9;
        const INSTRUCTION_FETCH     = 1 << 10;
    }
}

impl EsrDefs {
    // ESR.ISS.xxSC[5:2], Data/Instruction Fault Status Code WITHOUT translation level.
    pub const SC_MASK: u64 = 0x3c;
    pub const SC_ADDRESS_SIZE_FAULT: u64 = 0x0;
    pub const SC_TRANSLATION_FAULT:  u64 = 0x4;
    pub const SC_ACCESS_FLAG_FAULT:  u64 = 0x8;
    pub const SC_PERMISSION_FAULT:   u64 = 0xc;
}

impl PageFaultErrorCode {
    pub fn get_type(&self) -> Self{
        *self & Self::TYPE_MASK
    }

    pub fn new(from_user: bool, is_instr: bool, esr: u64) -> Self {
        let mut fault_flags = Self::empty();

        if from_user {
           fault_flags.insert(Self::USER_MODE);
        }

        if is_instr {
            fault_flags.insert(Self::INSTRUCTION_FETCH);
        } else {
            fault_flags.set(Self::CAUSED_BY_WRITE, EsrDefs::IsWrite(esr));
        }

        let xxsc = esr & EsrDefs::SC_MASK;
        match xxsc {
            EsrDefs::SC_PERMISSION_FAULT  => {
                fault_flags.insert(Self::PROTECTION_VIOLATION);
            },
            EsrDefs::SC_ACCESS_FLAG_FAULT => {
                fault_flags.insert(Self::ACCESS_FLAG);
            },
            EsrDefs::SC_ADDRESS_SIZE_FAULT => {
                fault_flags.insert(Self::ADDRESS_SIZE);
            },
            EsrDefs::SC_TRANSLATION_FAULT => {
                fault_flags.insert(Self::TRANSLATION);
            },
            _ => {
                fault_flags.insert(Self::NO_HANDLER);
            }
        };

        fault_flags
    }
}


pub fn PageFaultHandler(ptRegs: &mut PtRegs, fault_address: u64,
                        error_code: PageFaultErrorCode) {
    CPULocal::Myself().SetMode(VcpuMode::Kernel);
    let currTask = Task::Current();
    // is this call from user
    let fromUser: bool = error_code
        .contains(PageFaultErrorCode::USER_MODE);
    //
    // NOTE: ATM only 0x0000xx...x address-space is used.
    //
    let ttbr = CurrentUserTable();

    if !SHARESPACE.config.read().CopyDataWithPf && !fromUser {
        error!("VM: PageFault in kernel FAR: {:#x}, TTBR: {:#x}, PtRegs: {:#x}, error_code: {:#x}",
               fault_address, ttbr, ptRegs, error_code);
       //backtracer::trace(ptRegs.pc, ptRegs.get_stack_pointer(), ptRegs.rbp, &mut |frame| {
       //      print!("pagefault frame is {:#x?}", frame);
        //     true
        // });
        panic!("VM: PageFault from kernel non recuperable.");
    }

    //
    // Ignore as PerfStuff...
    //
    //currTask.PerfGoto(PerfType::PageFault);
    //defer!(Task::Current().PerfGofrom(PerfType::PageFault));

    let PRINT_EXECPTION: bool = SHARESPACE.config.read().PrintException;
    if PRINT_EXECPTION {
        info!("VM: PageFaultHandler - FAR: {:#x}, PC: {:#x},\
               TTBR: {:#x}, is-user: {}, error code: {:#x}.",
               fault_address, ptRegs.pc, ttbr, fromUser, error_code);
    }

    let signal;
    // no need loop, just need to enable break
    loop {
        let _ml = currTask.mm.MappingWriteLock();
        let (vma, range) = match currTask.mm
            .GetVmaAndRangeLocked(fault_address) {
            None => {
                if fault_address > MemoryDef::PAGE_SIZE {
                    let map = currTask.mm.GetSnapshotLocked(currTask, false);
                    error!("VM: The map is {}, fault address is not part of it.", &map);
                }
                signal = Signal::SIGSEGV;
                break;
            }
            Some(vma) => vma.clone(),
        };

        if error_code.contains(PageFaultErrorCode::ACCESS_FLAG) {
           let bind = currTask
                     .mm
                     .pagetable
                     .write();
           let pte = bind
                     .pt
                     .VirtualToEntry(fault_address).unwrap();
           let mut flags = pte.flags();
           if flags.contains(PageTableFlags::ACCESSED) {
                panic!("VM: Error - PF with Accessed-Flag not set while flag set in PTE.");
           } else {
               flags.insert(PageTableFlags::ACCESSED);
               let fault_addr_alligned = Addr(fault_address)
                                         .RoundDown()
                                         .unwrap();
               bind.pt.SetPageFlags(fault_addr_alligned, flags);
               return;
           }
        }

        //
        // PF in Kernel VMA cannot be handlet => Should not happen!
        //
        if vma.kernel == true {
            let k_map = currTask.mm.GetSnapshotLocked(currTask, false);
            info!("VM: vma_kernel:True - k_map:{}", &k_map);
            signal = Signal::SIGSEGV;
            break;
        }

        if !vma.effectivePerms.Read() {
            error!("VM: No Read-Permission on mem-area.");
            signal = Signal::SIGSEGV;
            break;
        }

        let pageAddr = Addr(fault_address).RoundDown().unwrap().0;
        assert!(range.Contains(pageAddr),
            "PageFaultHandler vm-addr is not in the VM-Area range"
        );

        //
        // Fault for not mapped page.
        //
        if error_code.contains(PageFaultErrorCode::TRANSLATION) {
            info!("VM: InstallPage 1, range is {:x?}, address is {:#x}, vma.growsDown is {}",
               &range, pageAddr, vma.growsDown);
            //let startTime = TSC.Rdtsc();
            let addr = currTask
                .mm
                .pagetable
                .write()
                .pt
                .SwapInPage(Addr(pageAddr))
                .unwrap();
            //let endtime = TSC.Rdtsc();
            if addr > 0 {
                //use crate::qlib::kernel::Tsc;
                info!("VM: Page {:x?}/{:x} is mapped", Addr(pageAddr).RoundDown().unwrap(), addr/*, Tsc::Scale(endtime - startTime)*/);
                //
                // Check if PAGE-/VALID-flagbits are set.
                // We take this path after being sure from above that the page
                // is present.
                //
                {
                    let bind = currTask
                        .mm
                        .pagetable
                        .write();
                    let pte = bind.pt
                        .VirtualToEntry(pageAddr).unwrap();
                    debug!("VM: Found virt-addr - {:#x}; PTE - {:?}", pageAddr, *pte);
                    let mut flags = pte.flags();
                    let page_bit_set = flags.contains(PageTableFlags::PAGE);
                    let valid_bit_set = flags.contains(PageTableFlags::VALID);
                    if valid_bit_set && page_bit_set {
                         panic!("VM: Error - Translation-PF with mapped page, PAGE-/VALID-Flag are set in PTE.");
                    } else {
                        if !page_bit_set {
                            flags.insert(PageTableFlags::PAGE);
                        }
                        if !valid_bit_set {
                            flags.insert(PageTableFlags::VALID);
                        }
                    }
                    bind.pt.SetPageFlags(Addr(pageAddr), flags);
                }
                return;
            }
            //
            // Could not swap in page
            //
            match currTask.mm
                .InstallPageLocked(currTask, &vma, pageAddr, &range)
            {
                Err(Error::FileMapError) => {
                    error!("VM: Installing page failed with FILE_MAP_ERROR.");
                    signal = Signal::SIGBUS;
                    break;
                }
                Err(e) => {
                    panic!("VM: Installing page failed with panic. PageFaultHandler error is {:?}", e)
                }
                _ => (),
            };

            for i in 1..8 {
                let addr = if vma.growsDown {
                    pageAddr - i * MemoryDef::PAGE_SIZE
                } else {
                    pageAddr + i * MemoryDef::PAGE_SIZE
                };

                if range.Contains(addr) {
                    match currTask.mm.InstallPageLocked(currTask, &vma, addr, &range) {
                        Err(_) => {
                            break;
                        }
                        _ => (),
                    };
                } else {
                    break;
                }
            }

            if fromUser {
                //PerfGoto(PerfType::User);
                currTask.AccountTaskEnter(SchedState::RunningApp);
            }
            CPULocal::Myself().SetMode(VcpuMode::User);
            currTask.mm.HandleTlbShootdown();
            return;
        }

        //
        // NOTE: Handle possible COW-modyfie events.
        // NOTE: COW flags RO set in ForkRange() for private mappings.
        //
        if vma.private == true {
            if error_code.contains(PageFaultErrorCode::PROTECTION_VIOLATION) &&
                error_code.contains(PageFaultErrorCode::CAUSED_BY_WRITE) {
                    if vma.effectivePerms.Write() == false  ||
                    (vma.kernel == true &&
                        error_code
                        .contains(PageFaultErrorCode::USER_MODE)) {
                        debug!("VM: Fault on private vma - Write access on \
                               VMA-perms:{:?}/kernel:{:#}.", vma.realPerms.Effective(), vma.kernel);
                        signal = Signal::SIGSEGV;
                        break;
                    } else {
                       //
                       // Handle COW write-event
                       //
                       debug!("VM: Handle COW-Write event.");
                       currTask.mm.CopyOnWriteLocked(pageAddr, &vma);
                       currTask.mm.TlbShootdown();
                       if fromUser {
                           //PerfGoto(PerfType::User);
                           currTask.AccountTaskEnter(SchedState::RunningApp);
                       }
                    }
            }
        }

        CPULocal::Myself().SetMode(VcpuMode::User);
        currTask.mm.HandleTlbShootdown();
        return;
    }

    HandleFault(currTask, fromUser, error_code, fault_address,
                ptRegs, signal);
}

pub fn HandleFault(
    task: &mut Task,
    user: bool,
    error_code: PageFaultErrorCode,
    fault_address: u64,
    sf: &mut PtRegs,
    signal: i32,
    ) -> ! {
    {
        debug!("VM: Entered HandleFault.");
        if !user {
            let map = task.mm.GetSnapshotLocked(task, false);
            print!("VM: Unhandled EXCEPTION: PageFault in Kernel -
                        error code: {:#x},
                        FAR: {:#x},
                        GPR: {:#x}",
                        error_code, fault_address, task.GetPtRegs());
            panic!("the k_map is {:?}", &map);
        }

        let mut info = SignalInfo {
            Signo: signal, //Signal::SIGBUS,
            ..Default::default()
        };

        let sigfault = info.SigFault();
        sigfault.addr = fault_address;

        if error_code.contains(PageFaultErrorCode::CAUSED_BY_WRITE)
           || error_code.contains(PageFaultErrorCode::INSTRUCTION_FETCH) {
            info.Code = 2; // SEGV_ACCERR
        } else {
            info.Code = 1; // SEGV_MAPPER
        }

        let thread = task.Thread();
        // Synchronous signal. Send it to ourselves. Assume the signal is
        // legitimate and force it (work around the signal being ignored or
        // blocked) like Linux does. Conveniently, this is even the correct
        // behavior for SIGTRAP from single-stepping.
        thread.forceSignal(Signal(Signal::SIGSEGV), false);
        thread
            .SendSignal(&info)
            .expect("PageFaultHandler send signal fail");
        MainRun(task, TaskRunState::RunApp);

        task.RestoreFp();
        CPULocal::Myself().SetMode(VcpuMode::User);
        task.mm.HandleTlbShootdown();
    }

    unsafe{
        ReturnToApp(sf);
    }
}
