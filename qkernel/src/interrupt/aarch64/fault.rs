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


pub fn PageFaultHandler(ptRegs: &mut PtRegs, fault_address: u64, error_code: PageFaultErrorCode) {
    use self::PageFaultErrorCode as PFEC;
    CPULocal::Myself().SetMode(VcpuMode::Kernel);
    let currTask = Task::Current();
    let fromUser: bool = error_code.contains(PFEC::USER_MODE);
    let ttbr = CurrentUserTable();

    if !SHARESPACE.config.read().CopyDataWithPf && !fromUser {
        error!(
            "VM: PageFault in kernel FAR: {:#x}, TTBR: {:#x}, PtRegs: {:#x}, error_code: {:#x}",
            fault_address, ttbr, ptRegs, error_code
        );
        panic!("VM: PageFault from kernel non recuperable.");
    }

    let PRINT_EXECPTION: bool = SHARESPACE.config.read().PrintException;
    if PRINT_EXECPTION {
        info!(
            "VM: PageFaultHandler - FAR: {:#x}, PC: {:#x},\
               TTBR: {:#x}, is-user: {}, error code: {:#x}.",
            fault_address, ptRegs.pc, ttbr, fromUser, error_code
        );
    }

    let signal;
    // no need to loop, just need to enable break
    'pf_handle: loop {
        let _ml = currTask.mm.MappingWriteLock();
        let (vma, range) = match currTask.mm.GetVmaAndRangeLocked(fault_address) {
            None => {
                signal = Signal::SIGSEGV;
                break 'pf_handle;
            }
            Some(vma) => vma.clone(),
        };

        if error_code.contains(PFEC::ACCESS_FLAG) {
            // we don't utilize access flag for now. Simply insert the flag and continue.
            let bind = currTask.mm.pagetable.write();
            let pte = bind.pt.VirtualToEntry(fault_address).unwrap();
            let mut flags = pte.flags();
            assert!(!flags.contains(PageTableFlags::ACCESSED),
                    "Access flag fault while access flag is set.");
            flags.insert(PageTableFlags::ACCESSED);
            let fault_addr_alligned = Addr(fault_address).RoundDown().unwrap();
            bind.pt.SetPageFlags(fault_addr_alligned, flags);
            return;
        }

        // A PF happened, fault address within kernel VMAs
        // from user    => kill it
        // from kernel  => panic
        if vma.kernel == true {
            assert!(fromUser, "FATAL: kernel hits PF on kernel VMA.");
            signal = Signal::SIGSEGV;
            break 'pf_handle;
        }

        if !vma.effectivePerms.Read() {
            signal = Signal::SIGSEGV;
            break 'pf_handle;
        }

        let pageAddr = Addr(fault_address).RoundDown().unwrap().0;
        assert!(
            range.Contains(pageAddr),
            "PageFaultHandler vm-addr is not in the VM-Area range"
        );

        // handle translation fault
        // NOTE: swap not enabled for aarch64 atm.
        // let addr = currTask.mm.pagetable.write().pt.SwapInPage(Addr(pageAddr)).unwrap();
        if error_code.contains(PFEC::TRANSLATION) {
            let res = currTask.mm.InstallPageLocked(currTask, &vma, pageAddr, &range);
            match res {
                Err(Error::FileMapError) => {
                    error!("VM: failed to install page: FILE_MAP_ERROR.");
                    signal = Signal::SIGBUS;
                    break 'pf_handle;
                }
                Err(e) => {
                    panic!("VM: failed to install page. Err: {:?}",e)
                }
                _ => (),
            };

            // proactively install subsequential pages.
            for i in 1..16 {
                let addr = if vma.growsDown {
                    pageAddr - i * MemoryDef::PAGE_SIZE
                } else {
                    pageAddr + i * MemoryDef::PAGE_SIZE
                };

                if !range.Contains(addr) {break;}

                if let Err(_) = currTask.mm.InstallPageLocked(currTask, &vma, addr, &range) {
                    break;
                }
            }

            if fromUser {
                currTask.AccountTaskEnter(SchedState::RunningApp);
            }
            CPULocal::Myself().SetMode(VcpuMode::User);
            currTask.mm.HandleTlbShootdown();
            return;
        }

        //
        // NOTE: Handle possible COW-modify events.
        // NOTE: COW flags RO set in ForkRange() for private mappings.
        //
        if !vma.private {
            signal = Signal::SIGSEGV;
            break 'pf_handle;
        }

        if error_code.contains(PFEC::PROTECTION_VIOLATION | PFEC::CAUSED_BY_WRITE) {
            if !vma.effectivePerms.Write() {
                signal = Signal::SIGSEGV;
                break 'pf_handle;
            }
            currTask.mm.CopyOnWriteLocked(pageAddr, &vma);
            currTask.mm.TlbShootdown();
            if fromUser {
                currTask.AccountTaskEnter(SchedState::RunningApp);
            }
        } else {
            // unexpected fault from user.
            signal = Signal::SIGSEGV;
            break 'pf_handle
        }

        CPULocal::Myself().SetMode(VcpuMode::User);
        currTask.mm.HandleTlbShootdown();
        return;
    } // end pf_handle loop

    HandleFault(currTask, fromUser, error_code, fault_address, ptRegs, signal);
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
        if !user {
            let map = task.mm.GetSnapshotLocked(task, false);
            panic!("unhandled pagefault in kernel.\
                    error code: {:#?}, FAR: {:#x}.\n\
                    PtRegs: {:#x}\n\
                    kernel map is {}\n",
                   error_code, fault_address, sf, &map);
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
