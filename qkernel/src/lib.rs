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

//#![feature(macro_rules)]
#![feature(lang_items)]
#![no_std]
#![feature(proc_macro_hygiene, asm)]
#![feature(alloc_error_handler)]
#![feature(abi_x86_interrupt)]
#![allow(dead_code)]
#![allow(deref_nullptr)]
#![allow(non_snake_case)]
#![allow(bare_trait_objects)]
//#![feature(const_raw_ptr_to_usize_cast)]
//#![feature(const_fn)]
#![feature(allocator_api)]
#![feature(associated_type_bounds)]
#![feature(core_intrinsics)]
#![feature(llvm_asm, naked_functions)]
#![feature(maybe_uninit_uninit_array)]
#![feature(panic_info_message)]

#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate serde;

#[macro_use]
extern crate alloc;

#[macro_use]
extern crate scopeguard;

//extern crate rusty_asm;
extern crate spin;
extern crate lazy_static;
extern crate x86_64;
//extern crate pic8259_simple;
extern crate xmas_elf;
extern crate bit_field;
//extern crate linked_list_allocator;
extern crate buddy_system_allocator;
#[macro_use]
extern crate bitflags;
//#[macro_use]
extern crate x86;
extern crate backtracer;
extern crate ringbuf;

#[macro_use]
pub mod asm;
mod print;
mod taskMgr;
#[macro_use]
mod qlib;
mod gdt;
#[macro_use]
mod interrupt;
mod Kernel;
mod syscalls;
mod arch;
pub mod kernel;
pub mod kernel_util;
pub mod guestfdnotifier;
pub mod threadmgr;
pub mod boot;
pub mod fs;
pub mod socket;
pub mod memmgr;
pub mod mm;
pub mod SignalDef;
pub mod fd;
pub mod task;
pub mod aqcall;
pub mod vcpu;
pub mod loader;
//pub mod ucall_server;
pub mod tcpip;
pub mod uid;
pub mod version;
pub mod id_mgr;
pub mod util;
pub mod perflog;
pub mod seqcount;
pub mod quring;
pub mod stack;

use core::panic::PanicInfo;
use lazy_static::lazy_static;
use core::{ptr, mem};
use alloc::vec::Vec;

//use linked_list_allocator::LockedHeap;
//use buddy_system_allocator::LockedHeap;
use taskMgr::{CreateTask, WaitFn, IOWait};
use self::qlib::{ShareSpace, SysCallID};
use self::qlib::buddyallocator::*;
use self::qlib::pagetable::*;
use self::qlib::control_msg::*;
use self::qlib::common::*;
use self::qlib::range::*;
use self::qlib::loader::*;
use self::qlib::config::*;
use self::qlib::vcpu_mgr::*;
use self::vcpu::*;
use self::boot::loader::*;
use self::loader::vdso::*;
use self::syscalls::syscalls::*;
use self::memmgr::pma::*;
use self::asm::*;
use self::kernel::timer::*;
use self::boot::controller::*;
use self::task::*;
use self::threadmgr::task_sched::*;
use self::qlib::perf_tunning::*;
//use self::memmgr::buf_allocator::*;
use self::memmgr::list_allocator::*;
use self::quring::*;
use self::print::SCALE;

pub const HEAP_START: usize = 0x70_2000_0000;
pub const HEAP_SIZE: usize = 0x1000_0000;

#[global_allocator]
//static ALLOCATOR: StackHeap = StackHeap::Empty();
static ALLOCATOR: ListAllocator = ListAllocator::Empty();
//static ALLOCATOR: BufHeap = BufHeap::Empty();
//static ALLOCATOR: LockedHeap = LockedHeap::empty();

pub const KERNELTABLE : bool = false;

pub fn AllocatorPrint() {
    //ALLOCATOR.Print();
}

lazy_static! {
    pub static ref SHARESPACE: ShareSpace = ShareSpace::New();
    pub static ref PAGE_ALLOCATOR: MemAllocator = MemAllocator::New();
    pub static ref KERNEL_PAGETABLE: PageTables = PageTables::Init(0);
    pub static ref PAGE_MGR: PageMgr = PageMgr::New();
    pub static ref LOADER: Loader = Loader::default();
    //pub static ref BUF_MGR: Mutex<BufMgr> = Mutex::new(BufMgr::default());
    pub static ref BUF_MGR: BufMgr = BufMgr::New();
    pub static ref IOURING: QUring = QUring::New(1024);
    pub static ref KERNEL_STACK_ALLOCATOR : AlignedAllocator = AlignedAllocator::New(DEFAULT_STACK_SIZE, DEFAULT_STACK_SIZE);
}

extern "C" {
    pub fn syscall_entry();
}

pub fn Init() {
    self::fs::Init();
    self::socket::Init();
}

#[no_mangle]
pub extern fn syscall_handler(arg0: u64, arg1: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> ! {
    //PerfGofrom(PerfType::User);

    let currTask = task::Task::Current();
    currTask.PerfGofrom(PerfType::User);

    currTask.PerfGoto(PerfType::Kernel);

    if KERNELTABLE {
        Task::SetKernelPageTable();
    }

    currTask.AccountTaskLeave(SchedState::RunningApp);
    let pt = currTask.GetPtRegs();
    pt.rip = 0; // set rip as 0 as the syscall will set cs as ret ipaddr

    let nr = pt.orig_rax;
    assert!(nr < SysCallID::maxsupport as u64, "get supported syscall id {:x}", nr);

    //SHARESPACE.SetValue(CPULocal::CpuId(), 0, nr);
    let callId: SysCallID = unsafe { mem::transmute(nr as u64) };

    let mut rflags = GetRflags();
    rflags &= !KERNEL_FLAGS_CLEAR;
    rflags |= KERNEL_FLAGS_SET;
    SetRflags(rflags);

    currTask.SaveFp();

    //let tid = currTask.Thread().lock().id;
    let mut tid = 0;
    let mut pid = 0;
    let startTime = Rdtsc();

    let llevel = SHARESPACE.config.LogLevel;
    if llevel == LogLevel::Complex {
        tid = currTask.Thread().lock().id;
        pid = currTask.Thread().ThreadGroup().ID();
        info!("({}/{})------get call id {:?} arg0:{:x}, 1:{:x}, 2:{:x}, 3:{:x}, 4:{:x}, 5:{:x}, userstack:{:x}, return address:{:x}, fs:{:x}",
            tid, pid, callId, arg0, arg1, arg2, arg3, arg4, arg5, currTask.GetPtRegs().rsp, currTask.GetPtRegs().rcx, GetFs());
    } else if llevel == LogLevel::Simple {
        tid = currTask.Thread().lock().id;
        pid = currTask.Thread().ThreadGroup().ID();
        info!("({}/{})------get call id {:?} arg0:{:x}",
            tid, pid, callId, arg0);
    }

    let res;
    let args = SyscallArguments {
        arg0: arg0,
        arg1: arg1,
        arg2: arg2,
        arg3: arg3,
        arg4: arg4,
        arg5: arg5,
    };

    let currTask = task::Task::Current();
    currTask.DoStop();

    currTask.PerfGoto(PerfType::SysCall);
    let state = SysCall(currTask, nr, &args);
    currTask.PerfGofrom(PerfType::SysCall);

    res = currTask.Return();
    //HostInputProcess();
    //ProcessOne();

    //currTask.PerfGoto(PerfType::KernelHandling);
    MainRun(currTask, state);
    //currTask.PerfGofrom(PerfType::KernelHandling);

    if llevel == LogLevel::Simple || llevel == LogLevel::Complex {
        let gap = Rdtsc() - startTime;
        info!("({}/{})------Return[{}] res is {:x}: call id {:?} ",
        tid, pid, gap / SCALE, res, callId);
    }

    let kernalRsp = pt as *const _ as u64;

    //PerfGoto(PerfType::User);
    currTask.PerfGofrom(PerfType::Kernel);
    currTask.PerfGoto(PerfType::User);

    let mut rflags = pt.eflags;
    rflags &= !USER_FLAGS_CLEAR;
    rflags |= USER_FLAGS_SET;
    SetRflags(rflags);
    currTask.RestoreFp();

    if KERNELTABLE {
        currTask.SwitchPageTable();
    }

    //SHARESPACE.SetValue(CPULocal::CpuId(), 0, 0);
    if pt.rip != 0 { // if it is from signal trigger from kernel, e.g. page fault
        pt.eflags = rflags;
        IRet(kernalRsp)
    } else {
        SyscallRet(kernalRsp)
    }
}

#[inline]
pub fn MainRun(currTask: &mut Task, mut state: TaskRunState) {
    //PerfGoto(PerfType::KernelHandling);
    loop {
        state = match state {
            TaskRunState::RunApp => currTask.RunApp(),
            TaskRunState::RunInterrupt => {
                info!("RunInterrupt[{:x}] ...", currTask.taskId);
                currTask.RunInterrupt()
            },
            TaskRunState::RunExit => {
                info!("RunExit[{:x}] ...", currTask.taskId);
                currTask.RunExit()
            },
            TaskRunState::RunExitNotify => {
                info!("RunExitNotify[{:x}] ...", currTask.taskId);
                currTask.RunExitNotify();

                // !!! make sure there is no object hold on stack

                TaskRunState::RunExitDone
            },
            TaskRunState::RunThreadExit => {
                info!("RunThreadExit[{:x}] ...", currTask.taskId);
                currTask.RunThreadExit()
            },
            TaskRunState::RunTreadExitNotify => {
                info!("RunTreadExitNotify[{:x}] ...", currTask.taskId);
                currTask.RunThreadExitNotify()
            },
            TaskRunState::RunExitDone => {
                {
                    error!("RunExitDone 1 [{:x}] ...", currTask.taskId);
                    let thread = currTask.Thread();
                    currTask.PerfStop();
                    currTask.SetDummy();

                    thread.lock().fdTbl = currTask.fdTbl.clone();
                    let mm = thread.lock().memoryMgr.clone();
                    thread.lock().memoryMgr = currTask.mm.clone();
                    CPULocal::SetPendingFreeStack(currTask.taskId);

                    error!("RunExitDone xxx 2 [{:x}] ...", currTask.taskId);
                    KERNEL_PAGETABLE.SwitchTo();

                    // mm needs to be clean as last function before SwitchToNewTask
                    // after this is called, another vcpu might drop the pagetable
                    core::mem::drop(mm);
                }

                self::taskMgr::SwitchToNewTask();
                //panic!("RunExitDone: can't reach here")
            }
            TaskRunState::RunNoneReachAble => panic!("unreadhable TaskRunState::RunNoneReachAble"),
            TaskRunState::RunSyscallRet => panic!("unreadhable TaskRunState::RunSyscallRet"),
        };

        if state == TaskRunState::RunSyscallRet {
            break;
        }
    }

    currTask.DoStop();

    let pt = currTask.GetPtRegs();

    CPULocal::SetUserStack(pt.rsp);
    CPULocal::SetKernelStack(currTask.GetKernelSp());

    currTask.AccountTaskEnter(SchedState::RunningApp);
    //PerfGofrom(PerfType::KernelHandling);
}

fn InitGs(id: u64) {
    SetGs(&CPU_LOCAL[id as usize] as *const _ as u64);
    SwapGs();
}

#[no_mangle]
pub extern fn rust_main(heapStart: u64, heapLen: u64, id: u64, vdsoParamAddr: u64, vcpuCnt: u64, autoStart: bool) {
    Kernel::HostSpace::KernelMsg(0x10, 1);
    if id == 0 {
        ALLOCATOR.Add(heapStart as usize, heapLen as usize);

        Kernel::HostSpace::KernelMsg(0x10, 2);
        PAGE_MGR.lock().Init();

        {
            //to initial the SHARESPACE
            let _tmp = &SHARESPACE;
        }

        // InitGS rely on SHARESPACE
        InitGs(id);
        //PerfGoto(PerfType::Kernel);

        {
            // init the IOURING
            IOURING.submission.lock();
        }

        Kernel::HostSpace::KernelMsg(0x10, 3);
        SHARESPACE.scheduler.SetVcpuCnt(vcpuCnt as usize);
        HyperCall64(qlib::HYPERCALL_INIT, (&(*SHARESPACE) as *const ShareSpace) as u64, 0);

        Kernel::HostSpace::KernelMsg(0x10, 4);
        {
            let root = CurrentCr3();
            let kpt = &KERNEL_PAGETABLE;
            Kernel::HostSpace::KernelMsg(0x11, 1);
            kpt.SetRoot(root);

            Kernel::HostSpace::KernelMsg(0x11, 2);
            let mut lock = PAGE_MGR.lock();
            Kernel::HostSpace::KernelMsg(0x11, 3);
            let vsyscallPages = lock.VsyscallPages();

            Kernel::HostSpace::KernelMsg(0x11, 4);
            kpt.InitVsyscall(vsyscallPages);
        }

        Kernel::HostSpace::KernelMsg(0x10, 5);
        SetVCPCount(vcpuCnt as usize);
        InitTimeKeeper(vdsoParamAddr);
        VDSO.Init(vdsoParamAddr);
    } else {
        InitGs(id);
        //PerfGoto(PerfType::Kernel);
    }

    Kernel::HostSpace::KernelMsg(0x10, 6);
    taskMgr::AddNewCpu();
    RegisterSysCall(syscall_entry as u64);

    //interrupts::init_idt();
    interrupt::init();

    /***************** can't run any qcall before this point ************************************/

    if id == 0 {
        IOWait();
    };

    if id == 1 {
        error!("heap start is {:x}/{:x}", heapStart, heapStart + heapLen);

        if autoStart {
            CreateTask(StartRootContainer, ptr::null(), false);
        }

        CreateTask(ControllerProcess, ptr::null(), true);
    }

    WaitFn();
}

fn Print() {
    let cr2: u64;
    unsafe { llvm_asm!("mov %cr2, $0" : "=r" (cr2) ) };

    let cr3: u64;
    unsafe { llvm_asm!("mov %cr3, $0" : "=r" (cr3) ) };

    let cs: u64;
    unsafe { llvm_asm!("mov %cs, $0" : "=r" (cs) ) };
    let ss: u64;
    unsafe { llvm_asm!("mov %ss, $0" : "=r" (ss) ) };

    info!("cr2 is {:x}, cr3 is {:x}, cs is {}, ss is {}", cr2, cr3, cs, ss);
}

fn StartExecProcess(msgId: u64, process: Process) {
    let (tid, entry, userStackAddr, kernelStackAddr) = {
        LOADER.ExecProcess(process).unwrap()
    };

    ControlMsgRet(msgId, &UCallResp::ExecProcessResp(tid));

    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);

    EnterUser(entry, userStackAddr, kernelStackAddr);
}

fn ControllerProcess(_para: *const u8) {
    Run().expect("ControllerProcess crash");
}

pub fn StartRootProcess() {
    CreateTask(StartRootContainer, ptr::null(), false);
}

fn StartRootContainer(_para: *const u8) {
    //Print();

    //let bits = VirtualAddressBits();
    //info!("bits is {}", bits);

    /*let fs = self::qlib::cpuid::HostFeatureSet();
      info!("{}", fs.CPUInfo(0));

      let (size, align) = fs.ExtendedStateSize();
      info!("ExtendedStateSize size is {}, align is {}", size, align);*/

    //waiter + timer example, todo: remove it
    /*let waiter = Waiter::default();
    let we = waiter.NewWaitEntry(1);
    let mut timer = NewTimer(&we);
    timer.Reset(1000_000_000);
    info!("before wait");
    waiter.Wait(&[we.clone()]);
    info!("after wait");
    timer.Reset(111111);
    timer.Stop();
    timer.Reset(111111);
    timer.Drop();*/


    //let vdsopage : [u8; 4096] = [0; 4096];
    //let timekeeper = kernel::time::timekeeper::TimeKeeper::New(&vdsopage[0] as * const _ as u64);

    //todo: the exception's stack is different than the task, so it can't be use for taskId
    //info!("before int3");
    //x86_64::instructions::interrupts::int3();
    //info!("after int3");

    self::Init();
    info!("StartRootContainer ....");
    let task = Task::Current();

    let process = {
        defer!(info!("after process"));
        let mut buf: [u8; 8192] = [0; 8192];
        let addr = &mut buf[0] as * mut _ as u64;
        let size = Kernel::HostSpace::LoadProcessKernel(addr, buf.len()) as usize;

        let process : Process = serde_json::from_slice(&buf[0..size]).expect("StartRootContainer: LoadProcessKernel des fail");
        process
    };


    let (_tid, entry, userStackAddr, kernelStackAddr) = {
        let mut processArgs = LOADER.Lock(task).unwrap().Init(process);
        LOADER.LoadRootProcess(&mut processArgs).unwrap()
    };

    //CreateTask(StartExecProcess, ptr::null());
    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);
    EnterUser(entry, userStackAddr, kernelStackAddr);

    //can't reach this
    WaitFn();
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // bug https://github.com/QuarkContainer/Quark/issues/26.
    // todo: enable this after the issue is fixed
    //print!("get panic: {:?}", info);

    /*backtracer::trace(|frame| {
        print!("panic frame is {:#x?}", frame);
        true
    });

    for i in 0..CPU_LOCAL.len() {
        error!("CPU#{} is {:#x?}", i, CPU_LOCAL[i]);
    }*/

    print!("get panic : {:?}", info.message());
    if let Some(location) = info.location() {
        print!("panic occurred in file '{}' at line {}",
                 location.file(),
                 location.line(),
        );
    } else {
        print!("panic occurred but can't get location information...");
    }

    backtracer::trace(|frame| {
        print!("panic frame is {:#x?}", frame);
        true
    });

    //self::Kernel::HostSpace::Panic(&format!("get panic: {:?}", info));
    self::Kernel::HostSpace::Panic("get panic ...");
    loop {}
}

#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    self::Kernel::HostSpace::Panic(&format!("alloc_error_handler layout: {:?}", layout));
    loop {}
}

#[lang = "eh_personality"]
extern fn eh_personality() {}
//#[lang = "panic_fmt"] #[no_mangle] pub extern fn panic_fmt() -> ! {loop{}}

