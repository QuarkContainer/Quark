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
#![feature(proc_macro_hygiene)]
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
#![feature(map_first_last)]
#![allow(deprecated)]
#![feature(asm)]

#[macro_use]
extern crate serde_derive;
extern crate cache_padded;

#[macro_use]
extern crate alloc;

#[macro_use]
extern crate scopeguard;

extern crate bit_field;

#[macro_use]
extern crate lazy_static;
extern crate spin;
extern crate x86_64;
extern crate xmas_elf;
#[macro_use]
extern crate bitflags;
extern crate hashbrown;

#[macro_use]
mod print;

//#[macro_use]
//pub mod asm;
//mod taskMgr;
#[macro_use]
mod qlib;
#[macro_use]
mod interrupt;
pub mod kernel_def;
pub mod rdma_def;
mod syscalls;

use self::kernel_def::*;
use self::interrupt::virtualization_handler;
use self::qlib::kernel::arch;
use self::qlib::kernel::asm;
use self::qlib::kernel::boot;
use self::qlib::kernel::fd;
use self::qlib::kernel::fs;
use self::qlib::kernel::kernel;
use self::qlib::kernel::loader;
use self::qlib::kernel::memmgr;
use self::qlib::kernel::perflog;
use self::qlib::kernel::quring;
use self::qlib::kernel::socket;
use self::qlib::kernel::task;
use self::qlib::kernel::taskMgr;
use self::qlib::kernel::threadmgr;
use self::qlib::kernel::util;
use self::qlib::kernel::vcpu;
use self::qlib::kernel::version;
use self::qlib::kernel::Kernel;
use self::qlib::kernel::SignalDef;
use self::qlib::kernel::TSC;
use crate::qlib::kernel::GlobalIOMgr;

use self::qlib::kernel::vcpu::*;
use vcpu::CPU_LOCAL;

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use core::{mem, ptr};
use qlib::mutex::*;
use spin::mutex::Mutex;

//use linked_list_allocator::LockedHeap;
//use buddy_system_allocator::LockedHeap;
use self::qlib::{ShareSpaceRef, SysCallID};
use taskMgr::{CreateTask, IOWait, WaitFn};
//use self::qlib::buddyallocator::*;
use self::asm::*;
use self::boot::controller::*;
use self::boot::loader::*;
use self::kernel::timer::*;
use self::loader::vdso::*;
use self::qlib::common::*;
use self::qlib::config::*;
use self::qlib::control_msg::*;
use self::qlib::cpuid::*;
use self::qlib::linux::time::*;
use self::qlib::linux_def::MemoryDef;
use self::qlib::loader::*;
use self::qlib::mem::list_allocator::*;
use self::qlib::pagetable::*;
//use self::qlib::perf_tunning::*;
use self::qlib::vcpu_mgr::*;
use self::syscalls::syscalls::*;
use self::task::*;
use self::threadmgr::task_sched::*;
//use self::vcpu::*;
use self::qlib::kernel::Scale;
use self::qlib::kernel::VcpuFreqInit;
use self::quring::*;
//use self::heap::QAllocator;

pub const HEAP_START: usize = 0x70_2000_0000;
pub const HEAP_SIZE: usize = 0x1000_0000;

//use buddy_system_allocator::*;
//#[global_allocator]
//pub static GLOBAL_ALLOCATOR: HostAllocator = HostAllocator::New();

#[global_allocator]
pub static VCPU_ALLOCATOR: GlobalVcpuAllocator = GlobalVcpuAllocator::New();

lazy_static! {
    pub static ref GLOBAL_LOCK : Mutex<()> = Mutex::new(());
}

//static ALLOCATOR: QAllocator = QAllocator::New();
//static ALLOCATOR: StackHeap = StackHeap::Empty();
//static ALLOCATOR: ListAllocator = ListAllocator::Empty();
//static ALLOCATOR: GuestAllocator = GuestAllocator::New();
//static ALLOCATOR: BufHeap = BufHeap::Empty();
//static ALLOCATOR: LockedHeap<33> = LockedHeap::empty();

/*pub fn AllocatorPrint(_class: usize) -> String {
    let class = 6;
    return ALLOCATOR.Print(class);
}*/

use self::qlib::kernel::*;

pub fn SingletonInit() {
    unsafe {
        vcpu::VCPU_COUNT.Init(AtomicUsize::new(0));
        vcpu::CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
        InitGs(0);
        KERNEL_PAGETABLE.Init(PageTables::Init(CurrentCr3()));
        //init fp state with current fp state as it is brand new vcpu
        FP_STATE.Reset();
        SHARESPACE.SetSignalHandlerAddr(SignalHandler as u64);
        SHARESPACE.SetvirtualizationHandlerAddr(virtualization_handler as u64);
        IOURING.SetValue(SHARESPACE.GetIOUringAddr());

        // the error! can run after this point
        //error!("error message");

        PAGE_MGR.SetValue(SHARESPACE.GetPageMgrAddr());
        LOADER.Init(Loader::default());
        KERNEL_STACK_ALLOCATOR.Init(AlignedAllocator::New(
            MemoryDef::DEFAULT_STACK_SIZE as usize,
            MemoryDef::DEFAULT_STACK_SIZE as usize,
        ));
        EXIT_CODE.Init(AtomicI32::new(0));

        let featureSet = HostFeatureSet();
        SUPPORT_XSAVE.store(
            featureSet.HasFeature(Feature(X86Feature::X86FeatureXSAVE as i32)),
            Ordering::Release,
        );
        SUPPORT_XSAVEOPT.store(
            featureSet.HasFeature(Feature(X86Feature::X86FeatureXSAVEOPT as i32)),
            Ordering::Release,
        );

        perflog::THREAD_COUNTS.Init(QMutex::new(perflog::ThreadPerfCounters::default()));

        fs::file::InitSingleton();
        fs::filesystems::InitSingleton();
        interrupt::InitSingleton();
        kernel::futex::InitSingleton();
        kernel::semaphore::InitSingleton();
        kernel::epoll::epoll::InitSingleton();
        kernel::timer::InitSingleton();
        loader::vdso::InitSingleton();
        socket::socket::InitSingleton();
        syscalls::sys_rlimit::InitSingleton();
        task::InitSingleton();

        qlib::InitSingleton();
    }
}

extern "C" {
    pub fn syscall_entry();
}

pub fn Init() {
    self::fs::Init();
    self::socket::Init();
}

#[no_mangle]
pub extern "C" fn syscall_handler(
    arg0: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
) -> ! {
    CPULocal::Myself().SetMode(VcpuMode::Kernel);

    let currTask = task::Task::Current();
    //currTask.PerfGofrom(PerfType::User);

    //currTask.PerfGoto(PerfType::Kernel);

    if SHARESPACE.config.read().KernelPagetable {
        Task::SetKernelPageTable();
    }

    currTask.AccountTaskLeave(SchedState::RunningApp);
    let pt = currTask.GetPtRegs();
    //pt.rip = 0; // set rip as 0 as the syscall will set cs as ret ipaddr

    let mut rflags = pt.eflags;
    rflags &= !USER_FLAGS_CLEAR;
    rflags |= USER_FLAGS_SET;
    pt.eflags = rflags;
    pt.r11 = rflags;
    pt.rip = pt.rcx;

    let mut nr = pt.orig_rax;
    let callId: SysCallID = if nr < SysCallID::UnknowSyscall as u64 {
        unsafe { mem::transmute(nr as u64) }
    } else if SysCallID::SysSocketProduce as u64 <= nr && nr < SysCallID::EXTENSION_MAX as u64 {
        unsafe { mem::transmute(nr as u64) }
    } else {
        nr = SysCallID::UnknowSyscall as _;
        SysCallID::UnknowSyscall
    };

    //let tid = currTask.Thread().lock().id;
    let mut tid = 0;
    let mut pid = 0;

    let llevel = SHARESPACE.config.read().LogLevel;
    if llevel == LogLevel::Complex {
        tid = currTask.Thread().lock().id;
        pid = currTask.Thread().ThreadGroup().ID();
        info!("({}/{})------get call id {:?} arg0:{:x}, 1:{:x}, 2:{:x}, 3:{:x}, 4:{:x}, 5:{:x}, userstack:{:x}, return address:{:x}, fs:{:x}",
            tid, pid, callId, arg0, arg1, arg2, arg3, arg4, arg5, currTask.GetPtRegs().rsp, currTask.GetPtRegs().rcx, GetFs());
    } else if llevel == LogLevel::Simple {
        tid = currTask.Thread().lock().id;
        pid = currTask.Thread().ThreadGroup().ID();
        info!(
            "({}/{})------get call id {:?} arg0:{:x}",
            tid, pid, callId, arg0
        );
    }

    //currTask.SaveFp();

    let startTime = TSC.Rdtsc();
    let enterAppTimestamp = CPULocal::Myself().ResetEnterAppTimestamp() as i64;
    let worktime = Tsc::Scale(startTime - enterAppTimestamp) * 1000; // the thread has used up time slot
    if worktime > CLOCK_TICK {
        taskMgr::Yield();
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

    //currTask.PerfGoto(PerfType::SysCall);
    let state = SysCall(currTask, nr, &args);
    //currTask.PerfGofrom(PerfType::SysCall);

    res = currTask.Return();
    //HostInputProcess();
    //ProcessOne();

    MainRun(currTask, state);

    currTask.RestoreFp();
   //error!("syscall_handler: {}", ::AllocatorPrint(10));
    if llevel == LogLevel::Simple || llevel == LogLevel::Complex {
        let gap = if self::SHARESPACE.config.read().PerfDebug {
            TSC.Rdtsc() - startTime
        } else {
            0
        };
        info!(
            "({}/{})------Return[{}] res is {:x}: call id {:?} ",
            tid,
            pid,
            Scale(gap),
            res,
            callId
        );
    }

    let kernalRsp = pt as *const _ as u64;

    //PerfGoto(PerfType::User);
    //currTask.PerfGofrom(PerfType::Kernel);
    //currTask.PerfGoto(PerfType::User);

    //currTask.Check();
    if SHARESPACE.config.read().KernelPagetable {
        currTask.SwitchPageTable();
    }
    CPULocal::Myself().SetMode(VcpuMode::User);
    currTask.mm.HandleTlbShootdown();
    CPULocal::Myself().SetEnterAppTimestamp(TSC.Rdtsc());
    if !(pt.rip == pt.rcx && pt.r11 == pt.eflags) {
        //error!("iret *****, pt is {:x?}", pt);
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
            }
            TaskRunState::RunExit => {
                info!("RunExit[{:x}] ...", currTask.taskId);
                currTask.RunExit()
            }
            TaskRunState::RunExitNotify => {
                info!("RunExitNotify[{:x}] ...", currTask.taskId);
                currTask.RunExitNotify();

                // !!! make sure there is no object hold on stack

                TaskRunState::RunExitDone
            }
            TaskRunState::RunThreadExit => {
                info!("RunThreadExit[{:x}] ...", currTask.taskId);
                currTask.RunThreadExit()
            }
            TaskRunState::RunThreadExitNotify => {
                info!("RunTreadExitNotify[{:x}] ...", currTask.taskId);
                currTask.RunThreadExitNotify()
            }
            TaskRunState::RunExitDone => {
                {
                    error!("RunExitDone 1 [{:x}] ...", currTask.taskId);
                    let thread = currTask.Thread();
                    //currTask.PerfStop();
                    currTask.SetDummy();

                    let fdtbl = thread.lock().fdTbl.clone();
                    thread.lock().fdTbl = currTask.fdTbl.clone();

                    // we have to clone fdtbl at first to avoid lock the thread when drop fdtbl
                    drop(fdtbl);
                    let mm = thread.lock().memoryMgr.clone();
                    thread.lock().memoryMgr = currTask.mm.clone();
                    CPULocal::SetPendingFreeStack(currTask.taskId);

                    error!("RunExitDone xxx 2 [{:x}] ...", currTask.taskId);
                    if !SHARESPACE.config.read().KernelPagetable {
                        KERNEL_PAGETABLE.SwitchTo();
                    }
                    // mm needs to be clean as last function before SwitchToNewTask
                    // after this is called, another vcpu might drop the pagetable
                    core::mem::drop(mm);
                    CPULocal::Myself().pageAllocator.lock().Clean();
                }

                self::taskMgr::SwitchToNewTask();
                //panic!("RunExitDone: can't reach here")
            }
            TaskRunState::RunNoneReachAble => panic!("unreadhable TaskRunState::RunNoneReachAble"),
            TaskRunState::RunSyscallRet => TaskRunState::RunSyscallRet, //panic!("unreadhable TaskRunState::RunSyscallRet"),
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

pub fn LogInit(pages: u64) {
    let bs = self::qlib::bytestream::ByteStream::Init(pages); // 4MB
    *SHARESPACE.logBuf.lock() = Some(bs);
}

pub fn InitTsc() {
    let _hosttsc1 = Kernel::HostSpace::Rdtsc();
    let tsc1 = TSC.Rdtsc();
    let hosttsc2 = Kernel::HostSpace::Rdtsc();
    let tsc2 = TSC.Rdtsc();
    let hosttsc3 = Kernel::HostSpace::Rdtsc();
    let tsc3 = TSC.Rdtsc();
    Kernel::HostSpace::SetTscOffset((hosttsc2 + hosttsc3) / 2 - (tsc1 + tsc2 + tsc3) / 3);
    VcpuFreqInit();
}

#[no_mangle]
pub extern "C" fn rust_main(
    heapStart: u64,
    shareSpaceAddr: u64,
    id: u64,
    vdsoParamAddr: u64,
    vcpuCnt: u64,
    autoStart: bool,
) {
    self::qlib::kernel::asm::fninit();
    if id == 0 {
        GLOBAL_ALLOCATOR.Init(heapStart);
        SHARESPACE.SetValue(shareSpaceAddr);
        SingletonInit();

        VCPU_ALLOCATOR.Initializated();
        InitTsc();
        InitTimeKeeper(vdsoParamAddr);

        {
            let kpt = &KERNEL_PAGETABLE;

            let vsyscallPages = PAGE_MGR.VsyscallPages();
            kpt.InitVsyscall(vsyscallPages);
        }

        GlobalIOMgr().InitPollHostEpoll(SHARESPACE.HostHostEpollfd());
        SetVCPCount(vcpuCnt as usize);
        VDSO.Initialization(vdsoParamAddr);

        // release other vcpus
        HyperCall64(qlib::HYPERCALL_RELEASE_VCPU, 0, 0, 0, 0);
    } else {
        InitGs(id);
        //PerfGoto(PerfType::Kernel);
    }

    SHARESPACE.IncrVcpuSearching();
    taskMgr::AddNewCpu();
    RegisterSysCall(syscall_entry as u64);

    //interrupts::init_idt();
    interrupt::init();

    /***************** can't run any qcall before this point ************************************/

    if id == 0 {
        //error!("start main: {}", ::AllocatorPrint(10));

        //ALLOCATOR.Print();
        IOWait();
    };

    if id == 1 {
        error!("heap start is {:x}", heapStart);

        if autoStart {
            CreateTask(StartRootContainer as u64, ptr::null(), false);
        }

        CreateTask(ControllerProcess as u64, ptr::null(), true);
    }

    WaitFn();
}

fn StartExecProcess(fd: i32, process: Process) -> ! {
    let (tid, entry, userStackAddr, kernelStackAddr) = { LOADER.ExecProcess(process).unwrap() };

    {
        WriteControlMsgResp(fd, &UCallResp::ExecProcessResp(tid), true);
    }

    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);

    EnterUser(entry, userStackAddr, kernelStackAddr);
}

fn StartSubContainerProcess(elfEntry: u64, userStackAddr: u64, kernelStackAddr: u64) -> ! {
    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);

    EnterUser(elfEntry, userStackAddr, kernelStackAddr);
}

fn ControllerProcess(_para: *const u8) {
    ControllerProcessHandler().expect("ControllerProcess crash");
}

pub fn StartRootProcess() {
    CreateTask(StartRootContainer as u64, ptr::null(), false);
}

fn StartRootContainer(_para: *const u8) -> ! {
    self::Init();
    info!("StartRootContainer ....");
    let task = Task::Current();
    let mut process = Process::default();
    Kernel::HostSpace::LoadProcessKernel(&mut process as * mut _ as u64) as usize;

    let (_tid, entry, userStackAddr, kernelStackAddr) = {
        let mut processArgs = LOADER.Lock(task).unwrap().Init(process);
        match LOADER.LoadRootProcess(&mut processArgs) {
            Err(e) => {
                error!("load root process failure with error {:?}, shutting down...", e);
                SHARESPACE.StoreShutdown();
                Kernel::HostSpace::ExitVM(2);
                panic!("exiting ...");
            }
            Ok(r) => r,
        }
    };

    //CreateTask(StartExecProcess, ptr::null());
    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);
    EnterUser(entry, userStackAddr, kernelStackAddr);
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // bug https://github.com/QuarkContainer/Quark/issues/26.
    // todo: enable this after the issue is fixed
    //print!("get panic: {:?}", info);

    /*backtracer::trace(|frame| {
        print!("panic frame is {:#x?}", frame);
        true
    });*/

    print!("get panic : {:?}", info.message());
    if let Some(location) = info.location() {
        print!(
            "panic occurred in file '{}' at line {}",
            location.file(),
            location.line(),
        );
    } else {
        print!("panic occurred but can't get location information...");
    }

    /*for i in 0..CPU_LOCAL.len() {
        error!("CPU  #{} is {:#x?}", i, CPU_LOCAL[i]);
    }*/

    /*backtracer::trace(&mut |frame| {
        print!("panic frame is {:#x?}", frame);
        true
    });*/

    //self::Kernel::HostSpace::Panic(&format!("get panic: {:?}", info));
    //self::Kernel::HostSpace::Panic("get panic ...");
    loop {}
}

#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    self::Kernel::HostSpace::Panic(&format!("alloc_error_handler layout: {:?}", layout));
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn eh_personality() {}
//#[lang = "panic_fmt"] #[no_mangle] pub extern fn panic_fmt() -> ! {loop{}}
