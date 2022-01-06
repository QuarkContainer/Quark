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
#![feature(map_first_last)]
#![allow(deprecated)]

#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate serde;
extern crate cache_padded;

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
extern crate ringbuf;

#[macro_use]
mod print;

#[macro_use]
pub mod asm;
mod taskMgr;
#[macro_use]
mod qlib;
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
pub mod util;
pub mod perflog;
pub mod seqcount;
pub mod quring;
pub mod stack;
pub mod backtracer;
pub mod heap;

use core::panic::PanicInfo;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicI32;
use core::{ptr, mem};
use alloc::vec::Vec;
use ::qlib::mutex::*;
use alloc::string::String;

//use linked_list_allocator::LockedHeap;
//use buddy_system_allocator::LockedHeap;
use taskMgr::{CreateTask, WaitFn, IOWait};
use self::qlib::{SysCallID, ShareSpaceRef};
//use self::qlib::buddyallocator::*;
use self::qlib::pagetable::*;
use self::qlib::control_msg::*;
use self::qlib::common::*;
use self::qlib::linux_def::MemoryDef;
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
//use self::qlib::mem::list_allocator::*;
use self::quring::*;
use self::print::SCALE;
//use self::heap::QAllocator;
use self::heap::GuestAllocator;
use self::qlib::singleton::*;
use self::uid::*;

pub const HEAP_START: usize = 0x70_2000_0000;
pub const HEAP_SIZE: usize = 0x1000_0000;

//use buddy_system_allocator::*;
#[global_allocator]
//static ALLOCATOR: QAllocator = QAllocator::New();
//static ALLOCATOR: StackHeap = StackHeap::Empty();
//static ALLOCATOR: ListAllocator = ListAllocator::Empty();
static ALLOCATOR: GuestAllocator = GuestAllocator::New();
//static ALLOCATOR: BufHeap = BufHeap::Empty();
//static ALLOCATOR: LockedHeap<33> = LockedHeap::empty();

pub fn AllocatorPrint(_class: usize) -> String {
    let class = 6;
    return ALLOCATOR.Print(class);
}

pub static SHARESPACE : ShareSpaceRef = ShareSpaceRef::New();

pub static KERNEL_PAGETABLE : Singleton<PageTables> = Singleton::<PageTables>::New();
pub static PAGE_MGR : Singleton<PageMgr> = Singleton::<PageMgr>::New();
pub static LOADER : Singleton<Loader> = Singleton::<Loader>::New();
pub static IOURING : Singleton<QUring> = Singleton::<QUring>::New();
pub static KERNEL_STACK_ALLOCATOR : Singleton<AlignedAllocator> = Singleton::<AlignedAllocator>::New();
pub static SHUTDOWN : Singleton<AtomicBool> = Singleton::<AtomicBool>::New();
pub static EXIT_CODE : Singleton<AtomicI32> = Singleton::<AtomicI32>::New();

pub fn SingletonInit() {
    unsafe {
        KERNEL_PAGETABLE.Init(PageTables::Init(CurrentCr3()));
        vcpu::VCPU_COUNT.Init(AtomicUsize::new(0));
        vcpu::CPU_LOCAL.Init(&SHARESPACE.scheduler.VcpuArr);
        InitGs(0);
        IOURING.Init(QUring::New(MemoryDef::QURING_SIZE));
        IOURING.SetIOUringsAddr(SHARESPACE.IOUringsAddr());

        // the error! can run after this point
        //error!("error message");

        PAGE_MGR.Init(PageMgr::New());
        LOADER.Init(Loader::default());
        KERNEL_STACK_ALLOCATOR.Init( AlignedAllocator::New(MemoryDef::DEFAULT_STACK_SIZE as usize, MemoryDef::DEFAULT_STACK_SIZE as usize));
        SHUTDOWN.Init(AtomicBool::new(false));
        EXIT_CODE.Init(AtomicI32::new(0));

        guestfdnotifier::GUEST_NOTIFIER.Init(guestfdnotifier::Notifier::New());
        UID.Init(AtomicU64::new(1));
        perflog::THREAD_COUNTS.Init(QMutex::new(perflog::ThreadPerfCounters::default()));
        boot::controller::MSG.Init(QMutex::new(None));

        fs::file::InitSingleton();
        fs::filesystems::InitSingleton();
        interrupt::InitSingleton();
        kernel::abstract_socket_namespace::InitSingleton();
        kernel::futex::InitSingleton();
        kernel::kernel::InitSingleton();
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

pub fn Shutdown() -> bool {
    return SHUTDOWN.load(self::qlib::linux_def::QOrdering::RELAXED);
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

    let nr = pt.orig_rax;
    assert!(nr < SysCallID::maxsupport as u64, "get supported syscall id {:x}", nr);

    //SHARESPACE.SetValue(CPULocal::CpuId(), 0, nr);
    let callId: SysCallID = unsafe { mem::transmute(nr as u64) };

    currTask.SaveFp();

    //let tid = currTask.Thread().lock().id;
    let mut tid = 0;
    let mut pid = 0;
    let startTime = Rdtsc();

    let llevel = SHARESPACE.config.read().LogLevel;
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

    //error!("syscall_handler: {}", ::AllocatorPrint(10));
    if llevel == LogLevel::Simple || llevel == LogLevel::Complex {
        let gap = if self::SHARESPACE.config.read().PerfDebug {
            Rdtsc() - startTime
        } else {
            0
        };
        info!("({}/{})------Return[{}] res is {:x}: call id {:?} ",
        tid, pid, gap / SCALE, res, callId);
    }

    let kernalRsp = pt as *const _ as u64;

    //PerfGoto(PerfType::User);
    currTask.PerfGofrom(PerfType::Kernel);
    currTask.PerfGoto(PerfType::User);

    currTask.RestoreFp();

    currTask.Check();
    if SHARESPACE.config.read().KernelPagetable {
        currTask.SwitchPageTable();
    }

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
            TaskRunState::RunThreadExitNotify => {
                info!("RunTreadExitNotify[{:x}] ...", currTask.taskId);
                currTask.RunThreadExitNotify()
            }
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
                    if !SHARESPACE.config.read().KernelPagetable {
                        KERNEL_PAGETABLE.SwitchTo();
                    }
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

pub fn LogInit(pages: u64) {
    let bs = self::qlib::bytestream::ByteStream::Init(pages); // 4MB
    *SHARESPACE.logBuf.lock() = Some(bs);
}

#[no_mangle]
pub extern fn rust_main(heapStart: u64, shareSpaceAddr: u64, id: u64, vdsoParamAddr: u64, vcpuCnt: u64, autoStart: bool) {
    if id == 0 {
        ALLOCATOR.Init(heapStart);
        SHARESPACE.SetValue(shareSpaceAddr);
        SingletonInit();
        InitTimeKeeper(vdsoParamAddr);


        //Kernel::HostSpace::KernelMsg(0, 0, 1);
        {
            let kpt = &KERNEL_PAGETABLE;

            let mut lock = PAGE_MGR.lock();
            let vsyscallPages = lock.VsyscallPages();

            kpt.InitVsyscall(vsyscallPages);
        }

        self::guestfdnotifier::GUEST_NOTIFIER.InitPollHostEpoll(SHARESPACE.HostHostEpollfd());
        SetVCPCount(vcpuCnt as usize);
        VDSO.Initialization(vdsoParamAddr);

        // release other vcpus
        HyperCall64(qlib::HYPERCALL_RELEASE_VCPU, 0, 0, 0);
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

fn StartExecProcess(fd: i32, process: Process) {
    let (tid, entry, userStackAddr, kernelStackAddr) = {
        LOADER.ExecProcess(process).unwrap()
    };

    WriteControlMsgResp(fd, &UCallResp::ExecProcessResp(tid));

    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);

    EnterUser(entry, userStackAddr, kernelStackAddr);
}

fn StartSubContainerProcess(elfEntry: u64, userStackAddr: u64, kernelStackAddr: u64) {
    let currTask = Task::Current();
    currTask.AccountTaskEnter(SchedState::RunningApp);

    EnterUser(elfEntry, userStackAddr, kernelStackAddr);
}

fn ControllerProcess(_para: *const u8) {
    ControllerProcessHandler().expect("ControllerProcess crash");
}

pub fn StartRootProcess() {
    CreateTask(StartRootContainer, ptr::null(), false);
}

fn StartRootContainer(_para: *const u8) {
    self::Init();
    info!("StartRootContainer ....");
    let task = Task::Current();

    let process = {
        defer!(info!("after process"));
        let mut buf: [u8; 8192] = [0; 8192];
        let addr = &mut buf[0] as * mut _ as u64;
        let size = Kernel::HostSpace::LoadProcessKernel(addr, buf.len()) as usize;
        let process  = serde_json::from_slice(&buf[0..size]);
        let process = match process {
            Ok(p) => p,
            Err(e) => {
                error!("StartRootContainer: failed to LoadProcessKernel, cause: {:?}", e);
                panic!("failed to load Process");
            }
        };
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
    });*/

    print!("get panic : {:?}", info.message());
    if let Some(location) = info.location() {
        print!("panic occurred in file '{}' at line {}",
                 location.file(),
                 location.line(),
        );
    } else {
        print!("panic occurred but can't get location information...");
    }

    for i in 0..CPU_LOCAL.len() {
        error!("CPU  #{} is {:#x?}", i, CPU_LOCAL[i]);
    }

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
extern fn eh_personality() {}
//#[lang = "panic_fmt"] #[no_mangle] pub extern fn panic_fmt() -> ! {loop{}}

