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
//

use core::alloc::{GlobalAlloc, Layout};
use core::arch::asm;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use crate::qlib::fileinfo::*;

use self::kernel::socket::hostinet::tsot_mgr::TsotSocketMgr;
use self::tsot_msg::TsotMessage;
use self::kernel::dns::dns_svc::DnsSvc;
use super::qlib::kernel::asm::*;
use super::qlib::kernel::quring::uring_async::UringAsyncMgr;
use super::qlib::kernel::taskMgr::*;
use super::qlib::kernel::threadmgr::task_sched::*;
use super::qlib::kernel::SHARESPACE;
use super::qlib::kernel::TSC;

use super::qlib::common::*;
use super::qlib::kernel::memmgr::pma::*;
use super::qlib::kernel::task::*;
use super::qlib::kernel::taskMgr;
use super::qlib::linux_def::*;
use super::qlib::loader::*;
use super::qlib::mem::list_allocator::*;
use super::qlib::mutex::*;
use super::qlib::perf_tunning::*;
use super::qlib::qmsg::*;
use super::qlib::task_mgr::*;
use super::qlib::vcpu_mgr::*;
use super::qlib::ShareSpace;
use super::qlib::*;
use super::syscalls::sys_file::*;
use super::Kernel::HostSpace;

use super::PRIVATE_VCPU_LOCAL_HOLDER;
use crate::GLOBAL_ALLOCATOR;

impl OOMHandler for ListAllocator {
    fn handleError(&self, size: u64, alignment: u64) {
        HostSpace::KernelOOM(size, alignment);
    }
}

impl ListAllocator {
    pub fn initialize(&self) -> () {
        self.initialized.store(true, Ordering::Relaxed);
    }

    pub fn Check(&self) {
        Task::StackOverflowCheck();
    }
}

impl CPULocal {
    pub fn Wakeup(&self) {
        super::Kernel::HostSpace::EventfdWrite(self.eventfd);
    }
}

impl<'a> ShareSpace {
    pub fn AQCall(&self, msg: &HostOutputMsg) {
        loop {
            match self.QOutput.Push(msg) {
                Ok(()) => {
                    break;
                }
                Err(_) => (),
            };
        }

        if self.HostProcessor() == 0 {
            self.scheduler.VcpuArr[0].Wakeup();
        }
    }

    pub fn Yield() {
        HostSpace::VcpuYield();
    }
}

impl<T: ?Sized> QMutexIntern<T> {
    pub fn GetID() -> u64 {
        return Task::TaskAddress();
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfType {
    Start,
    Kernel,
    User,
    Read,
    Write,
    Open,
    KernelHandling,
    Print,
    Idle,
    PageFault,
    QCall,
    SysCall,
    Blocked,
    HostInputProcess,
    End,
}

impl CounterSet {
    pub const PERM_COUNTER_SET_SIZE: usize = 8;

    pub fn GetPerfId(&self) -> usize {
        CPULocal::CpuId() as usize
    }

    pub fn PerfType(&self) -> &str {
        return "PerfPrint::Kernel";
    }
}

#[inline]
pub fn switch(from: TaskId, to: TaskId) {
    //Task::Current().PerfGoto(PerfType::Blocked);
    Task::Current().AccountTaskEnter(SchedState::Blocked);

    CPULocal::SetCurrentTask(to.Addr());
    let fromCtx = from.GetTask();
    let toCtx = to.GetTask();

    if !SHARESPACE.config.read().KernelPagetable {
        toCtx.SwitchPageTable();
    }
    toCtx.SetFS();

    fromCtx.mm.VcpuLeave();
    toCtx.mm.VcpuEnter();

    //fromCtx.Check();
    //toCtx.Check();
    //debug!("switch {:x}->{:x}", from.data, to.data);

    unsafe {
        context_swap(fromCtx.GetContext(), toCtx.GetContext(), 1, 0);
    }

    //Task::Current().PerfGofrom(PerfType::Blocked);
    Task::Current().AccountTaskLeave(SchedState::Blocked);
}

pub fn OpenAt(task: &Task, dirFd: i32, addr: u64, flags: u32) -> Result<i32> {
    return openAt(task, dirFd, addr, flags);
}

pub fn StartRootContainer(para: *const u8) {
    super::StartRootContainer(para)
}

pub fn StartExecProcess(fd: i32, process: Process) {
    super::StartExecProcess(fd, process)
}
pub fn StartSubContainerProcess(elfEntry: u64, userStackAddr: u64, kernelStackAddr: u64) {
    super::StartSubContainerProcess(elfEntry, userStackAddr, kernelStackAddr)
}

extern "C" {
    pub fn CopyPageUnsafe(to: u64, from: u64);
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn Invlpg(addr: u64) {
    if !super::SHARESPACE.config.read().KernelPagetable {
        unsafe {
            asm!("
            invlpg [{0}]
            ",
            in(reg) addr)
        };
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn Invlpg(addr: u64) {
    if !super::SHARESPACE.config.read().KernelPagetable {
        unsafe {
            asm!("
            dsb ishst
            tlbi vaae1is, {}
            dsb ish
            isb
        ", in(reg) (addr >> PAGE_SHIFT));
        };
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn HyperCall64(type_: u16, para1: u64, para2: u64, para3: u64, para4: u64) {
    unsafe {
        let data: u8 = 0;
        asm!("
            out dx, al
            ",
            in("dx") type_,
            in("al") data,
            in("rsi") para1,
            in("rcx") para2,
            in("rdi") para3,
            in("r10") para4
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn HyperCall64(type_: u16, para1: u64, para2: u64, para3: u64, para4: u64) {
    // TODO add HyperCall64
}

impl CPULocal {
    pub fn CpuId() -> usize {
        return GetVcpuId();
    }
}

impl PageMgr {
    pub fn CopyVsysCallPages(&self, addr: u64) {
        CopyPage(addr, __vsyscall_page as u64);
    }
}

pub fn ClockGetTime(clockId: i32) -> i64 {
    return HostSpace::KernelGetTime(clockId).unwrap();
}

pub fn VcpuFreq() -> i64 {
    return HostSpace::KernelVcpuFreq();
}

pub fn NewSocket(fd: i32) -> i64 {
    return HostSpace::NewSocket(fd);
}

impl HostSpace {
    pub fn Close(fd: i32) -> i64 {
        let mut msg = Msg::Close(qcall::Close { fd });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Call(msg: &mut Msg, _mustAsync: bool) -> u64 {
        let current = Task::Current().GetTaskId();

        let qMsg = QMsg {
            taskId: current,
            globalLock: true,
            ret: 0,
            msg: msg,
        };

        let addr = &qMsg as *const _ as u64;
        let om = HostOutputMsg::QCall(addr);

        super::SHARESPACE.AQCall(&om);
        taskMgr::Wait();
        return qMsg.ret;
    }

    pub fn HCall(msg: &mut Msg, lock: bool) -> u64 {
        let taskId = Task::Current().GetTaskId();

        let mut event = QMsg {
            taskId: taskId,
            globalLock: lock,
            ret: 0,
            msg: msg,
        };

        HyperCall64(HYPERCALL_HCALL, &mut event as *const _ as u64, 0, 0, 0);

        return event.ret;
    }
}

#[inline]
pub fn child_clone(userSp: u64) {
    let currTask = Task::Current();
    CPULocal::SetUserStack(userSp);
    CPULocal::SetKernelStack(currTask.GetKernelSp());

    currTask.AccountTaskEnter(SchedState::RunningApp);
    let pt = currTask.GetPtRegs();

    let kernelRsp = pt as *const _ as u64;
    CPULocal::Myself().SetEnterAppTimestamp(TSC.Rdtsc());
    //currTask.mm.VcpuEnter();
    CPULocal::Myself().SetMode(VcpuMode::User);
    currTask.mm.HandleTlbShootdown();
    SyscallRet(kernelRsp)
}

extern "C" {
    pub fn initX86FPState(data: u64, useXsave: bool);
}

pub fn InitX86FPState(data: u64, useXsave: bool) {
    unsafe { initX86FPState(data, useXsave) }
}







impl HostAllocator {
    pub const fn New() -> Self {
        return Self {
            host_initialization_heap: AtomicU64::new(0),
            host_guest_shared_heap: AtomicU64::new(0),
            guest_private_heap: AtomicU64::new(0),
            initialized: AtomicBool::new(true),
            is_vm_lauched: AtomicBool::new(true),
            is_host_allocator: AtomicBool::new(false),
        };
    }

    pub fn Init(&self, privateHeapAddr: u64, sharedHeapAddr: u64) {
        self.host_guest_shared_heap.store(sharedHeapAddr, Ordering::SeqCst);
        self.guest_private_heap.store(privateHeapAddr, Ordering::SeqCst);
    }
}

unsafe impl GlobalAlloc for HostAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        return self.GuestPrivateAllocator().alloc(layout);
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let addr = ptr as u64;
        if Self::IsGuestPrivateHeapAddr(addr) {
            self.GuestPrivateAllocator().dealloc(ptr, layout);
        } else {
            self.GuestHostSharedAllocator().dealloc(ptr, layout);
        }
    }
}




#[inline]
pub fn VcpuId() -> usize {
    return CPULocal::CpuId();
}

pub fn HugepageDontNeed(addr: u64) {
    let ret = HostSpace::Madvise(
        addr,
        MemoryDef::HUGE_PAGE_SIZE as usize,
        MAdviseOp::MADV_DONTNEED,
    );
    assert!(ret == 0, "HugepageDontNeed fail with {}", ret)
}

impl IOMgr {
    pub fn Init() -> Result<Self> {
        return Err(Error::Common(format!("IOMgr can't init in kernel")));
    }
}

unsafe impl GlobalAlloc for GlobalVcpuAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !self.init.load(Ordering::Relaxed) {
            return GLOBAL_ALLOCATOR.alloc(layout);
        }

        return PRIVATE_VCPU_LOCAL_HOLDER.AllocatorMut().alloc(layout);
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if !self.init.load(Ordering::Relaxed) {
            return GLOBAL_ALLOCATOR.dealloc(ptr, layout);
        }

        return PRIVATE_VCPU_LOCAL_HOLDER.AllocatorMut().dealloc(ptr, layout);
    }
}

impl UringAsyncMgr {
    pub fn FreeSlot(&self, id: usize) {
        self.freeSlot(id);
    }

    pub fn Clear(&self) {
        loop {
            let id = match self.freeids.lock().pop_front() {
                None => break,
                Some(id) => id,
            };
            self.freeSlot(id as _);
        }
    }
}

pub fn IsKernel() -> bool {
    return true;
}

pub fn ReapSwapIn() {
    HostSpace::SwapIn();
}


impl TsotSocketMgr {
    pub fn SendMsg(m: &TsotMessage) -> Result<()> {
        let res = HostSpace::TsotSendMsg(m as * const _ as u64);
        if res == 0 {
            return Ok(())
        }

        return Err(Error::SysError(SysErr::EINVAL));
    }

    pub fn RecvMsg() -> Result<TsotMessage> {
        let mut m = TsotMessage::default();
        let res = HostSpace::TsotRecvMsg(&mut m as * mut _ as u64);
        if res == 0 {
            return Ok(m)
        }

        return Err(Error::SysError(SysErr::EINVAL));
    }
}

impl DnsSvc {
    pub fn Init(&self) -> Result<()> {
        panic!("impossible");
    }
}