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

use crate::qlib::mutex::*;
use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
use core::mem;
use core::ops::Deref;
use core::ptr;
use core::sync::atomic::Ordering;

//use super::arch::x86_64::arch_x86::*;
use super::super::super::kernel_def::*;
use super::super::auth::*;
use super::super::common::*;
use super::super::linux_def::*;
use super::super::perf_tunning::*;
use super::super::task_mgr::*;
use super::super::usage::io::*;
use super::super::vcpu_mgr::*;
use super::fs::dirent::*;
use super::kernel::fd_table::*;
use super::kernel::futex::*;
use super::kernel::ipc_namespace::*;
use super::kernel::time::*;
use super::kernel::timer::*;
use super::kernel::uts_namespace::*;
use super::kernel::waiter::*;
use super::memmgr::mm::*;
use super::perflog::*;
use super::threadmgr::task_block::*;
use super::threadmgr::task_exit::*;
use super::threadmgr::task_sched::*;
use super::threadmgr::task_syscall::*;
use super::threadmgr::thread::*;
use super::vcpu::*;
use super::SignalDef::*;
use super::*;

use super::fs::file::*;
use super::fs::mount::*;
use super::kernel::fs_context::*;

use super::super::SysCallID;
use super::asm::*;

const DEFAULT_STACK_SIZE: usize = MemoryDef::DEFAULT_STACK_SIZE as usize;
pub const DEFAULT_STACK_PAGES: u64 = DEFAULT_STACK_SIZE as u64 / (4 * 1024);
pub const DEFAULT_STACK_MAST: u64 = !(DEFAULT_STACK_SIZE as u64 - 1);

pub static DUMMY_TASK: Singleton<DummyTask> = Singleton::<DummyTask>::New();

pub unsafe fn InitSingleton() {
    DUMMY_TASK.Init(DummyTask::default());
}

pub struct DummyTask(Arc<QRwLock<Task>>);

impl Deref for DummyTask {
    type Target = Arc<QRwLock<Task>>;

    fn deref(&self) -> &Arc<QRwLock<Task>> {
        &self.0
    }
}

impl Default for DummyTask {
    fn default() -> Self {
        return Self(Arc::new(QRwLock::new(Task::DummyTask())));
    }
}

impl DummyTask {
    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }
}

pub struct TaskStore {}

impl TaskStore {
    pub fn New() -> Self {
        return TaskStore {};
    }

    pub fn CreateTask(runFnAddr: u64, para: *const u8, kernel: bool) -> TaskId {
        let t = Task::Create(runFnAddr as u64, para, kernel);
        return TaskId::New(t.taskId);
    }

    pub fn CreateFromThread() -> TaskId {
        let t = Task::CreateFromThread();

        return TaskId::New(t.taskId);
    }
}

impl TaskId {
    #[inline]
    pub fn GetTask(&self) -> &'static mut Task {
        return unsafe { &mut *(self.Addr() as *mut Task) };
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TaskState {
    Running,
    Ready,
    Waiting,
    Done,
    Saving,
}

fn guard() {}

pub type TaskFn = fn(*const u8);

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub struct TidInfo {
    pub set_child_tid: u64,
    pub clear_child_tid: Option<u64>,
    pub robust_list_head: u64,
}

#[derive(Debug)]
pub struct Guard(u64);

impl Default for Guard {
    fn default() -> Self {
        return Self(Self::MAGIC_GUILD);
    }
}

impl Guard {
    const MAGIC_GUILD: u64 = 0x1234567890abcd;

    #[inline(always)]
    pub fn Check(&self) {
        if self.0 != Self::MAGIC_GUILD {
            let task = Task::Current();
            raw!(0x240, task.taskId, self.0, 0);
            super::Kernel::HostSpace::VcpuDebug();
            loop {}
        }
        //assert!(self.0==Self::MAGIC_GUILD)
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        info!("Task::Drop...");
    }
}

#[repr(C)]
pub struct Task {
    pub context: Context,
    pub taskId: u64,
    pub mm: MemoryManager,
    pub tidInfo: TidInfo,
    pub isWaitThread: bool,
    pub signalStack: SignalStack,
    pub creds: Credentials,
    pub utsns: UTSNamespace,
    pub ipcns: IPCNamespace,

    pub fdTbl: FDTable,

    pub fsContext: FSContext,

    pub mountNS: MountNs,
    pub blocker: Blocker,

    pub thread: Option<Thread>,
    pub haveSyscallReturn: bool,
    pub syscallRestartBlock: Option<Box<SyscallRestartBlock>>,
    pub futexMgr: FutexMgr,
    pub ioUsage: IO,
    pub sched: TaskSchedInfo,
    pub exiting: bool,

    pub perfcounters: Option<Arc<Counters>>,

    pub guard: Guard,
    //check whether the stack overflow
}

unsafe impl Sync for Task {}

impl Default for Task {
    fn default() -> Self {
        return Self::DummyTask();
    }
}

impl Task {
    #[inline(always)]
    pub fn Check(&self) {
        self.guard.Check();
    }

    pub fn Addr(&self) -> u64 {
        return self as *const _ as u64;
    }

    pub fn IPCNamespace(&self) -> IPCNamespace {
        return self.ipcns.clone();
    }

    //clean object on stack
    pub fn SetDummy(&mut self) {
        let dummyTask = DUMMY_TASK.read();
        self.mm = dummyTask.mm.clone();
        self.mountNS = dummyTask.mountNS.clone();
        self.creds = dummyTask.creds.clone();
        self.utsns = dummyTask.utsns.clone();
        self.ipcns = dummyTask.ipcns.clone();

        self.fsContext = dummyTask.fsContext.clone();

        self.fdTbl = dummyTask.fdTbl.clone();
        //self.blocker = dummyTask.blocker.clone();
        self.thread = None;
        self.syscallRestartBlock = None;
        self.futexMgr = dummyTask.futexMgr.clone();
        self.perfcounters = None;
        self.sched = dummyTask.sched.clone();
        self.exiting = true;
        self.ioUsage = dummyTask.ioUsage.clone();
    }

    pub fn SaveFp(&mut self) {
        self.context.X86fpstate.as_ref().unwrap().SaveFp();
        self.context.savefpsate = true;
    }

    pub fn RestoreFp(&mut self) {
        if self.context.savefpsate {
            self.context.X86fpstate.as_ref().unwrap().RestoreFp();
            self.context.savefpsate = false;
        }
    }

    pub fn QueueId(&self) -> usize {
        return self.context.queueId.load(Ordering::Acquire);
    }

    pub fn SetQueueId(&self, queueId: usize) {
        return self.context.queueId.store(queueId, Ordering::Release);
    }

    #[inline(always)]
    pub fn TaskAddress() -> u64 {
        let rsp = GetRsp();
        let task = rsp & DEFAULT_STACK_MAST;
        if rsp - task < 0x2000 {
            raw!(0x238, rsp, task, 0);
            super::Kernel::HostSpace::VcpuDebug();
            loop {}
            //panic!("TaskAddress panic");
        }
        return rsp; //& DEFAULT_STACK_MAST;
    }

    pub fn DummyTask() -> Self {
        let creds = Credentials::default();
        let userns = creds.lock().UserNamespace.clone();

        let futexMgr = FutexMgr::default();
        let mm = MemoryManager::Init(true);
        let mountNS = MountNs::default();
        let fsContext = FSContext::default();
        let utsns = UTSNamespace::New("".to_string(), "".to_string(), userns.clone());
        let ipcns = IPCNamespace::New(&userns);
        let fdTbl = FDTable::default();
        let blocker = Blocker::Dummy();

        let ret = Task {
            context: Context::New(),
            taskId: 0,
            //mm: MemoryMgr::default(),
            mm: mm,
            tidInfo: Default::default(),
            isWaitThread: false,
            signalStack: Default::default(),
            mountNS: mountNS,
            creds: creds.clone(),
            utsns: utsns,
            ipcns: ipcns,

            fsContext: fsContext,

            fdTbl: fdTbl,
            blocker: blocker,
            thread: None,
            haveSyscallReturn: false,
            syscallRestartBlock: None,
            futexMgr: futexMgr,
            ioUsage: IO::default(),
            sched: TaskSchedInfo::default(),
            exiting: false,
            perfcounters: None,
            guard: Guard::default(),
        };

        return ret;
    }

    pub fn AccountTaskEnter(&self, state: SchedState) {
        if self.taskId == CPULocal::WaitTask() || self.exiting == true {
            return;
        }

        let now = TSC.Rdtsc();
        let mut t = self.sched.lock();
        let current = t.State;

        match current {
            SchedState::RunningSys => {
                t.SysTicks += now - t.Timestamp;
            }
            SchedState::Nonexistent => {}
            SchedState::Stopped => {}
            _ => {
                panic!("AccountTaskEnter: Task[{:x}] switching from state {:?} (expected {:?}) to {:?}",
                       self.taskId, t.State, SchedState::RunningSys, state)
            }
        }

        t.Timestamp = now;
        t.State = state;
    }

    pub fn AccountTaskLeave(&self, state: SchedState) {
        //print!("AccountTaskLeave current task is {:x}, state is {:?}", self.taskId, state);
        if self.taskId == CPULocal::WaitTask() || self.exiting == true {
            return;
        }

        let now = TSC.Rdtsc();
        let mut t = self.sched.lock();
        //info!("AccountTaskLeave[{:x}] current state is {:?} -> {:?}, now is {}",
        //    self.taskId, t.State, SchedState::RunningSys, now);
        if t.State != state &&
            t.State != SchedState::Nonexistent &&
            // when doing clone, there is no good way to change new thread stat to runapp. todo: fix this
            t.State != SchedState::RunningSys
        {
            panic!(
                "AccountTaskLeave: Task[{:x}] switching from state {:?} (expected {:?}) to {:?}",
                self.taskId,
                t.State,
                SchedState::RunningSys,
                state
            )
        }

        if state == SchedState::RunningApp && t.State != SchedState::Nonexistent {
            t.UserTicks += now - t.Timestamp;
        }

        t.Timestamp = now;
        t.State = SchedState::RunningSys;
    }

    pub fn StackOverflowCheck() {
        let rsp = GetRsp();
        let task = rsp & DEFAULT_STACK_MAST;
        if rsp - task < 0x2000 {
            raw!(0x237, rsp, task, 0);
            super::Kernel::HostSpace::VcpuDebug();
            loop {}
            //panic!("TaskAddress panic");
        }
    }

    // doStop is called to block until the task is not stopped.
    pub fn DoStop(&self) {
        let thread = match &self.thread {
            None => return,
            Some(t) => t.clone(),
        };

        if thread.lock().stopCount.Count() == 0 {
            return;
        }

        let stopCount = thread.lock().stopCount.clone();
        self.AccountTaskEnter(SchedState::Stopped);
        self.blocker.WaitGroupWait(self, &stopCount);
        self.AccountTaskLeave(SchedState::Stopped)
    }

    pub fn SetSyscallRestartBlock(&mut self, b: Box<SyscallRestartBlock>) {
        self.syscallRestartBlock = Some(b)
    }

    pub fn TakeSyscallRestartBlock(&mut self) -> Option<Box<SyscallRestartBlock>> {
        return self.syscallRestartBlock.take();
    }

    // TODO(Cong): double check the semantics of this
    pub fn IsChrooted(&self) -> bool {
        let realRoot = self.mountNS.root.clone();
        let root = self.fsContext.RootDirectory();
        return root != realRoot;
    }

    pub fn Root(&self) -> Dirent {
        return self.fsContext.RootDirectory();
    }

    pub fn Workdir(&self) -> Dirent {
        return self.fsContext.WorkDirectory();
    }

    pub fn Umask(&self) -> u32 {
        return self.fsContext.Umask();
    }

    pub fn Creds(&self) -> Credentials {
        return self.creds.clone();
    }

    pub fn GetFile(&self, fd: i32) -> Result<File> {
        match self.fdTbl.Get(fd) {
            Err(e) => return Err(e),
            Ok(f) => return Ok(f.0),
        }
    }

    pub fn GetDescriptor(&self, fd: i32) -> Result<(File, FDFlags)> {
        match self.fdTbl.Get(fd) {
            Err(e) => return Err(e),
            Ok(f) => return Ok((f.0, f.1)),
        }
    }

    pub fn GetFileAll(&self, fd: i32) -> Result<(File, FDFlags)> {
        return self.fdTbl.Get(fd);
    }

    pub fn SetFlags(&self, fd: i32, flags: &FDFlags) -> Result<()> {
        return self.fdTbl.SetFlags(fd, flags);
    }

    pub fn NewFDAt(&mut self, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        return self.fdTbl.NewFDAt(self, fd, file, flags);
    }

    pub fn FileOwner(&self) -> FileOwner {
        let creds = self.creds.lock();
        let ret = FileOwner {
            UID: creds.EffectiveKUID.clone(),
            GID: creds.EffectiveKGID.clone(),
        };

        return ret;
    }

    pub fn NewStdFds(&mut self, stdfds: &[i32], isTTY: bool) -> Result<()> {
        for i in 0..stdfds.len() {
            let file = self.NewFileFromHostStdioFd(i as i32, stdfds[i], isTTY)?;
            file.flags.lock().0.NonBlocking = false; //need to clean the stdio nonblocking
        }

        return Ok(());
    }

    pub fn NewFileFromHostStdioFd(&mut self, fd: i32, hostfd: i32, isTTY: bool) -> Result<File> {
        let fileOwner = self.FileOwner();
        let file = File::NewFileFromFd(self, hostfd, &fileOwner, true, isTTY)?;
        self.NewFDAt(fd, &Arc::new(file.clone()), &FDFlags::default())?;
        return Ok(file);
    }

    pub fn NewFDFrom(&self, fd: i32, file: &File, flags: &FDFlags) -> Result<i32> {
        return self.fdTbl.NewFDFrom(self, fd, file, flags);
    }

    pub fn RemoveFile(&self, fd: i32) -> Result<File> {
        match self.fdTbl.Remove(fd) {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(f) => return Ok(f),
        }
    }

    pub fn Dup(&mut self, oldfd: u64) -> i64 {
        match self.fdTbl.Dup(self, oldfd as i32) {
            Ok(fd) => fd as i64,
            Err(Error::SysError(e)) => -e as i64,
            Err(e) => panic!("unsupport error {:?}", e),
        }
    }

    pub fn Dup2(&mut self, oldfd: u64, newfd: u64) -> i64 {
        match self.fdTbl.Dup2(self, oldfd as i32, newfd as i32) {
            Ok(fd) => fd as i64,
            Err(Error::SysError(e)) => -e as i64,
            Err(e) => panic!("unsupport error {:?}", e),
        }
    }

    pub fn Dup3(&mut self, oldfd: u64, newfd: u64, flags: u64) -> i64 {
        match self
            .fdTbl
            .Dup3(self, oldfd as i32, newfd as i32, flags as i32)
        {
            Ok(fd) => fd as i64,
            Err(Error::SysError(e)) => -e as i64,
            Err(e) => panic!("unsupport error {:?}", e),
        }
    }

    #[inline(always)]
    pub fn TaskId() -> TaskId {
        let rsp = GetRsp();
        return TaskId::New(rsp & DEFAULT_STACK_MAST);
    }

    #[inline(always)]
    pub fn GetPtr(&self) -> &'static mut Task {
        return unsafe { &mut *(self.taskId as *mut Task) };
    }

    #[inline(always)]
    pub fn GetMut(&self) -> &'static mut Task {
        return unsafe { &mut *(self.taskId as *mut Task) };
    }

    #[inline(always)]
    pub fn GetKernelSp(&self) -> u64 {
        return self.taskId + DEFAULT_STACK_SIZE as u64 - 0x10;
    }

    #[inline(always)]
    pub fn GetPtRegs(&self) -> &'static mut PtRegs {
        //let addr = self.kernelsp - mem::size_of::<PtRegs>() as u64;
        let addr = self.GetKernelSp() - mem::size_of::<PtRegs>() as u64;
        return unsafe { &mut *(addr as *mut PtRegs) };
    }

    #[inline(always)]
    pub fn SetReturn(&self, val: u64) {
        let pt = self.GetPtRegs();
        pt.rax = val;
    }

    #[inline(always)]
    pub fn Return(&self) -> u64 {
        return self.GetPtRegs().rax;
    }

    const SYSCALL_WIDTH: u64 = 2;
    pub fn RestartSyscall(&self) {
        let pt = self.GetPtRegs();
        pt.rcx -= Self::SYSCALL_WIDTH;
        pt.rip = pt.rcx;
        pt.rax = pt.orig_rax;
    }

    pub fn RestartSyscallWithRestartBlock(&self) {
        let pt = self.GetPtRegs();
        pt.rcx -= Self::SYSCALL_WIDTH;
        pt.rip = pt.rcx;
        pt.rax = SysCallID::sys_restart_syscall as u64;
    }

    #[inline]
    pub fn RealTimeNow() -> Time {
        let clock = REALTIME_CLOCK.clone();
        return clock.Now();
    }

    #[inline]
    pub fn MonoTimeNow() -> Time {
        let clock = MONOTONIC_CLOCK.clone();
        return clock.Now();
    }

    pub fn Now(&self) -> Time {
        return Self::RealTimeNow();
    }

    #[inline(always)]
    pub fn Current() -> &'static mut Task {
        let rsp = GetRsp();

        return Self::GetTask(rsp);
    }

    #[inline(always)]
    pub fn GetTask(addr: u64) -> &'static mut Task {
        let addr = addr & DEFAULT_STACK_MAST;
        unsafe {
            return &mut *(addr as *mut Task);
        }
    }

    pub fn GetTaskId(&self) -> TaskId {
        return TaskId::New(self.taskId);
    }

    pub fn Create(runFnAddr: u64, para: *const u8, kernel: bool) -> &'static mut Self {
        //let s_ptr = pa.Alloc(DEFAULT_STACK_PAGES).unwrap() as *mut u8;
        let s_ptr = KERNEL_STACK_ALLOCATOR.Allocate().unwrap() as *mut u8;

        let size = DEFAULT_STACK_SIZE;

        let mut ctx = Context::New();

        unsafe {
            //ptr::write(s_ptr.offset((size - 24) as isize) as *mut u64, guard as u64);
            ptr::write(s_ptr.offset((size - 32) as isize) as *mut u64, runFnAddr);
            ctx.rsp = s_ptr.offset((size - 32) as isize) as u64;
            ctx.rdi = para as u64;
        }

        //let ioUsage = DUMMY_TASK.read().ioUsage.clone();
        let ioUsage = DUMMY_TASK.read().ioUsage.clone();
        let perfcounters = Some(THREAD_COUNTS.lock().NewCounters());
        let futexMgr = FUTEX_MGR.Fork();
        let blocker = Blocker::New(s_ptr as u64);
        let mm = MemoryManager::Init(kernel);
        let creds = Credentials::default();
        let userns = creds.lock().UserNamespace.clone();
        let utsns = UTSNamespace::New("".to_string(), "".to_string(), userns.clone());
        let ipcns = IPCNamespace::New(&userns);

        //put Task on the task as Linux
        let taskPtr = s_ptr as *mut Task;
        unsafe {
            ptr::write(
                taskPtr,
                Task {
                    context: ctx,
                    taskId: s_ptr as u64,
                    mm: mm,
                    tidInfo: Default::default(),
                    isWaitThread: false,
                    signalStack: Default::default(),
                    mountNS: MountNs::default(),
                    creds: creds.clone(),
                    utsns: utsns,
                    ipcns: ipcns,

                    fsContext: FSContext::default(),

                    fdTbl: FDTable::default(),
                    blocker: blocker,
                    thread: None,
                    haveSyscallReturn: false,
                    syscallRestartBlock: None,
                    futexMgr: futexMgr,
                    ioUsage: ioUsage,
                    sched: TaskSchedInfo::default(),
                    exiting: false,
                    perfcounters: perfcounters,
                    guard: Guard::default(),
                },
            );

            //let new = &mut *taskPtr;
            //new.PerfGoto(PerfType::Blocked);
            //new.PerfGoto(PerfType::Kernel);
            return &mut (*taskPtr);
        }
    }

    pub fn Thread(&self) -> Thread {
        match self.thread.clone() {
            None => panic!("Task::Thread panic..."),
            Some(t) => t,
        }
    }

    // Wait waits for an event from a thread group that is a child of t's thread
    // group, or a task in such a thread group, or a task that is ptraced by t,
    // subject to the options specified in opts.
    pub fn Wait(&self, opts: &WaitOptions) -> Result<WaitResult> {
        if opts.BlockInterruptErr.is_none() {
            return self.Thread().waitOnce(opts);
        }

        let tg = self.Thread().lock().tg.clone();
        let queue = tg.lock().eventQueue.clone();
        queue.EventRegister(self, &self.blocker.generalEntry, opts.Events);
        defer!(queue.EventUnregister(self, &self.blocker.generalEntry));
        loop {
            match self.Thread().waitOnce(opts) {
                Ok(wr) => {
                    return Ok(wr);
                }
                Err(Error::ErrNoWaitableEvent) => {}
                Err(e) => return Err(e),
            };

            match self.blocker.BlockGeneral() {
                Err(Error::ErrInterrupted) => {
                    return Err(opts.BlockInterruptErr.clone().unwrap());
                }
                _ => (),
            }
        }
    }

    pub fn Exit(&mut self) {
        self.blocker.Drop();
        self.ExitWithCode(ExitStatus::default());
    }

    pub fn ExitWithCode(&mut self, _exitCode: ExitStatus) {
        if self.isWaitThread {
            panic!("Exit from wait thread!")
        }

        match self.tidInfo.clear_child_tid {
            None => {
                //println!("there is no clear_child_tid");
            }
            Some(addr) => {
                let val: i32 = 0;
                self.CopyOutObj(&val, addr).ok();
            }
        }
    }

    pub fn CreateFromThread() -> &'static mut Self {
        let baseStackAddr = Self::TaskId().Addr();
        let taskPtr = baseStackAddr as *mut Task;

        unsafe {
            let creds = Credentials::default();
            let userns = creds.lock().UserNamespace.clone();
            let dummyTask = DUMMY_TASK.read();
            ptr::write(
                taskPtr,
                Task {
                    context: Context::New(),
                    taskId: baseStackAddr,
                    mm: dummyTask.mm.clone(),
                    tidInfo: Default::default(),
                    isWaitThread: true,
                    signalStack: Default::default(),
                    mountNS: MountNs::default(),
                    creds: creds.clone(),
                    utsns: UTSNamespace::New("".to_string(), "".to_string(), userns.clone()),
                    ipcns: IPCNamespace::New(&userns),

                    fsContext: FSContext::default(),

                    fdTbl: FDTable::default(),
                    blocker: Blocker::New(baseStackAddr),
                    thread: None,
                    haveSyscallReturn: false,
                    syscallRestartBlock: None,
                    futexMgr: FUTEX_MGR.clone(),
                    ioUsage: dummyTask.ioUsage.clone(),
                    sched: TaskSchedInfo::default(),
                    exiting: false,
                    perfcounters: None,
                    guard: Guard::default(),
                },
            );

            return &mut (*taskPtr);
        }
    }

    #[inline]
    pub fn SwitchPageTable(&self) {
        let root = self.mm.GetRoot();
        let curr = super::asm::CurrentUserTable();
        if curr != root {
            CPULocal::Myself()
                .tlbEpoch
                .store(self.mm.TLBEpoch(), Ordering::Relaxed);
            super::super::pagetable::PageTables::Switch(root);
        }
    }

    #[cfg(target_arch="x86_64")]
    pub fn SetKernelPageTable() {
        KERNEL_PAGETABLE.SwitchTo();
    }

    #[inline]
    pub fn SetFS(&self) {
        SetFs(self.context.fs);
    }

    #[inline]
    pub fn GetContext(&self) -> u64 {
        return (&self.context as *const Context) as u64;
    }

    //todo: remove this
    pub fn Open(&mut self, fileName: u64, flags: u64, _mode: u64) -> i64 {
        //todo: mode?
        match OpenAt(self, ATType::AT_FDCWD, fileName, flags as u32) {
            Ok(fd) => return fd as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Open get unknown failure"),
        }
    }

    pub fn SignalStack(&self) -> SignalStack {
        let mut alt = self.signalStack;
        if self.OnSignalStack(&alt) {
            alt.flags |= SignalStack::FLAG_ON_STACK
        }

        return alt;
    }

    pub fn OnSignalStack(&self, alt: &SignalStack) -> bool {
        let sp = self.GetPtRegs().rsp;
        return alt.Contains(sp);
    }

    pub fn SetSignalStack(&mut self, alt: SignalStack) -> bool {
        let mut alt = alt;
        if self.OnSignalStack(&self.signalStack) {
            return false; //I am on the signal stack, can't change
        }

        if !alt.IsEnable() {
            self.signalStack = SignalStack {
                flags: SignalStack::FLAG_DISABLE,
                ..Default::default()
            }
        } else {
            alt.flags &= SignalStack::FLAG_DISABLE;
            self.signalStack = alt;
        }

        return true;
    }

    // CloneSignalStack sets the task-private signal stack.
    //
    // This value may not be changed if the task is currently executing on the
    // signal stack, i.e. if t.onSignalStack returns true. In this case, this
    // function will return false. Otherwise, true is returned.
    pub fn CloneSignalStack(&self) -> SignalStack {
        let mut alt = self.signalStack;
        let mut ret = SignalStack::default();

        // Check that we're not executing on the stack.
        if self.OnSignalStack(&alt) {
            return ret;
        }

        if alt.flags & SignalStack::FLAG_DISABLE != 0 {
            // Don't record anything beyond the flags.
            ret = SignalStack {
                flags: SignalStack::FLAG_DISABLE,
                ..Default::default()
            };
        } else {
            // Mask out irrelevant parts: only disable matters.
            alt.flags &= SignalStack::FLAG_DISABLE;
            ret = alt;
        }

        return ret;
    }
}
