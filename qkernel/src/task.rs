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

use alloc::string::ToString;
use alloc::vec::Vec;
use core::ptr;
use alloc::sync::Arc;
use spin::RwLock;
use core::mem;
use alloc::boxed::Box;
use lazy_static::lazy_static;

use super::qlib::buddyallocator::MemAllocator;
use super::qlib::linux_def::*;
use super::qlib::common::*;
use super::SignalDef::*;
use super::*;
use super::vcpu::*;
use super::qlib::auth::*;
use super::qlib::task_mgr::*;
use super::qlib::perf_tunning::*;
use super::kernel::time::*;
use super::syscalls::*;
use super::qlib::usage::io::*;
use super::fs::dirent::*;
use super::kernel::uts_namespace::*;
use super::kernel::ipc_namespace::*;
use super::kernel::fd_table::*;
use super::kernel::timer::timer::*;
use super::threadmgr::task_exit::*;
use super::threadmgr::task_block::*;
use super::threadmgr::task_syscall::*;
use super::threadmgr::task_sched::*;
use super::threadmgr::thread::*;
use super::kernel::waiter::*;
use super::kernel::futex::*;
use super::kernel::kernel::GetKernelOption;
use super::memmgr::mm::*;
use super::perflog::*;

use super::fs::file::*;
use super::fs::mount::*;
use super::kernel::fs_context::*;

use super::asm::*;
use super::qlib::SysCallID;

pub const DEFAULT_STACK_SIZE: usize = MemoryDef::DEFAULT_STACK_SIZE as usize;
pub const DEFAULT_STACK_PAGES: u64 = DEFAULT_STACK_SIZE as u64 / (4 * 1024);
pub const DEFAULT_STACK_MAST: u64 = !(DEFAULT_STACK_SIZE as u64 - 1);

lazy_static! {
    pub static ref DUMMY_TASK : RwLock<Task> = RwLock::new(Task::DummyTask());
}

pub struct TaskStore {}

impl TaskStore {
    pub fn New() -> Self {
        return TaskStore {}
    }

    pub fn CreateTask(runFn: TaskFn, para: *const u8, pa: &MemAllocator) -> TaskId {
        let t = Task::Create(runFn, para, pa);
        return TaskId::New(t.taskId);
    }

    pub fn CreateFromThread() -> TaskId {
        let t = Task::CreateFromThread();

        return TaskId::New(t.taskId);
    }

    pub fn FreeTask(taskId: TaskId, pa: &mut MemAllocator) {
        let task = taskId.GetTask();
        task.Free(pa);
    }
}


impl TaskId {
    #[inline]
    pub fn GetTask(&self) -> &'static mut Task {
        return unsafe { &mut *(self.Addr() as *mut Task) };
    }
}

#[derive(Debug, Default, Copy, Clone)]
#[repr(C)]
pub struct Context {
    pub rsp: u64,
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub rbx: u64,
    pub rbp: u64,
    pub rdi: u64,

    pub ready: u64,

    pub fs: u64,
    pub gs: u64,
}

impl Context {
    pub fn New() -> Self {
        return Self {
            ready: 1,
            ..Default::default()
        }
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
        return Self(Self::MAGIC_GUILD)
    }
}

impl Guard {
    const MAGIC_GUILD: u64 = 0x1234567890abcd;
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
    // job queue id
    pub queueId: usize,
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
    pub iovs: Vec<IoVec>,

    pub perfcounters: Option<Arc<Counters>>,

    pub guard: Guard,
    //check whether the stack overflow
}

unsafe impl Sync for Task {}

impl Task {
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
        self.blocker = dummyTask.blocker.clone();
        self.thread = None;
        self.syscallRestartBlock = None;
        self.futexMgr = dummyTask.futexMgr.clone();
        self.perfcounters = None;
        self.ioUsage = dummyTask.ioUsage.clone();
    }

    pub fn DummyTask() -> Self {
        let creds = Credentials::default();
        let userns = creds.lock().UserNamespace.clone();

        return Task {
            context: Context::default(),
            taskId: 0,
            queueId: 0,
            //mm: MemoryMgr::default(),
            mm: MemoryManager::Init(),
            tidInfo: Default::default(),
            isWaitThread: false,
            signalStack: Default::default(),
            mountNS: MountNs::default(),
            creds: creds.clone(),
            utsns: UTSNamespace::New("".to_string(), "".to_string(), userns.clone()),
            ipcns: IPCNamespace::New(&userns),

            fsContext: FSContext::default(),

            fdTbl: FDTable::default(),
            blocker: Blocker::default(),
            thread: None,
            haveSyscallReturn: false,
            syscallRestartBlock: None,
            futexMgr: FUTEX_MGR.clone(),
            ioUsage: IO::default(),
            sched: TaskSchedInfo::default(),
            iovs: Vec::new(),
            perfcounters: None,
            guard: Guard::default(),
        }
    }

    pub fn AccountTaskEnter(&self, state: SchedState) {
        //print!("AccountTaskEnter current task is {:x}", self.taskId);
        if self.taskId == CPULocal::WaitTask() {
            return
        }

        let kernel = match GetKernelOption() {
            None => return, //kernel is not initialized
            Some(k) => k,
        };

        let now = kernel.CPUClockNow();
        let mut t = self.sched.lock();
        //info!("AccountTaskEnter[{:x}] current state is {:?} -> {:?}, address is {:x}",
        //    self.taskId, t.State, state, &t.State as * const _ as u64);
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
        if self.taskId == CPULocal::WaitTask() {
            return
        }

        let kernel = match GetKernelOption() {
            None => return, //kernel is not initialized
            Some(k) => k,
        };

        let now = kernel.CPUClockNow();
        let mut t = self.sched.lock();
        //info!("AccountTaskLeave[{:x}] current state is {:?} -> {:?}, address is {:x}",
        //    self.taskId, t.State, SchedState::RunningSys, &t.State as * const _ as u64);
        if t.State != state &&
            t.State != SchedState::Nonexistent &&
            // when doing clone, there is no good way to change new thread stat to runapp. todo: fix this
            t.State != SchedState::RunningSys {
            panic!("AccountTaskLeave: Task[{:x}] switching from state {:?} (expected {:?}) to {:?}",
                   self.taskId, t.State, SchedState::RunningSys, state)
        }

        if state == SchedState::RunningApp {
            t.UserTicks += now - t.Timestamp
        }

        t.Timestamp = now;
        t.State = SchedState::RunningSys;
    }

    // doStop is called to block until the task is not stopped.
    pub fn DoStop(&self) {
        let thread = match &self.thread {
            None => return,
            Some(t) => t.clone(),
        };

        if thread.lock().stopCount.Count() == 0 {
            return
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

    pub fn IsChrooted(&self) -> bool {
        let kernel = self.Thread().lock().k.clone();
        let realRoot = kernel.RootDir();
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

    pub fn SwapUmask(&self, mask: u32) -> i64 {
        //return self.fsContext.SwapUmask(mask);
        return Kernel::HostSpace::Umask(mask as u32)
    }

    pub fn Creds(&self) -> Credentials {
        return self.creds.clone();
    }

    pub fn GetFile(&self, fd: i32) -> Result<File> {
        match self.fdTbl.lock().Get(fd) {
            Err(e) => return Err(e),
            Ok(f) => return Ok(f.0),
        }
    }

    pub fn GetDescriptor(&self, fd: i32) -> Result<(File, FDFlags)> {
        match self.fdTbl.lock().Get(fd) {
            Err(e) => return Err(e),
            Ok(f) => return Ok((f.0, f.1)),
        }
    }

    pub fn GetFileAll(&self, fd: i32) -> Result<(File, FDFlags)> {
        return self.fdTbl.lock().Get(fd);
    }

    pub fn SetFlags(&self, fd: i32, flags: &FDFlags) -> Result<()> {
        return self.fdTbl.lock().SetFlags(fd, flags);
    }

    pub fn NewFDs(&mut self, fd: i32, file: &[File], flags: &FDFlags) -> Result<Vec<i32>> {
        return self.fdTbl.lock().NewFDs(fd, file, flags)
    }

    pub fn NewFDAt(&mut self, fd: i32, file: &File, flags: &FDFlags) -> Result<()> {
        return self.fdTbl.lock().NewFDAt(fd, file, flags)
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
            let file = self.NewFileFromHostFd(i as i32, stdfds[i], isTTY)?;
            file.flags.lock().0.NonBlocking = false; //need to clean the stdio nonblocking
        }

        return Ok(())
    }

    pub fn NewFileFromHostFd(&mut self, fd: i32, hostfd: i32, isTTY: bool) -> Result<File> {
        let fileOwner = self.FileOwner();
        let file = File::NewFileFromFd(self, hostfd, &fileOwner, isTTY)?;
        self.NewFDAt(fd, &Arc::new(file.clone()), &FDFlags::default())?;
        return Ok(file);
    }

    pub fn NewFDFromHostFd(&mut self, hostfd: i32, isTTY: bool, wouldBlock: bool) -> Result<i32> {
        let fileOwner = self.FileOwner();
        let file = File::NewFileFromFd(self, hostfd, &fileOwner, isTTY)?;
        file.flags.lock().0.NonBlocking = !wouldBlock;
        let fds = self.NewFDs(0, &[file.clone()], &FDFlags::default())?;
        return Ok(fds[0]);
    }

    pub fn NewFDFrom(&self, fd: i32, file: &File, flags: &FDFlags) -> Result<i32> {
        //let fds = self.fdTbl.lock().NewFDs(fd, vec![file.clone()], flags)?;
        //return Ok(fds[0])
        return self.fdTbl.lock().NewFDFrom(fd, file, flags)
    }

    pub fn RemoveFile(&self, fd: i32) -> Result<File> {
        match self.fdTbl.lock().Remove(fd) {
            None => return Err(Error::SysError(SysErr::EBADF)),
            Some(f) => {
                return Ok(f)
            },
        }
    }

    pub fn Dup(&mut self, oldfd: u64) -> i64 {
        match self.fdTbl.lock().Dup(oldfd as i32) {
            Ok(fd) => fd as i64,
            Err(Error::SysError(e)) => -e as i64,
            Err(e) => panic!("unsupport error {:?}", e),
        }
    }

    pub fn Dup2(&mut self, oldfd: u64, newfd: u64) -> i64 {
        match self.fdTbl.lock().Dup2(oldfd as i32, newfd as i32) {
            Ok(fd) => fd as i64,
            Err(Error::SysError(e)) => -e as i64,
            Err(e) => panic!("unsupport error {:?}", e),
        }
    }

    pub fn Dup3(&mut self, oldfd: u64, newfd: u64, flags: u64) -> i64 {
        match self.fdTbl.lock().Dup3(oldfd as i32, newfd as i32, flags as i32) {
            Ok(fd) => fd as i64,
            Err(Error::SysError(e)) => -e as i64,
            Err(e) => panic!("unsupport error {:?}", e),
        }
    }

    #[inline(always)]
    pub fn TaskId() -> TaskId {
        //let rsp: u64;
        //unsafe { llvm_asm!("mov %rsp, $0" : "=r" (rsp) ) };
        let rsp = GetRsp();
        return TaskId::New(rsp & DEFAULT_STACK_MAST);
    }

    #[inline(always)]
    pub fn GetPtr(&self) -> &'static mut Task {
        return unsafe {
            &mut *(self.taskId as *mut Task)
        }
    }

    #[inline(always)]
    pub fn GetMut(&self) -> &'static mut Task {
        return unsafe {
            &mut *(self.taskId as *mut Task)
        }
    }

    #[inline(always)]
    pub fn GetKernelSp(&self) -> u64 {
        return self.taskId + DEFAULT_STACK_SIZE as u64 - 0x10;
    }

    #[inline(always)]
    pub fn GetPtRegs(&self) -> &'static mut PtRegs {
        //let addr = self.kernelsp - mem::size_of::<PtRegs>() as u64;
        let addr = self.GetKernelSp() - mem::size_of::<PtRegs>() as u64;
        return unsafe {
            &mut *(addr as *mut PtRegs)
        }
    }

    #[inline(always)]
    pub fn SetReturn(&self, val: u64) {
        let pt = self.GetPtRegs();
        pt.rax = val;
    }

    #[inline(always)]
    pub fn Return(&self) -> u64 {
        return self.GetPtRegs().rax
    }

    const SYSCALL_WIDTH: u64 = 2;
    pub fn RestartSyscall(&self) {
        let pt = self.GetPtRegs();
        pt.rcx -= Self::SYSCALL_WIDTH;
        pt.rax = pt.orig_rax;
    }

    pub fn RestartSyscallWithRestartBlock(&self) {
        let pt = self.GetPtRegs();
        pt.rcx -= Self::SYSCALL_WIDTH;
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
        //let rsp: u64;
        //unsafe { llvm_asm!("mov %rsp, $0" : "=r" (rsp) ) };
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

    pub fn GetTaskIdQ(&self) -> TaskIdQ {
        return TaskIdQ::New(self.taskId, self.queueId as u64)
    }

    pub fn Create(runFn: TaskFn, para: *const u8, pa: &MemAllocator) -> &'static mut Self {
        //let s_ptr = super::super::PAGE_ALLOCATOR.lock().Alloc(DEFAULT_STACK_PAGES).unwrap() as *mut u8; //16*4KB = 64KB

        let s_ptr = pa.Alloc(DEFAULT_STACK_PAGES).unwrap() as *mut u8;

        //let s_ptr = super::super::STACK_MGR.lock().Alloc(mm) as *mut u8;

        let size = DEFAULT_STACK_SIZE;

        let mut ctx = Context::New();

        unsafe {
            //ptr::write(s_ptr.offset((size - 24) as isize) as *mut u64, guard as u64);
            ptr::write(s_ptr.offset((size - 32) as isize) as *mut u64, runFn as u64);
            ctx.rsp = s_ptr.offset((size - 32) as isize) as u64;
            ctx.rdi = para as u64;
        }

        //put Task on the task as Linux
        let taskPtr = s_ptr as *mut Task;
        unsafe {
            let creds = Credentials::default();
            let userns = creds.lock().UserNamespace.clone();

            ptr::write(taskPtr, Task {
                context: ctx,
                taskId: s_ptr as u64,
                queueId: 0,
                mm: MemoryManager::Init(),
                tidInfo: Default::default(),
                isWaitThread: false,
                signalStack: Default::default(),
                mountNS: MountNs::default(),
                creds: creds.clone(),
                utsns: UTSNamespace::New("".to_string(), "".to_string(), userns.clone()),
                ipcns: IPCNamespace::New(&userns),

                fsContext: FSContext::default(),

                fdTbl: FDTable::default(),
                blocker: Blocker::New(s_ptr as u64),
                thread: None,
                haveSyscallReturn: false,
                syscallRestartBlock: None,
                futexMgr: FUTEX_MGR.Fork(),
                ioUsage: DUMMY_TASK.read().ioUsage.clone(),
                sched: TaskSchedInfo::default(),
                iovs: Vec::with_capacity(4),
                perfcounters: Some(THREAD_COUNTS.lock().NewCounters()),
                guard: Guard::default(),
            });

            let new = &mut *taskPtr;
            new.PerfGoto(PerfType::Blocked);
            new.PerfGoto(PerfType::Kernel);
            return &mut (*taskPtr)
        }
    }

    pub fn Thread(&self) -> Thread {
        return self.thread.clone().unwrap();
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
                Err(e) => {
                    return Err(e)
                }
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
                //todo: FutexWake
                match self.GetTypeMut::<i32>(addr) {
                    Err(_) => (),
                    Ok(v) => *v = 0,
                }
                /*unsafe {
                    *(addr as *mut u64) = 0;
                }*/

                //self.FutexWake(addr);
            }
        }
    }

    pub fn CreateFromThread() -> &'static mut Self {
        let baseStackAddr = Self::TaskId().Addr();
        let taskPtr = baseStackAddr as *mut Task;

        unsafe {
            //let root = super::KERNEL_PAGETABLE.read().GetRoot();
            let creds = Credentials::default();
            let userns = creds.lock().UserNamespace.clone();

            ptr::write(taskPtr, Task {
                context: Context::New(),
                taskId: baseStackAddr,
                queueId: 0,
                mm: MemoryManager::Init(),
                tidInfo: Default::default(),
                isWaitThread: true,
                signalStack: Default::default(),
                mountNS: MountNs::default(),
                creds: creds.clone(),
                utsns: UTSNamespace::New("".to_string(), "".to_string(), userns.clone()),
                ipcns: IPCNamespace::New(&userns),

                fsContext: FSContext::default(),

                fdTbl: FDTable::default(),
                blocker: Blocker::New1(baseStackAddr),
                thread: None,
                haveSyscallReturn: false,
                syscallRestartBlock: None,
                futexMgr: FUTEX_MGR.clone(),
                ioUsage: DUMMY_TASK.read().ioUsage.clone(),
                sched: TaskSchedInfo::default(),
                iovs: Vec::new(),
                perfcounters: None,
                guard: Guard::default(),
            });

            return &mut (*taskPtr)
        }
    }

    #[inline]
    pub fn SwitchPageTable(&self) {
        let root = self.mm.GetRoot();
        super::qlib::pagetable::PageTables::Switch(root);
    }

    #[inline]
    pub fn SetFS(&self) {
        SetFs(self.context.fs);
    }

    #[inline]
    pub fn GetContext(&self) -> u64 {
        return (&self.context as *const Context) as u64;
    }

    pub fn Free(&mut self, pa: &MemAllocator) {
        //super::super::PAGE_ALLOCATOR.lock().Free(self.taskId, DEFAULT_STACK_PAGES).unwrap();
        pa.Free(self.taskId, DEFAULT_STACK_PAGES).unwrap();
    }

    /*pub fn SetTidAddr(&mut self, tidptr: u64) -> u64 {
        self.tidInfo.clear_child_tid = Some(tidptr);
        return self.tid.unwrap() as u64;
        //return self.GetFs();
    }*/

    const ROBUST_LIST_LEN: u64 = 0x18;
    pub fn SetRobustList(&mut self, robust_list_head: u64, robust_list_len: u64) -> i64 {
        if robust_list_len != Self::ROBUST_LIST_LEN {
            return -SysErr::EINVAL as i64;
        }

        self.tidInfo.robust_list_head = robust_list_head;
        return 0;
    }

    pub fn PRLimit(&mut self, pid: u64, resource: u64, newLimit: u64, oldLimit: u64) -> i64 {
        let newLimit = if let Ok(newLimit) = self.CheckedV2P(newLimit) {
            newLimit
        } else {
            return SysErr::EINVAL as i64;
        };

        let oldLimit = if let Ok(oldLimit) = self.CheckedV2P(oldLimit) {
            oldLimit
        } else {
            return SysErr::EINVAL as i64;
        };

        return Kernel::HostSpace::PRLimit(pid as i32, resource as i32, newLimit, oldLimit);
    }

    pub fn GetRLimit(&mut self, resource: u64, rlimit: u64) -> i64 {
        let rlimit = if let Ok(rlimit) = self.CheckedV2P(rlimit) {
            rlimit
        } else {
            return SysErr::EINVAL as i64;
        };

        return Kernel::HostSpace::GetRLimit(resource as u32, rlimit);
    }

    pub fn SetRLimit(&mut self, resource: u64, rlimit: u64) -> i64 {
        let rlimit = if let Ok(rlimit) = self.CheckedV2P(rlimit) {
            rlimit
        } else {
            return SysErr::EINVAL as i64;
        };

        return Kernel::HostSpace::SetRLimit(resource as u32, rlimit);
    }

    pub fn MAdvise(&mut self, _addr: u64, _len: u64, _advise: u64) -> i64 {
        return 0; //todo: imple MAdvise
    }

    pub fn SetFs(&mut self, val: u64) {
        self.context.fs = val;
    }

    pub fn GetFs(&self) -> u64 {
        return self.context.fs;
    }

    pub fn SetGs(&mut self, val: u64) {
        self.context.gs = val;
    }

    pub fn GetGs(&self) -> u64 {
        return self.context.gs;
    }

    pub fn Truncate(&self, path: u64, len: u64) -> i64 {
        let path = if let Ok(path) = self.CheckedV2P(path) {
            path
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::Truncate(path, len as i64);
    }

    pub fn Open(&mut self, fileName: u64, flags: u64, _mode: u64) -> i64 {
        //todo: mode?
        match sys_file::openAt(self, ATType::AT_FDCWD, fileName, flags as u32) {
            Ok(fd) => return fd as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Open get unknown failure"),
        }

        /*if let Ok(fileNameAddr) = self.VirtualToPhy(fileName) {
            let hostfd = Kernel::Kernel::Open(fileNameAddr, flags as i32, mode as i32);
            if hostfd < 0 {
                return hostfd
            }

            let fd = self.AddFd(hostfd as i32, false).expect("open fail") as i64;

            return fd;
        } else {
            return -SysErr::EFAULT as i64;
        }*/
    }

    pub fn Close(&mut self, fd: u64) -> i64 {
        match sys_file::close(self, fd as i32) {
            Ok(()) => return 0,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Write: get unexpected error"),
        }

        /*//todo: fix tty
        if fd <=2 { //stdin, stdout, stderr
            return 0
        }

        match self.RemoveFd(fd as i32) {
            None => {
                info!("close fd {} not found", fd);
                -SysErr::EBADFD as i64
            },
            Some(_) => 0,
        }*/
    }

    pub fn Unlink(&self, pathname: u64) -> i64 {
        let pathname = if let Ok(pathname) = self.CheckedV2P(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64;
        };

        Kernel::HostSpace::Unlink(pathname)
    }

    /*pub fn Unlinkat(&mut self, dirfd: u64, pathname: u64, flags: u64) -> i64 {
        let dirfd = if dirfd as i32 == super::libcDef::AT_FDCWD {
            dirfd as i32
        } else {
            match self.GetHostfd(dirfd as i32) {
                None => return -SysErr::EINVAL as i64,
                Some(dirfd) => dirfd,
            }
        };

        let pathname = if let Ok(pathname) = self.CheckedV2P(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64;
        };

        Kernel::Kernel::Unlinkat(dirfd as i32, pathname, flags as i32)
    }*/

    pub fn Mkdir(&self, pathname: u64, mode_: u64) -> i64 {
        let pathname = if let Ok(pathname) = self.CheckedV2P(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64;
        };

        Kernel::HostSpace::Mkdir(pathname, mode_ as u32)
    }

    /*pub fn Mkdirat(&mut self, dirfd: u64, pathname: u64, mode_ : u64) -> i64 {
        let dirfd = if dirfd as i32 == super::libcDef::AT_FDCWD {
            dirfd as i32
        } else {
            match self.GetHostfd(dirfd as i32) {
                None => return -SysErr::EINVAL as i64,
                Some(dirfd) => dirfd,
            }
        };

        let pathname = if let Ok(pathname) = self.CheckedV2P(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64;
        };

        Kernel::Kernel::Mkdirat(dirfd as i32, pathname, mode_ as u32)
    }

    pub fn FSync(&mut self, fd: u64) -> i64 {
        let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EBADFD as i64,
            Some(fd) => fd,
        };

        Kernel::Kernel::FSync(fd as i32)
    }*/

    pub fn MSync(&self, addr: u64, len: u64, flags: u64) -> i64 {
        let addr = if let Ok(addr) = self.CheckedV2P(addr) {
            addr
        } else {
            return -SysErr::EFAULT as i64;
        };

        Kernel::HostSpace::MSync(addr, len as usize, flags as i32)
    }

    /*pub fn FDataSync(&mut self, fd: u64) -> i64 {
        let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EBADFD as i64,
            Some(fd) => fd,
        };

        Kernel::Kernel::FDataSync(fd as i32)
    }

    pub fn Flock(&mut self, fd: u64, operation: u64) -> i64 {
        let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EBADFD as i64,
            Some(fd) => fd,
        };

        Kernel::Kernel::Flock(fd as i32, operation as i32)
    }*/

    pub fn Uname(&self, buff: u64) -> i64 {
        let buff = {
            if let Ok(buff) = self.VirtualToPhy(buff) {
                buff
            } else {
                return -SysErr::EFAULT as i64;
            }
        };

        return Kernel::HostSpace::Uname(buff);
    }

    pub fn Prctl(&self, option: u64, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i64 {
        //for PR_SET_NAME only
        //todo: implement it
        let arg2 = {
            if let Ok(arg2) = self.VirtualToPhy(arg2) {
                arg2
            } else {
                return -SysErr::EFAULT as i64;
            }
        };

        return Kernel::HostSpace::Prctl(option as i32, arg2, arg3, arg4, arg5)
    }

    pub fn Pause(&self) -> i64 {
        return Kernel::HostSpace::Pause();
    }

    pub fn Sigtimedwait(&self, _set: u64, _info: u64, timeout: u64) -> i64 {
        assert!(timeout == 0, "Sigtimedwait not implement, timeout should be zero");
        self.Pause();
        panic!("sigtimedwait not implemented")
    }

    pub fn OpenAt(&mut self, dirfd: u64, fileName: u64, flags: u64, _mode: u64) -> i64 {
        //todo: mode?
        match sys_file::openAt(self, dirfd as i32, fileName, flags as u32) {
            Ok(fd) => return fd as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("OpenAt get unknown failure"),
        }
    }

    pub fn Read(&mut self, fd: u64, buf: u64, count: u64) -> i64 {
        match sys_read::Read(self, fd as i32, buf, count as i64) {
            Ok(res) => return res as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Read: get unexpected error"),
        }
    }

    pub fn Pread(&mut self, fd: u64, buf: u64, count: u64, offset: u64) -> i64 {
        match sys_read::Pread64(self, fd as i32, buf, count as i64, offset as i64) {
            Ok(res) => return res as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Pread: get unexpected error"),
        }
    }

    pub fn ReadLink(&self, _task: &Task,path: u64, buf: u64, bufsize: u64) -> i64 {
        let buf = if let Ok(buf) = self.VirtualToPhy(buf) {
            buf
        } else {
            return -SysErr::EFAULT as i64;
        };

        let path = if let Ok(path) = self.VirtualToPhy(path) {
            path
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::ReadLink(path, buf, bufsize);
    }

    pub fn Write(&mut self, fd: u64, buf: u64, count: u64) -> i64 {
        match sys_write::Write(self, fd as i32, buf, count as i64) {
            Ok(res) => return res as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Write: get unexpected error"),
        }
    }

    pub fn Pwrite(&mut self, fd: u64, buf: u64, count: u64, offset: u64) -> i64 {
        match sys_write::Pwrite64(self, fd as i32, buf, count as i64, offset as i64) {
            Ok(res) => return res as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Pwrite: get unexpected error"),
        }

        /*let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EINVAL as i64,
            Some(fd) => fd,
        };

        let buf = if let Ok(buf) = self.VirtualToPhy(buf) {
            buf
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::Kernel::Pwrite(fd as i32, buf, count, offset);*/
    }

    pub fn Writev(&mut self, fd: u64, iov: u64, iovcnt: u64) -> i64 {
        match sys_write::Writev(self, fd as i32, iov, iovcnt as i32) {
            Ok(res) => return res as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Writev: get unexpected error"),
        }

        /*let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EINVAL as i64,
            Some(fd) => fd,
        };

        let mut nIovVec : [IoVec; 32] = [IoVec {start:0, len:0}; 32];

        if iovcnt > 32 {
            panic!("Writev: too many iovcnt");
        }

        unsafe {
            for i in 0..iovcnt {
                let iov = (iov + i*16) as *const IoVec;
                let v_addr = (*iov).start as u64;
                let len = (*iov).len;

                match self.VirtualToPhy(v_addr) {
                    Err(e) => {
                        info!("convert to phyaddress fail, addr = {:x} e={:?}", v_addr, e);
                        return -SysErr::EFAULT as i64
                    },
                    Ok(pAddr) => {
                        nIovVec[i as usize].start = pAddr;
                        nIovVec[i as usize].len = len;
                    }
                }
            }

            return Kernel::Kernel::Writev(fd as i32, &nIovVec[0] as *const _ as u64, iovcnt as i32);
        }*/
    }

    const SELECT_READ_EVENTS: i32 = PollConst::POLLIN | PollConst::POLLHUP | PollConst::POLLERR;
    const SELECT_WRITE_EVENTS: i32 = PollConst::POLLOUT | PollConst::POLLERR;
    const SELECT_EXECPT_EVENTS: i32 = PollConst::POLLPRI;

    pub fn GetTimeOfDay(&mut self, tv: u64, tz: u64) -> i64 {
        let tv = if let Ok(tv) = self.CheckedV2P(tv) {
            tv
        } else {
            return -SysErr::EFAULT as i64;
        };

        let tz = if let Ok(tz) = self.CheckedV2P(tz) {
            tz
        } else {
            return -SysErr::EFAULT as i64;
        };

        return super::Kernel::HostSpace::GetTimeOfDay(tv, tz);
    }

    pub fn ClockGetRes(&mut self, clkId: u64, ts: u64) -> i64 {
        let ts = if let Ok(ts) = self.CheckedV2P(ts) {
            ts
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::ClockGetRes(clkId as i32, ts);
    }

    pub fn ClockGetTime(&mut self, clkId: u64, ts: u64) -> i64 {
        let ts = if let Ok(ts) = self.CheckedV2P(ts) {
            ts
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::ClockGetTime(clkId as i32, ts);
    }

    pub fn ClockSetTime(&mut self, clkId: u64, ts: u64) -> i64 {
        let ts = if let Ok(ts) = self.CheckedV2P(ts) {
            ts
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::ClockGetRes(clkId as i32, ts);
    }

    pub fn Times(&mut self, tms: u64) -> i64 {
        let tms = if let Ok(tms) = self.CheckedV2P(tms) {
            tms
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::Times(tms);
    }

    pub fn NanoSleep(&mut self, req: u64, rem: u64) -> i64 {
        let req = if let Ok(req) = self.CheckedV2P(req) {
            req
        } else {
            return -SysErr::EFAULT as i64
        };

        let rem = if let Ok(rem) = self.CheckedV2P(rem) {
            rem
        } else {
            return -SysErr::EFAULT as i64
        };

        Kernel::HostSpace::NanoSleep(req, rem)
    }

    pub fn Time(&self, tloc: u64) -> i64 {
        let tloc = if let Ok(tloc) = self.CheckedV2P(tloc) {
            tloc
        } else {
            return -SysErr::EFAULT as i64
        };

        Kernel::HostSpace::Time(tloc)
    }

    pub fn MinCore(&self, addr: u64, len: u64, vec: u64) -> i64 {
        let addr = if let Ok(addr) = self.VirtualToPhy(addr) {
            addr
        } else {
            return -SysErr::EFAULT as i64;
        };

        let vec = if let Ok(vec) = self.VirtualToPhy(vec) {
            vec
        } else {
            return -SysErr::EFAULT as i64;
        };

        Kernel::HostSpace::MinCore(addr, len, vec)
    }

    pub fn Seek(&mut self, fd: u64, offset: u64, whence: u64) -> i64 {
        match sys_file::Lseek(self, fd as i32, offset as i64, whence as i32) {
            Ok(res) => return res as i64,
            Err(Error::SysError(e)) => return -e as i64,
            _ => panic!("Seek: get unexpected error"),
        }
        /*let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EINVAL as i64,
            Some(fd) => fd,
        };

        Kernel::Kernel::Seek(fd as i32, offset as i64, whence as i32)*/
    }


    /*pub fn Pipe(&mut self, fds: u64) -> i64 {
        let oldfds = fds;
        let fds = if let Ok(fds) = self.CheckedV2P(fds) {
            fds
        } else {
            return -SysErr::EFAULT as i64
        };

        let res = Kernel::Kernel::Pipe(fds);

        if res < 0 {
            return res;
        }

        let ptr = oldfds as * mut i32;

        unsafe {
            let fds = slice::from_raw_parts_mut(ptr, 2);

            fds[0] = self.AddFd(fds[0], false).expect("Pipe fail");
            fds[1] = self.AddFd(fds[1], false).expect("Pipe fail");
        }

        return res;
    }*/

    pub fn Getxattr(&self, _path: u64, _name: u64, _value: u64, _size: u64) -> i64 {
        //return -SysErr::ENODATA as i64
        return -SysErr::ENOTSUP as i64
        /*let path = if let Ok(path) = self.CheckedV2P(path) {
            path
        } else {
            return -SysErr::EFAULT as i64
        };

        let name = if let Ok(name) = self.CheckedV2P(name) {
            name
        } else {
            return -SysErr::EFAULT as i64
        };

        let value = if let Ok(value) = self.CheckedV2P(value) {
            value
        } else {
            return -SysErr::EFAULT as i64
        };

        Kernel::Kernel::Getxattr(path, name, value, size)*/
    }

    pub fn Lgetxattr(&self, _path: u64, _name: u64, _value: u64, _size: u64) -> i64 {
        //return -SysErr::ENODATA as i64
        return -SysErr::ENOTSUP as i64
        /*let path = if let Ok(path) = self.CheckedV2P(path) {
            path
        } else {
            return -SysErr::EFAULT as i64
        };

        let name = if let Ok(name) = self.CheckedV2P(name) {
            name
        } else {
            return -SysErr::EFAULT as i64
        };

        let value = if let Ok(value) = self.CheckedV2P(value) {
            value
        } else {
            return -SysErr::EFAULT as i64
        };

        Kernel::Kernel::Lgetxattr(path, name, value, size)*/
    }

    pub fn Fgetxattr(&mut self, _fd: u64, _name: u64, _value: u64, _size: u64) -> i64 {
        //return -SysErr::ENODATA as i64
        return -SysErr::ENOTSUP as i64
        /*let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EINVAL as i64,
            Some(fd) => fd,
        };

        let name = if let Ok(name) = self.CheckedV2P(name) {
            name
        } else {
            return -SysErr::EFAULT as i64
        };

        let value = if let Ok(value) = self.CheckedV2P(value) {
            value
        } else {
            return -SysErr::EFAULT as i64
        };

        Kernel::Kernel::Fgetxattr(fd as i32, name, value, size)*/
    }

    pub fn GetUid() -> i64 {
        return Kernel::HostSpace::GetUid()
    }

    pub fn GetEUid() -> i64 {
        return Kernel::HostSpace::GetEUid()
    }

    pub fn GetGid() -> i64 {
        return Kernel::HostSpace::GetGid()
    }

    pub fn SetGid(gid: u64) -> i64 {
        return Kernel::HostSpace::SetGid(gid as u32)
    }

    pub fn GetEGid() -> i64 {
        return Kernel::HostSpace::GetEGid()
    }

    pub fn GetGroups(&self, size: u64, list: u64) -> i64 {
        let list = if let Ok(list) = self.CheckedV2P(list) {
            list
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::GetGroups(size as i32, list)
    }

    pub fn SetGroups(&self, size: u64, list: u64) -> i64 {
        let list = if let Ok(list) = self.CheckedV2P(list) {
            list
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::SetGroups(size as usize, list)
    }

    pub fn Sysinfo(&self, info: u64) -> i64 {
        let info = if let Ok(info) = self.CheckedV2P(info) {
            info
        } else {
            return -SysErr::EFAULT as i64
        };
        return Kernel::HostSpace::Sysinfo(info)
    }

    pub fn GetCwd(&self, buf: u64, size: u64) -> i64 {
        let buf = if let Ok(buf) = self.CheckedV2P(buf) {
            buf
        } else {
            return -SysErr::EFAULT as i64
        };


        let ret = Kernel::HostSpace::GetCwd(buf, size);

        /*if ret == buf as i64 {
            return backBuff as i64;
        }*/

        return ret;
    }

    pub fn SchedGetAffinity(&self, pid: u64, cpuSetSize: u64, mask: u64) -> i64 {
        let mask = if let Ok(mask) = self.CheckedV2P(mask) {
            mask
        } else {
            return -SysErr::EFAULT as i64
        };

        //todo: fix SchedGetAffinity
        /*let ptr = mask as *mut u64;
        let arr = unsafe { slice::from_raw_parts_mut(ptr, cpuSetSize as usize/8 as usize) };
        arr[0]=0xff;
        return 8;*/

        return Kernel::HostSpace::SchedGetAffinity(pid as i32, cpuSetSize, mask);
    }

    pub fn GetRandom(&self, buf: u64, len: u64, flags: u64) -> i64 {
        let buf = if let Ok(buf) = self.CheckedV2P(buf) {
            buf
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::GetRandom(buf, len, flags as u32);
    }

    pub fn SignalStack(&self) -> SignalStack {
        let mut alt = self.signalStack;
        if self.OnSignalStack(&alt) {
            alt.flags |= SignalStack::FLAG_ON_STACK
        }

        return alt
    }

    pub fn OnSignalStack(&self, alt: &SignalStack) -> bool {
        let sp = self.context.rsp;
        return alt.Contains(sp)
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

    pub fn Chdir(&self, path: u64) -> i64 {
        let path = if let Ok(path) = self.CheckedV2P(path) {
            path
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::Chdir(path);
    }

    pub fn Fchdir(&self, fd: u64) -> i64 {
        assert!(false, "Fchdir not impl");
        return Kernel::HostSpace::Fchdir(fd as i32);
    }

    pub fn Mlock(&self, addr: u64, len: u64) -> i64 {
        let addr = if let Ok(addr) = self.CheckedV2P(addr) {
            addr
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::Mlock(addr, len);
    }

    pub fn MUnlock(&self, addr: u64, len: u64) -> i64 {
        let addr = if let Ok(addr) = self.CheckedV2P(addr) {
            addr
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::MUnlock(addr, len);
    }

    pub fn IOSetup(&self, nr_events: u64, ctx_idp: u64) -> i64 {
        let ctx_idp = if let Ok(ctx_idp) = self.CheckedV2P(ctx_idp) {
            ctx_idp
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::IOSetup(nr_events, ctx_idp);
    }

    pub fn IOSubmit(&self, ctx_id: u64, nr: u64, iocbpp: u64) -> i64 {
        let mut nIocbpp: [iocb; 32] = [iocb::default(); 32];
        let mut nAddrArr: [u64; 32] = [0; 32];

        if nr > 32 {
            panic!("IOSubmit: too many iocb");
        }

        //todo: the parameter is wrong, check it later
        unsafe {
            for i in 0..nr {
                let addr = (iocbpp + i * 8) as *const u64;
                let iocbp = (*addr) as *const iocb;
                info!("IOSubmit: nIocbpp[{}] is {:?}, addr is {:x}, iocbpp is {:x}", i, *iocbp, *addr, iocbpp);
                nIocbpp[i as usize] = *iocbp;
                nAddrArr[i as usize] = &nIocbpp[i as usize] as *const _ as u64;
            }
        }

        for i in 0..nr {
            let buf = nIocbpp[i as usize].aio_buf;
            nIocbpp[i as usize].aio_buf = if let Ok(buf) = self.CheckedV2P(buf) {
                buf
            } else {
                return -SysErr::EFAULT as i64
            };
            info!("IOSubmit: nIocbpp[{}] is {:?}, aio_buf is {:x}, iocbpp is {:x}", i, &nIocbpp[i as usize], nIocbpp[i as usize].aio_buf, iocbpp);
        }

        return Kernel::HostSpace::IOSubmit(ctx_id, nr, &nAddrArr[0] as *const _ as u64);
        //panic!("end of IOSubmit")
    }

    pub fn Rename(&self, oldpath: u64, newpath: u64) -> i64 {
        let oldpath = if let Ok(oldpath) = self.CheckedV2P(oldpath) {
            oldpath
        } else {
            return -SysErr::EFAULT as i64
        };

        let newpath = if let Ok(newpath) = self.CheckedV2P(newpath) {
            newpath
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::Rename(oldpath, newpath);
    }

    pub fn Rmdir(&self, pathname: u64) -> i64 {
        let pathname = if let Ok(pathname) = self.CheckedV2P(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::Rmdir(pathname);
    }

    pub fn Chown(&self, pathname: u64, owner: u64, group: u64) -> i64 {
        let pathname = if let Ok(pathname) = self.CheckedV2P(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64
        };

        return Kernel::HostSpace::Chown(pathname, owner as u32, group as u32);
    }

    /*pub fn FChown(&mut self, fd: u64, owner: u64, group: u64) -> i64 {
        let fd = match self.GetHostfd(fd as i32) {
            None => {
                return -SysErr::EBADF as i64
            },
            Some(fd) => fd,
        };

        return Kernel::Kernel::FChown(fd, owner as u32, group as u32);
    }

    pub fn TimerFdCreate(&mut self, clockId: u64, flags: u64) -> i64 {
        let hostfd = Kernel::Kernel::TimerFdCreate(clockId as i32, flags as i32);

        if hostfd <=0 {
            return hostfd;
        }

        let fd = self.AddFd(hostfd as i32, false).expect("open fail") as i64;

        return fd;
    }

    pub fn TimerFdSetTime(&mut self, fd: u64, flags: u64, newValue: u64, oldValue: u64) -> i64 {
        let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EBADFD as i64,
            Some(fd) => fd,
        };

        let newValue = if let Ok(newValue) = self.VirtualToPhy(newValue) {
            newValue
        } else {
            return -SysErr::EFAULT as i64;
        };

        let oldValue = if let Ok(oldValue) = self.VirtualToPhy(oldValue) {
            oldValue
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::Kernel::TimerFdSetTime(fd as i32, flags as i32, newValue, oldValue);
    }

    pub fn TimerFdGetTime(&mut self, fd: u64, currVal: u64) -> i64 {
        let fd = match self.GetHostfd(fd as i32) {
            None => return -SysErr::EBADFD as i64,
            Some(fd) => fd,
        };

        let currVal = if let Ok(currVal) = self.VirtualToPhy(currVal) {
            currVal
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::Kernel::TimerFdGetTime(fd as i32, currVal);
    }*/

    pub fn Chmod(&mut self, pathname: u64, mode: u64) -> i64 {
        let pathname = if let Ok(pathname) = self.VirtualToPhy(pathname) {
            pathname
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::Chmod(pathname, mode as u32);
    }

    pub fn SetHostName(&mut self, name: u64, len: u64) -> i64 {
        let name = if let Ok(name) = self.VirtualToPhy(name) {
            name
        } else {
            return -SysErr::EFAULT as i64;
        };

        return Kernel::HostSpace::SetHostName(name, len as usize);
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
