use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use super::qlib::*;
use super::qlib::loader::*;
use super::qlib::mutex::*;
use super::qlib::common::*;
use super::qlib::task_mgr::*;
use super::qlib::qmsg::*;
use super::qlib::control_msg::*;
use super::qlib::kernel::task::*;
use super::qlib::perf_tunning::*;
use super::qlib::vcpu_mgr::*;
use super::FD_NOTIFIER;
use super::QUARK_CONFIG;
use super::KERNEL_IO_THREAD;
use super::ThreadId;

impl<'a> ShareSpace {
    pub fn AQCall(&self, _msg: &HostOutputMsg) {
    }

    pub fn Schedule(&self, _taskId: u64) {
    }
}

impl<'a> ShareSpace {
    pub fn AQHostInputCall(&self, item: &HostInputMsg) {
        loop {
            if self.QInput.IsFull() {
                continue;
            }

            self.QInput.Push(&item).unwrap();
            break;
        }
        //SyncMgr::WakeVcpu(self, TaskIdQ::default());

        //SyncMgr::WakeVcpu(self, TaskIdQ::New(1<<12, 0));
        KERNEL_IO_THREAD.Wakeup(self);
    }

    pub fn LogFlush(&self, partial: bool) {
        let lock = self.logLock.try_lock();
        if lock.is_none() {
            return;
        }

        let logfd = self.logfd.load(Ordering::Relaxed);

        let mut cnt = 0;
        if partial {
            let (addr, len) = self.ConsumeAndGetAvailableWriteBuf(cnt);
            if len == 0 {
                return
            }

            /*if len > 16 * 1024 {
                len = 16 * 1024
            };*/

            let ret = unsafe {
                libc::write(logfd, addr as _, len)
            };
            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            cnt = ret as usize;
            self.ConsumeAndGetAvailableWriteBuf(cnt);
            return
        }

        loop {
            let (addr, len) = self.ConsumeAndGetAvailableWriteBuf(cnt);
            if len == 0 {
                return
            }

            let ret = unsafe {
                libc::write(logfd, addr as _, len)
            };
            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            cnt = ret as usize;
        }
    }
}

impl ShareSpace {
    pub fn Init(&mut self, vcpuCount: usize, controlSock: i32) {
        *self.config.write() = *QUARK_CONFIG.lock();
        let mut values = Vec::with_capacity(vcpuCount);
        for _i in 0..vcpuCount {
            values.push([AtomicU64::new(0), AtomicU64::new(0)])
        };

        let SyncLog= self.config.read().SyncPrint();
        if !SyncLog {
            let bs = super::qlib::bytestream::ByteStream::Init(128 * 1024); // 128 MB
            *self.logBuf.lock() = Some(bs);
        }

        self.scheduler = Scheduler::New(vcpuCount);
        self.values = values;

        self.scheduler.Init();
        self.SetLogfd(super::print::LOG.lock().Logfd());
        self.hostEpollfd.store(FD_NOTIFIER.Epollfd(), Ordering::SeqCst);
        self.controlSock = controlSock;
        super::vmspace::VMSpace::BlockFd(controlSock);
    }

    pub fn Yield() {
        std::thread::yield_now();
        std::thread::yield_now();
        std::thread::yield_now();
        std::thread::yield_now();
        std::thread::yield_now();
        std::thread::yield_now();
    }
}



impl<T: ?Sized> QMutexIntern<T> {
    pub fn GetID() -> u64 {
        return super::ThreadId() as u64;
    }
}

/*
impl Scheduler {
    // steal scheduling
    pub fn GetNext(&self) -> Option<TaskId> {
        return None
    }

    pub fn Count(&self) -> u64 {
        return 0
    }

    pub fn Print(&self) -> String {
        return "".to_string();
    }

    #[inline]
    pub fn GetNextForCpu(&self, currentCpuId: usize, vcpuId: usize) -> Option<TaskId> {
        return None
    }

    pub fn Schedule(&self, taskId: TaskId) {
    }

    pub fn KScheduleQ(&self, task: TaskId, vcpuId: usize) {
    }

    pub fn NewTask(&self, taskId: TaskId) -> usize {
        return 0;
    }
}*/


#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfType {
    Start,
    Other,
    QCall,
    AQCall,
    AQHostCall,
    BusyWait,
    IdleWait,
    BufWrite,
    End,
    User, //work around for kernel clone
    Idle, //work around for kernel clone

    ////////////////////////////////////////
    Blocked,
    Kernel
}

impl CounterSet {
    pub const PERM_COUNTER_SET_SIZE : usize = 1;
    pub fn GetPerfId(&self) -> usize {
        0
    }

    pub fn PerfType(&self) -> &str {
        return "PerfPrint::Host"
    }
}

pub fn switch(_from: TaskId, _to: TaskId) {}

pub fn OpenAt(_task: &Task, _dirFd: i32, _addr: u64, _flags: u32) -> Result<i32> {
    return Ok(0)
}

pub fn SignalProcess(_signalArgs: &SignalArgs) {}

pub fn StartRootContainer(_para: *const u8) {}
pub fn StartExecProcess(_fd: i32, _process: Process) {}
pub fn StartSubContainerProcess(_elfEntry: u64, _userStackAddr: u64, _kernelStackAddr: u64) {}

pub unsafe fn CopyPageUnsafe(_to: u64, _from: u64){}

impl CPULocal {
    pub fn CpuId() -> usize {
        return ThreadId() as _;
    }
}