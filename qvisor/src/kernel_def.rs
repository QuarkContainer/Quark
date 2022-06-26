use cache_padded::CachePadded;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use libc::*;
use std::fmt;

use super::qlib::kernel::quring::uring_async::UringAsyncMgr;
use super::qlib::common::*;
use super::qlib::control_msg::*;
use super::qlib::kernel::memmgr::pma::*;
use super::qlib::kernel::task::*;
use super::qlib::kernel::Kernel::*;
use super::qlib::kernel::Tsc;
use super::qlib::kernel::TSC;
use super::qlib::linux::time::*;
use super::qlib::linux_def::*;
use super::qlib::loader::*;
use super::qlib::mutex::*;
use super::qlib::perf_tunning::*;
use super::qlib::qmsg::*;
use super::qlib::rdma_svc_cli::*;
use super::qlib::task_mgr::*;
use super::qlib::vcpu_mgr::*;
use super::qlib::*;
use super::vmspace::*;
use super::ThreadId;
use super::FD_NOTIFIER;
use super::QUARK_CONFIG;
use super::URING_MGR;
use super::VMS;
use crate::SHARE_SPACE;

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SuperError is here!")
    }
}

impl<'a> ShareSpace {
    pub fn AQCall(&self, msg: &HostOutputMsg) {
        panic!("ShareSpace::AQCall {:x?}", msg);
    }

    pub fn Schedule(&self, _taskId: u64) {}
}

impl<'a> ShareSpace {
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
                return;
            }

            /*if len > 16 * 1024 {
                len = 16 * 1024
            };*/

            let ret = unsafe { libc::write(logfd, addr as _, len) };
            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            cnt = ret as usize;
            self.ConsumeAndGetAvailableWriteBuf(cnt);
            return;
        }

        loop {
            let (addr, len) = self.ConsumeAndGetAvailableWriteBuf(cnt);
            if len == 0 {
                return;
            }

            let ret = unsafe { libc::write(logfd, addr as _, len) };
            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            cnt = ret as usize;
        }
    }
}

impl ShareSpace {
    pub fn Init(&mut self, vcpuCount: usize, controlSock: i32, rdmaSvcCliSock: i32) {
        *self.config.write() = *QUARK_CONFIG.lock();
        let mut values = Vec::with_capacity(vcpuCount);
        for _i in 0..vcpuCount {
            values.push([AtomicU64::new(0), AtomicU64::new(0)])
        }

        if self.config.read().EnableRDMA {
            self.rdmaSvcCli = CachePadded::new(RDMASvcClient::initialize(
                rdmaSvcCliSock,
                MemoryDef::RDMA_LOCAL_SHARE_OFFSET,
                MemoryDef::RDMA_GLOBAL_SHARE_OFFSET,
            ));
        }

        let SyncLog = self.config.read().SyncPrint();
        if !SyncLog {
            let bs = super::qlib::bytestream::ByteStream::Init(128 * 1024); // 128 MB
            *self.logBuf.lock() = Some(bs);
        }

        self.scheduler = Scheduler::New(vcpuCount);
        self.values = values;

        self.scheduler.Init();
        self.SetLogfd(super::print::LOG.Logfd());
        self.hostEpollfd
            .store(FD_NOTIFIER.Epollfd(), Ordering::SeqCst);
        self.controlSock = controlSock;
        self.supportMemoryBarrier = VMS.lock().haveMembarrierGlobal;
        super::vmspace::VMSpace::BlockFd(controlSock);
    }

    pub fn TlbShootdown(&self, vcpuMask: u64) -> u64 {
        let vcpu_len = self.scheduler.VcpuArr.len();
        for i in 1..vcpu_len {
            if ((1 << i) & vcpuMask != 0)
                && SHARE_SPACE.scheduler.VcpuArr[i].GetMode() == VcpuMode::User
            {
                let cpu = VMS.lock().vcpus[i].clone();
                SHARE_SPACE.scheduler.VcpuArr[i].InterruptTlbShootdown();
                cpu.interrupt();
            }
        }
        return 0;
    }

    pub fn Yield() {
        use std::{thread, time};
        let dur = time::Duration::new(0, 1000);
        thread::sleep(dur);
    }

    pub fn CheckVcpuTimeout(&self) {
        let now = TSC.Rdtsc();
        for i in 1..self.scheduler.VcpuArr.len() {
            let enterAppTimestamp = self.scheduler.VcpuArr[i].EnterAppTimestamp();
            if enterAppTimestamp == 0 {
                continue;
            }

            //error!("CheckVcpuTimeout {}/{}/{}/{}", i, enterAppTimestamp, now, Tsc::Scale(now - enterAppTimestamp));
            if Tsc::Scale(now - enterAppTimestamp) * 1000 > 2 * CLOCK_TICK {
                //self.scheduler.VcpuArr[i].ResetEnterAppTimestamp();

                // retry to send signal for each 2 ms
                self.scheduler.VcpuArr[i].SetEnterAppTimestamp(enterAppTimestamp + CLOCK_TICK / 5);
                self.scheduler.VcpuArr[i].InterruptThreadTimeout();
                //error!("CheckVcpuTimeout {}/{}/{}/{}", i, enterAppTimestamp, now, Tsc::Scale(now - enterAppTimestamp));
                let vcpu = VMS.lock().vcpus[i].clone();
                vcpu.interrupt();
            }
        }
    }
}

impl<T: ?Sized> QMutexIntern<T> {
    pub fn GetID() -> u64 {
        return super::ThreadId() as u64;
    }
}

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
    Kernel,
}

impl CounterSet {
    pub const PERM_COUNTER_SET_SIZE: usize = 1;
    pub fn GetPerfId(&self) -> usize {
        0
    }

    pub fn PerfType(&self) -> &str {
        return "PerfPrint::Host";
    }
}

pub fn switch(_from: TaskId, _to: TaskId) {}

pub fn OpenAt(_task: &Task, _dirFd: i32, _addr: u64, _flags: u32) -> Result<i32> {
    return Ok(0);
}

pub fn SignalProcess(_signalArgs: &SignalArgs) {}

pub fn StartRootContainer(_para: *const u8) {}
pub fn StartExecProcess(_fd: i32, _process: Process) {}
pub fn StartSubContainerProcess(_elfEntry: u64, _userStackAddr: u64, _kernelStackAddr: u64) {}

pub unsafe fn CopyPageUnsafe(_to: u64, _from: u64) {}

impl CPULocal {
    pub fn CpuId() -> usize {
        return ThreadId() as _;
    }

    pub fn Wakeup(&self) {
        let val: u64 = 8;
        let ret = unsafe { libc::write(self.eventfd, &val as *const _ as *const libc::c_void, 8) };
        if ret < 0 {
            panic!("KIOThread::Wakeup fail...");
        }
    }
}

impl PageMgrInternal {
    pub fn CopyVsysCallPages(&self) {}
}

pub fn ClockGetTime(clockId: i32) -> i64 {
    let ts = Timespec::default();
    let res = unsafe {
        clock_gettime(
            clockId as clockid_t,
            &ts as *const _ as u64 as *mut timespec,
        ) as i64
    };

    if res == -1 {
        return errno::errno().0 as i64;
    } else {
        return ts.ToNs().unwrap();
    }
}

pub fn VcpuFreq() -> i64 {
    return VMS.lock().GetVcpuFreq();
}

pub fn NewSocket(fd: i32) -> i64 {
    return VMSpace::NewSocket(fd);
}

pub fn UringWake(minCompleted: u64) {
    URING_MGR
        .lock()
        .Wake(minCompleted as _)
        .expect("qlib::HYPER CALL_URING_WAKE fail");
}

impl HostSpace {
    pub fn Close(fd: i32) -> i64 {
        return VMSpace::Close(fd);
    }

    pub fn Call(msg: &mut Msg, _mustAsync: bool) -> u64 {
        panic!("HostSpace::Call msg {:x?}", msg);
    }

    pub fn HCall(msg: &mut Msg, _lock: bool) -> u64 {
        panic!("HostSpace::HCall msg {:x?}", msg);
    }
}

#[inline]
pub fn child_clone(_userSp: u64) {}

pub fn InitX86FPState(_data: u64, _useXsave: bool) {}

#[inline]
pub fn VcpuId() -> usize {
    return ThreadId() as usize;
}

pub fn HugepageDontNeed(addr: u64) {
    let ret = unsafe {
        libc::madvise(
            addr as _,
            MemoryDef::HUGE_PAGE_SIZE as usize,
            MAdviseOp::MADV_DONTNEED,
        )
    };
    assert!(ret == 0, "HugepageDontNeed::Host fail with {}", ret)
}

impl UringAsyncMgr {
    pub fn FreeSlot(&self, id: usize) {
        self.freeids.lock().push_back(id as _);
    }
}