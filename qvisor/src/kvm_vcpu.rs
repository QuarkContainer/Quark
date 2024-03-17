use crate::host_uring::HostSubmit;

use crate::arch::{vCPU, __cpu_arch::ArchvCPU};
use super::qcall::AQHostCall;
use super::qlib::buddyallocator::ZeroPage;
use super::qlib::common::Allocator;
use super::qlib::common::RefMgr;
use super::qlib::common::{Error, Result};
use super::qlib::kernel::IOURING;
use super::qlib::linux_def::{MemoryDef, SysErr};
use super::qlib::linux_def::{Signal, EVENT_READ};
use super::qlib::pagetable::AlignedAllocator;
use super::qlib::qmsg::qcall::{HostOutputMsg, QMsg};
use super::qlib::task_mgr::Scheduler;
use super::qlib::vcpu_mgr::CPULocal;
use super::qlib::ShareSpace;
use super::FD_NOTIFIER;
use super::VMS;
use super::vmspace::VMSpace;
use alloc::alloc::alloc;
use libc::ioctl;
use libc::gettid;
use core::alloc::Layout;
use core::slice;
use core::sync::atomic::AtomicU64;
use nix::sys::signal;
use std::sync::atomic::{fence, Ordering};
use std::sync::mpsc::Sender;
use std::os::unix::io::AsRawFd;

pub struct HostPageAllocator {
    pub allocator: AlignedAllocator,
}

impl HostPageAllocator {
    pub fn New() -> Self {
        return Self {
            allocator: AlignedAllocator::New(0x1000, 0x10000),
        };
    }
}

impl Allocator for HostPageAllocator {
    fn AllocPage(&self, _incrRef: bool) -> Result<u64> {
        let ret = self.allocator.Allocate()?;
        ZeroPage(ret);
        return Ok(ret);
    }
}

impl RefMgr for HostPageAllocator {
    fn Ref(&self, _addr: u64) -> Result<u64> {
        //panic!("HostPageAllocator doesn't support Ref");
        return Ok(1);
    }

    fn Deref(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support Deref");
    }

    fn GetRef(&self, _addr: u64) -> Result<u64> {
        panic!("HostPageAllocator doesn't support GetRef");
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
#[repr(u64)]
pub enum KVMVcpuState {
    HOST,
    GUEST,
}

#[repr(C)]
pub struct SignalMaskStruct {
    length: u32,
    mask1: u32,
    mask2: u32,
    _pad: u32,
}

pub struct KVMVcpu {
    pub id: usize,
    pub cordId: isize,
    pub threadid: AtomicU64,
    pub tgid: AtomicU64,
    pub state: AtomicU64,
    pub vcpuCnt: usize,
    pub topStackAddr: u64,
    pub entry: u64,
    pub arch_vcpu: ArchvCPU,
    pub heapStartAddr: u64,
    pub shareSpaceAddr: u64,
    pub autoStart: bool
}

//for pub shareSpace: * mut Mutex<ShareSpace>
unsafe impl Send for KVMVcpu {}

impl KVMVcpu {
    pub fn Init(
        id: usize,
        vcpuCnt: usize,
        vm_fd: &kvm_ioctls::VmFd,
        entry: u64,
        pageAllocatorBaseAddr: u64,
        shareSpaceAddr: u64,
        autoStart: bool,
    ) -> Result<Self> {
        const DEFAULT_STACK_PAGES: u64 = MemoryDef::DEFAULT_STACK_PAGES; //64KB
        let stackSize = DEFAULT_STACK_PAGES << 12;
        let stackAddr = AlignedAllocate(stackSize as usize, stackSize as usize, false).unwrap();
        let topStackAddr = stackAddr + (DEFAULT_STACK_PAGES << 12);

        let mut _arch_vcpu: ArchvCPU = ArchvCPU::new(&vm_fd, id);
        _arch_vcpu.init(id)?;
        let cpuAffinit = VMS.lock().cpuAffinit;
        let vcpuCoreId = if !cpuAffinit {
            -1
        } else {
            VMS.lock().ComputeVcpuCoreId(id) as isize
        };

        return Ok(Self {
            id: id,
            cordId: vcpuCoreId,
            threadid: AtomicU64::new(0),
            tgid: AtomicU64::new(0),
            state: AtomicU64::new(KVMVcpuState::HOST as u64),
            vcpuCnt,
            topStackAddr: topStackAddr,
            entry: entry,
            arch_vcpu: _arch_vcpu,
            heapStartAddr: pageAllocatorBaseAddr,
            shareSpaceAddr: shareSpaceAddr,
            autoStart: autoStart,
        });
    }

    pub fn run(&self, tgid: i32) -> Result<()> {
        SetExitSignal();
        self.SignalMask();
        let tid = unsafe { gettid() };
        self.threadid.store(tid as u64, Ordering::SeqCst);
        self.tgid.store(tgid as u64, Ordering::SeqCst);

        if self.cordId > 0 {
            let coreid = core_affinity::CoreId {
                id: self.cordId as usize,
            };
            // print cpu id
            core_affinity::set_for_current(coreid);
        }

        if !super::runc::runtime::vm::IsRunning() {
            info!("The VM is not running.");
            return Ok(());
        }

        info!(
            "Start enter guest[{}]: entry is {:x}, stack is {:x}",
            self.id, self.entry, self.topStackAddr
        );
        self.arch_vcpu.run(self.entry, self.topStackAddr, self.heapStartAddr,
                           self.shareSpaceAddr, self.id as u64, VMS.lock().vdsoAddr,
                           self.vcpuCnt as u64, self.autoStart)?;

        Ok(())
    }

    pub fn GuestMsgProcess(sharespace: &ShareSpace) -> usize {
        let mut count = 0;
        loop {
            let msg = sharespace.AQHostOutputPop();

            match msg {
                None => break,
                Some(HostOutputMsg::QCall(addr)) => {
                    count += 1;
                    let eventAddr = addr as *mut QMsg; // as &mut qlib::Event;
                    let qmsg = unsafe { &mut (*eventAddr) };
                    let currTaskId = qmsg.taskId;

                    {
                        let _l = if qmsg.globalLock {
                            Some(super::GLOCK.lock())
                        } else {
                            None
                        };

                        qmsg.ret = Self::qCall(qmsg.msg);
                    }

                    if currTaskId.Addr() != 0 {
                        sharespace
                            .scheduler
                            .ScheduleQ(currTaskId, currTaskId.Queue(), true)
                    }
                }
                Some(msg) => {
                    count += 1;
                    AQHostCall(msg, sharespace);
                }
            }
        }

        return count;
    }

    pub fn interrupt(&self, waitCh: Option<Sender<()>>) {
        let mut interrupting = self.arch_vcpu
                                   .get_interrupt_lock();
        if let Some(w) = waitCh {
            interrupting.1.push(w);
        }
        if !interrupting.0 {
            interrupting.0 = true;
            self.Signal(Signal::SIGCHLD);
        }
    }

    pub fn Signal(&self, signal: i32) {
        loop {
            let ret = VMSpace::TgKill(
                self.tgid.load(Ordering::Relaxed) as i32,
                self.threadid.load(Ordering::Relaxed) as i32,
                signal,
            );

            if ret == 0 {
                break;
            }

            if ret < 0 {
                let errno = errno::errno().0;
                if errno == SysErr::EAGAIN {
                    continue;
                }

                panic!("vcpu tgkill fail with error {}", errno);
            }
        }
    }

    pub const KVM_SET_SIGNAL_MASK: u64 = 0x4004ae8b;
    pub fn SignalMask(&self) {
        let boundSignal = Signal::SIGSEGV; // Signal::SIGCHLD;
        let bounceSignalMask: u64 = 1 << (boundSignal as u64 - 1);

        let data = SignalMaskStruct {
            length: 8,
            mask1: (bounceSignalMask & 0xffffffff) as _,
            mask2: (bounceSignalMask >> 32) as _,
            _pad: 0,
        };

        let ret = unsafe {
            ioctl(
                self.arch_vcpu
                    .vcpu_fd()
                    .unwrap()
                    .as_raw_fd(),
                Self::KVM_SET_SIGNAL_MASK,
                &data as *const _ as u64,
            )
        };

        assert!(
            ret == 0,
            "SignalMask ret is {}/{}/{}",
            ret,
            errno::errno().0,
            self.arch_vcpu
                .vcpu_fd()
                .unwrap()
                .as_raw_fd()
        );
    }
}

pub fn AlignedAllocate(size: usize, align: usize, zeroData: bool) -> Result<u64> {
    assert!(
        size % 8 == 0,
        "AlignedAllocate get unaligned size {:x}",
        size
    );
    let layout = Layout::from_size_align(size, align);
    match layout {
        Err(_e) => Err(Error::UnallignedAddress(format!("AlignedAllocate {:?}", align))),
        Ok(l) => unsafe {
            let addr = alloc(l);
            if zeroData {
                let arr = slice::from_raw_parts_mut(addr as *mut u64, size / 8);
                for i in 0..512 {
                    arr[i] = 0
                }
            }

            Ok(addr as u64)
        },
    }
}

// SetVmExitSigAction set SIGCHLD as the vm exit signal,
// the signal handler will set_kvm_immediate_exit to 1,
// which will force the vcpu running exit with Intr.
pub fn SetExitSignal() {
    let sig_action = signal::SigAction::new(
        signal::SigHandler::Handler(handleSigChild),
        signal::SaFlags::empty(),
        signal::SigSet::empty(),
    );

    unsafe {
        signal::sigaction(signal::Signal::SIGCHLD, &sig_action).expect("sigaction set fail");
        let mut sigset = signal::SigSet::empty();
        signal::pthread_sigmask(signal::SigmaskHow::SIG_BLOCK, None, Some(&mut sigset))
            .expect("sigmask set fail");
        sigset.remove(signal::Signal::SIGCHLD);
        signal::pthread_sigmask(signal::SigmaskHow::SIG_SETMASK, Some(&sigset), None)
            .expect("sigmask set fail");
    }
}

extern "C" fn handleSigChild(signal: i32) {
    if signal == Signal::SIGCHLD {
        // used for tlb shootdown
        if let Some(vcpu) = super::LocalVcpu() {
            vcpu.arch_vcpu
                .vcpu_fd()
                .unwrap()
                .set_kvm_immediate_exit(1);
            fence(Ordering::SeqCst);
        }
    }
}

impl Scheduler {
    pub fn Init(&mut self) {
        for i in 0..self.vcpuCnt {
            self.VcpuArr[i].Init(i);
        }
    }

    pub fn VcpWaitMaskSet(&self, vcpuId: usize) -> bool {
        let mask = 1 << vcpuId;
        let prev = self.vcpuWaitMask.fetch_or(mask, Ordering::SeqCst);
        return (prev & mask) != 0;
    }

    pub fn VcpWaitMaskClear(&self, vcpuId: usize) -> bool {
        let mask = 1 << vcpuId;
        let prev = self
            .vcpuWaitMask
            .fetch_and(!(1 << vcpuId), Ordering::SeqCst);
        return (prev & mask) != 0;
    }

    pub fn WaitVcpu(&self, sharespace: &ShareSpace, vcpuId: usize, block: bool) -> Result<u64> {
        return self.VcpuArr[vcpuId].VcpuWait(sharespace, block);
    }
}

impl CPULocal {
    pub fn Init(&mut self, vcpuId: usize) {
        let epfd = unsafe { libc::epoll_create1(0) };

        if epfd == -1 {
            panic!(
                "CPULocal::Init {} create epollfd fail, error is {}",
                self.vcpuId,
                errno::errno().0
            );
        }

        let eventfd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };

        if eventfd < 0 {
            panic!("Vcpu::Init fail...");
        }

        let mut ev = libc::epoll_event {
            events: EVENT_READ as u32 | libc::EPOLLET as u32,
            u64: eventfd as u64,
        };

        let ret = unsafe {
            libc::epoll_ctl(
                epfd,
                libc::EPOLL_CTL_ADD,
                eventfd,
                &mut ev as *mut libc::epoll_event,
            )
        };

        if ret == -1 {
            panic!(
                "CPULocal::Init {} add eventfd fail, error is {}",
                self.vcpuId,
                errno::errno().0
            );
        }

        let mut ev = libc::epoll_event {
            events: EVENT_READ as u32 | libc::EPOLLET as u32,
            u64: FD_NOTIFIER.Epollfd() as u64,
        };

        let ret = unsafe {
            libc::epoll_ctl(
                epfd,
                libc::EPOLL_CTL_ADD,
                FD_NOTIFIER.Epollfd(),
                &mut ev as *mut libc::epoll_event,
            )
        };

        if ret == -1 {
            panic!(
                "CPULocal::Init {} add host epollfd fail, error is {}",
                self.vcpuId,
                errno::errno().0
            );
        }

        self.eventfd = eventfd;
        self.epollfd = epfd;
        self.vcpuId = vcpuId;
        self.data = 1;
    }

    pub fn ProcessOnce(sharespace: &ShareSpace) -> usize {
        let mut count = 0;

        loop {
            let cnt = HostSubmit().unwrap();
            if cnt == 0 {
                break;
            }
            count += cnt;
        }

        count += IOURING.DrainCompletionQueue();
        count += KVMVcpu::GuestMsgProcess(sharespace);
        count += FD_NOTIFIER.HostEpollWait() as usize;

        return count;
    }

    pub fn Process(&self, sharespace: &ShareSpace) -> Option<u64> {
        match sharespace.scheduler.GetNext() {
            None => (),
            Some(newTask) => return Some(newTask.data),
        }
        return None;
    }

    pub fn VcpuWait(&self, sharespace: &ShareSpace, block: bool) -> Result<u64> {
        let mut events = [libc::epoll_event { events: 0, u64: 0 }; 2];

        let time = if block { -1 } else { 0 };

        sharespace.scheduler.VcpWaitMaskSet(self.vcpuId);
        defer!(sharespace.scheduler.VcpWaitMaskClear(self.vcpuId););

        match self.Process(sharespace) {
            None => (),
            Some(newTask) => return Ok(newTask),
        }

        super::GLOBAL_ALLOCATOR.Clear();
        self.ToWaiting(sharespace);
        defer!(self.ToSearch(sharespace););

        while !sharespace.Shutdown() {
            match self.Process(sharespace) {
                None => (),
                Some(newTask) => {
                    return Ok(newTask);
                }
            }

            if sharespace.scheduler.VcpWaitMaskSet(self.vcpuId) {
                match sharespace.scheduler.GetNext() {
                    None => (),
                    Some(newTask) => return Ok(newTask.data),
                }
            }

            super::GLOBAL_ALLOCATOR.Clear();

            let _nfds = unsafe { libc::epoll_wait(self.epollfd, &mut events[0], 2, time) };
            {
                let mut data: u64 = 0;
                let ret = unsafe {
                    libc::read(self.eventfd, &mut data as *mut _ as *mut libc::c_void, 8)
                };

                if ret < 0 && errno::errno().0 != SysErr::EINTR {
                    panic!(
                        "Vcppu::Wakeup fail... eventfd is {}, errno is {}",
                        self.eventfd,
                        errno::errno().0
                    );
                }
            }
        }
        return Err(Error::Exit);
    }
}
