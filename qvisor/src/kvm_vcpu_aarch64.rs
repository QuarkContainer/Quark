use std::os::fd::AsRawFd;
use std::sync::atomic::Ordering;

use kvm_bindings::{kvm_vcpu_events, kvm_vcpu_events__bindgen_ty_1, kvm_vcpu_init};
use libc::{gettid, clock_gettime, clockid_t, timespec};
use kvm_ioctls::VcpuExit;

use super::qlib::common::{Result, Error};
use crate::qlib::kernel::IOURING;
use crate::qlib::linux::time::Timespec;
use crate::qlib::qmsg::{Print, QMsg};
use crate::{KVMVcpu, VMS, kvm_vcpu::SetExitSignal};
use crate::qlib::singleton::Singleton;
use crate::kvm_vcpu::KVMVcpuState;
use crate::qlib::linux_def::SysErr;
use crate::qlib::{backtracer, VcpuFeq, GetTimeCall};
use crate::{qlib, URING_MGR};
use crate::KERNEL_IO_THREAD;
use crate::SHARE_SPACE;
use crate::syncmgr::SyncMgr;
use crate::qlib::perf_tunning::PerfPrint;
use crate::runc::runtime::vm::{SetExitStatus, VirtualMachine};

const _KVM_ARM_VCPU_PSCI_0_2: u32 = 2;
const _KVM_ARM_VCPU_INIT: u64 = 0x4020aeae;
const _KVM_ARM64_REGS_PSTATE     :u64 = 0x6030000000100042;
const _KVM_ARM64_REGS_SP_EL1     :u64 = 0x6030000000100044;
const _KVM_ARM64_REGS_R1         :u64 = 0x6030000000100002;
const _KVM_ARM64_REGS_R0         :u64 = 0x6030000000100000;
const _KVM_ARM64_REGS_R2         :u64 = 0x6030000000100004;
const _KVM_ARM64_REGS_R3         :u64 = 0x6030000000100006;
const _KVM_ARM64_REGS_R4         :u64 = 0x6030000000100008;
const _KVM_ARM64_REGS_R5         :u64 = 0x603000000010000a;
const _KVM_ARM64_REGS_R6         :u64 = 0x603000000010000c;
const _KVM_ARM64_REGS_R7         :u64 = 0x603000000010000e;
const _KVM_ARM64_REGS_R8         :u64 = 0x6030000000100010;
const _KVM_ARM64_REGS_R18        :u64 = 0x6030000000100024;
const _KVM_ARM64_REGS_R29        :u64 = 0x6030000000100036;
const _KVM_ARM64_REGS_PC         :u64 = 0x6030000000100040;
const _KVM_ARM64_REGS_MAIR_EL1   :u64 = 0x603000000013c510;
const _KVM_ARM64_REGS_TCR_EL1    :u64 = 0x603000000013c102;
const _KVM_ARM64_REGS_TTBR0_EL1  :u64 = 0x603000000013c100;
const _KVM_ARM64_REGS_TTBR1_EL1  :u64 = 0x603000000013c101;
const _KVM_ARM64_REGS_SCTLR_EL1  :u64 = 0x603000000013c080;
const _KVM_ARM64_REGS_CPACR_EL1  :u64 = 0x603000000013c082;
const _KVM_ARM64_REGS_VBAR_EL1   :u64 = 0x603000000013c600;
const _KVM_ARM64_REGS_TIMER_CNT  :u64 = 0x603000000013df1a;
const _KVM_ARM64_REGS_CNTFRQ_EL0 :u64 = 0x603000000013df00;
const _KVM_ARM64_REGS_MDSCR_EL1  :u64 = 0x6030000000138012;
const _KVM_ARM64_REGS_CNTKCTL_EL1:u64 = 0x603000000013c708;
const _KVM_ARM64_REGS_TPIDR_EL1  :u64 = 0x603000000013c684;

const _TCR_IPS_40BITS:u64 = 2 << 32; // PA=40
const _TCR_IPS_48BITS :u64 = 5 << 32;// PA=48

const _TCR_T0SZ_OFFSET :u64 = 0;
const _TCR_T1SZ_OFFSET :u64 = 16;
const _TCR_IRGN0_SHIFT :u64 = 8;
const _TCR_IRGN1_SHIFT :u64 = 24;
const _TCR_ORGN0_SHIFT :u64 = 10;
const _TCR_ORGN1_SHIFT :u64 = 26;
const _TCR_SH0_SHIFT   :u64 = 12;
const _TCR_SH1_SHIFT   :u64 = 28;
const _TCR_TG0_SHIFT   :u64 = 14;
const _TCR_TG1_SHIFT   :u64 = 30;

const _TCR_T0SZ_VA48 :u64 = 64 - 48; // VA=48
const _TCR_T1SZ_VA48 :u64 = 64 - 48; // VA=48

const _TCR_A1     :u64 = 1 << 22;
const _TCR_ASID16 :u64 = 1 << 36;
const _TCR_TBI0   :u64 = 1 << 37;

const _TCR_TXSZ_VA48 :u64 = (_TCR_T0SZ_VA48 << _TCR_T0SZ_OFFSET) | (_TCR_T1SZ_VA48 <<  _TCR_T1SZ_OFFSET);

const _TCR_TG0_4K  :u64 = 0 << _TCR_TG0_SHIFT; // 4K
const _TCR_TG0_64K :u64 = 1 << _TCR_TG0_SHIFT; // 64K

const _TCR_TG1_4K :u64 = 2 << _TCR_TG1_SHIFT;

const _TCR_TG_FLAGS :u64 = _TCR_TG0_4K |  _TCR_TG1_4K;

const _TCR_IRGN0_WBWA :u64 = 1 << _TCR_IRGN0_SHIFT;
const _TCR_IRGN1_WBWA :u64 = 1 << _TCR_IRGN1_SHIFT;
const _TCR_IRGN_WBWA  :u64 = _TCR_IRGN0_WBWA |  _TCR_IRGN1_WBWA;

const _TCR_ORGN0_WBWA :u64 = 1 << _TCR_ORGN0_SHIFT;
const _TCR_ORGN1_WBWA :u64 = 1 << _TCR_ORGN1_SHIFT;

const _TCR_ORGN_WBWA :u64 = _TCR_ORGN0_WBWA |  _TCR_ORGN1_WBWA;

const _TCR_SHARED :u64 = (3 << _TCR_SH0_SHIFT) | (3 << _TCR_SH1_SHIFT);

const _TCR_CACHE_FLAGS :u64 = _TCR_IRGN_WBWA |  _TCR_ORGN_WBWA;

const _MT_DEVICE_nGnRnE     :u64 = 0;
const _MT_DEVICE_nGnRE      :u64 = 1;
const _MT_DEVICE_GRE        :u64 = 2;
const _MT_NORMAL_NC         :u64 = 3;
const _MT_NORMAL            :u64 = 4;
const _MT_NORMAL_WT         :u64 = 5;
const _MT_ATTR_DEVICE_nGnRnE:u64 = 0x00;
const _MT_ATTR_DEVICE_nGnRE :u64 = 0x04;
const _MT_ATTR_DEVICE_GRE   :u64 = 0x0c;
const _MT_ATTR_NORMAL_NC    :u64 = 0x44;
const _MT_ATTR_NORMAL_WT    :u64 = 0xbb;
const _MT_ATTR_NORMAL       :u64 = 0xff;
const _MT_ATTR_MASK         :u64 = 0xff;
const _MT_EL1_INIT          :u64 = (_MT_ATTR_DEVICE_nGnRnE << (_MT_DEVICE_nGnRnE * 8)) | (_MT_ATTR_DEVICE_nGnRE << (_MT_DEVICE_nGnRE * 8)) | (_MT_ATTR_DEVICE_GRE << (_MT_DEVICE_GRE * 8)) | (_MT_ATTR_NORMAL_NC << (_MT_NORMAL_NC * 8)) | (_MT_ATTR_NORMAL << (_MT_NORMAL * 8)) | (_MT_ATTR_NORMAL_WT << (_MT_NORMAL_WT * 8));

const _CNTKCTL_EL0PCTEN:u64 = 1 << 0;
const _CNTKCTL_EL0VCTEN:u64 = 1 << 1;
const _CNTKCTL_EL1_DEFAULT:u64 = _CNTKCTL_EL0PCTEN | _CNTKCTL_EL0VCTEN;

const _SCTLR_M          :u64 = 1 << 0;
const _SCTLR_C          :u64 = 1 << 2;
const _SCTLR_I          :u64 = 1 << 12;
const _SCTLR_DZE        :u64 = 1 << 14;
const _SCTLR_UCT        :u64 = 1 << 15;
const _SCTLR_UCI        :u64 = 1 << 26;
const _SCTLR_EL1_DEFAULT:u64 = _SCTLR_M | _SCTLR_C | _SCTLR_I | _SCTLR_UCT | _SCTLR_UCI | _SCTLR_DZE;

lazy_static! {
    pub static ref KVM_VCPU_INIT: Singleton<kvm_vcpu_init> = Singleton::<kvm_vcpu_init>::New();
}

impl KVMVcpu {
    pub fn run(&self, tgid: i32) -> Result<()> {
        SetExitSignal();
        self.setup_registers()?;
        let tid = unsafe { gettid() };
        self.threadid.store(tid as u64, Ordering::SeqCst);
        self.tgid.store(tgid as u64, Ordering::SeqCst);
        self.SignalMask();
        info!(
            "start enter guest[{}]: entry is {:x}, stack is {:x}",
            self.id, self.entry, self.topStackAddr
        );

        loop {
            if !super::runc::runtime::vm::IsRunning() {
                return Ok(());
            }
            self.state
                .store(KVMVcpuState::GUEST as u64, Ordering::Release);
            let kvmRet = match self.vcpu.run() {
                Ok(ret) => ret,
                Err(e) => {
                    if e.errno() == SysErr::EINTR {
                        self.vcpu.set_kvm_immediate_exit(0);
                        self.dump()?;
                        if self.vcpu.get_ready_for_interrupt_injection() > 0 {
                            VcpuExit::IrqWindowOpen
                        } else {
                            VcpuExit::Intr
                        }
                    } else {
                        let pc = self.vcpu.get_one_reg(_KVM_ARM64_REGS_PC).map_err(|e| Error::SysError(e.errno()))?;
                        let rsp = self.vcpu.get_one_reg(_KVM_ARM64_REGS_SP_EL1).map_err(|e| Error::SysError(e.errno()))?;
                        let rbp = self.vcpu.get_one_reg(_KVM_ARM64_REGS_R29).map_err(|e| Error::SysError(e.errno()))?;
                        error!("vcpu error: {:?}", e);
                        backtracer::trace(pc, rsp, rbp, &mut |frame| {
                            print!("host frame is {:#x?}", frame);
                            true
                        });

                        panic!("kvm virtual cpu[{}] run failed: Error {:?}", self.id, e)
                    }
                }
            };
            self.state
            .store(KVMVcpuState::HOST as u64, Ordering::Release);
            match kvmRet {
                VcpuExit::MmioRead(addr, data) => {
                    panic!(
                        "CPU[{}] Received an MMIO Read Request for the address {:#x}.",
                        self.id, addr,
                    );
                }
                VcpuExit::MmioWrite(addr, data) => {
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }
                    let para1 = self.vcpu.get_one_reg(_KVM_ARM64_REGS_R0).map_err(|e| Error::SysError(e.errno()))?;
                    let para2 = self.vcpu.get_one_reg(_KVM_ARM64_REGS_R1).map_err(|e| Error::SysError(e.errno()))?;
                    let para3 = self.vcpu.get_one_reg(_KVM_ARM64_REGS_R2).map_err(|e| Error::SysError(e.errno()))?;
                    let para4 = self.vcpu.get_one_reg(_KVM_ARM64_REGS_R3).map_err(|e| Error::SysError(e.errno()))?;
                    if addr > (u16::MAX as u64) {
                        panic!("cpu[{}] Received hypercall id max than 255");
                    }
                    self.handle_hypercall(addr as u16, data, para1, para2, para3, para4)?;
                }
                VcpuExit::Hlt => {
                    error!("in hlt....");
                }
                VcpuExit::FailEntry => {
                    info!("get fail entry***********************************");
                    break;
                }
                VcpuExit::Exception => {
                    info!("get exception");
                }
                VcpuExit::IrqWindowOpen => {
                    self.InterruptGuest();
                    self.vcpu.set_kvm_request_interrupt_window(0);
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }
                }
                VcpuExit::Intr => {
                    self.vcpu.set_kvm_request_interrupt_window(1);
                    {
                        let mut interrupting = self.interrupting.lock();
                        interrupting.0 = false;
                        interrupting.1.clear();
                    }
                }
                r => {
                    error!("Panic: CPU[{}] Unexpected exit reason: {:?}", self.id, r);
                    unsafe {
                        libc::exit(0);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn dump(&self) -> Result<()> {
        Ok(())
    }

    fn setup_registers(&self) -> Result<()> {
        self.vcpu.vcpu_init(&KVM_VCPU_INIT).map_err(|e|Error::SysError(e.errno()))?;
        // tcr_el1
        let data = _TCR_TXSZ_VA48 |  _TCR_CACHE_FLAGS | _TCR_SHARED | _TCR_TG_FLAGS |  _TCR_ASID16 |  _TCR_IPS_40BITS;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_TCR_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // mair_el1
        let data = _MT_EL1_INIT;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_MAIR_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // ttbr0_el1
        let data = VMS.lock().pageTables.GetRoot();
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_TTBR0_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // TODO set ttbr1_el1
        // cntkctl_el1
        let data = _CNTKCTL_EL1_DEFAULT;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_CNTKCTL_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // cpacr_el1
        let data = 0;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_CPACR_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // sctlr_el1
        let data = _SCTLR_EL1_DEFAULT;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_SCTLR_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // tpidr_el1 has to be set in kernel
        // sp_el1
        let data = self.topStackAddr;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_SP_EL1, data).map_err(|e| Error::SysError(e.errno()))?;
        // pc
        let data = self.entry;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_PC, data).map_err(|e| Error::SysError(e.errno()))?;
        // vbar_el1 holds exception vector base address, should be set in kernel
        // system time, is it necessary?
        let data = self.heapStartAddr;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_R0, data).map_err(|e| Error::SysError(e.errno()))?;
        let data = self.shareSpaceAddr;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_R1, data).map_err(|e| Error::SysError(e.errno()))?;
        let data = self.id as u64;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_R2, data).map_err(|e| Error::SysError(e.errno()))?;
        let data = VMS.lock().vdsoAddr;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_R3, data).map_err(|e| Error::SysError(e.errno()))?;
        let data = self.vcpuCnt as u64;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_R4, data).map_err(|e| Error::SysError(e.errno()))?;
        let data = self.autoStart as u64;
        self.vcpu.set_one_reg(_KVM_ARM64_REGS_R5, data).map_err(|e| Error::SysError(e.errno()))?;
        
        Ok(())
    }

    fn ioctl_set_kvm_vcpu_init(&self) -> Result<()> {
        let ret = unsafe { libc::ioctl(self.vcpu.as_raw_fd(), _KVM_ARM_VCPU_INIT, &KVM_VCPU_INIT as *const _ as u64) };
        if ret != 0 {
            return Err(Error::SysError(ret));
        }
        Ok(())

    }

    pub fn InterruptGuest(&self) {
        let mut vcpu_events = kvm_vcpu_events::default();
        vcpu_events.exception.serror_pending = 1;
        if let Err(e) = self.vcpu.set_vcpu_events(&vcpu_events) {
            panic!("Interrupt Guest Error {}", e);
        }
    }

    fn handle_hypercall(&self, addr: u16, data: &[u8], para1: u64, para2: u64,para3: u64, para4: u64) -> Result<()> {
        match addr {
            qlib::HYPERCALL_IOWAIT => {
                if !super::runc::runtime::vm::IsRunning() {
                    return Ok(());
                }

                match KERNEL_IO_THREAD.Wait(&SHARE_SPACE) {
                    Ok(()) => (),
                    Err(Error::Exit) => {
                        if !super::runc::runtime::vm::IsRunning() {
                            return Ok(());
                        }

                        return Ok(());
                    }
                    Err(e) => {
                        panic!("KERNEL_IO_THREAD get error {:?}", e);
                    }
                };
            }
            qlib::HYPERCALL_URING_WAKE => {
                let minComplete = para1 as usize;

                URING_MGR
                    .lock()
                    .Wake(minComplete)
                    .expect("qlib::HYPER CALL_URING_WAKE fail");
            }
            qlib::HYPERCALL_RELEASE_VCPU => {
                SyncMgr::WakeShareSpaceReady();
            }
            qlib::HYPERCALL_EXIT_VM => {
                let exitCode = para1 as i32;

                super::print::LOG.Clear();
                PerfPrint();

                SetExitStatus(exitCode);

                //wake up Kernel io thread
                KERNEL_IO_THREAD.Wakeup(&SHARE_SPACE);

                //wake up workthread
                VirtualMachine::WakeAll(&SHARE_SPACE);
            }

            qlib::HYPERCALL_PANIC => {
                let addr = para1;
                let msg = unsafe { &*(addr as *const Print) };

                eprintln!("Application error: {}", msg.str);
                ::std::process::exit(1);
            }

            qlib::HYPERCALL_WAKEUP_VCPU => {
                let vcpuId = para1 as usize;

                //error!("HYPERCALL_WAKEUP_VCPU vcpu id is {:x}", vcpuId);
                SyncMgr::WakeVcpu(vcpuId);
            }

            qlib::HYPERCALL_PRINT => {
                let addr = para1;
                let msg = unsafe { &*(addr as *const Print) };

                log!("{}", msg.str);
            }

            qlib::HYPERCALL_MSG => {
                let data1 = para1;
                let data2 = para2;
                let data3 = para3;
                let data4 = para4;
                raw!(data1, data2, data3, data4);
                /*info!(
                    "[{}] get kernel msg [rsp {:x}/rip {:x}]: {:x}, {:x}, {:x}",
                    self.id, regs.rsp, regs.rip, data1, data2, data3
                );*/
            }

            qlib::HYPERCALL_OOM => {
                let data1 = para1;
                let data2 = para2;
                error!(
                    "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
                    self.id, data1, data2
                );
                eprintln!(
                    "OOM!!! cpu [{}], size is {:x}, alignment is {:x}",
                    self.id, data1, data2
                );
                ::std::process::exit(1);
            }

            qlib::HYPERCALL_EXIT => {
                info!("call in HYPERCALL_EXIT");
                unsafe { libc::_exit(0) }
            }

            qlib::HYPERCALL_U64 => unsafe {
                
            },

            qlib::HYPERCALL_GETTIME => {
                let data = para1;

                unsafe {
                    let call = &mut *(data as *mut GetTimeCall);

                    let clockId = call.clockId;
                    let ts = Timespec::default();

                    let res = clock_gettime(
                        clockId as clockid_t,
                        &ts as *const _ as u64 as *mut timespec,
                    ) as i64;

                    if res == -1 {
                        call.res = errno::errno().0 as i64;
                    } else {
                        call.res = ts.ToNs()?;
                    }
                }
            }

            qlib::HYPERCALL_VCPU_FREQ => {
                let data = para1;

                // TODO get cpu freq
                let freq = 1000 * 1000 * 1000;
                unsafe {
                    let call = &mut *(data as *mut VcpuFeq);
                    call.res = freq as i64;
                }
            }

            qlib::HYPERCALL_VCPU_YIELD => {
                let _ret = IOURING.IOUring().HostSubmit().unwrap();
            }

            qlib::HYPERCALL_VCPU_DEBUG => {
                error!("HYPERCALL_VCPU_DEBUG");
            }

            qlib::HYPERCALL_VCPU_PRINT => {
                error!("[{}] HYPERCALL_VCPU_PRINT", self.id);
            }

            qlib::HYPERCALL_QCALL => {
                Self::GuestMsgProcess(&SHARE_SPACE);
                // last processor in host
                if SHARE_SPACE.DecrHostProcessor() == 0 {
                    Self::GuestMsgProcess(&SHARE_SPACE);
                }
            }

            qlib::HYPERCALL_HCALL => {
                let addr = para1;

                let eventAddr = addr as *mut QMsg; // as &mut qlib::Event;
                let qmsg = unsafe { &mut (*eventAddr) };

                {
                    let _l = if qmsg.globalLock {
                        Some(super::GLOCK.lock())
                    } else {
                        None
                    };

                    qmsg.ret = Self::qCall(qmsg.msg);
                }

                SHARE_SPACE.IncrHostProcessor();

                Self::GuestMsgProcess(&SHARE_SPACE);
                // last processor in host
                if SHARE_SPACE.DecrHostProcessor() == 0 {
                    Self::GuestMsgProcess(&SHARE_SPACE);
                }
            }

            qlib::HYPERCALL_VCPU_WAIT => {
                let retAddr = para3;

                let ret = SHARE_SPACE.scheduler.WaitVcpu(&SHARE_SPACE, self.id, true);
                match ret {
                    Ok(taskId) => unsafe {
                        *(retAddr as *mut u64) = taskId as u64;
                    },
                    Err(Error::Exit) => return Ok(()),
                    Err(e) => {
                        panic!("HYPERCALL_HLT wait fail with error {:?}", e);
                    }
                }
            }

            _ => info!("Unknow hyper call!!!!! address is {}", addr),
        }
        Ok(())
    }
}
