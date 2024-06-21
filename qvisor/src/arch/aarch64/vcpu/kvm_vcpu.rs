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

#![allow(non_upper_case_globals)]
use kvm_bindings::kvm_vcpu_events;
use crate::{KVMVcpu, qlib::common::Error, vmspace::VMSpace, qlib::backtracer};

const _TCR_IPS_40BITS: u64 = 2 << 32; // PA=40
const _TCR_IPS_48BITS: u64 = 5 << 32; // PA=48
const _TCR_T0SZ_OFFSET: u64 = 0;
const _TCR_T1SZ_OFFSET: u64 = 16;
const _TCR_IRGN0_SHIFT: u64 = 8;
const _TCR_IRGN1_SHIFT: u64 = 24;
const _TCR_ORGN0_SHIFT: u64 = 10;
const _TCR_ORGN1_SHIFT: u64 = 26;
const _TCR_SH0_SHIFT: u64 = 12;
const _TCR_SH1_SHIFT: u64 = 28;
const _TCR_TG0_SHIFT: u64 = 14;
const _TCR_TG1_SHIFT: u64 = 30;
const _TCR_T0SZ_VA48: u64 = 64 - 48; // VA=48
const _TCR_T1SZ_VA48: u64 = 64 - 48; // VA=48
const _TCR_A1: u64 = 1 << 22;
const _TCR_ASID16: u64 = 1 << 36;
const _TCR_TBI0: u64 = 1 << 37;
const _TCR_TXSZ_VA48: u64 = (_TCR_T0SZ_VA48 << _TCR_T0SZ_OFFSET)
                            | (_TCR_T1SZ_VA48 <<  _TCR_T1SZ_OFFSET);
const _TCR_TG0_4K: u64 = 0 << _TCR_TG0_SHIFT; // 4K
const _TCR_TG0_64K: u64 = 1 << _TCR_TG0_SHIFT; // 64K
const _TCR_TG1_4K: u64 = 2 << _TCR_TG1_SHIFT;
const _TCR_TG_FLAGS: u64 = _TCR_TG0_4K |  _TCR_TG1_4K;
const _TCR_IRGN0_WBWA: u64 = 1 << _TCR_IRGN0_SHIFT;
const _TCR_IRGN1_WBWA: u64 = 1 << _TCR_IRGN1_SHIFT;
const _TCR_IRGN_WBWA: u64 = _TCR_IRGN0_WBWA |  _TCR_IRGN1_WBWA;
const _TCR_ORGN0_WBWA: u64 = 1 << _TCR_ORGN0_SHIFT;
const _TCR_ORGN1_WBWA: u64 = 1 << _TCR_ORGN1_SHIFT;
const _TCR_ORGN_WBWA: u64 = _TCR_ORGN0_WBWA |  _TCR_ORGN1_WBWA;
const _TCR_SHARED: u64 = (3 << _TCR_SH0_SHIFT) | (3 << _TCR_SH1_SHIFT);
const _TCR_CACHE_FLAGS: u64 = _TCR_IRGN_WBWA |  _TCR_ORGN_WBWA;
pub const TCR_EL1_DEFAULT: u64 = _TCR_TXSZ_VA48 | _TCR_CACHE_FLAGS | _TCR_SHARED
                                | _TCR_TG_FLAGS | _TCR_ASID16 | _TCR_IPS_40BITS;

const _MT_DEVICE_nGnRnE: u64 = 0;
const _MT_DEVICE_nGnRE: u64 = 1;
const _MT_DEVICE_GRE: u64 = 2;
const _MT_NORMAL_NC: u64 = 3;
const _MT_NORMAL: u64 = 4;
const _MT_NORMAL_WT: u64 = 5;
const _MT_ATTR_DEVICE_nGnRnE: u64 = 0x00;
const _MT_ATTR_DEVICE_nGnRE: u64 = 0x04;
const _MT_ATTR_DEVICE_GRE: u64 = 0x0c;
const _MT_ATTR_NORMAL_NC: u64 = 0x44;
const _MT_ATTR_NORMAL_WT: u64 = 0xbb;
const _MT_ATTR_NORMAL: u64 = 0xff;
const _MT_ATTR_MASK: u64 = 0xff;
pub const MT_EL1_DEFAULT: u64 = (_MT_ATTR_DEVICE_nGnRnE << (_MT_DEVICE_nGnRnE * 8))
                        | (_MT_ATTR_DEVICE_nGnRE << (_MT_DEVICE_nGnRE * 8))
                        | (_MT_ATTR_DEVICE_GRE << (_MT_DEVICE_GRE * 8))
                        | (_MT_ATTR_NORMAL_NC << (_MT_NORMAL_NC * 8))
                        | (_MT_ATTR_NORMAL << (_MT_NORMAL * 8))
                        | (_MT_ATTR_NORMAL_WT << (_MT_NORMAL_WT * 8));

const _CNTKCTL_EL0PCTEN: u64 = 1 << 0;
const _CNTKCTL_EL0VCTEN: u64 = 1 << 1;
pub const CNTKCTL_EL1_DEFAULT: u64 = _CNTKCTL_EL0PCTEN | _CNTKCTL_EL0VCTEN;

const _SCTLR_M: u64 = 1 << 0;
const _SCTLR_C: u64 = 1 << 2;
const _SCTLR_I: u64 = 1 << 12;
const _SCTLR_DZE: u64 = 1 << 14;
const _SCTLR_UCT: u64 = 1 << 15;
const _SCTLR_SPAN: u64 = 1 << 23;
const _SCTLR_UCI: u64 = 1 << 26;
// NOTE: before PAN is fully supported, we disable this feature by setting SCTLR_EL1.SPAN
// to 1, preventing PSTATE.PAN from being set to 1 upon exception to EL1
pub const SCTLR_EL1_DEFAULT: u64 = _SCTLR_M | _SCTLR_C | _SCTLR_I | _SCTLR_UCT
                                  | _SCTLR_UCI | _SCTLR_DZE | _SCTLR_SPAN;
// NOTE: FPEN[21:20] - Do not cause instructions related
//                     to FP registers to be trapped.
const _CPACR_EL1_FPEN: u64 = 3 << 20;
pub const CPACR_EL1_DEFAULT: u64 = _CPACR_EL1_FPEN;

#[repr(u64)]
#[derive(Copy, Clone)]
pub enum KvmAarch64Reg {
    X0     = 0x6030000000100000,
    X1     = 0x6030000000100002,
    X2     = 0x6030000000100004,
    X3     = 0x6030000000100006,
    X4     = 0x6030000000100008,
    X5     = 0x603000000010000a,
    X6     = 0x603000000010000c,
    X7     = 0x603000000010000e,
    X8     = 0x6030000000100010,
    X18    = 0x6030000000100024,
    X29    = 0x603000000010003a,
    PC     = 0x6030000000100040,
    Pstate = 0x6030000000100042,
    SpEl1  = 0x6030000000100044,
    MairEl1     = 0x603000000013c510,
    Ttbr0El1    = 0x603000000013c100,
    Ttbr1El1    = 0x603000000013c101,
    TcrEl1      = 0x603000000013c102,
    SctlrEl1    = 0x603000000013c080,
    CpacrEl1    = 0x603000000013c082,
    VbarEl1     = 0x603000000013c600,
    TimerCnt    = 0x603000000013df1a,
    CntfrqEl0   = 0x603000000013df00,
    MdscrEl1    = 0x6030000000138012,
    CntkctlEl1  = 0x603000000013c708,
    TpidrEl1    = 0x603000000013c684,
}

pub enum Register{
    Reg(KvmAarch64Reg, u64),
}

impl KVMVcpu {
    pub fn set_regs(&self, reg_list: Vec<Register>) -> Result<(), Error> {
        for reg in reg_list.iter() {
            match reg {
                Register::Reg(reg_addr, reg_val) => {
                    self.vcpu_fd.set_one_reg(*reg_addr as u64, *reg_val)
                        .map_err(|e| Error::SysError(e.errno()))?;
                }
            }
        }
        Ok(())
    }

    pub fn dump(&self) -> Result<(), Error> {
        Ok(())
    }

    pub fn get_frequency(&self) -> Result<u64, Error> {
        Ok(VMSpace::get_cpu_frequency())
    }

    pub fn backtrace(&self) -> Result<(), Error> {
        use KvmAarch64Reg::{PC, SpEl1, X29};
        let pc = self.vcpu_fd.get_one_reg(PC as u64).map_err(|e| Error::SysError(e.errno()))?;
        let rsp = self.vcpu_fd.get_one_reg(SpEl1 as u64).map_err(|e| Error::SysError(e.errno()))?;
        let rbp = self.vcpu_fd.get_one_reg(X29 as u64).map_err(|e| Error::SysError(e.errno()))?;

        backtracer::trace(pc, rsp, rbp, &mut |frame| {
            print!("host frame is {:#x?}", frame);
            true
        });
        Ok(())
    }

    pub fn InterruptGuest(&self) {
        let mut vcpu_events = kvm_vcpu_events::default();
        vcpu_events.exception.serror_pending = 1;
        if let Err(e) = self.vcpu_fd.set_vcpu_events(&vcpu_events) {
            panic!("Interrupt Guest Error {}", e);
        }
    }
}
