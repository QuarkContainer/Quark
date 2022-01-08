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

use core::mem;

use super::super::asm::*;
use super::super::super::common::*;
use super::super::Kernel::HostSpace;
use super::timer::*;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct VdsoParams {
    pub seq_count: u64,

    pub monotonicReady: u64,
    pub monotonicBaseCycles: i64,
    pub monotonicBaseRef: i64,
    pub monotonicFrequency: u64,

    pub realtimeReady: u64,
    pub realtimeBaseCycles: i64,
    pub realtimeBaseRef: i64,
    pub realtimeFrequency: u64,
}

impl VdsoParams {
    pub fn ClockRealTime(&self) -> Result<i64> {
        let mut ready;
        let mut baseRef;
        let mut baseCycle;
        let mut frequency;
        let mut now;

        loop {
            let seq = self.ReadSeqBegin();

            ready = self.realtimeReady;
            baseRef = self.realtimeBaseRef;
            baseCycle = self.realtimeBaseCycles;
            frequency = self.realtimeFrequency;
            now = Rdtsc();

            if self.ReadSeqReady(seq) {
                break;
            };
        };

        if ready == 0 {
            return HostSpace::KernelGetTime(REALTIME);
        }

        let delta = if now < baseCycle {
            0
        } else {
            now - baseCycle
        };

        let nowNs = baseRef + CyclesToNs(frequency, delta);

        return Ok(nowNs)
    }

    pub fn ClockMonotonicTime(&self) -> Result<i64> {
        let mut ready;
        let mut baseRef;
        let mut baseCycle;
        let mut frequency;
        let mut now;

        loop {
            let seq = self.ReadSeqBegin();

            ready = self.monotonicReady;
            baseRef = self.monotonicBaseRef;
            baseCycle = self.monotonicBaseCycles;
            frequency = self.monotonicFrequency;
            now = Rdtsc();

            if self.ReadSeqReady(seq) {
                break;
            };
        };

        if ready == 0 {
            return HostSpace::KernelGetTime(MONOTONIC);
        }

        let delta = if now < baseCycle {
            0
        } else {
            now - baseCycle
        };

        let nowNs = baseRef + CyclesToNs(frequency, delta);

        return Ok(nowNs)
    }

    fn ReadSeqBegin(&self) -> u64 {
        let seq = self.seq_count;
        ReadBarrier();
        return seq & !1;
    }

    //read success: return true, need retry: return false
    fn ReadSeqReady(&self, seq: u64) -> bool {
        ReadBarrier();
        return self.seq_count == seq;
    }
}

pub struct VDSOParamPage {
    //pub vdsoParams: &'static mut VdsoParams,
    pub vdsoParams: &'static mut VdsoParams,

    // seq is the current sequence count written to the page.
    //
    // A write is in progress if bit 1 of the counter is set.
    pub seq: u64,
    pub paramPageAddr: u64,
}

impl Default for VDSOParamPage {
    fn default() -> VDSOParamPage {
        return unsafe {
            VDSOParamPage {
                vdsoParams: &mut *(0 as * mut VdsoParams),
                //vdsoParams: VdsoParams::default(),
                seq: 0,
                paramPageAddr: 0,
            }
        }
    }
}

impl VDSOParamPage {
    pub fn SetParamPageAddr(&mut self, paramPageAddr: u64) {
        unsafe {
            self.vdsoParams = &mut *(paramPageAddr as * mut VdsoParams);
        }
        self.paramPageAddr = paramPageAddr;
    }

    pub fn GetParamPageAddr(&self) -> u64 {
        return self.paramPageAddr
    }

    fn IncrementSeq(&mut self) -> Result<()> {
        let next = self.seq + 1;

        let mut old: u64 = next;
        mem::swap(&mut old, &mut self.vdsoParams.seq_count);

        if old != self.seq {
            return Err(Error::Common(format!("unexpected VDSOParamPage seq value: got {} expected {}. Application may hang or get incorrect time from the VDSO.",
                                             old, self.seq)));
        }

        self.seq = next;
        return Ok(())
    }

    pub fn Write(&mut self, para: &VdsoParams) -> Result<()> {
        let next = self.seq + 1;
        if next % 2 != 1 {
            //let str = format!("Out-of-order sequence count: {}", self.seq);
            panic!("Out-of-order sequence count");
        }

        self.IncrementSeq()?;

        self.vdsoParams.monotonicReady = para.monotonicReady;
        self.vdsoParams.monotonicBaseCycles = para.monotonicBaseCycles;
        self.vdsoParams.monotonicBaseRef = para.monotonicBaseRef;
        self.vdsoParams.monotonicFrequency = para.monotonicFrequency;

        self.vdsoParams.realtimeReady = para.realtimeReady;
        self.vdsoParams.realtimeBaseCycles = para.realtimeBaseCycles;
        self.vdsoParams.realtimeBaseRef = para.realtimeBaseRef;
        self.vdsoParams.realtimeFrequency = para.realtimeFrequency;

        return self.IncrementSeq();
    }
}

const NS_PER_SEC: i128 = 1000_000_000;

fn CyclesToNs(freq: u64, cycles: i64) -> i64 {
    let mult = NS_PER_SEC << 32 / freq as i128;
    return ((cycles as i128 * mult) >> 32) as i64;
}