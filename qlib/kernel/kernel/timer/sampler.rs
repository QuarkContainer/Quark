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

use alloc::collections::vec_deque::VecDeque;

use super::super::super::asm::*;
use super::super::super::super::common::*;
use super::super::super::Kernel::HostSpace;

use super::*;

// defaultOverheadTSC is the default estimated syscall overhead in TSC cycles.
// It is further refined as syscalls are made.
pub const DEFAULT_OVERHEAD_CYCLES: TSCValue = 1 * 1000;

// maxOverheadCycles is the maximum allowed syscall overhead in TSC cycles.
pub const MAX_OVERHEAD_CYCLES: TSCValue = 100 * DEFAULT_OVERHEAD_CYCLES;

// maxSampleLoops is the maximum number of times to try to get a clock sample
// under the expected overhead.
pub const MAX_SAMPLE_LOOPS: usize = 5;

// maxSamples is the maximum number of samples to collect.
pub const MAX_SAMPLES: usize = 11;

// TSCValue is a value from the TSC.
pub type TSCValue = i64;


// ReferenceNS are nanoseconds in the reference clock domain.
// int64 gives us ~290 years before this overflows.
pub type ReferenceNS = i64;

pub fn Magnitude(r: ReferenceNS) -> ReferenceNS {
    if r < 0 {
        return -r
    }
    return r
}

pub fn Sample(c: ClockID) -> Result<Sample> {
    let before = Rdtsc();

    let time = HostSpace::KernelGetTime(c)?;

    if time < 0 {
        return Err(Error::SysError(-time as i32));
    }

    let after = Rdtsc();
    let res = Sample {
        Before: before,
        After: after,
        Ref: time,
    };

    return Ok(res)
}

pub fn Cycles() -> TSCValue {
    return Rdtsc()
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Sample {
    pub Before: TSCValue,
    pub After: TSCValue,
    pub Ref: ReferenceNS,
}

impl Sample {
    pub fn Overhead(&self) -> TSCValue {
        return self.After - self.Before
    }
}

// sampler collects samples from a reference system clock, minimizing
// the overhead in each sample.
pub struct Sampler {
    // clockID is the reference clock ID (e.g., CLOCK_MONOTONIC).
    pub clockID: ClockID,

    // overhead is the estimated sample overhead in TSC cycles.
    pub overhead: TSCValue,

    // samples is a ring buffer of the latest samples collected.
    pub samples: VecDeque<Sample>,
}

impl Sampler {
    pub fn New(c: ClockID) -> Self {
        return Sampler {
            clockID: c,
            overhead: DEFAULT_OVERHEAD_CYCLES,
            samples: VecDeque::with_capacity(MAX_SAMPLES),
        }
    }

    pub fn Reset(&mut self) {
        self.overhead = DEFAULT_OVERHEAD_CYCLES;
        self.samples.clear();
    }

    fn LowOverheadSample(&mut self) -> Result<Sample> {
        loop {
            for _i in 0..MAX_SAMPLE_LOOPS {
                let sample = Sample(self.clockID)?;

                if sample.Before > sample.After {
                    info!("TSC went backwards: {:x} > {:x}", sample.Before, sample.After);
                    continue;
                }

                if sample.Overhead() < self.overhead {
                    return Ok(sample)
                }
            }

            let mut newOverhead = 2 * self.overhead;
            if newOverhead > MAX_OVERHEAD_CYCLES {
                if self.overhead == MAX_OVERHEAD_CYCLES {
                    return Err(Error::Common(format!("time syscall overhead exceeds maximum, overhead is {}, MAX_OVERHEAD_CYCLES is {}, newOverhead is {}",
                                                     self.overhead, MAX_OVERHEAD_CYCLES, newOverhead)));
                }

                newOverhead = MAX_OVERHEAD_CYCLES;
            }

            self.overhead = newOverhead;
            //info!("Time: Adjusting syscall overhead up to {}", self.overhead);
        }
    }

    pub fn Sample(&mut self) -> Result<()> {
        let sample = self.LowOverheadSample()?;

        if self.samples.len() == MAX_SAMPLES {
            self.samples.pop_front();
        }

        self.samples.push_back(sample);

        // If the 4 most recent samples all have an overhead less than half the
        // expected overhead, adjust downwards.
        if self.samples.len() < 4 {
            return Ok(())
        }

        for sample in self.samples.iter().skip(self.samples.len() - 4) {
            if sample.Overhead() > self.overhead / 2 {
                return Ok(())
            }
        }

        self.overhead -= self.overhead / 8;
        //info!("Time: Adjusting syscall overhead down to {}", self.overhead);

        return Ok(())
    }

    pub fn Syscall(&self) -> Result<ReferenceNS> {
        let sample = Sample(self.clockID)?;

        return Ok(sample.Ref);
    }

    pub fn Cycles(&self) -> TSCValue {
        return Rdtsc()
    }

    pub fn Range(&self) -> Option<(Sample, Sample)> {
        if self.samples.len() < 2 {
            return None
        }

        return Some((*self.samples.front().unwrap(), *self.samples.back().unwrap()))
    }
}