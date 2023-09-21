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

use core::sync::atomic::AtomicI64;
use core::sync::atomic::Ordering;

use super::super::super::linux::time::*;
use super::super::super::mutex::*;
use super::super::task::*;
use super::super::LoadVcpuFreq;
use super::super::TSC;
use super::kernel::*;

pub struct AsyncProcess {
    pub lastTsc: AtomicI64,
    pub lastProcessTime: QMutex<i64>,
}

const TSC_GAP: i64 = 2_000_000; // for 2 GHZ process, it is 1 ms
static CLOCK_TICK_MS: i64 = CLOCK_TICK / MILLISECOND;

pub fn CyclesPerTick() -> i64 {
    CLOCK_TICK_MS * LoadVcpuFreq() / 1000
}

impl AsyncProcess {
    pub const fn New() -> Self {
        return Self {
            lastTsc: AtomicI64::new(0),
            lastProcessTime: QMutex::new(0),
        };
    }

    pub fn Init(&self) {
        let curr = TSC.Rdtsc();
        *self.lastProcessTime.lock() = curr;
        self.lastTsc.store(curr, Ordering::SeqCst);
    }

    pub fn Process(&self) {
        let curr = TSC.Rdtsc();
        if curr - self.lastTsc.load(Ordering::Relaxed) > TSC_GAP {
            self.lastTsc.store(curr, Ordering::Relaxed);
            if let Some(mut processTime) = self.lastProcessTime.try_lock() {
                let currTime = Task::MonoTimeNow().0 / MILLISECOND;
                if currTime - *processTime >= CLOCK_TICK_MS {
                    let tick = (currTime - *processTime) / CLOCK_TICK_MS;
                    if let Some(kernel) = GetKernelOption() {
                        let ticker = kernel.cpuClockTicker.clone();
                        ticker.Notify(tick as u64);
                        *processTime = currTime;
                    }
                }
            }
        }
    }

    pub fn Atomically(&self, mut f: impl FnMut()) {
        let _t = self.lastProcessTime.lock();
        f();
    }
}
