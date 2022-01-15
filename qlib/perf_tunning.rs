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

use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use alloc::vec::Vec;
use super::mutex::*;
use core::mem;

use super::kernel::TSC;
pub use super::super::kernel_def::*;
use super::singleton::*;

pub static COUNTS : Singleton<CounterSet> = Singleton::<CounterSet>::New();

pub unsafe fn InitSingleton() {
    COUNTS.Init(CounterSet::default());
}

pub fn PerfGoto(typ: PerfType) {
    COUNTS.Goto(typ)
}

pub fn PerfGofrom(typ: PerfType) {
    COUNTS.Gofrom(typ)
}

pub fn PerfStop() {
    COUNTS.Stop();
}

pub fn PerfPrint() {
    COUNTS.Print(true);
}

#[derive(Debug)]
pub struct Counter {
    pub count: AtomicU64,
    pub lastVal: AtomicU64,
}

impl Default for Counter {
    fn default() -> Self {
        return Self {
            count: AtomicU64::new(0),
            lastVal: AtomicU64::new(0),
        }
    }
}

impl Counter {
    pub fn Enter(&self) {
        self.lastVal.store(TSC.Rdtsc() as u64, Ordering::SeqCst);
    }

    pub fn Leave(&self) {
        let currentVal = TSC.Rdtsc() as u64;
        let lastVal = self.lastVal.load(Ordering::SeqCst);
        if lastVal != 0 {
            self.count.fetch_add(currentVal - self.lastVal.load(Ordering::SeqCst), Ordering::SeqCst);
            self.lastVal.store(0, Ordering::SeqCst);
        }
    }

    pub fn Val(&self) -> u64 {
        let ret = self.count.load(Ordering::SeqCst);

        let last = self.lastVal.load(Ordering::SeqCst);
        if last == 0 {
            return ret
        } else {
            ret + (TSC.Rdtsc() as u64 - last)
        }
    }
}


#[derive(Debug)]
pub struct Counters {
    pub data: [Counter; 32],
    pub state: QMutex<Vec<PerfType>>,
}

impl Default for Counters {
    fn default() -> Self {
        let ret = Self {
            data: Default::default(),
            state: QMutex::new(Vec::with_capacity(8)),
        };

        ret.state.lock().push(PerfType::Start);
        return ret;
    }
}

impl Counters {
    pub fn Stop(&self) {
        loop {
            let top = self.state.lock().pop();
            match top {
                None => return,
                Some(t) => {
                    self.Leave(t);
                }
            }
        }
    }

    pub fn Goto(&self, _id: usize, typ: PerfType) {
        let top;
        {
            let mut state = self.state.lock();
            if state.len() > 0 {
                top = Some(state[state.len()-1])
            } else {
                top = None;
            }
            state.push(typ);
        };


        if let Some(t) = top {
            self.Leave(t)
        }
        self.Enter(typ);
    }

    pub fn Gofrom(&self, id: usize, typ: PerfType) {
        let current;
        let top;

        {
            let mut state = self.state.lock();
            current = state.pop().expect("Counters::Gofrom pop fail");
            top = state[state.len()-1]
        };

        if current != typ {
            // work around for clone, the state can't chagne to PerfType::User with current implementation
            // todo: fix this
            if typ == PerfType::User { //|| typ == PerfType::Idle {
                self.state.lock().push(current);
            } else {
                panic!("Counters[{}]::Gofrom fail current stat is {:?}, expect {:?}", id, current, typ)
            }
        }

        self.Leave(typ);
        self.Enter(top);
    }

    pub fn Enter(&self, _typ: PerfType) {
        //self.data[typ as usize].Enter()
    }

    pub fn Leave(&self, _typ: PerfType) {
        //self.data[typ as usize].Leave()
    }
}

#[derive(Default)]
pub struct CounterSet {
    pub data: [Counters; Self::PERM_COUNTER_SET_SIZE]
}

impl CounterSet {
    pub fn Goto(&self, typ: PerfType) {
        let id = self.GetPerfId();
        self.data[id].Goto(id, typ);
    }

    pub fn Gofrom(&self, typ: PerfType) {
        let id = self.GetPerfId();
        self.data[id].Gofrom(id, typ);
    }

    pub fn Stop(&self) {
        for id in 0..Self::PERM_COUNTER_SET_SIZE {
            self.data[id].Stop();
        }
    }

    pub fn Print(&self, onlySum: bool) {
        let mut sum = vec![0; PerfType::End as usize];
        for idx in 0..Self::PERM_COUNTER_SET_SIZE {
            let mut total = 0;
            let counts = &self.data[idx];
            for i in 1..PerfType::End as usize {
                let val = counts.data[i].Val();
                sum[i] += val;
                total += val;
            }

            sum[0] += total;

            if total < 1000 {
                continue;
            }

            if !onlySum {
                let mut line = format!("{}#{} \t", self.PerfType(), idx);
                line += &format!("total->{} \t", total / 100_000);
                for i in 1..PerfType::End as usize {
                    let t: PerfType = unsafe { mem::transmute(i) };
                    let val = counts.data[i].Val();
                    line += &format!("{:?}->{}/{} \t", t, val / 100_000, val / (total/1000));
                }

                error!("{}", line);
            }
        }

        if sum[0] < 1000 {
            error!("PerfPrint::Kernel not ready ....");
            return
        }

        let mut line = format!("{} \t", self.PerfType());
        line += &format!("total->{} \t", sum[0] / 100_000);
        for i in 1..PerfType::End as usize {
            let t: PerfType = unsafe { mem::transmute(i) };
            let val = sum[i];
            line += &format!("{:?}->{}/{} \t", t, val / 100_000, val / (sum[0]/1000));
        }
        error!("{}", line);
    }
}