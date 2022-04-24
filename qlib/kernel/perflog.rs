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

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use core::mem;

use super::super::perf_tunning::*;
use super::super::singleton::*;
use super::task::*;
use super::uid::*;

pub static THREAD_COUNTS: Singleton<QMutex<ThreadPerfCounters>> =
    Singleton::<QMutex<ThreadPerfCounters>>::New();

#[derive(Default)]
pub struct ThreadPerfCounters {
    data: BTreeMap<u64, Arc<Counters>>,
}

impl ThreadPerfCounters {
    pub fn NewCounters(&mut self) -> Arc<Counters> {
        let uid = NewUID();
        let counters = Arc::new(Counters::default());
        self.data.insert(uid, counters.clone());
        return counters;
    }

    pub fn PerfType(&self) -> &str {
        return "PerfPrint::Thread";
    }

    pub fn Print(&self, onlySum: bool) {
        let mut sum = vec![0; PerfType::End as usize];
        for (idx, counts) in &self.data {
            let mut total = 0;
            for i in 1..PerfType::End as usize {
                if i == PerfType::Blocked as usize {
                    continue;
                }
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
                    if t == PerfType::Blocked {
                        continue;
                    }
                    let val = counts.data[i].Val();
                    line += &format!("{:?}->{}/{} \t", t, val / 100_000, val / (total / 1000));
                }

                error!("{}", line);
            }
        }

        if sum[0] < 1000 {
            error!("PerfPrint::Threads not ready ....");
            return;
        }

        let mut line = format!("{} \t", self.PerfType());
        line += &format!("total->{} \t", sum[0] / 100_000);
        for i in 1..PerfType::End as usize {
            let t: PerfType = unsafe { mem::transmute(i) };
            if t == PerfType::Blocked {
                continue;
            }
            let val = sum[i];
            line += &format!("{:?}->{}/{} \t", t, val / 100_000, val / (sum[0] / 1000));
        }
        error!("{}", line);
    }
}

impl Task {
    pub fn PerfGoto(&self, typ: PerfType) {
        if let Some(counts) = &self.perfcounters {
            counts.Goto(0, typ)
        }
    }

    pub fn PerfGofrom(&self, typ: PerfType) {
        if let Some(counts) = &self.perfcounters {
            counts.Gofrom(0, typ)
        }
    }

    pub fn PerfStop(&self) {
        if let Some(counts) = &self.perfcounters {
            counts.Stop();
        }
    }
}
