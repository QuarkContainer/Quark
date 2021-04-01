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

use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::kernel::timer::timer::*;
use super::super::kernel::kernel::*;
use super::super::qlib::linux::time::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::thread_group::*;
use super::super::qlib::usage::io::*;
use super::super::qlib::linux::rusage::*;

impl Thread {
    pub fn Getitimer(&self, id: i32) -> Result<ItimerVal> {
        let tg = self.lock().tg.clone();
        let (tm, olds) = match id {
            ITIMER_REAL => {
                tg.lock().itimerRealTimer.Get()
            }
            ITIMER_VIRTUAL => {
                let tm = tg.UserCPUClock().Now();
                let lock = tg.lock().signalLock.clone();
                let _sl = lock.lock();
                let (s, _) = tg.lock().itimerVirtSetting.At(tm);
                (tm, s)
            }
            ITIMER_PROF => {
                let tm = tg.CPUClock().Now();
                let lock = tg.lock().signalLock.clone();
                let _sl = lock.lock();
                let (s, _) = tg.lock().itimerProfSetting.At(tm);
                (tm, s)
            }
            _ => {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        };

        let (oldval, oldiv) = SpecFromSetting(tm, olds);

        return Ok(ItimerVal {
            Value: Timeval::FromNs(oldval),
            Interval: Timeval::FromNs(oldiv)
        })
    }

    pub fn Setitimer(&self, id: i32, newitv: &ItimerVal) -> Result<ItimerVal> {
        let tg = self.ThreadGroup();
        let (tm, olds) = match id {
            ITIMER_REAL => {
                let timer = tg.lock().itimerRealTimer.clone();
                let news = Setting::FromSpec(newitv.Value.ToDuration(), newitv.Interval.ToDuration(), &Arc::new(timer.Clock()))?;
                let (tm, olds) = timer.Swap(&news);
                (tm, olds)
            }
            ITIMER_VIRTUAL => {
                let c = tg.UserCPUClock();
                let kernel = GetKernel();
                let ticker = kernel.cpuClockTicker.clone();
                let tm = c.Now();
                let mut olds = Setting::default();
                let mut err = None;
                ticker.Atomically(|| {
                    let news = match Setting::FromSpecAt(newitv.Value.ToDuration(), newitv.Interval.ToDuration(), tm) {
                        Ok(n) => n,
                        Err(e) => {
                            err = Some(Err(e));
                            return
                        },
                    };
                    let lock = tg.lock().signalLock.clone();
                    let _s = lock.lock();
                    olds = tg.lock().itimerVirtSetting;
                    tg.lock().itimerVirtSetting = news;
                    tg.lock().updateCPUTimersEnabledLocked();

                });

                match err {
                    None => (),
                    Some(e) => return e,
                }

                (tm, olds)
            }
            ITIMER_PROF => {
                let c = tg.CPUClock();
                let kernel = GetKernel();
                let ticker = kernel.cpuClockTicker.clone();
                let tm = c.Now();
                let mut olds = Setting::default();
                let mut err = None;
                ticker.Atomically(|| {
                    let news = match Setting::FromSpecAt(newitv.Value.ToDuration(), newitv.Interval.ToDuration(), tm) {
                        Ok(n) => n,
                        Err(e) => {
                            err = Some(Err(e));
                            return
                        },
                    };
                    let lock = tg.lock().signalLock.clone();
                    let _s = lock.lock();
                    olds = tg.lock().itimerProfSetting;
                    tg.lock().itimerProfSetting = news;
                    tg.lock().updateCPUTimersEnabledLocked();

                    //error!("itimerProfSetting(uid is {})  is {:?}", tg.uid, tg.lock().itimerProfSetting);
                });

                match err {
                    None => (),
                    Some(e) => return e,
                }

                (tm, olds)
            }
            _ => {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        };

        let (oldval, oldiv) = SpecFromSetting(tm, olds);

        return Ok(ItimerVal {
            Value: Timeval::FromNs(oldval),
            Interval: Timeval::FromNs(oldiv)
        })
    }

    pub fn Name(&self) -> String {
        return self.lock().name.to_string();
    }

    pub fn SetName(&self, name: &str) {
        self.lock().name = name.to_string();
    }

    // MaxRSS returns the maximum resident set size of the task in bytes. which
    // should be one of RUSAGE_SELF, RUSAGE_CHILDREN, RUSAGE_THREAD, or
    // RUSAGE_BOTH. See getrusage(2) for documentation on the behavior of these
    // flags.
    pub fn MaxRSS(&self, which: i32) -> u64 {
        let tg = self.ThreadGroup();
        let pidns = tg.PIDNamespace();
        let owner = pidns.Owner();

        let _= owner.ReadLock();

        match which {
            RUSAGE_SELF | RUSAGE_THREAD => {
                let mm = self.MemoryManager();
                let mmMaxRSS = mm.MaxResidentSetSize();
                if mmMaxRSS > tg.lock().maxRSS {
                    return mmMaxRSS
                }
                return tg.lock().maxRSS
            }
            RUSAGE_CHILDREN => {
                tg.lock().childMaxRSS
            }
            RUSAGE_BOTH => {
                let mut maxRSS = tg.lock().maxRSS;
                if maxRSS < tg.lock().childMaxRSS {
                    maxRSS = tg.lock().childMaxRSS;
                }

                let mm = self.MemoryManager();
                let mmMaxRSS = mm.MaxResidentSetSize();
                if mmMaxRSS > maxRSS {
                    return mmMaxRSS
                }

                return maxRSS
            }
            _ => return 0
        }
    }
}

pub trait IOUsage {
    fn IOUsage(&self) -> IO;
}

impl IOUsage for Thread {
    // IOUsage returns the io usage of the thread.
    fn IOUsage(&self) -> IO {
        return self.lock().ioUsage.clone();
    }
}

impl IOUsage for ThreadGroup {
    // IOUsage returns the total io usage of all dead and live threads in the group.
    fn IOUsage(&self) -> IO {
        let owner = self.TaskSet();
        let _r = owner.ReadLock();

        let io = self.lock().ioUsage.clone();
        for t in &self.lock().tasks {
            io.Accumulate(&t.IOUsage())
        }

        return io;
    }
}