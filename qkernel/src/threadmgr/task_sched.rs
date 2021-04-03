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

use core::sync::atomic::Ordering;
use spin::Mutex;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use super::super::asm::*;
use super::super::qlib::limits::*;
use super::super::threadmgr::thread_group::*;
use super::super::threadmgr::thread::*;
use super::super::qlib::usage::cpu::*;
use super::super::qlib::linux::time::*;
use super::super::SignalDef::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::vcpu_mgr::*;
use super::super::task::*;
use super::super::kernel::timer::timer::*;
use super::super::kernel::time::*;
use super::super::kernel::kernel::*;
use super::super::kernel::waiter::*;
use super::super::kernel::cpuset::*;
use super::task_exit::*;
use super::task_stop::*;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum SchedState {
    // Nonexistent indicates that the task has either
    // not yet been created by Task.Start() or has returned from Task.run().
    // This must be the zero value for State.
    Nonexistent,

    // RunningSys indicates that the task is executing
    RunningSys,

    // RunningApp indicates that the task is executing
    // application code.
    RunningApp,

    // includes both BlockedInterruptible/BlockedUninterruptible, todo: fix this later
    Blocked,

    // BlockedInterruptible indicates that the task is
    // blocked in Task.block(), and hence may be woken by Task.interrupt()
    // (e.g. due to signal delivery).
    BlockedInterruptible,

    // BlockedUninterruptible indicates that the task is
    // stopped outside of Task.block() and Task.doStop(), and hence cannot be
    // woken by Task.interrupt().
    BlockedUninterruptible,

    // Stopped indicates that the task is blocked in
    // Task.doStop(). Stopped is similar to
    // BlockedUninterruptible, but is a separate state to make it
    // possible to determine when Task.stop is meaningful.
    Stopped,
}

impl Default for SchedState {
    fn default() -> Self {
        return Self::Nonexistent
    }
}

// TaskGoroutineSchedInfo contains task scheduling state which must
// be read and updated atomically.
#[derive(Default, Debug, Copy, Clone)]
pub struct TaskSchedInfoInternal {
    // Timestamp was the value of cpu cycle when this
    // TaskSchedInfo was last updated.
    pub Timestamp: u64,

    // State is the current state of the task.
    pub State: SchedState,

    // UserTicks is the amount of time the task has spent executing
    // its associated Task's application code, in units of cpu cycle.
    pub UserTicks: u64,

    // SysTicks is the amount of time the task has spent executing
    pub SysTicks: u64,

    // yieldCount is the number of times the task  has called
    // Task.InterruptibleSleepStart, Task.UninterruptibleSleepStart, or
    // Task.Yield(), voluntarily ceasing execution.
    pub YieldCount: u64,
}

impl TaskSchedInfoInternal {
    // userTicksAt returns the extrapolated value of ts.UserTicks after
    // Kernel.CPUClockNow() indicates a time of now.
    //
    // Preconditions: now <= Kernel.CPUClockNow(). (Since Kernel.cpuClock is
    // monotonic, this is satisfied if now is the result of a previous call to
    // Kernel.CPUClockNow().) This requirement exists because otherwise a racing
    // change to t.gosched can cause userTicksAt to adjust stats by too much,
    // making the observed stats non-monotonic.
    pub fn userTicksAt(&self, now: u64) -> u64 {
        if self.Timestamp < now && self.State == SchedState::RunningApp {
            return self.UserTicks + now - self.Timestamp;
        }

        return self.UserTicks
    }

    // sysTicksAt returns the extrapolated value of ts.SysTicks after
    // Kernel.CPUClockNow() indicates a time of now.
    //
    // Preconditions: As for userTicksAt.
    pub fn sysTicksAt(&self, now: u64) -> u64 {
        if self.Timestamp < now && self.State == SchedState::RunningSys {
            return self.UserTicks + now - self.Timestamp;
        }

        return self.SysTicks;
    }
}

#[derive(Clone, Default)]
pub struct TaskSchedInfo(Arc<Mutex<TaskSchedInfoInternal>>);

impl Deref for TaskSchedInfo {
    type Target = Arc<Mutex<TaskSchedInfoInternal>>;

    fn deref(&self) -> &Arc<Mutex<TaskSchedInfoInternal>> {
        &self.0
    }
}

impl ThreadInternal {
    pub fn TaskSchedInfo(&self) -> TaskSchedInfoInternal {
        return *(self.sched.lock())
    }

    pub fn cpuStatsAt(&self, now: u64) -> CPUStats {
        let tsched = self.TaskSchedInfo();

        let userTime = tsched.userTicksAt(now) as i64;
        let sysTime = tsched.sysTicksAt(now) as i64;

        return CPUStats {
            UserTime: userTime * CLOCK_TICK,
            SysTime: sysTime * CLOCK_TICK,
            VoluntarySwitches: tsched.YieldCount,
        }
    }

    pub fn CPUStats(&self) -> CPUStats {
        let now = GetKernel().CPUClockNow();
        return self.cpuStatsAt(now)
    }

    // StateStatus returns a string representation of the task's current state,
    // appropriate for /proc/[pid]/status.
    pub fn StateStatus(&self) -> &str {
        let s = self.TaskSchedInfo().State;

        let tg = self.tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();

        match s {
            SchedState::Nonexistent => {
                let _r = owner.ReadLock();
                match self.exitState {
                    TaskExitState::TaskExitZombie => {
                        return "Z (zombie)"
                    }
                    TaskExitState::TaskExitDead => {
                        return "X (dead)"
                    }
                    _ => {
                        // The task can't exit before passing through
                        // runExitNotify, so this indicates that the task has been created,
                        // but the task goroutine hasn't yet started. The Linux equivalent
                        // is struct task_struct::state == TASK_NEW
                        // (kernel/fork.c:copy_process() =>
                        // kernel/sched/core.c:sched_fork()), but the TASK_NEW bit is
                        // masked out by TASK_REPORT for /proc/[pid]/status, leaving only
                        // TASK_RUNNING.
                        return "R (running)";
                    }
                }
            }
            SchedState::RunningSys | SchedState::RunningApp => {
                return "R (running)";
            }
            SchedState::BlockedInterruptible | SchedState::Blocked => {
                return "S (sleeping)";
            }
            SchedState::Stopped => {
                let lock = tg.lock().signalLock.clone();
                let _s = lock.lock();
                match self.stop.clone().unwrap().Type() {
                    TaskStopType::GROUPSTOP => {
                        return "T (stopped)";
                    }
                    TaskStopType::PTRACESTOP => {
                        return "t (tracing stop)";
                    }
                    _ => {
                        return "D (disk sleep)";
                    }
                }
            }
            SchedState::BlockedUninterruptible => {
                return "D (disk sleep)";
            }
        }
    }
}

impl Thread {
    pub fn NotifyRlimitCPUUpdated(&self) {
        //todo: fix this.
        info!("NotifyRlimitCPUUpdated: no more ticket, need fix");
        let ticker = self.lock().k.cpuClockTicker.clone();
        ticker.Atomically(|| {
            let tg = self.lock().tg.clone();
            let pidns = tg.PIDNamespace();
            let owner = pidns.lock().owner.clone();
            let _r = owner.read();
            let lock = tg.lock().signalLock.clone();
            let _s = lock.lock();

            let rlimitCPU = tg.lock().limits.Get(LimitType::CPU);
            tg.lock().rlimitCPUSoftSetting = Setting {
                Enabled: rlimitCPU.Cur != INFINITY,
                Next: Time::FromNs(rlimitCPU.Cur as i64 * SECOND),
                Period: SECOND,
            };

            if rlimitCPU.Max != INFINITY {
                // Check if tg is already over the hard limit.
                let now = self.lock().k.CPUClockNow();
                let tgcpu = tg.cpuStatsAtLocked(now);
                let tgProfNow = Time::FromNs(tgcpu.UserTime + tgcpu.SysTime);
                if !tgProfNow.Before(Time::FromNs(rlimitCPU.Max as i64)) {
                    self.sendSignalLocked(&SignalInfo::SignalInfoPriv(Signal(Signal::SIGKILL)), true).unwrap();
                }
            }

            tg.lock().updateCPUTimersEnabledLocked();
        });
    }

    pub fn CPUStats(&self) -> CPUStats {
        return self.lock().CPUStats();
    }

    // CPUMask returns a copy of t's allowed CPU mask.
    pub fn CPUMask(&self) -> CPUSet {
        let t = self.lock();
        return t.allowedCPUMask.Copy();
    }

    // SetCPUMask sets t's allowed CPU mask based on mask. It takes ownership of
    // mask.
    //
    // Preconditions: mask.Size() ==
    // sched.CPUSetSize(t.Kernel().ApplicationCores()).
    pub fn SetCPUMask(&self, mask: CPUSet) -> Result<()> {
        let mut mask = mask.Copy();
        let applicationCores = self.lock().k.staticInfo.lock().ApplicationCores as usize;
        let want = CPUSetSize(applicationCores);
        if mask.Size() != want {
            panic!("Invalid CPUSet {:?} (expected {} bytes)", &mask, want);
        }

        // Remove CPUs in mask above Kernel.applicationCores.
        mask.ClearAbove(applicationCores);

        // Ensure that at least 1 CPU is still allowed.
        if mask.NumCPUs() == 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if self.lock().k.staticInfo.lock().useHostCores {
            // No-op; pretend the mask was immediately changed back.
            return Ok(())
        }

        let tg = self.lock().tg.clone();
        let pidns = tg.PIDNamespace();
        let owner = pidns.lock().owner.clone();

        let rootTID = {
            let ts = owner.read();
            let pidns = ts.root.clone().unwrap();
            let tid = *pidns.lock().tids.get(self).unwrap();
            tid
        };

        let mut t = self.lock();
        t.allowedCPUMask = mask.Copy();
        t.cpu = assignCPU(&mask, rootTID);
        return Ok(())
    }

    // Niceness returns t's niceness.
    pub fn Niceness(&self) -> i32 {
        return self.lock().niceness
    }

    // Priority returns t's priority.
    pub fn Priority(&self) -> i32 {
        return self.lock().niceness + 20
    }

    // SetNiceness sets t's niceness to n.
    pub fn SetNiceness(&self, n: i32) {
        self.lock().niceness = n;
    }

    // NumaPolicy returns t's current numa policy.
    pub fn NumaPolicy(&self) -> (i32, u32) {
        let t = self.lock();
        return (t.numaPolicy, t.numaNodeMask)
    }

    // SetNumaPolicy sets t's numa policy.
    pub fn SetNumaPolicy(&self, policy: i32, nodeMask: u32) {
        let mut t = self.lock();
        t.numaPolicy = policy;
        t.numaNodeMask = nodeMask;
    }
}

// assignCPU returns the virtualized CPU number for the task with global TID
// tid and allowedCPUMask allowed.
fn assignCPU(allowed: &CPUSet, tid: ThreadID) -> i32 {
    // To pretend that threads are evenly distributed to allowed CPUs, choose n
    // to be less than the number of CPUs in allowed ...
    let mut n = tid % allowed.NumCPUs() as i32;
    // ... then pick the nth CPU in allowed.
    let mut cpu = 0;
    allowed.ForEachCPU(|c| {
        n -= 1;
        if n == 0 {
            cpu = c as i32;
        }
    });

    return cpu;
}

impl Task {
    pub fn CPU(&self) -> i32 {
        /*let k = self.Thread().lock().k.clone();
        let useHostCores = k.staticInfo.lock().useHostCores; //always false, "rdtscp" is not available in guest kernel
        if useHostCores {
            return GetCpu() as i32;
        }

        return k.staticInfo.lock().cpu;*/
        let cpuid = CPULocal::CpuId();
        return cpuid as i32;
    }
}

impl ThreadGroupInternal {
    // Preconditions: The signal mutex must be locked.
    pub fn updateCPUTimersEnabledLocked(&mut self) {
        let rlimitCPU = self.limits.Get(LimitType::CPU);
        if self.itimerVirtSetting.Enabled ||
            self.itimerProfSetting.Enabled ||
            self.rlimitCPUSoftSetting.Enabled ||
            rlimitCPU.Max != INFINITY {
            self.cpuTimersEnabled = 1;
        } else {
            self.cpuTimersEnabled = 0;
        }
    }
}

impl ThreadGroup {
    // Preconditions: As for TaskGoroutineSchedInfo.userTicksAt. The TaskSet mutex
    // must be locked.
    pub fn cpuStatsAtLocked(&self, now: u64) -> CPUStats {
        let mut stats = self.lock().exitedCPUStats;
        // Account for live tasks.
        let threads : Vec<Thread> = self.lock().tasks.iter().cloned().collect();
        for t in threads {
            stats.Accumulate(&t.lock().cpuStatsAt(now))
        }

        return stats;
    }

    /// CPUStats returns the combined CPU usage statistics of all past and present
    // threads in tg.
    pub fn CPUStats(&self) -> CPUStats {
        let pidns = self.lock().pidns.clone();
        let owner = pidns.lock().owner.clone();

        let _r = owner.read();

        let lead = self.lock().leader.Upgrade();
        match &lead {
            None => {
                return CPUStats::default()
            },
            Some(ref _leader) => {
                //let now = leader.lock().k.CPUClockNow();
                let now = GetKernel().CPUClockNow();
                let ret = self.cpuStatsAtLocked(now);
                return ret;
            }
        }
    }

    // JoinedChildCPUStats implements the semantics of RUSAGE_CHILDREN: "Return
    // resource usage statistics for all children of [tg] that have terminated and
    // been waited for. These statistics will include the resources used by
    // grandchildren, and further removed descendants, if all of the intervening
    // descendants waited on their terminated children."
    pub fn JoinedChildCPUStats(&self) -> CPUStats {
        let pidns = self.PIDNamespace();
        let owner = pidns.lock().owner.clone();
        let _r = owner.read();
        return self.lock().childCPUStats;
    }
}

// taskClock is a ktime.Clock that measures the time that a task has spent
// executing. taskClock is primarily used to implement CLOCK_THREAD_CPUTIME_ID.
#[derive(Clone)]
pub struct TaskClock {
    pub t: Thread,
    pub includeSys: bool,
}

impl Waitable for TaskClock {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        return 0
    }
}

impl TaskClock {
    pub fn Now(&self) -> Time {
        let stats = self.t.CPUStats();
        if self.includeSys {
            return Time::FromNs(stats.UserTime + stats.SysTime)
        }

        return Time::FromNs(stats.UserTime)
    }

    pub fn WallTimeUntil(&self, t: Time, now: Time) -> Duration {
        return t.Sub(now)
    }
}

#[derive(Clone)]
pub struct ThreadGroupClock {
    pub tg: ThreadGroup,
    pub includeSys: bool,
    pub queue: Queue,
}

impl Waitable for ThreadGroupClock {
    fn Readiness(&self, _task: &Task, _mask: EventMask) -> EventMask {
        return 0
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        return self.queue.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        return self.queue.EventUnregister(task, e)
    }
}

impl ThreadGroupClock {
    pub fn Now(&self) -> Time {
        let stats = self.tg.CPUStats();
        if self.includeSys {
            //error!("ThreadGroupClock usertime is {:x}, SysTime is {:x}", stats.UserTime, stats.SysTime);
            return Time::FromNs(stats.UserTime + stats.SysTime)
        }

        return Time::FromNs(stats.UserTime)
    }

    pub fn WallTimeUntil(&self, t: Time, now: Time) -> Duration {
        let ts = self.tg.TaskSet();
        let n = {
            let _r = ts.ReadLock();
            self.tg.lock().liveTasks as i64
        } ;

        if n == 0 {
            if t.Before(now) {
                return 0
            }

            // The timer tick raced with thread group exit, after which no more
            // tasks can enter the thread group. So tgc.Now() will never advance
            // again. Return a large delay; the timer should be stopped long before
            // it comes again anyway.
            return HOUR
        }

        // This is a lower bound on the amount of time that can elapse before an
        // associated timer expires, so returning this value tends to result in a
        // sequence of closely-spaced ticks just before timer expiry. To avoid
        // this, round up to the nearest ClockTick; CPU usage measurements are
        // limited to this resolution anyway.
        let remaining = t.Sub(now) / n;
        return ((remaining + CLOCK_TICK - NANOSECOND) / CLOCK_TICK) * CLOCK_TICK
    }
}

impl Thread {
    pub fn UserCPUClock(&self) -> Clock {
        let c = TaskClock {
            t: self.clone(),
            includeSys: false,
        };

        return Clock::TaskClock(c)
    }

    pub fn CPUClock(&self) -> Clock {
        let c = TaskClock {
            t: self.clone(),
            includeSys: true,
        };

        return Clock::TaskClock(c)
    }
}

impl ThreadGroup {
    pub fn UserCPUClock(&self) -> Clock {
        let c = ThreadGroupClock {
            tg: self.clone(),
            includeSys: false,
            queue: Queue::default(),
        };

        return Clock::ThreadGroupClock(c);
    }

    pub fn CPUClock(&self) -> Clock {
        let c = ThreadGroupClock {
            tg: self.clone(),
            includeSys: true,
            queue: Queue::default(),
        };

        return Clock::ThreadGroupClock(c);
    }
}

pub struct KernelCPUClockTicker {}

impl KernelCPUClockTicker {
    pub fn New() -> Self {
        return Self {}
    }
}

// Notify implements ktime.TimerListener.Notify.
impl TimerListener for KernelCPUClockTicker {
    fn Notify(&self, _exp: u64) {
        // Only increment cpuClock by 1 regardless of the number of expirations.
        // This approximately compensates for cases where thread throttling or bad
        // Go runtime scheduling prevents the kernelCPUClockTicker goroutine, and
        // presumably task goroutines as well, from executing for a long period of
        // time. It's also necessary to prevent CPU clocks from seeing large
        // discontinuous jumps.
        let kernel = GetKernel();
        let now = kernel.cpuClock.fetch_add(1, Ordering::SeqCst);
        let tasks = kernel.tasks.clone();
        let root = tasks.Root();
        let tgs = root.ThreadGroups();

        for tg in tgs {
            if tg.lock().cpuTimersEnabled == 0 {
                continue;
            }

            tasks.read();
            if tg.lock().leader.Upgrade().is_none() {
                // No tasks have ever run in this thread group.
                continue;
            }

            // Accumulate thread group CPU stats, and randomly select running tasks
            // using reservoir sampling to receive CPU timer signals.
            let mut virtReceiver = None;
            let mut nrVirtCandidates = 0;
            let mut profReceiver = None;
            let mut nrProfCandidates = 0;

            let mut tgUserTime = tg.lock().exitedCPUStats.UserTime;
            let mut tgSysTime = tg.lock().exitedCPUStats.SysTime;
            let tasks : Vec<Thread> = tg.lock().tasks.iter().cloned().collect();
            for t in &tasks {
                let tsched = t.lock().TaskSchedInfo();
                tgUserTime += tsched.userTicksAt(now) as i64 * CLOCK_TICK;
                tgSysTime += tsched.sysTicksAt(now) as i64 * CLOCK_TICK;

                if tsched.State == SchedState::RunningApp {
                    // Considered by ITIMER_VIRT, ITIMER_PROF, and RLIMIT_CPU
                    // timers.
                    nrVirtCandidates += 1;
                    //use rdtsc as random source
                    if Rdtsc() % nrVirtCandidates == 0 {
                        virtReceiver = Some(t.clone());
                    }
                }

                if tsched.State == SchedState::RunningApp || tsched.State == SchedState::RunningSys {
                    // Considered by ITIMER_PROF and RLIMIT_CPU timers.
                    nrProfCandidates += 1;
                    if Rdtsc() % nrProfCandidates == 0 {
                        profReceiver = Some(t.clone());
                    }
                }
            }

            let tgVirtNow = Time::FromNs(tgUserTime);
            let tgProfNow = Time::FromNs(tgUserTime + tgSysTime);

            // All of the following are standard (not real-time) signals, which are
            // automatically deduplicated, so we ignore the number of expirations.
            let lock = tg.lock().signalLock.clone();
            let _s = lock.lock();
            // It should only be possible for these timers to advance if we found
            // at least one running task.
            if virtReceiver.is_some() {
                // ITIMER_VIRTUAL
                let (newItimerVirtSetting, exp) = tg.lock().itimerVirtSetting.At(tgVirtNow);
                tg.lock().itimerVirtSetting = newItimerVirtSetting;
                if exp != 0 {
                    virtReceiver.clone().unwrap().sendSignalLocked(&SignalInfo::SignalInfoPriv(Signal(Signal::SIGVTALRM)), true).unwrap();
                }
            }

            if profReceiver.is_some() {
                // ITIMER_PROF
                let (newItimerProfSetting, exp) = tg.lock().itimerProfSetting.At(tgProfNow);
                //error!("profReceiver2 is some .... tgProfNow is {:?}, newItimerProfSetting is {:?}, exp is {}",
                //    tgProfNow, &newItimerProfSetting, exp);
                tg.lock().itimerProfSetting = newItimerProfSetting;
                if exp != 0 {
                    let receiver = profReceiver.clone().unwrap();
                    receiver.sendSignalLocked(&SignalInfo::SignalInfoPriv(Signal(Signal::SIGPROF)), true).unwrap();
                }

                // RLIMIT_CPU soft limit
                let (newRlimitCPUSoftSetting, exp) = tg.lock().rlimitCPUSoftSetting.At(tgProfNow);
                tg.lock().rlimitCPUSoftSetting = newRlimitCPUSoftSetting;
                if exp != 0 {
                    profReceiver.clone().unwrap().sendSignalLocked(&SignalInfo::SignalInfoPriv(Signal(Signal::SIGXCPU)), true).unwrap();
                }

                // RLIMIT_CPU hard limit
                let rlimitCPUMax = tg.lock().limits.Get(LimitType::CPU).Max;
                if rlimitCPUMax != INFINITY && !tgProfNow.Before(Time::FromSec(rlimitCPUMax as i64)) {
                    profReceiver.clone().unwrap().sendSignalLocked(&SignalInfo::SignalInfoPriv(Signal(Signal::SIGKILL)), true).unwrap();
                }
            }
        }
    }

    fn Destroy(&self) {}
}
