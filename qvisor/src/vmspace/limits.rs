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

use alloc::collections::btree_map::BTreeMap;
use lazy_static::lazy_static;

use super::super::qlib::common::*;
use super::super::qlib::limits::*;
use super::super::runc::oci::*;
//use super::super::qlib::linux::limits::*;

lazy_static! {
    pub static ref FROM_LINUX_SOURCE: BTreeMap<&'static str, LimitType> = [
        ("RLIMIT_AS", LimitType::AS),
        ("RLIMIT_CORE", LimitType::Core),
        ("RLIMIT_CPU", LimitType::CPU),
        ("RLIMIT_DATA", LimitType::Data),
        ("RLIMIT_FSIZE", LimitType::FileSize),
        ("RLIMIT_LOCKS", LimitType::Locks),
        ("RLIMIT_MEMLOCK", LimitType::MemoryLocked),
        ("RLIMIT_MSGQUEUE", LimitType::MessageQueueBytes),
        ("RLIMIT_NICE", LimitType::Nice),
        ("RLIMIT_NOFILE", LimitType::NumberOfFiles),
        ("RLIMIT_NPROC", LimitType::ProcessCount),
        ("RLIMIT_RSS", LimitType::Rss),
        ("RLIMIT_RTPRIO", LimitType::RealTimePriority),
        ("RLIMIT_RTTIME", LimitType::Rttime),
        ("RLIMIT_SIGPENDING", LimitType::SignalsPending),
        ("RLIMIT_STACK", LimitType::Stack),
    ]
    .iter()
    .cloned()
    .collect();
    pub static ref DEFAULT_LIMITS: LimitSet = DefaultLimits();
}

pub fn FindName(lt: LimitType) -> &'static str {
    for (k, v) in FROM_LINUX_SOURCE.iter() {
        if *v == lt {
            return *k;
        }
    }

    return "unknown";
}

pub fn DefaultLimits() -> LimitSet {
    let ls = NewLinuxLimitSet();

    // Set default limits based on what containers get by default, ex:
    // $ docker run --rm debian prlimit
    ls.SetUnchecked(
        LimitType::AS,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::Core,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::CPU,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::Data,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::FileSize,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::Locks,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::MemoryLocked,
        Limit {
            Cur: 65536,
            Max: 65536,
        },
    );
    ls.SetUnchecked(
        LimitType::MessageQueueBytes,
        Limit {
            Cur: 819200,
            Max: 819200,
        },
    );
    ls.SetUnchecked(LimitType::Nice, Limit { Cur: 0, Max: 0 });
    ls.SetUnchecked(
        LimitType::NumberOfFiles,
        Limit {
            Cur: 1048576,
            Max: 1048576,
        },
    );
    ls.SetUnchecked(
        LimitType::ProcessCount,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(
        LimitType::Rss,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(LimitType::RealTimePriority, Limit { Cur: 0, Max: 0 });
    ls.SetUnchecked(
        LimitType::Rttime,
        Limit {
            Cur: INFINITY,
            Max: INFINITY,
        },
    );
    ls.SetUnchecked(LimitType::SignalsPending, Limit { Cur: 0, Max: 0 });
    ls.SetUnchecked(
        LimitType::Stack,
        Limit {
            Cur: 8388608,
            Max: INFINITY,
        },
    );

    // Read host limits that directly affect the sandbox and adjust the defaults
    // based on them.
    for res in [libc::RLIMIT_FSIZE, libc::RLIMIT_NOFILE].iter() {
        let mut hl = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };

        let ret = unsafe { libc::getrlimit(*res, &mut hl) };

        let res = *res as i32;
        if ret < 0 {
            panic!("Getrlimit fail with err {}", errno::errno().0)
        }

        let lt = FROM_LINUX_RESOURCE
            .Get(res)
            .expect(&format!("unknown rlimit type {}", res));

        let hostLimit = Limit {
            Cur: FromLinux(hl.rlim_cur),
            Max: FromLinux(hl.rlim_max),
        };

        let defaultLimit = ls.Get(lt);
        if hostLimit.Cur != INFINITY && hostLimit.Cur < defaultLimit.Cur {
            error!(
                "Host limit is lower than recommended, resource: {}, host: {}, recommended: {}",
                FindName(lt),
                hostLimit.Cur,
                defaultLimit.Cur
            );
        }

        if hostLimit.Cur != defaultLimit.Cur || hostLimit.Max != defaultLimit.Max {
            info!(
                "Setting limit from host, resource: {} {{soft: {}, hard: {}}}",
                FindName(lt),
                hostLimit.Cur,
                hostLimit.Max
            );
            ls.SetUnchecked(lt, hostLimit);
        }
    }

    return ls;
}

pub fn CreateLimitSet(spec: &Spec) -> Result<LimitSet> {
    let ls = DEFAULT_LIMITS.GetCopy();

    for rl in &spec.process.rlimits {
        let lt = match FROM_LINUX_RESOURCE.Get(rl.typ as i32) {
            None => return Err(Error::Common(format!("unknown resource {:?}", rl.typ))),
            Some(lt) => lt,
        };

        ls.SetUnchecked(
            lt,
            Limit {
                Cur: rl.soft,
                Max: rl.hard,
            },
        )
    }

    return Ok(ls);
}
