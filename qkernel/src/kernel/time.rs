// Copyright (c) 2021 QuarkSoft LLC
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

use super::super::qlib::linux_def::*;
use super::super::qlib::linux::time::*;

pub const MIN_TIME: Time = Time(core::i64::MIN);
pub const MAX_TIME: Time = Time(core::i64::MAX);
pub const ZERO_TIME: Time = Time(0);

#[derive(Debug)]
pub struct InterTimeSpec {
    pub ATime: Time,
    pub ATimeOmit: bool,
    pub ATimeSetSystemTime: bool,
    pub MTime: Time,
    pub MTimeOmit: bool,
    pub MTimeSetSystemTime: bool,
}

impl Default for InterTimeSpec {
    fn default() -> Self {
        return Self {
            ATime: Default::default(),
            ATimeOmit: false,
            ATimeSetSystemTime: true,
            MTime: Default::default(),
            MTimeOmit: false,
            MTimeSetSystemTime: true,
        }
    }
}

pub const MIN_DURATION: Duration = core::i64::MIN;
pub const MAX_DURATION: Duration = core::i64::MAX;

#[derive(Debug, Default, Copy, Clone)]
pub struct Time(pub i64); //how many ns

impl Time {
    pub fn FromNs(ns: i64) -> Self {
        return Self(ns)
    }

    pub fn FromSec(s: i64) -> Self {
        return Self(s * 1000_000_000)
    }

    pub fn FromUnix(s: i64, ns: i64) -> Self {
        if s > core::i64::MAX / 1000_000_000 {
            return MAX_TIME.clone();
        }

        let t = s * 1000_000_000;
        if t > core::i64::MAX - ns {
            return MAX_TIME.clone();
        }

        return Time(t + ns)
    }

    pub fn FromStatxTimestamp(ts: &StatxTimestamp) -> Self {
        return Self::FromUnix(ts.tv_sec, ts.tv_nsec as i64)
    }

    pub fn FromTimespec(ts: &Timespec) -> Self {
        return Self::FromUnix(ts.tv_sec, ts.tv_nsec)
    }

    pub fn FromTimeval(tv: &Timeval) -> Self {
        let s = tv.Sec;
        let ns = tv.Usec * 1000;
        return Self::FromUnix(s, ns)
    }

    pub fn Nanoseconds(&self) -> i64 {
        return self.0;
    }

    pub fn Seconds(&self) -> i64 {
        return self.0 / 1000_000_000;
    }

    pub fn Timespec(&self) -> Timespec {
        return Timespec {
            tv_sec: self.0 / 1000_000_000,
            tv_nsec: self.0 % 1000_000_000,
        }
    }

    pub fn Unix(&self) -> (i64, i64) {
        return (self.0 / 1000_000_000, self.0 % 1000_000_000)
    }

    pub fn TimeT(&self) -> TimeT {
        return NsecToTimeT(self.0)
    }

    pub fn Timeval(&self) -> Timeval {
        return Timeval {
            Sec: self.0 / 1000_000_000,
            Usec: (self.0 % 1000_000_000) / 1000,
        }
    }

    pub fn Add(&self, ns: i64) -> Self {
        if self.0 > 0 && ns > core::i64::MAX - self.0 {
            return MAX_TIME
        }

        if self.0 < 0 && ns < core::i64::MIN - self.0 {
            return MIN_TIME
        }

        return Self(self.0 + ns)
    }

    pub fn AddTime(&self, u: Self) -> Self {
        return self.Add(u.0)
    }

    pub fn Equal(&self, u: Self) -> bool {
        return self.0 == u.0
    }

    pub fn Before(&self, u: Self) -> bool {
        return self.0 < u.0
    }

    pub fn After(&self, u: Self) -> bool {
        return self.0 > u.0
    }

    pub fn Sub(&self, u: Time) -> Duration {
        let dur = (self.0 - u.0) * NANOSECOND;
        if u.Add(dur).Equal(*self) {
            return dur
        } else if self.Before(u) {
            return MIN_DURATION;
        } else {
            return MAX_DURATION;
        }
    }

    pub fn IsMin(&self) -> bool {
        return self.Equal(MIN_TIME)
    }

    pub fn IsZero(&self) -> bool {
        return self.Equal(ZERO_TIME)
    }

    pub fn StatxTimestamp(&self) -> StatxTimestamp {
        return StatxTimestamp::FromNs(self.Nanoseconds())
    }
}
