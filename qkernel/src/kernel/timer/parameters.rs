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

use super::super::super::asm::muldiv64;
use super::super::super::qlib::linux::time::*;
use super::super::super::qlib::common::*;
use super::sampler::*;
use super::*;

// ApproxUpdateInterval is the approximate interval that parameters
// should be updated at.
//
// Error correction assumes that the next update will occur after this
// much time.
//
// If an update occurs before ApproxUpdateInterval passes, it has no
// adverse effect on error correction behavior.
//
// If an update occurs after ApproxUpdateInterval passes, the clock
// will overshoot its error correction target and begin accumulating
// error in the other direction.
//
// If updates occur after more than 2*ApproxUpdateInterval passes, the
// clock becomes unstable, accumulating more error than it had
// originally. Repeated updates after more than 2*ApproxUpdateInterval
// will cause unbounded increases in error.
//
// These statements assume that the host clock does not change. Actual
// error will depend upon host clock changes.
pub const APPROX_UPDATE_INTERVAL: Duration = 1 * SECOND;

// MaxClockError is the maximum amount of error that the clocks will
// try to correct.
//
// This limit:
//
//  * Puts a limit on cases of otherwise unbounded increases in error.
//
//  * Avoids unreasonably large frequency adjustments required to
//    correct large errors over a single update interval.
pub const MAX_CLOCK_ERROR: ReferenceNS = APPROX_UPDATE_INTERVAL / 4;

#[derive(Default, Debug, Clone, Copy)]
pub struct Parameters {
    // BaseCycles was the TSC counter value when the time was BaseRef.
    pub BaseCycles: TSCValue,

    // BaseRef is the reference clock time in nanoseconds corresponding to
    // BaseCycles.
    pub BaseRef: ReferenceNS,

    // Frequency is the frequency of the cycle clock in Hertz.
    pub Frequency: u64,
}

impl Parameters {
    // ComputeTime calculates the current time from a "now" TSC value.
    //
    // time = ref + (now - base) / f
    pub fn ComputeTime(&self, nowCycles: TSCValue) -> (i64, bool) {
        let mut diffCycles = nowCycles - self.BaseCycles;

        if diffCycles < 0 {
            info!("now cycles {} < base cycles {}", nowCycles, self.BaseCycles);
            diffCycles = 0;
        }

        // Overflow "won't ever happen". If diffCycles is the max value
        // (2^63 - 1), then to overflow,
        //
        // frequency <= ((2^63 - 1) * 10^9) / 2^64 = 500Mhz
        //
        // A TSC running at 2GHz takes 201 years to reach 2^63-1. 805 years at
        // 500MHz.
        let (diffNS, ok) = muldiv64(diffCycles as u64, SECOND as u64, self.Frequency);
        return ((self.BaseRef as u64 + diffNS) as i64, ok)
    }
}

// errorAdjust returns a new Parameters struct "adjusted" that satisfies:
//
// 1. adjusted.ComputeTime(now) = prevParams.ComputeTime(now)
//   * i.e., the current time does not jump.
//
// 2. adjusted.ComputeTime(TSC at next update) = newParams.ComputeTime(TSC at next update)
//   * i.e., Any error between prevParams and newParams will be corrected over
//     the course of the next update period.
//
// errorAdjust also returns the current clock error.
//
// Preconditions:
// * newParams.BaseCycles >= prevParams.BaseCycles; i.e., TSC must not go
//   backwards.
// * newParams.BaseCycles <= now; i.e., the new parameters be computed at or
//   before now.
pub fn ErrorAdjust(prevParams: &Parameters, newParams: &Parameters, now: TSCValue) -> Result<(Parameters, ReferenceNS)> {
    if newParams.BaseCycles < prevParams.BaseCycles {
        return Err(Error::Common(format!("TSC went backwards in updated clock params: {} < {}",
                                         newParams.BaseCycles, prevParams.BaseCycles)))
    }

    if newParams.BaseCycles > now {
        return Err(Error::Common(format!("parameters contain base cycles later than now: {} > {}",
                                         newParams.BaseCycles, now)))
    }

    let intervalNS = APPROX_UPDATE_INTERVAL as i64;
    let nsPerSec = SECOND as u64;

    // Current time as computed by prevParams.
    let (oldNowNS, ok) = prevParams.ComputeTime(now);
    if !ok {
        return Err(Error::Common(format!("old now time computation overflowed. params = {:?}, now = {}",
                                         prevParams, now)))
    }

    // We expect the update ticker to run based on this clock (i.e., it has
    // been using prevParams and will use the returned adjusted
    // parameters). Hence it will decide to fire intervalNS from the
    // current (oldNowNS) "now".
    let nextNS = oldNowNS + intervalNS;

    if nextNS <= newParams.BaseRef {
        // The next update time already passed before the new
        // parameters were created! We definitely can't correct the
        // error by then.
        return Err(Error::Common(format!("old now time computation overflowed. params = {:?}, now = {}",
                                         prevParams, now)))
    }

    // For what TSC value next will newParams.ComputeTime(next) = nextNS?
    //
    // Solve ComputeTime for next:
    //
    // next = newParams.Frequency * (nextNS - newParams.BaseRef) + newParams.BaseCycles
    let (c, ok) = muldiv64(newParams.Frequency, (nextNS - newParams.BaseRef) as u64, nsPerSec);
    if !ok {
        return Err(Error::Common(format!("{} * ({} - {}) / {} overflows",
                                         newParams.Frequency, nextNS, newParams.BaseRef, nsPerSec)))
    }

    let mut cycles = c as TSCValue;
    let next = cycles + newParams.BaseCycles;

    if next < now {
        // The next update time already passed now with the new
        // parameters! We can't correct the error in a single period.
        return Err(Error::Common(format!("unable to correct error in single period. oldNowNS = {}, nextNS = {}, now = {}, next = {}",
                                         oldNowNS, nextNS, now, next)))
    }

    // We want to solve for parameters that satisfy:
    //
    // adjusted.ComputeTime(now) = oldNowNS
    //
    // adjusted.ComputeTime(next) = nextNS
    //
    // i.e., the current time does not change, but by the time we reach
    // next we reach the same time as newParams.

    // We choose to keep BaseCycles fixed.
    let mut adjusted = Parameters {
        BaseCycles: newParams.BaseCycles,
        ..Default::default()
    };

    // We want a slope such that time goes from oldNowNS to nextNS when
    // we reach next.
    //
    // In other words, cycles should increase by next - now in the next
    // interval.
    cycles = next - now;
    let ns = intervalNS;

    // adjusted.Frequency = cycles / ns
    let (freq, ok) = muldiv64(cycles as u64, nsPerSec, ns as u64);
    if !ok {
        return Err(Error::Common(format!("{} * ({} - {}) / {} overflows",
                                         next, now, nsPerSec, ns)))
    }

    adjusted.Frequency = freq;

    // Now choose a base reference such that the current time remains the
    // same. Note that this is just ComputeTime, solving for BaseRef:
    //
    // oldNowNS = BaseRef + (now - BaseCycles) / Frequency
    // BaseRef = oldNowNS - (now - BaseCycles) / Frequency
    let (diffNS, ok) = muldiv64((now - adjusted.BaseCycles) as u64, nsPerSec, adjusted.Frequency);
    if !ok {
        return Err(Error::Common(format!("{} * ({} - {}) / {} overflows",
                                         now, adjusted.BaseCycles, nsPerSec, adjusted.Frequency)))
    }

    adjusted.BaseRef = oldNowNS - diffNS as i64;

    // The error is the difference between the current time and what the
    // new parameters say the current time should be.
    let (newNowNS, ok) = newParams.ComputeTime(now);
    if !ok {
        return Err(Error::Common(format!("new now time computation overflowed. params = {:?}, now = {}",
                                         newParams, now)))
    }

    let errorNS = oldNowNS - newNowNS;
    return Ok((adjusted, errorNS))
}

pub fn logErrorAdjustement(_clock: ClockID, _errorNS: ReferenceNS, _orig: &Parameters, _adjusted: &Parameters) {
    //info!("Clock({}): error: {} ns, adjusted frequency from {} Hz to {} Hz", clock, errorNS, orig.Frequency, adjusted.Frequency)
}
