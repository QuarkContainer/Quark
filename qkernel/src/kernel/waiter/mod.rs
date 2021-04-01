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

pub mod waiter;
pub mod entry;
pub mod waitlist;
pub mod queue;
pub mod waitgroup;
pub mod lock;
pub mod bufchan;
pub mod chan;
pub mod cond;
pub mod qlock;

use super::super::qlib::linux_def::*;
use super::super::task::*;
pub use self::entry::*;
pub use self::waiter::*;
pub use self::queue::*;


// EventMaskFromLinux returns an EventMask representing the supported events
// from the Linux events e, which is in the format used by poll(2).
pub fn EventMaskFromLinux(e: u32) -> EventMask {
    return e as EventMask & ALL_EVENTS;
}

// ToLinux returns e in the format used by Linux poll(2).
pub fn ToLinux(e: EventMask) -> u32 {
    return e as u32;
}

// Waitable contains the methods that need to be implemented by waitable
// objects.
// default:: Alway readable
pub trait Waitable {
    // Readiness returns what the object is currently ready for. If it's
    // not ready for a desired purpose, the caller may use EventRegister and
    // EventUnregister to get notifications once the object becomes ready.
    //
    // Implementations should allow for events like EventHUp and EventErr
    // to be returned regardless of whether they are in the input EventMask.
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        return mask
    }

    // EventRegister registers the given waiter entry to receive
    // notifications when an event occurs that makes the object ready for
    // at least one of the events in mask.
    fn EventRegister(&self, _task: &Task, _e: &WaitEntry, _mask: EventMask) {}

    // EventUnregister unregisters a waiter entry previously registered with
    // EventRegister().
    fn EventUnregister(&self, _task: &Task, _e: &WaitEntry) {}
}


