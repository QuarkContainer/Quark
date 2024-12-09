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

use enum_dispatch::enum_dispatch;

pub mod bufchan;
pub mod chan;
pub mod cond;
pub mod entry;
pub mod lock;
pub mod qlock;
pub mod queue;
pub mod waiter;
pub mod waitgroup;
pub mod waitlist;

pub use self::entry::*;
pub use self::queue::*;
pub use self::waiter::*;
use super::super::super::linux_def::*;
use super::super::task::*;
use super::async_wait::*;

use super::super::fs::file::File;
use crate::qlib::kernel::fs::file::FileOps;

// EventMaskFromLinux returns an EventMask representing the supported events
// from the Linux events e, which is in the format used by poll(2).
pub fn EventMaskFromLinux(e: u32) -> EventMask {
    return e as u64 & ALL_EVENTS;
}

// ToLinux returns e in the format used by Linux poll(2).
pub fn ToLinux(e: EventMask) -> u32 {
    return e as u32;
}

// Waitable contains the methods that need to be implemented by waitable
// objects.
// default:: Alway readable
#[enum_dispatch(FileOps)]
pub trait Waitable {
    fn AsyncReadiness(&self, task: &Task, mask: EventMask, _wait: &MultiWait) -> Future<EventMask> {
        //wait.AddWait();
        let future = Future::New(0 as EventMask);
        let ret = self.Readiness(task, mask);
        future.Set(Ok(ret));
        //wait.Done();
        return future;
    }

    // Readiness returns what the object is currently ready for. If it's
    // not ready for a desired purpose, the caller may use EventRegister and
    // EventUnregister to get notifications once the object becomes ready.
    //
    // Implementations should allow for events like EventHUp and EventErr
    // to be returned regardless of whether they are in the input EventMask.
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        return mask;
    }

    // EventRegister registers the given waiter entry to receive
    // notifications when an event occurs that makes the object ready for
    // at least one of the events in mask.
    fn EventRegister(&self, _task: &Task, _e: &WaitEntry, _mask: EventMask) {}

    // EventUnregister unregisters a waiter entry previously registered with
    // EventRegister().
    fn EventUnregister(&self, _task: &Task, _e: &WaitEntry) {}
}

pub struct PollStruct {
    pub f: File,
    pub event: EventMask,
    pub revent: EventMask,
    pub future: Option<Future<EventMask>>,
}

impl PollStruct {
    pub fn PollMulti(task: &Task, polls: &mut [PollStruct]) -> usize {
        let mw = MultiWait::New(task.GetTaskId());

        for i in 0..polls.len() {
            let poll = &mut polls[i];
            let future = poll.f.AsyncReadiness(task, poll.event, &mw);
            poll.future = Some(future)
        }

        mw.Wait();

        let mut cnt = 0;
        for i in 0..polls.len() {
            let poll = &mut polls[i];
            match poll.future.take().unwrap().Wait() {
                Err(_) => (),
                Ok(revent) => {
                    if revent > 0 {
                        poll.revent = revent;
                        cnt += 1;
                    }
                }
            };
        }

        return cnt;
    }
}
