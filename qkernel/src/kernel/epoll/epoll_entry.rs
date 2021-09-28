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

use alloc::sync::Arc;
use ::qlib::mutex::*;
use core::ops::Deref;

use super::super::super::qlib::linux_def::*;
use super::super::super::fs::file::*;
use super::super::waiter::*;
use super::epoll::*;

pub type EntryFlags = i32;

pub const ONE_SHOT : EntryFlags = 1 << 0;
pub const EDGE_TRIGGERED: EntryFlags = 1 << 1;

#[derive(Clone)]
pub struct FileIdentifier {
    pub File: FileWeak,
    pub Fd: i32,
}

impl Ord for FileIdentifier {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.File.Upgrade().cmp(&other.File.Upgrade())
    }
}

impl PartialOrd for FileIdentifier {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for FileIdentifier {
    fn eq(&self, other: &Self) -> bool {
        return self.File.Upgrade() == other.File.Upgrade()
    }
}

impl Eq for FileIdentifier {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PollEntryState {
    Ready,
    Waiting,
    Disabled,
}

pub struct PollEntryInternal {
    pub next: Option<PollEntry>,
    pub prev: Option<PollEntry>,

    pub id: FileIdentifier,
    pub userData: [i32; 2],
    pub waiter: WaitEntry,
    pub mask: EventMask,
    pub flags: EntryFlags,

    pub epoll: EventPoll,
    pub state: PollEntryState,
    pub readyTimeStamp: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ReadyState {
    pub readyTimeStamp: u64,
    pub mask: u32,
}

#[derive(Clone)]
pub struct PollEntry(pub Arc<QMutex<PollEntryInternal>>);

impl PollEntry {
    pub fn CallBack(&self) {
        let epoll = self.lock().epoll.clone();
        let mut lists = epoll.lists.lock();

        let state = self.SetReady();
        if state == PollEntryState::Waiting {
            lists.waitingList.Remove(self);
            lists.readyList.PushBack(self);

            epoll.queue.Notify(EVENT_IN);
        }
    }

    pub fn SetReady(&self) -> PollEntryState {
        let mut e = self.lock();
        let oldstate = e.state;
        e.state = PollEntryState::Ready;
        e.readyTimeStamp += 1;
        return oldstate;
    }

    pub fn ReadyState(&self) -> ReadyState {
        let e = self.lock();
        return ReadyState {
            readyTimeStamp: e.readyTimeStamp,
            mask: e.mask as u32,
        }
    }

    pub fn Id(&self) -> i32 {
        return self.lock().id.Fd
    }

    pub fn Reset(&self) {
        self.lock().prev = None;
        self.lock().next = None;
    }
}

impl Deref for PollEntry {
    type Target = Arc<QMutex<PollEntryInternal>>;

    fn deref(&self) -> &Arc<QMutex<PollEntryInternal>> {
        &self.0
    }
}

impl PollEntry {
    pub fn Next(&self) -> Option<PollEntry> {
        return self.lock().next.clone();
    }

    pub fn Prev(&self) -> Option<PollEntry> {
        return self.lock().prev.clone();
    }

    pub fn SetNext(&self, elem: Option<PollEntry>) {
        self.lock().next = elem
    }

    pub fn SetPrev(&self, elem: Option<PollEntry>) {
        self.lock().prev = elem
    }
}
