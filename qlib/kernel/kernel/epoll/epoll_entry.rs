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
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::super::super::linux_def::*;
use super::super::super::fs::file::*;
use super::super::waiter::*;
use super::epoll::*;

use crate::GuestHostSharedAllocator;

pub type EntryFlags = i32;

pub const ONE_SHOT: EntryFlags = 1 << 0;
pub const EDGE_TRIGGERED: EntryFlags = 1 << 1;

pub type FileIdentifier = u64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PollEntryState {
    Ready,
    Waiting,
    Disabled,
}

pub struct PollEntryInternal {
    pub next: Option<PollEntry>,
    pub prev: Option<PollEntry>,

    pub id: u64,
    pub file: FileWeak,
    pub userData: u64,
    pub waiter: WaitEntryWeak,
    pub mask: EventMask,
    pub flags: EntryFlags,

    pub epoll: EventPoll,
    pub state: PollEntryState,
}

#[derive(Debug, Clone, Copy)]
pub struct ReadyState {
    pub mask: u32,
}

#[derive(Clone)]
pub struct PollEntry(pub Arc<QMutex<PollEntryInternal>, GuestHostSharedAllocator>);

impl PollEntry {
    pub fn CallBack(&self) {
        let epoll = self.lock().epoll.clone();
        let mut lists = epoll.lists.lock();

        let state = self.lock().state;
        if state == PollEntryState::Waiting {
            self.SetReady();
            lists.PushBack(self);

            drop(lists);
            epoll.queue.Notify(READABLE_EVENT);
        }
    }

    pub fn SetReady(&self) {
        let mut e = self.lock();
        e.state = PollEntryState::Ready;
    }

    pub fn ReadyState(&self) -> u32 {
        return self.lock().mask as u32;
    }

    pub fn Id(&self) -> u64 {
        return self.lock().id;
    }

    pub fn Reset(&self) {
        self.lock().prev = None;
        self.lock().next = None;
    }
}

impl Deref for PollEntry {
    type Target = Arc<QMutex<PollEntryInternal>, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<QMutex<PollEntryInternal>, GuestHostSharedAllocator> {
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
