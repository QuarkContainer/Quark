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

use alloc::string::String;
use alloc::string::ToString;

use super::epoll_entry::*;

#[derive(Default)]
pub struct PollEntryList {
    head: Option<PollEntry>,
    tail: Option<PollEntry>,
}

impl PollEntryList {
    pub fn GetString(&self) -> String {
        let mut item = self.head.clone();
        let mut output = "".to_string();
        while item.is_some() {
            let node = item.unwrap();
            output += &format!("{} ", node.Id());

            item = node.lock().next.clone();
        }

        return output;
    }

    //remove all of the wait entries
    pub fn Reset(&mut self) {
        let mut cur = self.head.take();

        self.tail = None;

        while cur.is_some() {
            let tmp = cur.clone().unwrap();
            let next = tmp.lock().next.clone();
            cur = next;
        }
    }

    pub fn Empty(&self) -> bool {
        return self.head.is_none();
    }

    pub fn Front(&self) -> Option<PollEntry> {
        return self.head.clone();
    }

    pub fn Back(&self) -> Option<PollEntry> {
        return self.tail.clone();
    }

    pub fn PushFront(&mut self, e: &PollEntry) {
        if self.head.is_none() {
            //empty
            self.head = Some(e.clone());
            self.tail = Some(e.clone())
        } else {
            let head = self.head.take().unwrap();
            e.lock().next = Some(head.clone());
            head.lock().prev = Some(e.clone());
            self.head = Some(e.clone());
        }
    }

    pub fn PushBack(&mut self, e: &PollEntry) {
        if self.head.is_none() {
            //empty
            self.head = Some(e.clone());
            self.tail = Some(e.clone())
        } else {
            //info!("self.head.is_none() is {}, self.tail.is_none() is {}",
            //        self.head.is_none(), self.tail.is_none());
            let tail = self.tail.take().unwrap();
            e.lock().prev = Some(tail.clone());
            tail.lock().next = Some(e.clone());
            self.tail = Some(e.clone());
        }
    }

    pub fn PushBackList(&mut self, m: &mut PollEntryList) {
        let head = m.head.take();
        let tail = m.tail.take();

        if head.is_none() {
            return;
        }

        if self.head.is_none() {
            self.head = head;
            self.tail = tail;
        } else {
            self.tail.as_ref().unwrap().SetNext(head.clone());
            head.as_ref().unwrap().SetPrev(self.tail.clone());
            self.tail = tail;
        }
    }

    pub fn RemoveAll(&mut self) {
        self.Reset();
    }

    pub fn Remove(&mut self, e: &PollEntry) {
        let mut elock = e.lock();
        if elock.prev.is_none() {
            //head
            self.head = elock.next.clone();
        } else {
            elock.prev.clone().unwrap().lock().next = elock.next.clone();
        }

        if elock.next.is_none() {
            //tail
            self.tail = elock.prev.clone();
        } else {
            elock.next.clone().unwrap().lock().prev = elock.prev.clone();
        }

        elock.prev = None;
        elock.next = None;
        elock.state = PollEntryState::Waiting;
    }
}
