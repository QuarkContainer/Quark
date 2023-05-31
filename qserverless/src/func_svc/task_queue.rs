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

use std::time::SystemTime;
use std::collections::{BTreeMap, VecDeque};
use std::collections::btree_map::Iter;

use crate::func_call::*;
use crate::scheduler::Resource;


#[derive(Debug, Default)]
pub struct TaskQueue {
    pub queue: BTreeMap<SystemTime, FuncCall>,
    pub reqResource: Resource,
}

impl TaskQueue {
    pub fn Enq(&mut self, task: &FuncCall) {
        self.reqResource = self.reqResource + task.ReqResource();
        self.queue.insert(task.createTime, task.clone());
    }

    pub fn Peek(&self) -> Option<FuncCall> {
        match self.queue.iter().next() {
            None => return None,
            Some((_, item)) => {
                return Some(item.clone())
            }
        }
    }

    // return left count of the queue
    pub fn Remove(&mut self, task: &FuncCall) -> usize {
        self.queue.remove(&task.createTime);
        self.reqResource = self.reqResource - task.ReqResource();
        return self.queue.len();
    }

    pub fn Count(&self) -> usize {
        return self.queue.len();
    }
}

pub const PRIORITY_COUNT: usize = 10;
pub const START_BATCHTASK_PRI: usize = 5;
pub const KEEP_BATCHTASK_THRESHOLD: u64 = 10; // free memory < 10% total memory 


#[derive(Debug, Default)]
pub struct TaskQueues {
    pub queues: [TaskQueue; PRIORITY_COUNT],
    pub existingMask: u64,
}

impl TaskQueues {
    pub fn Enq(&mut self, task: &FuncCall) {
        let pri = task.Priority();
        assert!(pri < PRIORITY_COUNT);
        self.queues[pri].Enq(task);
        self.existingMask |= 1 << pri;
    }

    pub fn Peek(&self) -> Option<FuncCall> {
        let highestPriIdx = self.existingMask.trailing_zeros() as usize;
        if highestPriIdx > self.queues.len() {{
            return None;
        }}
        return self.queues[highestPriIdx].Peek();
    }

    pub fn Remove(&mut self, task: &FuncCall) {
        let pri = task.Priority();
        assert!(pri < PRIORITY_COUNT);
        let count = self.queues[pri].Remove(task);
        if count == 0 {
            self.existingMask &= !(1<<pri);
        }
    }

    pub fn ReqResourceForHigherPri(&self, pri: usize) -> Resource {
        let highestPri = self.TopPriority();
        let mut resource = Resource::default();
        if highestPri >= pri {
            return resource;
        }

        for i in highestPri..pri {
            resource = resource + self.queues[i].reqResource;
        }

        return resource;
    }

    pub fn TopPriority(&self) -> usize {
        return self.existingMask.trailing_zeros() as usize;
    }

    pub fn HasTask(&self) -> bool {
        return self.existingMask > 0;
    }

    pub fn Iter<'a>(&self) -> TaskQueuesIter {
        let pri = self.TopPriority();
        if pri > self.queues.len() {
            return TaskQueuesIter {
                queues: self,
                nextPri: pri,
                currentIter: None,
            }
        } else {
            return TaskQueuesIter {
                queues: self,
                nextPri: pri + 1,
                currentIter: Some(self.queues[pri].queue.iter()),
            }
        }
    }
}

pub struct TaskQueuesIter <'a>{
    pub queues: &'a TaskQueues,
    pub nextPri: usize,
    pub currentIter: Option<Iter<'a, SystemTime, FuncCall>>
}

impl <'a> Iterator for TaskQueuesIter <'a> {
    type Item = FuncCall;

    fn next(&mut self) -> Option<Self::Item> {
        let mut iter = match self.currentIter.take() {
            None => return None,
            Some(i) => i,
        };

        match iter.next() {
            Some((_, v)) => {
                self.currentIter = Some(iter);
                return Some(v.clone());
            }
            None => (),
        }

        if self.nextPri == self.queues.queues.len() {
            return None;
        }

        for i in self.nextPri .. self.queues.queues.len() {
            if self.queues.queues[i].queue.len() > 0 {
                self.currentIter = Some(self.queues.queues[i].queue.iter());
                self.nextPri = i + 1;
                return Some(self.currentIter.as_mut().unwrap().next().unwrap().1.clone());
            }
        }

        self.currentIter = None;
        self.nextPri = self.queues.queues.len();
        return None; 
    }
}

#[derive(Debug, Default)]
pub struct PackageTaskQueue {
    pub waitingTask: usize,
    pub waitingQueue: [VecDeque<FuncCall>; PRIORITY_COUNT],
    pub existingMask: u64,
}

impl PackageTaskQueue {
    // get the Nth task in the waitingQueue 
    pub fn GetWaitingTask(&self, n: usize) -> Option<FuncCall> {
        if n > self.waitingTask {
            return None;
        }

        let mut n: usize = n;
        for i in self.TopPriority()..self.waitingQueue.len() {
            if n > self.waitingQueue[i].len() {
                n -= self.waitingQueue[i].len();
            } else {
                let (s1, s2) = self.waitingQueue[i].as_slices();
                if s1.len() >= n {
                    return Some(s1[n].clone());
                } else {
                    return Some(s2[n-s1.len()].clone());
                }
            }
        }

        return None;
    }

    pub fn TopPriority(&self) -> usize {
        return self.existingMask.trailing_zeros() as usize;
    }

    pub fn Pop(&mut self) -> Option<FuncCall> {
        if self.waitingTask == 0 {
            return None;
        }

        self.waitingTask -= 1;

        let pri = self.TopPriority();
        let task = self.waitingQueue[pri].pop_front().clone().unwrap();
        if self.waitingQueue[pri].len() == 0 {
            self.existingMask &= !(1<<pri); // clean the mask
        }
        return Some(task);
    }

    pub fn Push(&mut self, task: &FuncCall) {
        self.waitingTask += 1;
        let pri = task.Priority();
        self.waitingQueue[pri].push_back(task.clone());
        self.existingMask |= 1 << pri;
    }
}