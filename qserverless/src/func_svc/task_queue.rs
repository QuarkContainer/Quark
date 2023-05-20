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

use std::sync::Arc;
use std::time::SystemTime;
use std::collections::{BTreeMap, VecDeque};
use core::ops::Deref;
use std::collections::btree_map::Iter;

use crate::func_context::*;
use crate::scheduler::Resource;

#[derive(Debug, Clone)]
pub struct TaskItemInner {
    pub priority: usize,
    pub createTime: SystemTime,
    pub context: FuncCallContext,
}

#[derive(Debug, Clone)]
pub struct TaskItem(Arc<TaskItemInner>);

impl TaskItem {
    pub fn ReqResource(&self) -> Resource {
        return self.context.ReqResource();
    }

    pub fn PackageId(&self) -> PackageId {
        return self.context.lock().unwrap().packageId.clone();
    }
}

impl Deref for TaskItem {
    type Target = Arc<TaskItemInner>;

    fn deref(&self) -> &Arc<TaskItemInner> {
        &self.0
    }
}

impl Ord for TaskItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.priority == other.priority {
            return other.createTime.cmp(&self.createTime);
        }

        return other.priority.cmp(&other.priority);
    }
}

impl PartialOrd for TaskItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TaskItem {
    fn eq(&self, other: &Self) -> bool {
        return self.priority == other.priority && self.createTime == other.createTime;
    }
}

impl Eq for TaskItem {}

#[derive(Debug)]
pub struct TaskQueue {
    pub queue: BTreeMap<SystemTime, TaskItem>,
    pub reqResource: Resource,
}

impl TaskQueue {
    pub fn Enq(&mut self, task: &TaskItem) {
        self.reqResource = self.reqResource + task.ReqResource();
        self.queue.insert(task.createTime, task.clone());
    }

    pub fn Peek(&self) -> Option<TaskItem> {
        match self.queue.iter().next() {
            None => return None,
            Some((_, item)) => {
                return Some(item.clone())
            }
        }
    }

    // return left count of the queue
    pub fn Remove(&mut self, task: &TaskItem) -> usize {
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

#[derive(Debug)]
pub struct TaskQueues {
    pub queues: [TaskQueue; PRIORITY_COUNT],
    pub existingMask: u64,
}

impl TaskQueues {
    pub fn Enq(&mut self, task: &TaskItem) {
        let pri = task.priority;
        assert!(pri < PRIORITY_COUNT);
        self.queues[pri].Enq(task);
        self.existingMask |= 1 << pri;
    }

    pub fn Peek(&self) -> Option<TaskItem> {
        let highestPriIdx = self.existingMask.trailing_zeros() as usize;
        if highestPriIdx > self.queues.len() {{
            return None;
        }}
        return self.queues[highestPriIdx].Peek();
    }

    pub fn Remove(&mut self, task: &TaskItem) {
        let pri = task.priority;
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

    // get the task priority after skip first N tasks
    // it is used to look for the first priority with N creating Pod in package
    pub fn PriAfterNTask(&self, n: usize) -> usize {
        let pri = self.TopPriority();
        let mut n = n;
        for i in pri..self.queues.len() {
            if n < self.queues[i].queue.len() {
                return i;
            }
            n -= self.queues[i].queue.len();
        }

        return self.queues.len();
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
    pub currentIter: Option<Iter<'a, SystemTime, TaskItem>>
}

impl <'a> Iterator for TaskQueuesIter <'a> {
    type Item = TaskItem;

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

#[derive(Debug)]
pub struct PackageTaskQueue {
    pub waitingTask: usize,
    pub waitingQueue: [VecDeque<TaskItem>; PRIORITY_COUNT],
    pub existingMask: u64,
}

impl PackageTaskQueue {
    // get the Nth task in the waitingQueue 
    pub fn GetWaitingTask(&self, n: usize) -> Option<TaskItem> {
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

    pub fn Pop(&mut self) -> Option<TaskItem> {
        if self.waitingTask > 0 {
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

    pub fn Push(&mut self, task: &TaskItem) {
        self.waitingTask += 1;
        let pri = task.priority;
        self.waitingQueue[pri].push_back(task.clone());
        self.existingMask |= 1 << pri;
    }
}