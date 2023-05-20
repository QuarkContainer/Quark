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
use std::collections::BTreeMap;
use core::ops::Deref;

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
        let highestPri = self.HighestPri();
        let mut resource = Resource::default();
        if highestPri >= pri {
            return resource;
        }

        for i in highestPri..pri {
            resource = resource + self.queues[i].reqResource;
        }

        return resource;
    }

    pub fn HighestPri(&self) -> usize {
        return self.existingMask.trailing_zeros() as usize;
    }

    pub fn HasTask(&self) -> bool {
        return self.existingMask > 0;
    }
}