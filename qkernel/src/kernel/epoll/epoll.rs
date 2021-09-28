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
use alloc::vec::Vec;
use ::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use core::any::Any;
use core::ops::Deref;
use alloc::string::String;
use alloc::string::ToString;

use super::super::super::task::*;
use super::super::super::qlib::singleton::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::dirent::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::fs::anon::*;
use super::super::waiter::*;
use super::epoll_entry::*;
use super::epoll_list::*;

pub static CYCLE_MU : Singleton<QMutex<()>> = Singleton::<QMutex<()>>::New();
pub unsafe fn InitSingleton() {
    CYCLE_MU.Init(QMutex::new(()));
}

// Event describes the event mask that was observed and the user data to be
// returned when one of the events occurs. It has this format to match the linux
// format to avoid extra copying/allocation when writing events to userspace.
#[derive(Default, Clone, Copy, Debug)]
pub struct Event {
    // Events is the event mask containing the set of events that have been
    // observed on an entry.
    pub Events: u32,

    // Data is an opaque 64-bit value provided by the caller when adding the
    // entry, and returned to the caller when the entry reports an event.
    pub Data: [i32; 2],
}

// An entry is always in one of the following lists:
//	readyList -- when there's a chance that it's ready to have
//		events delivered to epoll waiters. Given that being
//		ready is a transient state, the Readiness() and
//		readEvents() functions always call the entry's file
//		Readiness() function to confirm it's ready.
//	waitingList -- when there's no chance that the entry is ready,
//		so it's waiting for the readyCallback to be called
//		on it before it gets moved to the readyList.
//	disabledList -- when the entry is disabled. This happens when
//		a one-shot entry gets delivered via readEvents().
#[derive(Default)]
pub struct PollLists {
    pub readyList: PollEntryList,
    pub waitingList: PollEntryList,
    pub disabledList: PollEntryList,
}

impl PollLists {
    pub fn ToString(&self) -> String {
        let mut str = "ready: ".to_string();
        str += &self.readyList.GetString();
        str += "waitingList ";
        str += &self.waitingList.GetString();
        str += "disableedList ";
        str += &self.disabledList.GetString();
        return str;
    }
}

// EventPoll holds all the state associated with an event poll object, that is,
// collection of files to observe and their current state.
#[derive(Default)]
pub struct EventPollInternal {
    pub queue: Queue,
    pub files: QMutex<BTreeMap<FileIdentifier, PollEntry>>,

    pub lists: QMutex<PollLists>,
}

// NewEventPoll allocates and initializes a new event poll object.
pub fn NewEventPoll(task: &Task) -> File {
    let inode = NewAnonInode(task);
    let dirent = Dirent::New(&inode, "anon_inode:[eventpoll]");
    let epoll = EventPoll::default();
    return File::New(&dirent, &FileFlags::default(), epoll);
}

#[derive(Clone, Default)]
pub struct EventPoll (Arc<EventPollInternal>);

impl PartialEq for EventPoll {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl EventPoll {
    // eventsAvailable determines if 'e' has events available for delivery.
    pub fn ReadyEvents(&self, task: &Task) -> BTreeMap<File, ReadyState>{
        let mut map = BTreeMap::new();
        {
            let mut lists = self.lists.lock();
            let mut it = lists.readyList.Front();
            while it.is_some() {
                let entry = it.unwrap();
                it = entry.Next();

                // If the entry is ready, we know 'e' has at least one entry
                // ready for delivery.
                let readyState = entry.ReadyState();

                let file = entry.lock().id.File.clone();
                let file = match file.Upgrade() {
                    None => {
                        // the file has been dropped, just remove it
                        lists.readyList.Remove(&entry);
                        continue;
                    },
                    Some(f) => f,
                };
                map.insert(file, readyState);
            }
        }

        for (file, readyState) in map.iter_mut() {
            let ready = file.Readiness(task, readyState.mask as u64);
            readyState.mask = ready as u32;
        }

        return map;
    }

    pub fn EventsAvailable(&self, task: &Task) -> bool {
        let readyMap = self.ReadyEvents(task);

        let mut lists = self.lists.lock();

        let mut it = lists.readyList.Front();
        while it.is_some() {
            let entry = it.unwrap();
            it = entry.Next();

            // If the entry is ready, we know 'e' has at least one entry
            // ready for delivery.
            let file = match entry.lock().id.File.Upgrade() {
                None => continue,
                Some(f) => f,
            };

            //let ready = file.Readiness(task, mask);
            let ready = match readyMap.get(&file) {
                None => {
                    // last get of readiness doesn't find the file
                    continue
                },
                Some(ready) => *ready,
            };

            if ready.mask != 0 {
                return true;
            }

            // there is no epoll_entry callback set it ready since last get readiness state
            if entry.lock().readyTimeStamp == ready.readyTimeStamp {
                lists.readyList.Remove(&entry);
                lists.waitingList.PushBack(&entry);
                entry.lock().state = PollEntryState::Waiting;
            }
        }

        return false;
    }

    pub fn ReadEvents(&self, task: &Task, max: i32) -> Vec<Event> {
        let mut lists = self.lists.lock();

        let mut local = PollEntryList::default();
        let mut ret = Vec::new();
        let mut it = lists.readyList.Front();
        while it.is_some() && ret.len() < max as usize {
            let entry = it.unwrap();
            it = entry.Next();

            // Check the entry's readiness. It it's not really ready, we
            // just put it back in the waiting list and move on to the next
            // entry.
            let id = entry.lock().id.clone();
            let file = match id.File.Upgrade() {
                None => {
                    lists.readyList.Remove(&entry);
                    continue;
                },
                Some(f) => f,
            };

            let ready = file.Readiness(task, entry.lock().mask);
            if ready == 0 {
                lists.readyList.Remove(&entry);
                lists.waitingList.PushBack(&entry);
                entry.lock().state = PollEntryState::Waiting;
                continue;
            }

            //let mask = entry.lock().mask;
            //error!("ReadEvents event fd is {}, ready is {:x}, mask is {:x}", entry.lock().id.Fd, ready, mask);
            // Add event to the array that will be returned to caller.
            ret.push(Event {
                Events: ready as u32,
                Data: entry.lock().userData,
            });

            // The entry is consumed, so we must move it to the disabled
            // list in case it's one-shot, or back to the wait list if it's
            // edge-triggered. If it's neither, we leave it in the ready
            // list so that its readiness can be checked the next time
            // around; however, we must move it to the end of the list so
            // that other events can be delivered as well.
            lists.readyList.Remove(&entry);
            let flags = entry.lock().flags;
            if flags & ONE_SHOT != 0 {
                lists.disabledList.PushBack(&entry);
                entry.lock().state = PollEntryState::Disabled;
            } else if flags & EDGE_TRIGGERED != 0 {
                lists.waitingList.PushBack(&entry);
                entry.lock().state = PollEntryState::Waiting;
            } else {
                entry.lock().state = PollEntryState::Ready;
                local.PushBack(&entry);
            }
        }

        lists.readyList.PushBackList(&mut local);
        return ret;
    }

    // ReadEvents returns up to max available events.
    pub fn ReadEvents1(&self, task: &Task, max: i32) -> Vec<Event> {
        let readyMap = self.ReadyEvents(task);

        let mut local = PollEntryList::default();
        let mut ret = Vec::new();

        let mut lists = self.lists.lock();

        let mut it = lists.readyList.Front();
        while it.is_some() && ret.len() < max as usize {
            let entry = it.unwrap();
            it = entry.Next();

            // Check the entry's readiness. It it's not really ready, we
            // just put it back in the waiting list and move on to the next
            // entry.
            let id = entry.lock().id.clone();
            let file = match id.File.Upgrade() {
                None => {
                    lists.readyList.Remove(&entry);
                    continue;
                },
                Some(f) => f,
            };

            let ready = match readyMap.get(&file) {
                None => {
                    // the file show up between ReadyEvents and ReadEvents, skip this now and waiting for next step process
                    continue
                },
                Some(ready) => {
                    *ready
                },
            };

            // the last get readiness state doesn't trigger the epoll and
            // there is no epoll_entry callback set it ready since last get readiness state
            if ready.mask == 0 && ready.readyTimeStamp == entry.lock().readyTimeStamp {
                lists.readyList.Remove(&entry);
                lists.waitingList.PushBack(&entry);
                entry.lock().state = PollEntryState::Waiting;
                continue;
            }

            // Add event to the array that will be returned to caller.
            ret.push(Event {
                Events: ready.mask as u32,
                Data: entry.lock().userData,
            });

            // The entry is consumed, so we must move it to the disabled
            // list in case it's one-shot, or back to the wait list if it's
            // edge-triggered. If it's neither, we leave it in the ready
            // list so that its readiness can be checked the next time
            // around; however, we must move it to the end of the list so
            // that other events can be delivered as well.
            lists.readyList.Remove(&entry);
            let flags = entry.lock().flags;
            if flags & ONE_SHOT != 0 {
                lists.disabledList.PushBack(&entry);
                entry.lock().state = PollEntryState::Disabled;
            } else if flags & EDGE_TRIGGERED != 0 {
                lists.waitingList.PushBack(&entry);
                entry.lock().state = PollEntryState::Waiting;
            } else {
                entry.lock().state = PollEntryState::Ready;
                local.PushBack(&entry);
            }
        }

        lists.readyList.PushBackList(&mut local);
        return ret;
    }

    // initEntryReadiness initializes the entry's state with regards to its
    // readiness by placing it in the appropriate list and registering for
    // notifications.
    pub fn InitEntryReadiness(&self, task: &Task, entry: &PollEntry) {
        let (f, mask) = {
            {
                let mut lists = self.lists.lock();
                lists.waitingList.PushBack(entry);
                entry.lock().state = PollEntryState::Waiting;
            }

            let f = match entry.lock().id.File.Upgrade() {
                None => return,
                Some(f) => f,
            };

            // Register for event notifications.
            let waiter = entry.lock().waiter.clone();
            let mask = entry.lock().mask;

            f.EventRegister(task, &waiter, mask);
            (f, mask)
        };

        // Check if the file happens to already be in a ready state.
        let ready = f.Readiness(task, mask) & mask;
        if ready != 0 {
            entry.CallBack();
        }
    }

    // observes checks if event poll object e is directly or indirectly observing
    // event poll object ep. It uses a bounded recursive depth-first search.
    pub fn Observes(&self, ep: &EventPoll, depthLeft: i32) -> bool {
        // If we reached the maximum depth, we'll consider that we found it
        // because we don't want to allow chains that are too long.
        if depthLeft < 0 {
            return true;
        }

        let files = self.files.lock();
        for (id, _) in files.iter() {
            let file = match id.File.Upgrade() {
                None => continue,
                Some(f) => f,
            };

            let fops = file.FileOp.clone();
            let epollOps = match fops.as_any().downcast_ref::<EventPoll>() {
                None => continue,
                Some(epollOps) => epollOps,
            };

            if *epollOps == *ep || epollOps.Observes(ep, depthLeft - 1) {
                return true
            }
        }

        return false;
    }

    // AddEntry adds a new file to the collection of files observed by e.
    pub fn AddEntry(&self, task: &Task, id: FileIdentifier, flags: EntryFlags, mask: EventMask, data: [i32; 2]) -> Result<()> {
        // Acquire cycle check lock if another event poll is being added.
        let file = match id.File.Upgrade() {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(f) => f,
        };

        let fops = file.FileOp.clone();
        let ep = fops.as_any().downcast_ref::<EventPoll>();

        let _lock = if ep.is_some() {
            Some(CYCLE_MU.lock())
        } else {
            None
        };

        let mut files = self.files.lock();
        if files.contains_key(&id) {
            return Err(Error::SysError(SysErr::EEXIST))
        }

        // Check if a cycle would be created. We use 4 as the limit because
        // that's the value used by linux and we want to emulate it.
        if ep.is_some() {
            let ep = ep.unwrap();
            if *ep == *self {
                return Err(Error::SysError(SysErr::EINVAL))
            }

            // Check if a cycle would be created. We use 4 as the limit because
            // that's the value used by linux and we want to emulate it.
            if ep.Observes(self, 4) {
                return Err(Error::SysError(SysErr::ELOOP))
            }
        }

        // Create new entry and add it to map.
        let entryInternal = PollEntryInternal {
            next: None,
            prev: None,

            id: id.clone(),
            userData: data,
            waiter: WaitEntry::New(),
            mask: mask,
            flags: flags,

            epoll: self.clone(),
            state: PollEntryState::Waiting,
            readyTimeStamp: 0,
        };

        let entry = PollEntry(Arc::new(QMutex::new(entryInternal)));
        entry.lock().waiter.lock().context = WaitContext::EpollContext(entry.clone());
        files.insert(id, entry.clone());

        // Initialize the readiness state of the new entry.
        self.InitEntryReadiness(task, &entry);

        return Ok(())
    }

    pub fn UpdateEntry(&self, task: &Task, id: &FileIdentifier, flags: EntryFlags, mask: EventMask, data: [i32; 2]) -> Result<()> {
        let files = self.files.lock();

        // Fail if the file doesn't have an entry.
        let entry = match files.get(id) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(e) => e.clone(),
        };

        // Unregister the old mask and remove entry from the list it's in, so
        // readyCallback is guaranteed to not be called on this entry anymore.
        let file = entry.lock().id.File.Upgrade();
        let waiter = entry.lock().waiter.clone();
        match file {
            None => {
                self.RemoveEntry(task, id)?;
                return Err(Error::SysError(SysErr::ENOENT));
            }
            Some(f) => {
                f.EventUnregister(task, &waiter);
            }
        }

        // Remove entry from whatever list it's in. This ensure that no other
        // threads have access to this entry as the only way left to find it
        // is via e.files, but we hold e.mu, which prevents that.
        {
            let mut lists = self.lists.lock();
            let state = entry.lock().state;
            let list = match state {
                PollEntryState::Ready => &mut lists.readyList,
                PollEntryState::Waiting => &mut lists.waitingList,
                PollEntryState::Disabled => &mut lists.disabledList,
            };

            list.Remove(&entry);
        }

        entry.lock().flags = flags;
        entry.lock().mask = mask;
        entry.lock().userData = data;

        self.InitEntryReadiness(task, &entry);

        return Ok(())
    }

    pub fn RemoveEntry(&self, task: &Task, id: &FileIdentifier) -> Result<()> {
        let mut files = self.files.lock();

        // Fail if the file doesn't have an entry.
        let entry = match files.get(id) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(e) => e.clone(),
        };

        // Unregister the old mask and remove entry from the list it's in, so
        // readyCallback is guaranteed to not be called on this entry anymore.
        let file = entry.lock().id.File.Upgrade();
        let waiter = entry.lock().waiter.clone();
        match file {
            None => (),
            Some(f) => f.EventUnregister(task, &waiter),
        }

        // Remove entry from whatever list it's in. This ensure that no other
        // threads have access to this entry as the only way left to find it
        // is via e.files, but we hold e.mu, which prevents that.
        {
            let mut lists = self.lists.lock();
            let state = entry.lock().state;
            let list = match state {
                PollEntryState::Ready => &mut lists.readyList,
                PollEntryState::Waiting => &mut lists.waitingList,
                PollEntryState::Disabled => &mut lists.disabledList,
            };

            list.Remove(&entry);
        }

        // Remove file from map, and drop weak reference.
        files.remove(id);

        return Ok(())
    }

    // UnregisterEpollWaiters removes the epoll waiter objects from the waiting
    // queues. This is different from Release() as the file is not dereferenced.
    pub fn UnregisterEpollWaiters(&self, task: &Task) {
        let files = self.files.lock();

        for (_, entry) in files.iter() {
            let file = entry.lock().id.File.Upgrade();
            let waiter = entry.lock().waiter.clone();

            match file {
                None => (),
                Some(f) => f.EventUnregister(task, &waiter),
            }
        }
    }
}

impl Drop for EventPollInternal {
    fn drop(&mut self) {
        for (id, entry) in self.files.lock().iter() {
            let waiter = entry.lock().waiter.clone();

            let task = Task::Current();
            match id.File.Upgrade() {
                None => (),
                Some(f) =>  f.EventUnregister(task, &waiter),
            }
        }
    }
}

impl Deref for EventPoll {
    type Target = Arc<EventPollInternal>;

    fn deref(&self) -> &Arc<EventPollInternal> {
        &self.0
    }
}

impl Waitable for EventPoll {
    // Readiness determines if the event poll object is currently readable (i.e.,
    // if there are pending events for delivery).
    fn Readiness(&self, task: &Task,mask: EventMask) -> EventMask {
        let mut ready = 0;

        if (mask & EVENT_IN) != 0 && self.EventsAvailable(task) {
            ready |= EVENT_IN
        };

        return ready;
    }

    fn EventRegister(&self, task: &Task,e: &WaitEntry, mask: EventMask) {
        self.queue.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task,e: &WaitEntry) {
        self.queue.EventUnregister(task, e)
    }
}

impl SpliceOperations for EventPoll {}

impl FileOperations for EventPoll {
    fn as_any(&self) -> &Any {
        self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::EventPoll
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE))
    }

    fn ReadDir(&self, _task: &Task, _f: &File, _offset: i64, _serializer: &mut DentrySerializer) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn ReadAt(&self, _task: &Task, _f: &File, _dsts: &mut [IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    fn WriteAt(&self, _task: &Task, _f: &File, _srcs: &[IoVec], _offset: i64, _blocking: bool) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS))
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0))
    }

    fn Fsync(&self, _task: &Task, _f: &File, _start: i64, _end: i64, _syncType: SyncType) -> Result<()> {
        return Ok(())
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(())
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);

    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<()> {
        return Err(Error::SysError(SysErr::ENOTTY))
    }

    fn IterateDir(&self, _task: &Task, _d: &Dirent, _dirCtx: &mut DirCtx, _offset: i32) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)))
    }

    fn Mappable(&self) -> Result<HostInodeOp> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

impl SockOperations for EventPoll {}