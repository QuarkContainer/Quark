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

use crate::qlib::mem::stackvec::StackVec;
use crate::qlib::mutex::*;
//use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use core::any::Any;
use core::hash::BuildHasherDefault;
use core::ops::Deref;
use hashbrown::hash_map::Entry;
use hashbrown::HashMap;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::singleton::*;
use super::super::super::fs::anon::*;
use super::super::super::fs::attr::*;
use super::super::super::fs::dentry::*;
use super::super::super::fs::dirent::*;
use super::super::super::fs::file::*;
use super::super::super::fs::flags::*;
use super::super::super::fs::host::hostinodeop::*;
use super::super::super::task::*;
use super::super::waiter::*;
use super::epoll_entry::*;
use super::epoll_list::*;

use crate::GUEST_HOST_SHARED_ALLOCATOR;
use crate::GuestHostSharedAllocator;

pub static CYCLE_MU: Singleton<QMutex<()>> = Singleton::<QMutex<()>>::New();
pub unsafe fn InitSingleton() {
    CYCLE_MU.Init(QMutex::new(()));
}

// Event describes the event mask that was observed and the user data to be
// returned when one of the events occurs. It has this format to match the linux
// format to avoid extra copying/allocation when writing events to userspace.
#[cfg(target_arch = "x86_64")]
#[derive(Default, Clone, Copy, Debug)]
#[repr(packed)]
#[repr(C)]
pub struct Event {
    // Events is the event mask containing the set of events that have been
    // observed on an entry.
    pub Events: u32,

    // Data is an opaque 64-bit value provided by the caller when adding the
    // entry, and returned to the caller when the entry reports an event.
    pub Data: u64,
}

#[cfg(target_arch = "aarch64")]
#[derive(Default, Clone, Copy, Debug)]
#[repr(packed)]
#[repr(C)]
pub struct Event {
    // Events is the event mask containing the set of events that have been
    // observed on an entry.
    pub Events: u32,

    _pad: i32,
    // Data is an opaque 64-bit value provided by the caller when adding the
    // entry, and returned to the caller when the entry reports an event.
    pub Data: u64,
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
}

impl PollLists {
    pub fn ToString(&self) -> String {
        let mut str = "ready: ".to_string();
        str += &self.readyList.GetString();
        return str;
    }
}

#[derive(Default)]
pub struct RawHasher {
    state: u64,
}

impl core::hash::Hasher for RawHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state = self.state.rotate_left(8) ^ u64::from(byte);
        }
    }

    fn write_u64(&mut self, i: u64) {
        self.state = i;
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

// EventPoll holds all the state associated with an event poll object, that is,
// collection of files to observe and their current state.
pub struct EventPollInternal {
    pub queue: Queue,
    pub files: QMutex<HashMap<FileIdentifier, PollEntry, BuildHasherDefault<RawHasher>>>,

    pub lists: QMutex<PollEntryList>,
}

impl Default for EventPollInternal {
    fn default() -> Self {
        let hash_map =
            HashMap::<FileIdentifier, PollEntry, BuildHasherDefault<RawHasher>>::default();
        return Self {
            queue: Queue::default(),
            files: QMutex::new(hash_map),
            lists: Default::default(),
        };
    }
}

// NewEventPoll allocates and initializes a new event poll object.
pub fn NewEventPoll(task: &Task) -> File {
    let inode = NewAnonInode(task);
    let dirent = Dirent::New(&inode, "anon_inode:[eventpoll]");
    let epoll = EventPoll::default();
    return File::New(&dirent, &FileFlags::default(), epoll.into());
}

#[derive(Clone)]
pub struct EventPoll(Arc<EventPollInternal, GuestHostSharedAllocator>);

impl Default for EventPoll {
    fn default() -> Self {
        Self(Arc::new_in(
            EventPollInternal::default(),
            GUEST_HOST_SHARED_ALLOCATOR,
        ))
    }
}

impl PartialEq for EventPoll {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl EventPoll {
    // eventsAvailable determines if 'e' has events available for delivery.
    pub fn EventsAvailable(&self, task: &Task) -> bool {
        let mut lists = self.lists.lock();
        let mut it = lists.Front();
        while it.is_some() {
            let entry = it.unwrap();
            it = entry.Next();

            let file = entry.lock().file.clone();
            let file = match file.Upgrade() {
                None => {
                    // the file has been dropped, just remove it
                    lists.Remove(&entry);
                    continue;
                }
                Some(f) => f,
            };

            let mask = entry.lock().mask;
            let ready = file.Readiness(task, mask);
            if ready != 0 {
                return true;
            }

            lists.Remove(&entry);
            entry.lock().state = PollEntryState::Waiting;
        }

        return false;
    }

    pub fn ReadEvents(&self, task: &Task, max: i32, events: &mut StackVec<Event, 64>) {
        let mut lists = self.lists.lock();

        let mut it = lists.Front();
        while events.Len() < max as usize {
            let entry = if let Some(entry) = it {
                entry
            } else {
                break;
            };
            it = entry.Next();

            // Check the entry's readiness. It it's not really ready, we
            // just put it back in the waiting list and move on to the next
            // entry.
            let file = entry.lock().file.Upgrade();
            let file = match file {
                None => {
                    lists.Remove(&entry);
                    continue;
                }
                Some(f) => f,
            };

            let mask = entry.lock().mask;
            let ready = file.Readiness(task, mask);
            if ready == 0 {
                lists.Remove(&entry);
                entry.lock().state = PollEntryState::Waiting;
                continue;
            }

            //let mask = entry.lock().mask;
            //error!("ReadEvents event fd is {}, ready is {:x}, mask is {:x}", entry.lock().id.Fd, ready, mask);
            // Add event to the array that will be returned to caller.
            #[cfg(target_arch = "aarch64")]
            events.Push(Event {
                Events: ready as u32,
                _pad: 0,
                Data: entry.lock().userData,
            });
            #[cfg(target_arch = "x86_64")]
            events.Push(Event {
                Events: ready as u32,
                Data: entry.lock().userData,
            });

            // The entry is consumed, so we must move it to the disabled
            // list in case it's one-shot, or back to the wait list if it's
            // edge-triggered. If it's neither, we leave it in the ready
            // list so that its readiness can be checked the next time
            // around; however, we must move it to the end of the list so
            // that other events can be delivered as well.
            let flags = entry.lock().flags;
            if flags & ONE_SHOT != 0 {
                lists.Remove(&entry);
                entry.lock().state = PollEntryState::Disabled;
            } else if flags & EDGE_TRIGGERED != 0 {
                lists.Remove(&entry);
                entry.lock().state = PollEntryState::Waiting;
            }
        }
    }

    // initEntryReadiness initializes the entry's state with regards to its
    // readiness by placing it in the appropriate list and registering for
    // notifications.
    pub fn InitEntryReadiness(&self, task: &Task, f: &File, entry: &PollEntry) {
        let mask = {
            // Register for event notifications.
            let waiter = entry.lock().waiter.Upgrade().unwrap();
            let mask = entry.lock().mask;

            f.EventRegister(task, &waiter, mask);
            mask
        };

        // Check if the file happens to already be in a ready state.
        let ready = f.Readiness(task, mask);
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
        for (_, entry) in files.iter() {
            let file = entry.lock().file.Upgrade();
            let file = match file {
                None => continue,
                Some(f) => f,
            };

            let fops = file.FileOp.clone();
            let epollOps = match fops.as_any().downcast_ref::<EventPoll>() {
                None => continue,
                Some(epollOps) => epollOps,
            };

            if *epollOps == *ep || epollOps.Observes(ep, depthLeft - 1) {
                return true;
            }
        }

        return false;
    }

    // AddEntry adds a new file to the collection of files observed by e.
    pub fn AddEntry(
        &self,
        task: &Task,
        file: File,
        flags: EntryFlags,
        mask: EventMask,
        data: u64,
    ) -> Result<()> {
        let id = file.UniqueId();
        let fops = file.FileOp.clone();
        let ep = fops.EventPoll(); // fops.as_any().downcast_ref::<EventPoll>();

        let _lock = if ep.is_some() {
            Some(CYCLE_MU.lock())
        } else {
            None
        };

        let mut files = self.files.lock();

        let entry = files.entry(id);

        match entry {
            Entry::Occupied(_) => {
                return Err(Error::SysError(SysErr::EEXIST));
            }
            Entry::Vacant(e) => {
                // Check if a cycle would be created. We use 4 as the limit because
                // that's the value used by linux and we want to emulate it.
                if let Some(ep) = ep {
                    if ep == *self {
                        return Err(Error::SysError(SysErr::EINVAL));
                    }

                    // Check if a cycle would be created. We use 4 as the limit because
                    // that's the value used by linux and we want to emulate it.
                    if ep.Observes(self, 4) {
                        return Err(Error::SysError(SysErr::ELOOP));
                    }
                }

                let waiter = WaitEntry::New();
                // Create new entry and add it to map.
                let entryInternal = PollEntryInternal {
                    next: None,
                    prev: None,
                    id: id,
                    file: file.Downgrade(),
                    userData: data,
                    waiter: waiter.Downgrade(),
                    mask: mask,
                    flags: flags,

                    epoll: self.clone(),
                    state: PollEntryState::Waiting,
                };

                let entry = PollEntry(Arc::new_in(
                    QMutex::new(entryInternal),
                    crate::GUEST_HOST_SHARED_ALLOCATOR,
                ));
                waiter.lock().context = WaitContext::EpollContext(entry.clone());
                e.insert(entry.clone());

                file.EventRegister(task, &waiter, mask);

                // Check if the file happens to already be in a ready state.
                let ready = file.Readiness(task, mask);
                if ready != 0 {
                    entry.CallBack();
                }

                // need drop files before return
                drop(files);
                return Ok(());
            }
        }
    }

    pub fn UpdateEntry(
        &self,
        task: &Task,
        file: File,
        flags: EntryFlags,
        mask: EventMask,
        data: u64,
    ) -> Result<()> {
        let id = file.UniqueId();
        let files = self.files.lock();

        // Fail if the file doesn't have an entry.
        let entry = match files.get(&id) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(e) => e.clone(),
        };

        // Unregister the old mask and remove entry from the list it's in, so
        // readyCallback is guaranteed to not be called on this entry anymore.
        let waiter = entry.lock().waiter.Upgrade().unwrap();
        file.EventUnregister(task, &waiter);

        // Remove entry from whatever list it's in. This ensure that no other
        // threads have access to this entry as the only way left to find it
        // is via e.files, but we hold e.mu, which prevents that.
        {
            let mut lists = self.lists.lock();
            let state = entry.lock().state;
            match state {
                PollEntryState::Ready => lists.Remove(&entry),
                PollEntryState::Waiting => (),
                PollEntryState::Disabled => (),
            };

            let mut entryLock = entry.lock();
            entryLock.flags = flags;
            entryLock.mask = mask;
            entryLock.userData = data;
            entryLock.state = PollEntryState::Waiting;
        }

        file.EventRegister(task, &waiter, mask);

        // Check if the file happens to already be in a ready state.
        let ready = file.Readiness(task, mask);
        if ready != 0 {
            entry.CallBack();
        }

        return Ok(());
    }

    pub fn RemoveEntry(&self, task: &Task, file: File) -> Result<()> {
        let id = file.UniqueId();

        // Fail if the file doesn't have an entry.
        let entry = match self.files.lock().remove(&id) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(e) => e,
        };

        // Unregister the old mask and remove entry from the list it's in, so
        // readyCallback is guaranteed to not be called on this entry anymore.
        let waiter = entry.lock().waiter.Upgrade().unwrap();
        file.EventUnregister(task, &waiter);

        // Remove entry from whatever list it's in. This ensure that no other
        // threads have access to this entry as the only way left to find it
        // is via e.files, but we hold e.mu, which prevents that.
        {
            let mut lists = self.lists.lock();
            let state = entry.lock().state;
            match state {
                PollEntryState::Ready => lists.Remove(&entry),
                PollEntryState::Waiting => (),
                PollEntryState::Disabled => (),
            };
        }

        return Ok(());
    }
}

impl Drop for EventPollInternal {
    fn drop(&mut self) {
        let mut lists = self.lists.lock();
        for (_, entry) in self.files.lock().iter() {
            let task = Task::Current();
            let file = entry.lock().file.Upgrade();
            match file {
                None => (),
                Some(f) => {
                    let waiter = entry.lock().waiter.Upgrade().unwrap();
                    f.EventUnregister(task, &waiter);
                }
            }

            lists.Remove(entry);
        }
    }
}

impl Deref for EventPoll {
    type Target = Arc<EventPollInternal, GuestHostSharedAllocator>;

    fn deref(&self) -> &Arc<EventPollInternal, GuestHostSharedAllocator> {
        &self.0
    }
}

impl Waitable for EventPoll {
    // Readiness determines if the event poll object is currently readable (i.e.,
    // if there are pending events for delivery).
    fn Readiness(&self, task: &Task, mask: EventMask) -> EventMask {
        let mut ready = 0;

        if (mask & READABLE_EVENT) != 0 && self.EventsAvailable(task) {
            ready |= READABLE_EVENT
        };

        return ready;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        self.queue.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        self.queue.EventUnregister(task, e)
    }
}

impl SpliceOperations for EventPoll {}

impl FileOperations for EventPoll {
    fn as_any(&self) -> &Any {
        self
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::EventPoll;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(
        &self,
        _task: &Task,
        _f: &File,
        _whence: i32,
        _current: i64,
        _offset: i64,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE));
    }

    fn ReadDir(
        &self,
        _task: &Task,
        _f: &File,
        _offset: i64,
        _serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    fn ReadAt(
        &self,
        _task: &Task,
        _f: &File,
        _dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOSYS));
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        let n = self.WriteAt(task, f, srcs, 0, false)?;
        return Ok((n, 0));
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
        return Ok(());
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let inode = f.Dirent.Inode();
        return inode.UnstableAttr(task);
    }

    fn Ioctl(&self, _task: &Task, _f: &File, _fd: i32, _request: u64, _val: u64) -> Result<u64> {
        return Err(Error::SysError(SysErr::ENOTTY));
    }

    fn IterateDir(
        &self,
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for EventPoll {}
