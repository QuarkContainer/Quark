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
use alloc::collections::linked_list::LinkedList;
use alloc::collections::btree_map::BTreeMap;
use alloc::collections::btree_set::BTreeSet;
use spin::Mutex;
use core::ops::Deref;
use core::any::Any;
use alloc::string::String;

use crate::qlib::mutex::*;
use crate::qlib::kernel::kernel::waiter::*;
use crate::qlib::kernel::fs::dentry::*;
use crate::qlib::kernel::fs::attr::UnstableAttr;
use crate::qlib::kernel::memmgr::vma::HostIopsMappable;
use super::super::task::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::uid::*;
use super::super::kernel::waiter::Queue;
use super::super::fs::inode::*;
use super::super::fs::dirent::*;
use super::file::*;

// inotifyEventBaseSize is the base size of linux's struct inotify_event. This
// must be a power 2 for rounding below.
pub const INOTIFY_EVENT_BASE_SIZE: usize = 16;

// Event represents a struct inotify_event from linux.
#[repr(C)]
#[derive(Debug)]
pub struct Event {
    pub wd: i32,
    pub mask: u32,
    pub cookie: u32,

    // len is computed based on the name field is set automatically by
    // Event.setName. It should be 0 when no name is set; otherwise it is the
    // length of the name slice.
    pub len: u32,

    // The name field has special padding requirements and should only be set by
    // calling Event.setName.
    pub name: Vec<u8>
}

// paddedBytes converts a go string to a null-terminated c-string, padded with
// null bytes to a total size of 'l'. 'l' must be large enough for all the bytes
// in the 's' plus at least one null byte.
pub fn PaddedBytes(s: &str, l: usize) -> Vec<u8> {
    if l < s.len() + 1 {
        panic!("Converting string to byte array results in truncation, this can lead to buffer-overflow due to the missing null-byte!")
    }

    let bytes = s.as_bytes();

    let mut b = Vec::with_capacity(l);
    b.resize(l, 0);
    for i in 0..bytes.len() {
        b[i] = bytes[i];
    }

    return b;

}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        let eq = self.wd == other.wd &&
            self.mask == other.mask &&
            self.cookie == other.cookie &&
            self.len == other.len;
        if !eq {
            return false
        }

        for i in 0..self.name.len() {
            if self.name[i] != other.name[i] {
                return false
            }
        }

        return true
    }
}

impl Event {
    pub fn New(wd: i32, name: &str, events: u32, cookie: u32) -> Self {
        let mut e = Event {
            wd: wd,
            mask: events,
            cookie: cookie,
            len: 0,
            name: Vec::new(),
        };

        if name.len() != 0 {
            e.SetName(name);
        }

        return e;
    }

    // setName sets the optional name for this event.
    pub fn SetName(&mut self, name: &str) {
        // We need to pad the name such that the entire event length ends up a
        // multiple of inotifyEventBaseSize.
        let unpaddedLen = name.len() + 1;
        // Round up to nearest multiple of inotifyEventBaseSize.
        self.len = ((unpaddedLen + INOTIFY_EVENT_BASE_SIZE - 1) & !(INOTIFY_EVENT_BASE_SIZE - 1)) as u32;
        // Make sure we haven't overflowed and wrapped around when rounding.
        if unpaddedLen > self.len as usize {
            panic!("Overflow when rounding inotify event size, the 'name' field was too big.")
        }
        self.name = PaddedBytes(name, self.len as usize);
    }

    pub fn Sizeof(&self) -> usize {
        let s = INOTIFY_EVENT_BASE_SIZE + self.len as usize;
        assert!(s>=INOTIFY_EVENT_BASE_SIZE);
        return s;
    }

    pub fn CopyOut(&self, task: &Task, addr: u64) -> Result<()> {
        task.CopyDataOut(self as *const _ as u64, addr, INOTIFY_EVENT_BASE_SIZE, false)?;
        if self.len > 0 {
            task.CopyOutSlice(&self.name, addr + INOTIFY_EVENT_BASE_SIZE as u64, self.len as usize)?;
        }
        return Ok(())
    }
}

// Watch represent a particular inotify watch created by inotify_add_watch.
//
// While a watch is active, it ensures the target inode is pinned in memory by
// holding an extra ref on each dirent known (by inotify) to point to the
// inode. These are known as pins.
pub struct WatchIntern {
    // Inotify instance which owns this watch.
    pub owner: Inotify,

    // Descriptor for this watch. This is unique across an inotify instance.
    pub wd: i32,

    // The inode being watched. Note that we don't directly hold a reference on
    // this inode. Instead we hold a reference on the dirent(s) containing the
    // inode, which we record in pins.
    pub target: InodeWeak,

    // unpinned indicates whether we have a hard reference on target. This field
    // may only be modified through atomic ops.
    pub unpinned: u32,

    // Events being monitored via this watch. Must be accessed atomically,
    // writes are protected by mu.
    pub mask: u32,

    // pins is the set of dirents this watch is currently pinning in memory by
    // holding a reference to them. See Pin()/Unpin().
    pub pins: BTreeSet<Dirent>,
}

#[derive(Clone)]
pub struct Watch(Arc<Mutex<WatchIntern>>);

impl Deref for Watch {
    type Target = Arc<Mutex<WatchIntern>>;

    fn deref(&self) -> &Arc<Mutex<WatchIntern>> {
        &self.0
    }
}

impl Watch {
    // ID returns the id of the inotify instance that owns this watch.
    pub fn Id(&self) -> u64 {
        return self.lock().owner.id;
    }

    pub fn ToString(&self) -> String {
        let w = self.lock();
        let mut output = format!("watch {}/{}:", w.owner.id, w.wd);
        for d in &w.pins {
            output = format!("{} {}", output, d.ID());
        }

        return output;
    }

    // NotifyParentAfterUnlink indicates whether the parent of the watched object
    // should continue to be be notified of events after the target has been
    // unlinked.
    pub fn NotifyParentAfterUnlink(&self) -> bool {
        return self.lock().mask & InotifyEvent::IN_EXCL_UNLINK == 0;
    }

    // Notify queues a new event on this watch.
    pub fn Notify(&self, name: &str, events: u32, cookie: u32) {
        let (owner, wd, matchedEvents) = {
            let w = self.lock();
            if w.mask & events == 0 {
                // We weren't watching for this event.
                return;
            }

            // Event mask should include bits matched from the watch plus all control
            // event bits.
            let unmaskableBits = !InotifyEvent::IN_ALL_EVENTS;
            let effectiveMask = unmaskableBits | w.mask;
            let matchedEvents = effectiveMask & events;
            (w.owner.clone(), w.wd, matchedEvents)
        };

        owner.QueueEvent(Event::New(wd, name, matchedEvents, cookie));
    }

    // Pin acquires a new ref on dirent, which pins the dirent in memory while
    // the watch is active. Calling Pin for a second time on the same dirent for
    // the same watch is a no-op.
    pub fn Pin(&self, d: &Dirent) {
        let mut w = self.lock();
        w.pins.insert(d.clone());
    }

    // Unpin drops any extra refs held on dirent due to a previous Pin
    // call. Calling Unpin multiple times for the same dirent, or on a dirent
    // without a corresponding Pin call is a no-op.
    pub fn Unpin(&self, d: &Dirent) {
        let mut w = self.lock();
        w.pins.remove(d);
    }

    pub fn TargetDestroyed(&self) {
        let owner = self.lock().owner.clone();
        owner.TargetDestroyed(self);
    }

    pub fn Destroy(&self) {
        self.lock().pins.clear()
    }
}

#[derive(Default)]
pub struct WatchesIntern {
    // ws is the map of active watches in this collection, keyed by the inotify
    // instance id of the owner.
    pub ws: BTreeMap<u64, Watch>,

    pub destroyed: bool,

    // unlinked indicates whether the target inode was ever unlinked. This is a
    // hack to figure out if we should queue a IN_DELETE_SELF event when this
    // watches collection is being destroyed, since otherwise we have no way of
    // knowing if the target inode is going down due to a deletion or
    // revalidation.
    pub unlinked: bool,
}

#[derive(Default, Clone)]
pub struct Watches(Arc<QRwLock<WatchesIntern>>);

impl Deref for Watches {
    type Target = Arc<QRwLock<WatchesIntern>>;

    fn deref(&self) -> &Arc<QRwLock<WatchesIntern>> {
        &self.0
    }
}

impl Watches {
    // MarkUnlinked indicates the target for this set of watches to be unlinked.
    // This has implications for the IN_EXCL_UNLINK flag.
    pub fn MarkUnlinked(&self) {
        self.write().unlinked = true;
    }

    // Lookup returns a matching watch with the given id. Returns nil if no such
    // watch exists. Note that the result returned by this method only remains valid
    // if the inotify instance owning the watch is locked, preventing modification
    // of the returned watch and preventing the replacement of the watch by another
    // one from the same instance (since there may be at most one watch per
    // instance, per target).
    pub fn Lookup(&self, id: u64) -> Option<Watch> {
        match self.write().ws.get(&id) {
            None => return None,
            Some(w) => return Some(w.clone())
        }
    }

    // Add adds watch into this set of watches. The watch being added must be unique
    // - its ID() should not collide with any existing watches.
    pub fn Add(&self, watch: &Watch) {
        let mut ws = self.write();
        let id = watch.Id();

        if ws.ws.contains_key(&id) {
            panic!("Watch collision with ID {}", id)
        }

        ws.ws.insert(id, watch.clone());
    }

    // Remove removes a watch with the given id from this set of watches. The caller
    // is responsible for generating any watch removal event, as appropriate. The
    // provided id must match an existing watch in this collection.
    pub fn Remove(&self, id: u64) {
        let mut ws = self.write();

        match ws.ws.remove(&id) {
            None => {
                // todo: how to handle None?
                if ws.destroyed {
                    return
                }

                // While there's technically no problem with silently ignoring a missing
                // watch, this is almost certainly a bug.
                panic!("Attempt to remove a watch, but no watch found with provided id {}.", id)
            }
            Some(_) => {
                return
            }
        }
    }

    // Notify queues a new event with all watches in this set.
    pub fn Notify(&self, name: &str, events: u32, cookie: u32) {
        let mut watchArr = Vec::new();
        {
            let ws = self.read();
            for (_, w) in &ws.ws {
                if name.len() != 0 && ws.unlinked && !w.NotifyParentAfterUnlink() {
                    // IN_EXCL_UNLINK - By default, when watching events on the children
                    // of a directory, events are generated for children even after they
                    // have been unlinked from the directory. This can result in large
                    // numbers of uninteresting events for some applications (e.g., if
                    // watching /tmp, in which many applications create temporary files
                    // whose names are immediately unlinked). Specifying IN_EXCL_UNLINK
                    // changes the default behavior, so that events are not generated
                    // for children after they have been unlinked from the watched
                    // directory.  -- inotify(7)
                    //
                    // We know we're dealing with events for a parent when the name
                    // isn't empty.
                    continue;
                }
                watchArr.push(w.clone());

            }
        }

        for w in &watchArr {
            w.Notify(name, events, cookie);
        }

    }

    // Unpin unpins dirent from all watches in this set.
    pub fn Unpin(&self, d: &Dirent) {
        let ws = self.read();
        for (_, watch) in &ws.ws {
            watch.Unpin(d)
        }
    }

    // targetDestroyed is called by the inode destructor to notify the watch owners
    // of the impending destruction of the watch target.
    pub fn TargetDestroyed(&self) {
        let watchArr : Vec<Watch>;
        {
            let mut ws = self.write();
            watchArr = ws.ws.values().cloned().collect();
            ws.ws.clear();
            ws.destroyed = true;
        }

        for watch in &watchArr {
            watch.TargetDestroyed();
        }
    }
}

pub struct WatchList {
    // The next watch descriptor number to use for this inotify instance. Note
    // that Linux starts numbering watch descriptors from 1.
    pub nextWatch: i32,
    pub watches: BTreeMap<i32, Watch>,
}

impl WatchList {
    pub fn New() -> Self {
        return Self {
            nextWatch: 1,
            watches: BTreeMap::new(),
        }
    }
}

pub struct InotifyIntern {
    // Unique identifier for this inotify instance. We don't just reuse the
    // inotify fd because fds can be duped. These should not be exposed to the
    // user, since we may aggressively reuse an id on S/R.
    pub id: u64,

    pub queue: Queue,

    // A list of pending events for this inotify instance. Protected by evMu.
    pub events: Mutex<LinkedList<Event>>,

    // Map from watch descriptors to watch objects.
    pub watches: Mutex<WatchList>
}

#[derive(Clone)]
pub struct Inotify(Arc<InotifyIntern>);

impl Deref for Inotify {
    type Target = Arc<InotifyIntern>;

    fn deref(&self) -> &Arc<InotifyIntern> {
        &self.0
    }
}

impl Drop for Inotify {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            self.Release();
        }
    }
}

impl Inotify {
    pub fn New() -> Self {
        let internl = InotifyIntern {
            id: NewUID(),
            queue: Queue::default(),
            events: Mutex::new(LinkedList::new()),
            watches: Mutex::new(WatchList::New())
        };
        return Self(Arc::new(internl));
    }

    pub fn Release(&self) {
        let ws = self.watches.lock();
        for (_, w) in &ws.watches {
            let inode = w.lock().target.Upgrade();
            match inode {
                None => (),
                Some(i) => i.Watches().Remove(w.Id())
            }
        }
    }

    pub fn QueueEvent(&self, ev: Event) {
        {
            let mut evs = self.events.lock();
            match evs.back() {
                None => (),
                Some(last) => {
                    if last == &ev {
                        return
                    }
                }
            }

            evs.push_back(ev);
        }

        self.queue.Notify(READABLE_EVENT)
    }

    // newWatchLocked creates and adds a new watch to target.
    pub fn NewWatchLocked(&self, target: &Dirent, mask: u32) -> Watch {
        let mut ws = self.watches.lock();
        let wd = ws.nextWatch;
        ws.nextWatch += 1;

        let watch = Watch(Arc::new(Mutex::new(WatchIntern {
            owner: self.clone(),
            wd: wd,
            target: target.Inode().Downgrade(),
            unpinned: 0,
            mask: mask,
            pins: BTreeSet::new(),
        })));

        ws.watches.insert(wd, watch.clone());

        // Grab an extra reference to target to prevent it from being evicted from
        // memory. This ref is dropped during either watch removal, target
        // destruction, or inotify instance destruction. See callers of Watch.Unpin.
        watch.Pin(target);
        target.Inode().Watches().Add(&watch);
        return watch
    }

    // targetDestroyed is called by w to notify i that w's target is gone. This
    // automatically generates a watch removal event.
    pub fn TargetDestroyed(&self, w: &Watch) {
        let found = {
            let mut ws = self.watches.lock();
            match ws.watches.remove(&w.lock().wd) {
                None => false,
                Some(_) => true
            }
        };

        if found {
            self.QueueEvent(Event::New(w.lock().wd, "", InotifyEvent::IN_IGNORED, 0))
        }
    }

    // AddWatch constructs a new inotify watch and adds it to the target dirent. It
    // returns the watch descriptor returned by inotify_add_watch(2).
    pub fn AddWatch(&self, target: &Dirent, mask: u32) -> i32 {
        // Note: Locking this inotify instance protects the result returned by
        // Lookup() below. With the lock held, we know for sure the lookup result
        // won't become stale because it's impossible for *this* instance to
        // add/remove watches on target.
        let _events = self.events.lock();

        let watch = target.Inode().Watches().Lookup(self.id);
        match watch {
            None => (),
            Some(w) => {
                // This may be a watch on a different dirent pointing to the
                // same inode. Obtain an extra reference if necessary.

                w.Pin(target);
                let mut newmask = mask;
                if (mask & InotifyEvent::IN_MASK_ADD) != 0 {
                    newmask |= w.lock().mask;
                }

                w.lock().mask = newmask;
                return w.lock().wd;
            }
        }

        // No existing watch, create a new watch.
        let watch = self.NewWatchLocked(target, mask);
        return watch.lock().wd;
    }

    // RmWatch implements watcher.Watchable.RmWatch.
    //
    // RmWatch looks up an inotify watch for the given 'wd' and configures the
    // target dirent to stop sending events to this inotify instance.
    pub fn RmWatch(&self, wd: i32) -> Result<()> {
        let watchId;
        let watch;
        {
            let _events = self.events.lock();

            watch = match self.watches.lock().watches.remove(&wd) {
                None => return Err(Error::SysError(SysErr::EINVAL)),
                Some(w) => w
            };

            let target = watch.lock().target.Upgrade();
            if let Some(target) = target {
                watchId = watch.Id();
                // Remove the watch from the watch target.
                target.Watches().Remove(watchId);
            }
        }

        self.QueueEvent(Event::New(watch.lock().wd, "", InotifyEvent::IN_IGNORED, 0));
        watch.Destroy();
        return Ok(())
    }
}

impl FileOperations for Inotify {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::InotifyFileOperations;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(&self, _task: &Task, _f: &File, _whence: i32, _current: i64, _offset: i64) -> Result<i64> {
        return Err(Error::SysError(SysErr::ESPIPE))
    }

    fn ReadDir(
        &self,
        _task: &Task,
        _f: &File,
        _offset: i64,
        _serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::ENOTDIR))
    }

    fn ReadAt(
        &self,
        task: &Task,
        _f: &File,
        dsts: &mut [IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        let dsts = task.AdjustIOVecPermission(dsts, true, true)?;
        let size = IoVec::NumBytes(&dsts);

        if size < INOTIFY_EVENT_BASE_SIZE {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let size = if size >= MemoryDef::HUGE_PAGE_SIZE as usize {
            MemoryDef::HUGE_PAGE_SIZE as usize
        } else {
            size
        };
        let mut buf = DataBuff::New(size);

        let mut events = self.events.lock();
        if events.len() == 0 {
            return Err(Error::SysError(SysErr::EAGAIN))
        }

        let mut slice = &mut buf.buf[0..size];

        let mut writelen = 0;
        loop {
            let event = match events.pop_front() {
                None => break,
                Some(e) => e,
            };

            // Does the buffer have enough remaining space to hold the event we're
            // about to write out?
            let len = event.Sizeof();
            if slice.len() < len {
                events.push_front(event);
                if writelen > 0 {
                    break
                }
                return Err(Error::SysError(SysErr::EINVAL))
            }

            event.CopyOut(task, &mut slice[0] as * mut _ as u64)?;
            writelen += len;
            slice = &mut slice[len..];
        }

        task.CopyDataOutToIovs(&buf.buf[0..writelen], &dsts, false)?;
        return Ok(writelen as i64);
    }

    fn WriteAt(
        &self,
        _task: &Task,
        _f: &File,
        _srcs: &[IoVec],
        _offset: i64,
        _blocking: bool,
    ) -> Result<i64> {
        return Err(Error::SysError(SysErr::EBADF))
    }

    fn Append(&self, _task: &Task, _f: &File, _srcs: &[IoVec]) -> Result<(i64, i64)> {
        return Err(Error::SysError(SysErr::EBADF))
    }

    fn Fsync(
        &self,
        _task: &Task,
        _f: &File,
        _start: i64,
        _end: i64,
        _syncType: SyncType,
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    fn Flush(&self, _task: &Task, _f: &File) -> Result<()> {
        return Ok(());
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        return f.Dirent.Inode().UnstableAttr(task)
    }

    fn Ioctl(&self, task: &Task, _f: &File, _fd: i32, request: u64, val: u64) -> Result<()> {
        match request {
            IoCtlCmd::FIONREAD => {
                let events = self.events.lock();
                loop {
                    let mut size : u32 = 0;
                    for event in events.iter() {
                        size += event.Sizeof() as u32;
                    }

                    task.CopyOutObj(&size, val)?;
                    return Ok(())
                }
            }
            _ => {
                return Err(Error::SysError(SysErr::ENOTTY))
            }
        }
    }

    fn IterateDir(
        &self,
        _task: &Task,
        _d: &Dirent,
        _dirCtx: &mut DirCtx,
        _offset: i32,
    ) -> (i32, Result<i64>) {
        return (0, Ok(0));
    }

    fn Mappable(&self) -> Result<HostIopsMappable> {
        return Err(Error::SysError(SysErr::ENODEV))
    }
}

impl Waitable for Inotify {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let ready = if self.events.lock().len() > 0 {
            READABLE_EVENT
        } else {
            0
        };

        return mask & ready;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.queue.clone();
        queue.EventRegister(task, e, mask);
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.queue.clone();
        queue.EventUnregister(task, e);
    }
}

impl SockOperations for Inotify {}
impl SpliceOperations for Inotify {}
