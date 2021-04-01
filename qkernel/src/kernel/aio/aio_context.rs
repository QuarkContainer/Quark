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

use alloc::collections::vec_deque::VecDeque;
use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use alloc::string::String;
use alloc::string::ToString;
use spin::Mutex;
use core::ops::Deref;

use super::super::super::qlib::addr::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::task::*;
use super::super::super::memmgr::*;
use super::super::super::memmgr::mm::*;
use super::super::waiter::*;

pub struct AIOMapping {}

impl Mapping for AIOMapping {
    fn MappedName(&self, _task: &Task) -> String {
        return "[aio]".to_string();
    }

    fn DeviceID(&self) -> u64 {
        return 0
    }

    fn InodeID(&self) -> u64 {
        return 0;
    }
}

#[derive(Default)]
pub struct AIOManagerIntern {
    pub contexts: BTreeMap<u64, AIOContext>,
}

#[derive(Default, Clone)]
pub struct AIOManager(Arc<Mutex<AIOManagerIntern>>);

impl Deref for AIOManager {
    type Target = Arc<Mutex<AIOManagerIntern>>;

    fn deref(&self) -> &Arc<Mutex<AIOManagerIntern>> {
        &self.0
    }
}

impl AIOManager {
    pub fn Destroy(&self) {
        let a = self.lock();
        for (_, ctx) in &a.contexts {
            ctx.Destroy()
        }
    }

    // newAIOContext creates a new context for asynchronous I/O.
    //
    // Returns false if 'id' is currently in use.
    pub fn NewAIOContext(&self, events: usize, id: u64) -> bool {
        let mut a = self.lock();

        if a.contexts.contains_key(&id) {
            return false;
        }

        a.contexts.insert(id, AIOContext::New(events));

        return true;
    }

    // destroyAIOContext destroys an asynchronous I/O context.
    //
    // False is returned if the context does not exist.
    pub fn DestroyAIOContext(&self, id: u64) -> bool {
        let mut a = self.lock();

        let context = match a.contexts.remove(&id) {
            None => return false,
            Some(c) => c,
        };

        context.Destroy();
        return true;
    }

    // lookupAIOContext looks up the given context.
    //
    // Returns false if context does not exist.
    pub fn LookupAIOContext(&self, id: u64) -> Option<AIOContext> {
        let a = self.lock();
        match a.contexts.get(&id) {
            None => None,
            Some(c) => Some(c.clone())
        }
    }
}

impl MemoryManager {
    // NewAIOContext creates a new context for asynchronous I/O.
    //
    // NewAIOContext is analogous to Linux's fs/aio.c:ioctx_alloc().
    pub fn NewAIOContext(&self, task: &Task, events: usize) -> Result<u64> {
        let mut opts = MMapOpts {
            Length: AIOContext::AIO_RINGBUF_SIZE,
            Addr: 0,
            Offset: 0,
            Fixed: false,
            Unmap: false,
            Map32Bit: false,
            Private: true,
            VDSO: false,
            Perms: AccessType::ReadOnly(),
            MaxPerms: AccessType::ReadOnly(),
            GrowsDown: false,
            Precommit: false,
            MLockMode: 0,
            Kernel: false,
            Mapping: Some(Arc::new(AIOMapping{})),
            Mappable: None,
            Hint: "".to_string(),
        };

        let addr = match self.MMap(task, &mut opts) {
            Ok(addr) => addr,
            Err(e) => return Err(e),
        };

        let id = addr;
        let ret = self.write().aioManager.NewAIOContext(events, id);
        if !ret {
            self.MUnmap(task, addr, AIOContext::AIO_RINGBUF_SIZE)?;
            return Err(Error::SysError(SysErr::EINVAL))
        }

        return Ok(id)
    }

    // DestroyAIOContext destroys an asynchronous I/O context. It returns false if
    // the context does not exist.
    pub fn DestroyAIOContext(&self, task: &Task, id: u64) -> bool {
        let _aioCtx = match self.LookupAIOContext(task, id) {
            None => return false,
            Some(c) => c,
        };

        // Only unmaps after it assured that the address is a valid aio context to
        // prevent random memory from been unmapped.
        //
        // Note: It's possible to unmap this address and map something else into
        // the same address. Then it would be unmapping memory that it doesn't own.
        // This is, however, the way Linux implements AIO. Keeps the same [weird]
        // semantics in case anyone relies on it.

        self.MUnmap(task, id, AIOContext::AIO_RINGBUF_SIZE).ok();
        return self.read().aioManager.DestroyAIOContext(id);
    }

    pub fn LookupAIOContext(&self, task: &Task, id: u64) -> Option<AIOContext> {
        let aioCtx = match self.read().aioManager.LookupAIOContext(id) {
            None => return None,
            Some(c) => c,
        };

        // Protect against 'ids' that are inaccessible (Linux also reads 4 bytes
        // from id).
        let _buf : [u8; 4] = *match task.GetType(id) {
            Err(_) => return None,
            Ok(t) => t,
        };

        return Some(aioCtx)
    }
}


// I/O commands.
pub const IOCB_CMD_PREAD   : u16 = 0;
pub const IOCB_CMD_PWRITE  : u16 = 1;
pub const IOCB_CMD_FSYNC   : u16 = 2;
pub const IOCB_CMD_FDSYNC  : u16 = 3;
pub const IOCB_CMD_NOOP    : u16 = 6;
pub const IOCB_CMD_PREADV  : u16 = 7;
pub const IOCB_CMD_PWRITEV : u16 = 8;

// I/O flags.
pub const IOCB_FLAG_RESFD : i32 = 1;

// ioCallback describes an I/O request.
//
// The priority field is currently ignored in the implementation below.
#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
pub struct IOCallback {
    pub data: u64,
    pub key: u32,
    pub rw_flags: i32,

    pub opcode: u16,
    pub reqprio: i16,
    pub fd: u32,

    pub buf: u64,
    pub bytes: u64,
    pub offset: i64,

    pub reserved2: u64,
    pub flags: u32,

    // eventfd to signal if IOCB_FLAG_RESFD is set in flags.
    pub resfd: u32,
}

// ioEvent describes an I/O result.
#[repr(C)]
#[derive(Default, Debug)]
pub struct IOEvent {
    pub data    : u64,
    pub obj     : u64,
    pub result  : i64,
    pub result2 : i64,
}

pub const IOEVENT_SIZE : u64 = 32; // sizeof(IOEvent)

#[derive(Default)]
pub struct AIOContextIntern {
    // results is the set of completed requests.
    pub results: VecDeque<IOEvent>,

    // maxOutstanding is the maximum number of outstanding entries; this value
    // is immutable.
    pub maxOutstanding: usize,

    pub outstanding: usize,

    // dead is set when the context is destroyed.
    pub dead: bool,

    pub queue: Queue,
}

#[derive(Default, Clone)]
pub struct AIOContext(Arc<Mutex<AIOContextIntern>>);

impl Deref for AIOContext {
    type Target = Arc<Mutex<AIOContextIntern>>;

    fn deref(&self) -> &Arc<Mutex<AIOContextIntern>> {
        &self.0
    }
}

impl Waitable for AIOContext {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let mut currentMask = 0;
        let aio = self.lock();
        if aio.dead {
            currentMask |= EVENT_HUP;
        }

        if aio.results.len() > 0 {
            currentMask |= EVENT_IN;
        }

        return mask & currentMask;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let aio = self.lock();
        aio.queue.EventRegister(task, e, mask);
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let aio = self.lock();
        aio.queue.EventUnregister(task, e);
    }
}

impl AIOContext {
    pub const AIO_RING_SIZE : u64 = 32;
    pub const AIO_RINGBUF_SIZE : u64 = MemoryDef::PAGE_SIZE; //Addr(Self::AIO_RING_SIZE).RoundUp().unwrap().0;

    pub fn New(events: usize) -> Self {
        let intern = AIOContextIntern {
            maxOutstanding: events,
            ..Default::default()
        };

        return Self(Arc::new(Mutex::new(intern)))
    }

    // destroy marks the context dead.
    pub fn Destroy(&self) {
        let mut aio = self.lock();
        aio.dead = true;
        aio.queue.Notify(EVENT_HUP);
    }


    // Prepare reserves space for a new request, returning true if available.
    // Returns false if the context is busy.
    pub fn Prepare(&self) -> bool {
        let mut aio = self.lock();
        if aio.outstanding >= aio.maxOutstanding {
            return false;
        }

        aio.outstanding += 1;

        return true
    }


    // PopRequest pops a completed request if available, this function does not do
    // any blocking. Returns false if no request is available.
    pub fn PopRequest(&self) -> Option<IOEvent> {
        let mut aio = self.lock();
        let ret = aio.results.pop_front();
        if aio.results.len() == 0 && aio.dead {
            aio.queue.Notify(EVENT_HUP);
        }

        if ret.is_some() {
            aio.outstanding -= 1;
        }

        return ret;
    }

    // FinishRequest finishes a pending request. It queues up the data
    // and notifies listeners.
    pub fn FinishRequest(&self, data: IOEvent) {
        let mut aio = self.lock();
        aio.results.push_back(data);

        use alloc::vec::Vec;
        let mut v = Vec::new();
        for r in &aio.results {
            v.push(r.obj);
        }

        aio.queue.Notify(EVENT_IN);
    }
}