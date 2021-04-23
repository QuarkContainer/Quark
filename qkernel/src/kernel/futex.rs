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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use lazy_static::lazy_static;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::linux::futex::*;
use super::super::task::*;
use super::super::kernel::waiter::*;

lazy_static! {
    pub static ref FUTEX_MGR : FutexMgr = FutexMgr::default();
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
#[repr(i32)]
pub enum KeyKind {
    // KindPrivate indicates a private futex (a futex syscall with the
    // FUTEX_PRIVATE_FLAG set).
    KindPrivate = 0,

    // KindSharedPrivate indicates a shared futex on a private memory mapping.
    // Although KindPrivate and KindSharedPrivate futexes both use memory
    // addresses to identify futexes, they do not interoperate (in Linux, the
    // two are distinguished by the FUT_OFF_MMSHARED flag, which is used in key
    // comparison).
    KindSharedPrivate,

    // KindSharedMappable indicates a shared futex on a memory mapping other
    // than a private anonymous memory mapping.
    KindSharedMappable,
}

impl Default for KeyKind {
    fn default() -> Self {
        return Self::KindPrivate
    }
}

// Key represents something that a futex waiter may wait on.
#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Default)]
pub struct Key {
    // Kind is the type of the Key.
    pub Kind: KeyKind,

    // If Kind is KindPrivate or KindSharedPrivate, addr is the represented
    // physical memory address. Otherwise, is the user address
    pub Addr: u64,
}

pub const BUCKET_COUNT: usize = 1 << BUCKET_COUNT_BITS;
pub const BUCKET_COUNT_BITS: usize = 10;

impl Key {
    pub fn HashCode(&self) -> usize {
        let addr = self.Addr as usize;
        let h1 = (addr >> 2) + (addr >> 12) + (addr >> 22);
        let h2 = (addr >> 32) + (addr >> 42);
        return (h1 + h2) % BUCKET_COUNT;
    }
}

// Target abstracts memory accesses and keys.
pub trait Target {
    fn SwapU32(&self, addr: u64, new: u32) -> Result<u32>;
    fn CompareAndSwapU32(&self, addr: u64, old: u32, new: u32) -> Result<u32>;
    fn LoadU32(&self, addr: u64) -> Result<u32>;

    // GetSharedKey returns a Key with kind KindSharedPrivate or
    // KindSharedMappable corresponding to the memory mapped at address addr.
    //
    // If GetSharedKey returns a Key with a non-nil MappingIdentity, a
    // reference is held on the MappingIdentity, which must be dropped by the
    // caller when the Key is no longer in use.
    fn GetSharedKey(&self, addr: u64) -> Result<Key>;
}

impl Target for Task {
    fn SwapU32(&self, addr: u64, new: u32) -> Result<u32> {
        let val = self.GetTypeMut::<AtomicU32>(addr)?;

        val.swap(new, Ordering::SeqCst);
        return Ok(new)
    }

    fn CompareAndSwapU32(&self, addr: u64, old: u32, new: u32) -> Result<u32> {
        let pval = self.GetTypeMut::<AtomicU32>(addr)?;
        match pval.compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(v) => return Ok(v),
            Err(v) => return Ok(v),
        }
    }

    fn LoadU32(&self, addr: u64) -> Result<u32> {
        let val = self.GetType::<u32>(addr)?;

        return Ok(*val)
    }

    fn GetSharedKey(&self, addr: u64) -> Result<Key> {
        /*let vmas = self.mm.vmas.clone();
        let vma = match vmas.lock().Get(addr) {
            None => return Err(Error::SysError(SysErr::EFAULT)),
            Some(v) => v.clone(),
        };*/
        let vma = match self.mm.GetVma(addr) {
            None => return Err(Error::SysError(SysErr::EFAULT)),
            Some(v) => v.clone(),
        };

        let private = vma.private;
        if private {
            return Ok(Key {
                Kind: KeyKind::KindSharedPrivate,
                Addr: addr,
            })
        }

        let phyAdr = self.VirtualToPhy(addr)?;

        return Ok(Key {
            Kind: KeyKind::KindSharedMappable,
            Addr: phyAdr,
        })
    }
}

// check performs a basic equality check on the given address.
fn Check(t: &Target, addr: u64, val: u32) -> Result<()> {
    let cur = t.LoadU32(addr)?;

    if cur != val {
        return Err(Error::SysError(SysErr::EAGAIN));
    }

    return Ok(())
}

// atomicOp performs a complex operation on the given address.
fn AtomicOp(t: &Target, addr: u64, opIn: u32) -> Result<bool> {
    let mut opType = (opIn >> 28) & 0xf;
    let cmp = (opIn >> 24) & 0xf;
    let mut opArg = (opIn >> 12) & 0xfff;
    let cmpArg = opIn & 0xfff;

    if opType & FUTEX_OP_OPARG_SHIFT != 0 {
        opArg = 1 << opArg;
        opType &= !FUTEX_OP_OPARG_SHIFT; // Clear flag.
    }

    let mut oldVal;

    if opType == FUTEX_OP_SET {
        oldVal = t.SwapU32(addr, opArg)?;
    } else {
        loop {
            oldVal = t.LoadU32(addr)?;

            let newVal = match opType {
                FUTEX_OP_ADD => oldVal + opArg,
                FUTEX_OP_OR => oldVal | opArg,
                FUTEX_OP_ANDN => oldVal & !opArg,
                FUTEX_OP_XOR => oldVal ^ opArg,
                _ => return Err(Error::SysError(SysErr::ENOSYS))
            };

            let prev = t.CompareAndSwapU32(addr, oldVal, newVal)?;

            if prev == oldVal {
                break;
            }
        }
    }

    match cmp {
        FUTEX_OP_CMP_EQ => return Ok(oldVal == cmpArg),
        FUTEX_OP_CMP_NE => return Ok(oldVal != cmpArg),
        FUTEX_OP_CMP_LT => return Ok(oldVal < cmpArg),
        FUTEX_OP_CMP_LE => return Ok(oldVal <= cmpArg),
        FUTEX_OP_CMP_GT => return Ok(oldVal < cmpArg),
        FUTEX_OP_CMP_GE => return Ok(oldVal >= cmpArg),
        _ => return Err(Error::SysError(SysErr::ENOSYS)),
    }
}

// getKey returns a Key representing address addr in c.
fn Getkey(t: &Target, addr: u64, private: bool) -> Result<Key> {
    // Ensure the address is aligned.
    // It must be a DWORD boundary.
    if addr & 0x3 != 0 {
        return Err(Error::SysError(SysErr::EINVAL));
    }

    if private {
        return Ok(Key {
            Kind: KeyKind::KindPrivate,
            Addr: addr,
        })
    }

    return t.GetSharedKey(addr)
}

pub struct SpinLock {
    //0: unlock, 1:locked
    lock: AtomicU8,
}

impl Default for SpinLock {
    fn default() -> Self {
        return Self {
            lock: AtomicU8::new(0),
        }
    }
}

impl SpinLock {
    pub fn Lock(&self) {
        loop {
            let old = self.lock.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);
            if old == Ok(0) {
                break;
            }
        }
    }

    pub fn Unlock(&self) {
        self.lock.store(0, Ordering::SeqCst);
    }
}

// bucket holds a list of waiters
pub struct BucketInternal {
    pub map: Mutex<BTreeMap<Key, Queue>>,
    pub locks: Vec<SpinLock>,
}

impl Default for BucketInternal {
    fn default() -> Self {
        let mut ret = Self {
            map: Mutex::new(BTreeMap::new()),
            locks: Vec::with_capacity(BUCKET_COUNT),
        };

        for _i in 0..BUCKET_COUNT {
            ret.locks.push(SpinLock::default())
        }

        return ret;
    }
}

impl BucketInternal {
    pub fn Lock(&self, k: &Key) {
        let idx = k.HashCode();
        self.locks[idx].Lock();
    }

    pub fn Unlock(&self, k: &Key) {
        let idx = k.HashCode();
        self.locks[idx].Unlock();
    }

    pub fn GetQueue(&self, k: &Key) -> Option<Queue> {
        let map = self.map.lock();
        match map.get(k).clone() {
            None => return None,
            Some(q) => return Some(q.clone()),
        };
    }

    pub fn GetCreateQueues(&self, k: &Key) -> Queue {
        let mut map = self.map.lock();
        if !map.contains_key(k) {
            let newQ = Queue::default();
            map.insert(k.clone(), newQ);
        }

        match map.get(k).clone() {
            Some(q) => return q.clone(),
            None => panic!("impossible"),
        };
    }

    //hash lock must be locked before remove
    pub fn RemoveQueue(&self, k: &Key) {
        self.map.lock().remove(k);
    }
}

#[derive(Default, Clone)]
pub struct Bucket(Arc<BucketInternal>);

impl Deref for Bucket {
    type Target = Arc<BucketInternal>;

    fn deref(&self) -> &Arc<BucketInternal> {
        &self.0
    }
}

#[derive(Default)]
pub struct FutexMgrInternal {
    pub private: Bucket,
    pub shared: Bucket,
}

#[derive(Default, Clone)]
pub struct FutexMgr(Arc<FutexMgrInternal>);

impl Deref for FutexMgr {
    type Target = Arc<FutexMgrInternal>;

    fn deref(&self) -> &Arc<FutexMgrInternal> {
        &self.0
    }
}

impl FutexMgr {
    pub fn Fork(&self) -> Self {
        let internal = FutexMgrInternal {
            shared: self.shared.clone(),
            ..Default::default()
        };

        return Self(Arc::new(internal))
    }

    pub fn lockQueue(&self, k: &Key) -> Option<Queue> {
        let b = if k.Kind == KeyKind::KindSharedMappable {
            self.shared.clone()
        } else {
            self.private.clone()
        };

        b.Lock(&k);
        return b.GetQueue(&k);
    }

    pub fn lockQueueOnly(&self, k: &Key) {
        let b = if k.Kind == KeyKind::KindSharedMappable {
            self.shared.clone()
        } else {
            self.private.clone()
        };

        b.Lock(&k);
    }

    pub fn lockQueueWithCreate(&self, k: &Key) -> Queue {
        let b = if k.Kind == KeyKind::KindSharedMappable {
            self.shared.clone()
        } else {
            self.private.clone()
        };

        b.Lock(&k);
        return b.GetCreateQueues(&k);
    }

    pub fn GetCreateQueues(&self, k: &Key) -> Queue {
        let b = if k.Kind == KeyKind::KindSharedMappable {
            self.shared.clone()
        } else {
            self.private.clone()
        };

        return b.GetCreateQueues(&k);
    }

    pub fn removeQueue(&self, k: &Key) {
        let b = if k.Kind == KeyKind::KindSharedMappable {
            self.shared.clone()
        } else {
            self.private.clone()
        };

        b.RemoveQueue(&k);
    }

    pub fn unlock(&self, k: &Key) {
        let b = if k.Kind == KeyKind::KindSharedMappable {
            self.shared.clone()
        } else {
            self.private.clone()
        };

        b.Unlock(k);
    }

    // lockBuckets returns locked queues for the given keys.
    pub fn lockQueues(&self, k1: &Key, k2: &Key) -> (Queue, Queue) {
        //lock must be done in order to avoid deadlock

        assert!(k1 != k2, "FutexMgr::lockQueues, k1==k2");

        let i1 = k1.HashCode();
        let i2 = k2.HashCode();
        if i1 < i2 {
            self.lockQueueOnly(k1);
            self.lockQueueOnly(k2);
        } else if i1 > i2 {
            self.lockQueueOnly(k2);
            self.lockQueueOnly(k1);
        } else {
            self.lockQueueOnly(k1);
        }

        let q1 = self.GetCreateQueues(k1);
        let q2 = self.GetCreateQueues(k2);
        return (q1, q2)
    }

    pub fn Wake(&self, t: &Target, addr: u64, private: bool, bitmask: u32, n: i32) -> Result<i32> {
        let k = Getkey(t, addr, private)?;

        let temp = self.lockQueue(&k);

        let res = match temp {
            None => Ok(0),
            Some(q) => {
                let res = q.write().WakeLocked(bitmask as u64, n);
                if q.read().Empty() {
                    self.removeQueue(&k);
                }

                Ok(res)
            }
        };

        self.unlock(&k);
        return res;
    }

    pub fn doRequeue(&self, t: &Target, addr: u64, naddr: u64, private: bool,
                     checkval: bool, val: u32, nwake: i32, nreq: i32) -> Result<i32> {
        let k1 = Getkey(t, addr, private)?;
        let k2 = Getkey(t, naddr, private)?;

        let (q1, q2) = self.lockQueues(&k1, &k2);

        if checkval {
            match Check(t, addr, val) {
                Err(e) => {
                    self.unlock(&k1);
                    self.unlock(&k2);
                    return Err(e)
                }
                _ => ()
            }
        }

        let done = q1.write().WakeLocked(!0, nwake);

        // Requeue the number required.
        q1.write().RequeueLocked(&mut q2.write(), &k2, nreq);

        if q1.read().Empty() {
            self.removeQueue(&k1);
        }

        if q2.read().Empty() {
            self.removeQueue(&k2);
        }

        self.unlock(&k1);
        self.unlock(&k2);

        return Ok(done)
    }

    // Requeue wakes up to nwake waiters on the given addr, and unconditionally
    // requeues up to nreq waiters on naddr.
    pub fn Requeue(&self, t: &Target, addr: u64, naddr: u64, private: bool, nwake: i32, nreq: i32) -> Result<i32> {
        return self.doRequeue(t, addr, naddr, private, false, 0, nwake, nreq)
    }

    // RequeueCmp atomically checks that the addr contains val (via the Target),
    // wakes up to nwake waiters on addr and then unconditionally requeues nreq
    // waiters on naddr.
    pub fn RequeueCmp(&self, t: &Target, addr: u64, naddr: u64, private: bool, val: u32, nwake: i32, nreq: i32) -> Result<i32> {
        return self.doRequeue(t, addr, naddr, private, true, val, nwake, nreq)
    }

    // WakeOp atomically applies op to the memory address addr2, wakes up to nwake1
    // waiters unconditionally from addr1, and, based on the original value at addr2
    // and a comparison encoded in op, wakes up to nwake2 waiters from addr2.
    // It returns the total number of waiters woken.
    pub fn WakeOp(&self, t: &Target, addr1: u64, addr2: u64, private: bool, nwake1: i32, nwake2: i32, op: u32) -> Result<i32> {
        let k1 = Getkey(t, addr1, private)?;
        let k2 = Getkey(t, addr2, private)?;

        let (q1, q2) = self.lockQueues(&k1, &k2);

        let cond = match AtomicOp(t, addr2, op) {
            Err(e) => {
                self.unlock(&k1);
                self.unlock(&k2);
                return Err(e)
            }
            Ok(c) => c
        };

        // Wake up up to nwake1 entries from the first bucket.
        let mut done = q1.write().WakeLocked(!0, nwake1);

        // Wake up up to nwake2 entries from the second bucket if the
        // operation yielded true.
        if cond {
            done += q2.write().WakeLocked(!0, nwake2);
        }

        self.unlock(&k1);
        self.unlock(&k2);

        return Ok(done)
    }

    // WaitPrepare atomically checks that addr contains val (via the Checker), then
    // enqueues w to be woken by a send to w.C. If WaitPrepare returns nil, the
    // Waiter must be subsequently removed by calling WaitComplete, whether or not
    // a wakeup is received on w.
    pub fn WaitPrepare(&self, w: &WaitEntry, t: &Target, addr: u64, private: bool, val: u32, bitmask: u32) -> Result<()> {
        let k = Getkey(t, addr, private)?;

        w.Clear();
        w.SetMask(bitmask as u64);
        w.SetKey(&k);

        let q = self.lockQueueWithCreate(&k);

        // Perform our atomic check.
        match Check(t, addr, val) {
            Err(e) => {
                self.unlock(&k);
                return Err(e)
            }
            _ => ()
        }

        q.write().PushBack(&w);
        self.unlock(&k);

        return Ok(())
    }

    // WaitComplete must be called when a Waiter previously added by WaitPrepare is
    // no longer eligible to be woken.
    pub fn WaitComplete(&self, w: &WaitEntry) {
        loop {
            let key = w.lock().context.ThreadContext().key;

            //the w has been waked
            if key == Key::default() {
                break;
            }

            let q = self.lockQueueWithCreate(&key);
            if key != w.lock().context.ThreadContext().key {
                //the w has been requeued or waked
                self.unlock(&key);
                continue
            }

            q.write().Remove(w);
            if q.read().Empty() {
                self.removeQueue(&key);
            }

            w.lock().context.ThreadContext().key = Key::default();
            self.unlock(&key);
            break;
        }
    }

    // LockPI attempts to lock the futex following the Priority-inheritance futex
    // rules. The lock is acquired only when 'addr' points to 0. The TID of the
    // calling task is set to 'addr' to indicate the futex is owned. It returns true
    // if the futex was successfully acquired.
    //
    // FUTEX_OWNER_DIED is only set by the Linux when robust lists are in use (see
    // exit_robust_list()). Given we don't support robust lists, although handled
    // below, it's never set.
    pub fn LockPI(&self, w: &WaitEntry, t: &Target, addr: u64, tid: u32, private: bool, try: bool) -> Result<bool> {
        let k = Getkey(t, addr, private)?;

        w.Clear();
        w.SetKey(&k);
        w.SetTid(tid);
        w.lock().mask = !0;

        let q = self.lockQueueWithCreate(&k);

        let success = match self.lockPILocked(w, t, addr, tid, &q, try) {
            Err(e) => {
                self.unlock(&k);
                return Err(e)
            }
            Ok(s) => s,
        };

        self.unlock(&k);
        return Ok(success)
    }

    fn lockPILocked(&self, w: &WaitEntry, t: &Target, addr: u64, tid: u32, q: &Queue, try: bool) -> Result<bool> {
        loop {
            let cur = t.LoadU32(addr)?;

            if (cur & FUTEX_TID_MASK) == tid {
                return Err(Error::SysError(SysErr::EDEADLK))
            }

            if (cur & FUTEX_TID_MASK) == 0 {
                // No owner and no waiters, try to acquire the futex.

                // Set TID and preserve owner died status.
                let mut val = tid;
                val |= cur & FUTEX_OWNER_DIED;

                let prev = t.CompareAndSwapU32(addr, cur, val)?;

                if prev != cur {
                    // CAS failed, retry...
                    // Linux reacquires the bucket lock on retries, which will re-lookup the
                    // mapping at the futex address. However, retrying while holding the
                    // lock is more efficient and reduces the chance of another conflict.
                    continue
                }

                return Ok(true)
            }

            // Futex is already owned, prepare to wait.
            if try {
                // Caller doesn't want to wait.
                return Ok(false)
            }

            // Set waiters bit if not set yet.
            if cur & FUTEX_WAITERS == 0 {
                let prev = t.CompareAndSwapU32(addr, cur, cur | FUTEX_WAITERS)?;

                if prev != cur {
                    // CAS failed, retry...
                    continue;
                }
            }

            q.write().PushBack(w);
            return Ok(false)
        }
    }

    // UnlockPI unlock the futex following the Priority-inheritance futex
    // rules. The address provided must contain the caller's TID. If there are
    // waiters, TID of the next waiter (FIFO) is set to the given address, and the
    // waiter woken up. If there are no waiters, 0 is set to the address.
    pub fn UnlockPI(&self, t: &Target, addr: u64, tid: u32, private: bool) -> Result<()> {
        let k = Getkey(t, addr, private)?;

        let q = self.lockQueueWithCreate(&k);

        let err = self.unlockPILocked(t, addr, tid, &q);

        self.unlock(&k);
        return err
    }

    fn unlockPILocked(&self, t: &Target, addr: u64, tid: u32, q: &Queue) -> Result<()> {
        let cur = t.LoadU32(addr)?;

        if (cur & FUTEX_TID_MASK) != tid {
            return Err(Error::SysError(SysErr::EPERM))
        }

        if q.read().Empty() {
            // It's safe to set 0 because there are no waiters, no new owner, and the
            // executing task is the current owner (no owner died bit).
            let prev = t.CompareAndSwapU32(addr, cur, 0)?;

            if prev != cur {
                return Err(Error::SysError(SysErr::EAGAIN))
            }

            return Ok(())
        }

        let next = q.read().Front().unwrap();

        // Set next owner's TID, waiters if there are any. Resets owner died bit, if
        // set, because the executing task takes over as the owner.
        let mut val = next.lock().context.ThreadContext().tid;
        if next.lock().next.is_some() {
            val |= FUTEX_WAITERS;
        }

        let prev = t.CompareAndSwapU32(addr, cur, val)?;
        if prev != cur {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        q.write().WakeWaiterLocked(&next, !0);
        return Ok(())
    }
}