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

use alloc::collections::btree_set::BTreeSet;
use core::ops::Deref;
use alloc::sync::Arc;
use spin::Mutex;
use alloc::string::String;

use super::super::qlib::mem::areaset::*;
use super::super::qlib::range::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::kernel::waiter::*;

#[derive(Clone, Copy)]
pub enum LockType {
    // ReadLock describes a POSIX regional file lock to be taken
    // read only.  There may be multiple of these locks on a single
    // file region as long as there is no writer lock on the same
    // region.
    ReadLock,

    // WriteLock describes a POSIX regional file lock to be taken
    // write only.  There may be only a single holder of this lock
    // and no read locks.
    WriteLock,
}

type UniqueId = u64;

pub const READ_LOCK: u32 = 0;
pub const WRITE_LOCK: u32 = 1;

#[derive(Default)]
pub struct LockInternel {
    pub Readers: BTreeSet<UniqueId>,
    pub Writer: Option<UniqueId>,
}

// Lock is a regional file lock.  It consists of either a single writer
// or a set of readers.
//
// A Lock may be upgraded from a read lock to a write lock only if there
// is a single reader and that reader has the same uid as the write lock.
//
// A Lock may be downgraded from a write lock to a read lock only if
// the write lock's uid is the same as the read lock.
//
#[derive(Clone, Default)]
pub struct Lock(Arc<Mutex<LockInternel>>);

impl Deref for Lock {
    type Target = Arc<Mutex<LockInternel>>;

    fn deref(&self) -> &Arc<Mutex<LockInternel>> {
        &self.0
    }
}

impl Lock {
    // isHeld returns true if uid is a holder of Lock.
    pub fn IsHeld(&self, uid: UniqueId) -> bool {
        let l = self.lock();
        if l.Writer == Some(uid) {
            return true
        };

        return l.Readers.contains(&uid)
    }

    // lock sets uid as a holder of a typed lock on Lock.
    //
    // Preconditions: canLock is true for the range containing this Lock.
    pub fn Lock(&self, uid: UniqueId, t: LockType) {
        let mut l = self.lock();
        match t {
            LockType::ReadLock => {
                // If we are already a reader, then this is a no-op.
                if l.Readers.contains(&uid) {
                    return;
                }

                // We cannot downgrade a write lock to a read lock unless the
                // uid is the same.
                if l.Writer.is_some() {
                    if l.Writer != Some(uid) {
                        panic!("lock: cannot downgrade write lock to read lock for uid {}, writer is {:?}", uid, l.Writer)
                    }

                    // Ensure that there is only one reader if upgrading.
                    l.Readers.clear();
                    l.Writer = None;
                }

                l.Readers.insert(uid);
                return;
            }
            LockType::WriteLock => {
                // If we are already the writer, then this is a no-op.
                if l.Writer == Some(uid) {
                    return
                }

                // We can only upgrade a read lock to a write lock if there
                // is only one reader and that reader has the same uid as
                // the write lock.
                let readers = l.Readers.len();
                if readers > 0 {
                    if readers != 1 {
                        panic!("lock: cannot upgrade read lock to write lock for uid {}, too many readers {:?}", uid, l.Readers)
                    }

                    if !l.Readers.contains(&uid) {
                        panic!("lock: cannot upgrade read lock to write lock for uid {}, conflicting reader {:?}", uid, l.Readers)
                    }
                }

                // Ensure that there is only a writer.
                l.Readers.clear();
                l.Writer = Some(uid);
            }
        }
    }

}

pub fn MakeLock(uid: UniqueId, t: LockType) -> Lock {
    let mut val = LockInternel::default();
    match t {
        LockType::ReadLock => {
            val.Readers.insert(uid);
        }
        LockType::WriteLock => {
            val.Writer = Some(uid);
        }
    }

    return Lock(Arc::new(Mutex::new(val)))
}

impl AreaValue for Lock {
    fn Merge(&self, _r1: &Range, _r2: &Range, val2: &Self) -> Option<Self> {
        let v1 = self.lock();
        let v2 = val2.lock();

        // Merge only if the Readers/Writers are identical.
        if v1.Readers.len() != v2.Readers.len() {
            return None;
        }

        for id in v1.Readers.iter() {
            if !v2.Readers.contains(id) {
                return None
            }
        }

        if v1.Writer != v2.Writer {
            return None;
        }

        return Some(val2.clone())
    }

    fn Split(&self, _r: &Range, _split: u64) -> (Self, Self) {
        // Copy the segment so that split segments don't contain map references
        // to other segments.
        let mut v2 = LockInternel::default();

        let v1 = self.lock();
        for r in v1.Readers.iter() {
            v2.Readers.insert(*r);
        }

        v2.Writer = v1.Writer;

        return (self.clone(), Self(Arc::new(Mutex::new(v2))))
    }
}

pub struct LocksInternal {
    // locks is the set of region locks currently held on an Inode.
    pub locks: AreaSet<Lock>,

    // queue is the queue of waiters that are waiting on a lock.
    pub queue: Queue,
}

impl Default for LocksInternal {
    fn default() -> Self {
        return Self {
            locks: AreaSet::New(0, MAX_RANGE),
            queue: Queue::default(),
        }
    }
}

impl LocksInternal {
    pub fn Lock(&mut self, uid: UniqueId, t: LockType, r: &Range) -> bool {
        // Don't attempt to insert anything with a range of 0 and treat this
        // as a successful no-op.
        if r.Len() == 0 {
            return true;
        }

        // Do a first-pass check.  We *could* hold onto the segments we
        // checked if canLock would return true, but traversing the segment
        // set should be fast and this keeps things simple.
        if !self.CanLock(uid, t, r) {
            return false;
        }

        // Get our starting point.
        let (mut seg, gap) = self.locks.Find(r.Start());
        if gap.Ok() {
            // Fill in the gap and get the next segment to modify.
            seg = self.locks.Insert(&gap, &gap.Range().Intersect(r), MakeLock(uid, t)).NextSeg();
        } else if seg.Range().Start() < r.Start() {
            let (_, tmp) = self.locks.Split(&seg, r.Start());
            seg = tmp;
        }

        while seg.Ok() && seg.Range().Start() < r.End() {
            // Split the last one if necessary.
            if seg.Range().End()  > r.End() {
                let (tmp, _) = self.locks.SplitUnchecked(&seg, r.End());
                seg = tmp;
            }

            // Set the lock on the segment.  This is guaranteed to
            // always be safe, given canLock above.
            let value = seg.Value();
            value.Lock(uid, t);

            // Fill subsequent gaps.
            let gap = seg.NextGap();
            let gr = gap.Range().Intersect(r);
            if gr.Len() > 0 {
                seg = self.locks.Insert(&gap, &gr, MakeLock(uid, t)).NextSeg();
            } else {
                seg = gap.NextSeg()
            }
        }

        return true;
    }

    // unlock is always successful.  If uid has no locks held for the range LockRange,
    // unlock is a no-op.
    pub fn Unlock(&mut self, uid: UniqueId, r: &Range) {
        if r.Len() == 0{
            return;
        }

        // Get our starting point.
        let mut seg = self.locks.UpperBoundSeg(r.Start());
        while seg.Ok() && seg.Range().Start() < r.End() {
            // If this segment doesn't have a lock from uid then
            // there is no need to fragment the set with Isolate (below).
            // In this case just move on to the next segment.
            if !seg.Value().IsHeld(uid) {
                seg = seg.NextSeg();
                continue;
            }

            // Ensure that if we need to unlock a sub-segment that
            // we don't unlock/remove that entire segment.
            seg = self.locks.Isolate(&seg, r);

            let value = seg.Value();
            let value = value.lock();
            let mut remove = false;

            if value.Writer == Some(uid) {
                // If we are unlocking a writer, then since there can
                // only ever be one writer and no readers, then this
                // lock should always be removed from the set.
                remove = true;
            } else if value.Readers.contains(&uid) {
                // If uid is the last reader, then just remove the entire
                // segment.
                if value.Readers.len() == 1 {
                    remove = true;
                } else {
                    // Otherwise we need to remove this reader without
                    // affecting any other segment's readers.  To do
                    // this, we need to make a copy of the Readers map
                    // and not add this uid.
                    let newValue = Lock::default();
                    {
                        let mut newLock = newValue.lock();
                        for r in value.Readers.iter() {
                            if *r != uid {
                                newLock.Readers.insert(*r);
                            }
                        }
                    }

                    seg.SetValue(newValue)
                }
            }

            if remove {
                seg = self.locks.Remove(&seg).NextSeg();
            } else {
                seg = seg.NextSeg();
            }
        }
    }

    // lockable returns true if check returns true for every Lock in LockRange.
    // Further, check should return true if Lock meets the callers requirements
    // for locking Lock.
    pub fn Lockable(&self, r: &Range, check: &Fn(&Lock) -> bool) -> bool {
        // Get our starting point.
        let mut seg = self.locks.LowerBoundSeg(r.Start());
        while seg.Ok() && seg.Range().Start() < r.End() {
            // Note that we don't care about overruning the end of the
            // last segment because if everything checks out we'll just
            // split the last segment.
            let value = seg.Value();
            if !check(&value) {
                return false;
            }

            // Jump to the next segment, ignoring gaps, for the same
            // reason we ignored the first gap.
            seg = seg.NextSeg();
        }

        // No conflict, we can get a lock for uid over the entire range.
        return true;
    }

    pub fn CanLock(&self, uid: UniqueId, t: LockType, r: &Range) -> bool {
        match t {
            LockType::ReadLock => {
                return self.Lockable(r, &|value: &Lock| {
                    // If there is no writer, there's no problem adding
                    // another reader.
                    if value.lock().Writer.is_none() {
                        return true;
                    }

                    // If there is a writer, then it must be the same uid
                    // in order to downgrade the lock to a read lock.
                    return *(value.lock().Writer.as_ref().unwrap()) == uid
                })
            }
            LockType::WriteLock => {
                return self.Lockable(r, &|value: &Lock| {
                    // If there is no writer, there's no problem adding
                    // another reader.
                    let value = value.lock();
                    if value.Writer.is_none() {
                        // Then this uid can only take a write lock if
                        // this is a private upgrade, meaning that the
                        // only reader is uid.
                        return value.Readers.len() == 1 && value.Readers.contains(&uid);
                    }

                    // If the uid is already a writer on this region, then
                    // adding a write lock would be a no-op.
                    return value.Writer == Some(uid)
                })
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Locks(Arc<Mutex<LocksInternal>>);

impl Deref for Locks {
    type Target = Arc<Mutex<LocksInternal>>;

    fn deref(&self) -> &Arc<Mutex<LocksInternal>> {
        &self.0
    }
}

impl Locks {
    // LockRegion attempts to acquire a typed lock for the uid on a region
    // of a file. Returns true if successful in locking the region. If false
    // is returned, the caller should normally interpret this as "try again later" if
    // accquiring the lock in a non-blocking mode or "interrupted" if in a blocking mode.
    // Blocker is the interface used to provide blocking behavior, passing a nil Blocker
    // will result in non-blocking behavior.
    pub fn LockRegion(&self, task: &Task, uid: UniqueId, t: LockType, r: &Range, block: bool) -> Result<bool> {
        loop {
            let mut l = self.lock();

            // Blocking locks must run in a loop because we'll be woken up whenever an unlock event
            // happens for this lock. We will then attempt to take the lock again and if it fails
            // continue blocking.
            let res = l.Lock(uid, t, r);
            if !res && block {
                l.queue.EventRegister(task, &task.blocker.generalEntry, EVENTMASK_ALL);
                core::mem::drop(l);

                defer!(self.lock().queue.EventUnregister(task, &task.blocker.generalEntry));

                match task.blocker.BlockGeneral() {
                    Err(Error::ErrInterrupted) => return Err(Error::SysError(SysErr::ERESTARTSYS)),
                    Err(e) => return Err(e),
                    Ok(()) => (),
                }
                // Try again now that someone has unlocked.
                continue;
            }

            return Ok(res);
        }
    }

    pub fn Print(&self) -> String {
        return self.lock().locks.Print();
    }

    // UnlockRegion attempts to release a lock for the uid on a region of a file.
    // This operation is always successful, even if there did not exist a lock on
    // the requested region held by uid in the first place.
    pub fn UnlockRegion(&self, _task: &Task, uid: UniqueId, r: &Range) {
        let mut l = self.lock();

        l.Unlock(uid, r);

        // Now that we've released the lock, we need to wake up any waiters.
        l.queue.Notify(EVENTMASK_ALL)
    }
}

// ComputeRange takes a positive file offset and computes the start of a LockRange
// using start (relative to offset) and the end of the LockRange using length. The
// values of start and length may be negative but the resulting LockRange must
// preserve that LockRange.Start < LockRange.End and LockRange.Start > 0.
pub fn ComputeRange(start: i64, length: i64, offset: i64) -> Result<Range> {
    let mut offset = offset;
    offset += start;

    // fcntl(2): "l_start can be a negative number provided the offset
    // does not lie before the start of the file"
    if offset < 0 {
        return Err(Error::SysError(SysErr::EINVAL))
    }

    // fcntl(2): Specifying 0 for l_len has the  special meaning: lock all
    // bytes starting at the location specified by l_whence and l_start
    // through to the end of file, no matter how large the file grows.
    let mut end = MAX_RANGE;
    if length > 0 {
        // fcntl(2): If l_len is positive, then the range to be locked
        // covers bytes l_start up to and including l_start+l_len-1.
        //
        // Since LockRange.End is exclusive we need not -1 from length..
        end = (offset + length) as u64;
    } else if length < 0 {
        // fcntl(2): If l_len is negative, the interval described by
        // lock covers bytes l_start+l_len up to and including l_start-1.
        //
        // Since LockRange.End is exclusive we need not -1 from offset.
        let signedEnd = offset;
        // Add to offset using a negative length (subtract).
        offset += length;
        if offset < 0 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if signedEnd < offset {
            return Err(Error::SysError(SysErr::EOVERFLOW))
        }
        // At this point signedEnd cannot be negative,
        // since we asserted that offset is not negative
        // and it is not less than offset.
        end = signedEnd as u64;
    }

    let len = if end == MAX_RANGE {
        MAX_RANGE
    } else {
        end - offset as u64
    };

    return Ok(Range::New(offset as u64, len))
}