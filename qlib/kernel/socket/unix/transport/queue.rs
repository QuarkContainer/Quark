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

use crate::qlib::mutex::*;
use alloc::collections::linked_list::LinkedList;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Deref;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::kernel::waiter::queue::*;
use super::super::super::super::tcpip::tcpip::*;
use super::super::super::buffer::view::*;
use super::unix::*;

pub struct MessageInternal {
    // Data is the Message payload.
    pub Data: View,

    // Control is auxiliary control message data that goes along with the
    // data.
    pub Control: SCMControlMessages,

    // Address is the bound address of the endpoint that sent the message.
    //
    // If the endpoint that sent the message is not bound, the Address is
    // the empty string.
    pub Address: SockAddrUnix,
}

#[derive(Clone)]
pub struct Message(Arc<QMutex<MessageInternal>>);

impl Deref for Message {
    type Target = Arc<QMutex<MessageInternal>>;

    fn deref(&self) -> &Arc<QMutex<MessageInternal>> {
        &self.0
    }
}

impl Message {
    pub fn New(Data: Vec<u8>, Control: SCMControlMessages, Address: SockAddrUnix) -> Self {
        let internal = MessageInternal {
            Data: View(Data),
            Control: Control,
            Address: Address,
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn Length(&self) -> usize {
        return self.lock().Data.len();
    }

    pub fn Release(&self) {
        self.lock().Control.Release();
    }

    // Truncate reduces the length of the message payload to n bytes.
    //
    // Preconditions: n <= m.Length().
    pub fn Truncate(&self, n: usize) {
        self.lock().Data.CapLength(n)
    }
}

impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl Eq for Message {}

pub struct MsgQueueInternal {
    pub ReaderQueue: Queue,
    pub WriterQueue: Queue,

    pub closed: bool,
    pub unread: bool,
    pub used: usize,
    pub limit: usize,

    pub dataList: LinkedList<Message>,
}

impl MsgQueueInternal {
    // bufWritable returns true if there is space for writing.
    //
    // N.B. Linux only considers a unix socket "writable" if >75% of the buffer is
    // free.
    //
    // See net/unix/af_unix.c:unix_writeable.
    pub fn BufWritable(&self) -> bool {
        return 4 * self.used < self.limit;
    }
}

#[derive(Clone)]
pub struct MsgQueue(Arc<QMutex<MsgQueueInternal>>);

impl Deref for MsgQueue {
    type Target = Arc<QMutex<MsgQueueInternal>>;

    fn deref(&self) -> &Arc<QMutex<MsgQueueInternal>> {
        &self.0
    }
}

impl Drop for MsgQueue {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            self.Reset();
        }
    }
}

impl MsgQueue {
    pub fn New(readQueue: Queue, writeQueue: Queue, limit: usize) -> Self {
        let internal = MsgQueueInternal {
            ReaderQueue: readQueue,
            WriterQueue: writeQueue,
            closed: false,
            unread: false,
            used: 0,
            limit: limit,
            dataList: LinkedList::new(),
        };

        return Self(Arc::new(QMutex::new(internal)));
    }

    // Close closes q for reading and writing. It is immediately not writable and
    // will become unreadable when no more data is pending.
    //
    // Both the read and write queues must be notified after closing:
    // q.ReaderQueue.Notify(waiter.EventIn)
    // q.WriterQueue.Notify(waiter.EventOut)
    pub fn Close(&self) {
        self.lock().closed = true;
    }

    // Reset empties the queue and Releases all of the Entries.
    //
    // Both the read and write queues must be notified after resetting:
    // q.ReaderQueue.Notify(waiter.EventIn)
    // q.WriterQueue.Notify(waiter.EventOut)
    pub fn Reset(&self) {
        let mut q = self.lock();

        let mut cur = q.dataList.pop_front();
        while cur.is_some() {
            cur.as_ref().unwrap().Release();
            cur = q.dataList.pop_front();
        }

        q.dataList.clear();
        q.used = 0;
    }

    // IsReadable determines if q is currently readable.
    pub fn IsReadable(&self) -> bool {
        let q = self.lock();

        return q.closed || q.dataList.front().is_some();
    }

    // IsWritable determines if q is currently writable.
    pub fn IsWritable(&self) -> bool {
        let q = self.lock();

        return q.closed || q.BufWritable();
    }

    // Enqueue adds an entry to the data queue if room is available.
    //
    // If truncate is true, Enqueue may truncate the message beforing enqueuing it.
    // Otherwise, the entire message must fit. If n < e.Length(), err indicates why.
    //
    // If notify is true, ReaderQueue.Notify must be called:
    // q.ReaderQueue.Notify(waiter.EventIn)

    // return (length, notify)
    pub fn Enqueue(&self, e: Message, truncate: bool) -> Result<(usize, bool)> {
        let mut q = self.lock();

        if q.closed {
            //return Err(Error::ErrClosedForReceive)
            return Err(Error::SysError(SysErr::EPIPE));
        }

        let free = q.limit - q.used;

        let mut l = e.Length();

        if l > free && truncate {
            if free == 0 {
                // Message can't fit right now.
                return Err(Error::SysError(SysErr::EAGAIN));
            }

            e.Truncate(free);
            l = e.Length();
            //return Err(Error::SysError(SysErr::EAGAIN))
        }

        if l > q.limit {
            // Message is too big to ever fit.
            return Err(Error::SysError(SysErr::EMSGSIZE));
        }

        if l > free {
            // Message can't fit right now.
            return Err(Error::SysError(SysErr::EAGAIN));
        }

        let notify = q.dataList.front().is_none();
        q.used += l;
        q.dataList.push_back(e);

        return Ok((l, notify));
    }

    // Dequeue removes the first entry in the data queue, if one exists.
    //
    // If notify is true, WriterQueue.Notify must be called:
    // q.WriterQueue.Notify(waiter.EventOut)
    // return (message, notify)
    pub fn Dequeue(&self) -> Result<(Message, bool)> {
        let mut q = self.lock();

        if q.dataList.front().is_none() {
            if q.closed {
                if q.unread {
                    return Err(Error::SysError(SysErr::ECONNRESET));
                } else {
                    return Err(Error::ErrClosedForReceive);
                }
            } else {
                return Err(Error::SysError(SysErr::EAGAIN));
            }
        }

        let mut notify = !q.BufWritable();

        let e = q.dataList.pop_front().unwrap();
        q.used -= e.Length();

        notify = notify && q.BufWritable();

        return Ok((e, notify));
    }

    // Peek returns the first entry in the data queue, if one exists.
    pub fn Peek(&self) -> Result<Message> {
        let q = self.lock();

        if q.dataList.front().is_none() {
            if q.closed {
                if q.unread {
                    return Err(Error::SysError(SysErr::ECONNRESET));
                } else {
                    return Err(Error::ErrClosedForReceive);
                }
            } else {
                return Err(Error::SysError(SysErr::EAGAIN));
            }
        }

        return Ok(q.dataList.front().unwrap().clone());
    }

    // QueuedSize returns the number of bytes currently in the queue, that is, the
    // number of readable bytes.
    pub fn QueuedSize(&self) -> i64 {
        return self.lock().used as i64;
    }

    // MaxQueueSize returns the maximum number of bytes storable in the queue.
    pub fn MaxQueueSize(&self) -> i64 {
        return self.lock().limit as i64;
    }

    // SetMaxQueueSize sets the maximum number of bytes storable in the queue.
    pub fn SetMaxQueueSize(&self, v: i64) {
        self.lock().limit = v as _;
    }

    // CloseUnread sets flag to indicate that the peer is closed (not shutdown)
    // with unread data. So if read on this queue shall return ECONNRESET error.
    pub fn CloseUnread(&self) {
        self.lock().unread = true;
    }
}
