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
use crate::qlib::mutex::*;
use core::ops::Deref;
use alloc::collections::vec_deque::VecDeque;

use super::super::super::auth::userns::*;
use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::linux::ipc::*;
use super::super::super::linux::msgqueue::*;
use super::super::task::*;
use super::waiter::*;
use super::time::*;
use super::ipc_namespace::*;

// System-wide limit for maximum number of queues.
pub const MAX_QUEUES : usize = MSGMNI;

// Maximum size of a queue in bytes.
pub const MAX_QUEUE_BYTES : usize = MSGMNB;

// Maximum size of a message in bytes.
pub const MAX_MESSAGE_BYTES : usize = MSGMAX;

pub struct MQRegistryIntern(RegistryInternal<MsgQueue>);

impl Deref for MQRegistryIntern {
    type Target = RegistryInternal<MsgQueue>;

    fn deref(&self) -> &RegistryInternal<MsgQueue> {
        &self.0
    }
}

impl MQRegistryIntern {
    pub fn NewQueue(&mut self,
                    task: &Task,
                    r: &MQRegistry,
                    key: Key,
                    creator: &FileOwner,
                    perms: &FilePermissions) -> Result<Mechanism<MsgQueue>> {
        let intern = MsgQueueIntern {
            registry: r.clone(),
            dead: false,
            senders: Queue::default(),
            receivers: Queue::default(),
            messages: VecDeque::new(),
            sendTime: Time::default(),
            receiveTime: Time::default(),
            changeTime: task.Now(),
            byteCount: 0,
            maxBytes: MAX_QUEUE_BYTES as _,
            sendPID: 0,
            receivePID: 0
        };
        let queue = MsgQueue(Arc::new(QMutex::new(intern)));
        let mec = Mechanism::New(self.userNS.clone(), key, creator, creator, perms, queue);
        self.0.Register(mec.clone())?;
        return Ok(mec)
    }
}

#[derive(Clone)]
pub struct MQRegistry(Arc<QMutex<MQRegistryIntern>>);

impl Deref for MQRegistry {
    type Target = Arc<QMutex<MQRegistryIntern>>;

    fn deref(&self) -> &Arc<QMutex<MQRegistryIntern>> {
        &self.0
    }
}

impl MQRegistry {
    pub fn New(userNS: &UserNameSpace) -> Self {
        let intern = RegistryInternal::New(userNS);
        return Self(Arc::new(QMutex::new(MQRegistryIntern(intern))))
    }

    // FindOrCreate creates a new message queue or returns an existing one. See
    // msgget(2).
    pub fn FindOrCreate(&self,
                        task: &Task,
                        key: Key,
                        mode: &FileMode,
                        private: bool,
                        create: bool,
                        exclusive: bool) -> Result<Mechanism<MsgQueue>> {
        let mut me = self.lock();

        if !private {
            let queue = me.Find(task, key, *mode, create, exclusive)?;
            match queue {
                None => (),
                Some(q) => return Ok(q)
            }
        }

        // Check system-wide limits.
        if me.ObjectCount() > MAX_QUEUES {
            return Err(Error::SysError(SysErr::ENOSPC))
        }

        let creator = task.FileOwner();
        let perms = FilePermissions::FromMode(*mode);
        let queue = me.NewQueue(task, self, key, &creator, &perms)?;
        return Ok(queue)
    }

    // Remove removes the queue with specified ID. All waiters (readers and
    // writers) and writers will be awakened and fail. Remove will return an error
    // if the ID is invalid, or the the user doesn't have privileges.
    pub fn Remove(&self, id: ID, creds: &Credentials) -> Result<()> {
        let mut me = self.lock();
        me.0.Remove(id, creds)?;
        return Ok(())
    }

    // FindByID returns the queue with the specified ID and an error if the ID
    // doesn't exist.
    pub fn FindById(&self, id: ID) -> Result<Mechanism<MsgQueue>> {
        let me = self.lock();

        match me.FindById(id) {
            None => return Err(Error::SysError(SysErr::EINVAL)),
            Some(q) => return Ok(q)
        }
    }

    // IPCInfo reports global parameters for message queues. See msgctl(IPC_INFO).
    pub fn IPCInfo(&self, _task: &Task) -> MsgInfo {
        return MsgInfo {
            MsgPool: MSGPOOL as _,
            MsgMap:  MSGMAP as _,
            MsgMax:  MSGMAX as _,
            MsgMnb:  MSGMNB as _,
            MsgMni:  MSGMNI as _,
            MsgSsz:  MSGSSZ as _,
            MsgTql:  MSGTQL as _,
            MsgSeg: MSGSEG as _,
        }
    }

    // MsgInfo reports global parameters for message queues. See msgctl(MSG_INFO).
    pub fn MsgInfo(&self, _task: &Task) -> MsgInfo {
        let me = self.lock();

        let mut messages = 0;
        let mut bytes = 0;
        me.ForAllObjects(&mut |q : &Mechanism<MsgQueue>| {
            let q = q.Object();
            messages += q.lock().messages.len();
            bytes += q.lock().byteCount;
        });

        return MsgInfo {
            MsgPool: me.ObjectCount() as _,
            MsgMap:  messages as _,
            MsgTql:  bytes as _,
            MsgMax:  MSGMAX as _,
            MsgMnb:  MSGMNB as _,
            MsgMni:  MSGMNI as _,
            MsgSsz:  MSGSSZ as _,
            MsgSeg: MSGSEG as _,
        }
    }

}

// Queue represents a SysV message queue, described by sysvipc(7).
#[derive(Clone)]
pub struct MsgQueue(Arc<QMutex<MsgQueueIntern>>);

impl Deref for MsgQueue {
    type Target = Arc<QMutex<MsgQueueIntern>>;

    fn deref(&self) -> &Arc<QMutex<MsgQueueIntern>> {
        &self.0
    }
}

impl Object for MsgQueue {
    fn Destory(&self) {
        let mut q = self.lock();
        q.dead = true;

        q.senders.Notify(EVENT_OUT);
        q.receivers.Notify(EVENT_IN);
    }
}

impl Mechanism<MsgQueue> {
     // Send appends a message to the message queue, and returns an error if sending
    // fails. See msgsnd(2).
    pub fn Send(&self, task: &Task, m: &Message, wait:  bool, pid: i32) -> Result<()> {
        let creds = task.creds.clone();

        match self.Push(task, m, &creds, pid) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
            Err(e) => return Err(e),
            Ok(()) => return Ok(()),
        }

        if !wait {
            return Err(Error::SysError(SysErr::EWOULDBLOCK))
        }

        // Slow path: at this point, the queue was found to be full, and we were
        // asked to block.
        let general = task.blocker.generalEntry.clone();
        let senders = self.Object().lock().senders.clone();

        senders.EventRegister(task, &general, EVENT_WRITE);
        defer!(senders.EventUnregister(task, &general));
        loop {
            match self.Push(task, m, &creds, pid) {
                Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
                Err(e) => return Err(e),
                Ok(()) => return Ok(()),
            }

            match task.blocker.BlockWithMonoTimer(true, None) {
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTSYS))
                },
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }
        }
    }

    pub fn Copy(&self, mType: i64) -> Result<Message> {
        return self.Object().lock().Copy(mType);
    }

    // push appends a message to the queue's message list and notifies waiting
    // receivers that a message has been inserted. It returns an error if adding
    // the message would cause the queue to exceed its maximum capacity, which can
    // be used as a signal to block the task. Other errors should be returned as is.
    pub fn Push(&self, task: &Task, m: &Message, cred: &Credentials, pid: i32) -> Result<()> {
        if m.Type() <= 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let q = self.Object();
        let mech = self.lock();
        let mut q = q.lock();

        if !mech.checkPermission(cred, &PermMask {
            write: true,
            ..Default::default()
        }) {
            // The calling process does not have write permission on the message
            // queue, and does not have the CAP_IPC_OWNER capability in the user
            // namespace that governs its IPC namespace.
            return Err(Error::SysError(SysErr::EACCES));
        }

        // Queue was removed while the process was waiting.
        if q.dead {
            return Err(Error::SysError(SysErr::EIDRM));
        }

        // Check if sufficient space is available (the queue isn't full.) From
        // the man pages:
        //
        // "A message queue is considered to be full if either of the following
        // conditions is true:
        //
        //  • Adding a new message to the queue would cause the total number
        //    of bytes in the queue to exceed the queue's maximum size (the
        //    msg_qbytes field).
        //
        //  • Adding another message to the queue would cause the total
        //    number of messages in the queue to exceed the queue's maximum
        //    size (the msg_qbytes field).  This check is necessary to
        //    prevent an unlimited number of zero-length messages being
        //    placed on the queue.  Although such messages contain no data,
        //    they nevertheless consume (locked) kernel memory."
        //
        // The msg_qbytes field in our implementation is q.maxBytes.
        if m.Size() + q.byteCount > q.maxBytes || q.messages.len() + 1 > q.maxBytes as _ {
            return Err(Error::SysError(SysErr::EWOULDBLOCK));
        }

        q.byteCount += m.Size();
        q.sendPID = pid;
        q.sendTime = task.Now();

        // Copy the message into the queue.
        q.messages.push_back(m.clone());

        // Notify receivers about the new message.
        q.receivers.Notify(EVENT_IN);
        return Ok(())
    }

    pub fn Receive(&self,
                   task: &Task,
                   mType: i64,
                   maxSize: i64,
                   wait: bool,
                   truncate: bool,
                   except: bool,
                   pid: i32) -> Result<Message> {
        if maxSize < 0 || maxSize > MAX_MESSAGE_BYTES as _ {
            return Err(Error::SysError(SysErr::EINVAL));
        }
        let max = maxSize as u64;
        let creds = task.creds.clone();

        match self.Pop(task, &creds, mType, max, truncate, except, pid) {
            Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
            Err(e) => return Err(e),
            Ok(m) => return Ok(m)
        }

        if !wait {
            return Err(Error::SysError(SysErr::ENOMSG))
        }

        // Slow path: at this point, the queue was found to be full, and we were
        // asked to block.
        let general = task.blocker.generalEntry.clone();
        let receivers = self.Object().lock().receivers.clone();

        receivers.EventRegister(task, &general, EVENT_READ);
        defer!(receivers.EventUnregister(task, &general));

        loop {
            match self.Pop(task, &creds, mType, max, truncate, except, pid) {
                Err(Error::SysError(SysErr::EWOULDBLOCK)) => (),
                Err(e) => return Err(e),
                Ok(m) => return Ok(m)
            }

            match task.blocker.BlockWithMonoTimer(true, None) {
                Err(Error::ErrInterrupted) => {
                    return Err(Error::SysError(SysErr::ERESTARTSYS))
                },
                Err(e) => {
                    return Err(e);
                }
                _ => (),
            }
        }
    }

    // pop pops the first message from the queue that matches the given type. It
    // returns an error for all the cases specified in msgrcv(2). If the queue is
    // empty or no message of the specified type is available, a EWOULDBLOCK error
    // is returned, which can then be used as a signal to block the process or fail.
    pub fn Pop(&self,
               task: &Task,
               creds: &Credentials,
               mType: i64,
               maxSize: u64,
               truncate: bool,
               except: bool,
               pid: i32) -> Result<Message> {

        let q = self.Object();
        let mech = self.lock();
        let mut q = q.lock();

        if !mech.checkPermission(creds, &PermMask {
            read: true,
            ..Default::default()
        }) {
            // The calling process does not have read permission on the message
            // queue, and does not have the CAP_IPC_OWNER capability in the user
            // namespace that governs its IPC namespace.
            return Err(Error::SysError(SysErr::EACCES));
        }

        // Queue was removed while the process was waiting.
        if q.dead {
            return Err(Error::SysError(SysErr::EIDRM));
        }

        if q.messages.len() == 0 {
            return Err(Error::SysError(SysErr::EWOULDBLOCK));
        }

        let msg;
        let idx;
        if mType == 0 {
            msg = q.messages.front().unwrap().clone();
            idx = 0;
        } else if mType > 0 {
            match q.msgOfType(mType, except) {
                None => return Err(Error::SysError(SysErr::EWOULDBLOCK)),
                Some((m, i)) => {
                    msg = m;
                    idx = i;
                }
            }
        } else { //mType == 0
            match q.msgOfTypeLessThan(-mType) {
                None => return Err(Error::SysError(SysErr::EWOULDBLOCK)),
                Some((m, i)) => {
                    msg = m;
                    idx = i;
                }
            }
        }

        let msgSize = msg.Size();
        if maxSize < msg.Size() {
            if !truncate {
                return Err(Error::SysError(SysErr::E2BIG));
            }
            msg.Truncate(maxSize as usize);
        }

        q.messages.remove(idx);
        q.byteCount -= msgSize;
        q.receivePID = pid;
        q.receiveTime = task.Now();

        // Notify senders about available space.
        q.senders.Notify(EVENT_OUT);

        return Ok(msg)
    }

    // Set modifies some values of the queue. See msgctl(IPC_SET).
    pub fn Set(&self, task: &Task, ds: &MsqidDS) -> Result<()> {
        let q = self.Object();
        let mut mech = self.lock();
        let mut q = q.lock();

        let creds = task.creds.clone();
        if ds.MsgQbytes > MAX_QUEUE_BYTES as _
            && !creds.HasCapabilityIn(Capability::CAP_SYS_RESOURCE, &mech.userNS) {
            // "An attempt (IPC_SET) was made to increase msg_qbytes beyond the
            // system parameter MSGMNB, but the caller is not privileged (Linux:
            // does not have the CAP_SYS_RESOURCE capability)."
            return Err(Error::SysError(SysErr::EPERM));
        }

        mech.Set(task, &ds.MsgPerm)?;

        q.maxBytes = ds.MsgQbytes;
        q.changeTime = task.Now();
        return Ok(())
    }

    // Stat returns a MsqidDS object filled with information about the queue. See
    // msgctl(IPC_STAT) and msgctl(MSG_STAT).
    pub fn Stat(&self, task: &Task) -> Result<MsqidDS> {
        return self.stat(task, &PermMask {read: true, ..Default::default()});
    }

    // StatAny is similar to Queue.Stat, but doesn't require read permission. See
    // msgctl(MSG_STAT_ANY).
    pub fn StatAny(&self, task: &Task) -> Result<MsqidDS> {
        return self.stat(task, &PermMask::default());
    }

    pub fn stat(&self, task: &Task, mask: &PermMask) -> Result<MsqidDS> {
        let q = self.Object();
        let mech = self.lock();
        let q = q.lock();

        let creds = task.creds.clone();
        if !mech.checkPermission(&creds, mask) {
            // "The caller must have read permission on the message queue."
            return Err(Error::SysError(SysErr::EACCES));
        }

        let userns = creds.lock().UserNamespace.clone();
        let UID = userns.MapFromKUID(mech.owner.UID).0;
        let GID = userns.MapFromKGID(mech.owner.GID).0;
        let CUID = userns.MapFromKUID(mech.creator.UID).0;
        let CGID = userns.MapFromKGID(mech.creator.GID).0;

        return Ok(MsqidDS {
            MsgPerm: IPCPerm {
                Key: mech.key as _,
                UID: UID,
                GID: GID,
                CUID: CUID,
                CGID: CGID,
                Mode: mech.perms.LinuxMode() as _,
                Seq: 0,
                ..Default::default()
            },
            MsgStime:  q.sendTime.TimeT(),
            MsgRtime:  q.receiveTime.TimeT(),
            MsgCtime:  q.changeTime.TimeT(),
            MsgCbytes: q.byteCount,
            MsgQnum:   q.messages.len() as _,
            MsgQbytes: q.maxBytes,
            MsgLspid:  q.sendPID,
            MsgLrpid:  q.receivePID,
            ..Default::default()
        })
    }
}

pub struct MsgQueueIntern {
    // registry is the registry owning this queue. Immutable.
    pub registry: MQRegistry,

    // dead is set to true when a queue is removed from the registry and should
    // not be used. Operations on the queue should check dead, and return
    // EIDRM if set to true.
    pub dead: bool,

    // senders holds a queue of blocked message senders. Senders are notified
    // when enough space is available in the queue to insert their message.
    pub senders: Queue,

    // receivers holds a queue of blocked receivers. Receivers are notified
    // when a new message is inserted into the queue and can be received.
    pub receivers: Queue,

    // messages is an array of sent messages.
    pub messages: VecDeque<Message>,

    // sendTime is the last time a msgsnd was perfomed.
    pub sendTime: Time,

    // receiveTime is the last time a msgrcv was performed.
    pub receiveTime: Time,

    // changeTime is the last time the queue was modified using msgctl.
    pub changeTime: Time,

    // byteCount is the current number of message bytes in the queue.
    pub byteCount: u64,

    // maxBytes is the maximum allowed number of bytes in the queue, and is also
    // used as a limit for the number of total possible messages.
    pub maxBytes: u64,

    // sendPID is the PID of the process that performed the last msgsnd.
    pub sendPID: i32,

    // receivePID is the PID of the process that performed the last msgrcv.
    pub receivePID: i32,
}

impl MsgQueueIntern {
    // Copy copies a message from the queue without deleting it. If no message
    // exists, an error is returned. See msgrcv(MSG_COPY).
    pub fn Copy(&self, mType: i64) -> Result<Message> {
        if mType < 0 || self.messages.len() == 0 {
            return Err(Error::SysError(SysErr::ENOMSG));
        }

        match self.msgAtIndex(mType) {
            None => return Err(Error::SysError(SysErr::ENOMSG)),
            Some(msg) => {
                return Ok(msg.MakeCopy());
            }
        }
    }

    // msgOfType returns the first message with the specified type, nil if no
    // message is found. If except is true, the first message of a type not equal
    // to mType will be returned.
    pub fn msgOfType(&self, mType: i64, except: bool) -> Option<(Message, usize)> {
        if except {
            for i in 0..self.messages.len() {
                let m = self.messages[i].clone();
                if m.Type() != mType {
                    return Some((m, i))
                }
            }

            return None
        } else {
            for i in 0..self.messages.len() {
                let m = self.messages[i].clone();
                if m.Type() == mType {
                    return Some((m, i))
                }
            }

            return None
        }
    }

    // msgOfTypeLessThan return the the first message with the lowest type less
    // than or equal to mType, nil if no such message exists.
    pub fn msgOfTypeLessThan(&self, mType: i64) -> Option<(Message, usize)> {
        let mut min = mType;
        let mut m = None;
        for i in 0..self.messages.len() {
            let msg = self.messages[i].clone();
            if msg.Type() <= mType && msg.Type() < min {
                min = msg.Type();
                m = Some((msg, i));
            }
        }

        return m
    }

    // msgAtIndex returns a pointer to a message at given index, nil if non exits.
    pub fn msgAtIndex(&self, mType: i64) -> Option<Message> {
        if self.messages.len() < (mType+1) as _ {
            return None
        }
        return Some(self.messages[mType as usize].clone());
    }

}

// Message represents a message exchanged through a Queue via msgsnd(2) and
// msgrcv(2).
pub struct MessageIntern {
    // Type is an integer representing the type of the sent message.
    pub Type: i64,

    // Text is an untyped block of memory.
    pub Text: Vec<u8>,

    // Size is the size of Text.
    pub Size: u64,
}

#[derive(Clone)]
pub struct Message(Arc<QMutex<MessageIntern>>);

impl Deref for Message {
    type Target = Arc<QMutex<MessageIntern>>;

    fn deref(&self) -> &Arc<QMutex<MessageIntern>> {
        &self.0
    }
}

impl Message {
    pub fn New(Type: i64, Text: Vec<u8>) -> Self {
        let intern = MessageIntern {
            Type: Type,
            Size: Text.len() as _,
            Text: Text,
        };

        return Self(Arc::new(QMutex::new(intern)))
    }

    pub fn Type(&self) -> i64 {
        return self.lock().Type
    }

    pub fn Size(&self) -> u64 {
        return self.lock().Size
    }

    pub fn Truncate(&self, maxSize: usize) {
        let mut l = self.lock();
        l.Text.resize(maxSize, 0);
        l.Size = maxSize as _;
    }

    pub fn MakeCopy(&self) -> Self {
        let mut text = Vec::new();
        for d in &self.lock().Text {
            text.push(*d);
        }
        let intern = MessageIntern {
            Type: self.Type(),
            Text: text,
            Size: self.Size(),
        };

        return Self(Arc::new(QMutex::new(intern)))
    }
}