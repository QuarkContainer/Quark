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

use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::any::Any;
use core::ops::Deref;

use crate::qlib::mutex::*;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux::socket::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::kernel::waiter::bufchan::*;
use super::super::super::super::kernel::waiter::*;
use super::super::super::super::task::*;
use super::super::super::super::tcpip::tcpip::*;
use super::super::super::super::uid::*;
use super::super::super::socketopts::*;
//use super::super::super::control::*;
use super::queue::*;
use super::unix::*;

// A ConnectingEndpoint is a connectioned unix endpoint that is attempting to
// establish a bidirectional connection with a BoundEndpoint.
pub trait ConnectingEndpoint: PartialEndPoint {
    fn Lock(&self) -> QMutexGuard<()>;

    fn as_any(&self) -> &Any;

    // ID returns the endpoint's globally unique identifier. This identifier
    // must be used to determine locking order if more than one endpoint is
    // to be locked in the same codepath. The endpoint with the smaller
    // identifier must be locked before endpoints with larger identifiers.
    fn ID(&self) -> u64;

    /*// Passcred implements socket.Credentialer.Passcred.
    fn Passcred(&self) -> bool;

    // Type returns the socket type, typically either SockStream or
    // SockSeqpacket. The connection attempt must be aborted if this
    // value doesn't match the ConnectableEndpoint's type.
    fn Type(&self) -> i32;

    // GetLocalAddress returns the bound path.
    fn GetLocalAddress(&self) -> Result<FullAddr>;*/

    // Connected returns true iff the ConnectingEndpoint is in the connected
    // state. ConnectingEndpoints can only be connected to a single endpoint,
    // so the connection attempt must be aborted if this returns true.
    fn Connected(&self) -> bool;

    // Listening returns true iff the ConnectingEndpoint is in the listening
    // state. ConnectingEndpoints cannot make connections while listening, so
    // the connection attempt must be aborted if this returns true.
    fn Listening(&self) -> bool;

    // WaiterQueue returns a pointer to the endpoint's waiter queue.
    fn WaiterQueue(&self) -> Queue;
}

pub struct ConnectionedEndPointInternal {
    pub baseEndpoint: BaseEndpoint,

    // id is the unique endpoint identifier. This is used exclusively for
    // lock ordering within connect.
    pub id: u64,

    // stype is used by connecting sockets to ensure that they are the
    // same type. The value is typically either tcpip.SockSeqpacket or
    // tcpip.SockStream.
    pub stype: i32,

    //accept backlog. If backlog is 0, it is not listening
    pub backlog: i32,

    // acceptedChan is per the TCP endpoint implementation. Note that the
    // sockets in this channel are _already in the connected state_, and
    // have another associated connectionedEndpoint.
    //
    // If nil, then no listen call has been made.
    pub acceptedChan: QMutex<Option<BufChan<ConnectionedEndPoint>>>,
}

impl ConnectionedEndPointInternal {
    pub fn Connected(&self) -> bool {
        let e = self.baseEndpoint.lock();
        return e.receiver.is_some() && e.connected.is_some();
    }

    // isBound returns true iff the connectionedEndpoint is bound (but not
    // listening).
    pub fn IsBound(&self) -> bool {
        let e = self;
        return e.baseEndpoint.lock().path.len() != 0 && e.acceptedChan.lock().is_none();
    }

    pub fn Listening(&self) -> bool {
        return self.acceptedChan.lock().is_some();
    }
}

#[derive(Clone)]
pub struct ConnectionedEndPointWeak(Weak<(ConnectionedEndPointInternal, QMutex<()>)>);

impl ConnectionedEndPointWeak {
    pub fn Upgrade(&self) -> Option<ConnectionedEndPoint> {
        let c = match self.0.upgrade() {
            None => return None,
            Some(c) => c,
        };

        return Some(ConnectionedEndPoint(c));
    }
}

#[derive(Clone)]
pub struct ConnectionedEndPoint(Arc<(ConnectionedEndPointInternal, QMutex<()>)>);

impl ConnectionedEndPoint {
    pub fn Downgrade(&self) -> ConnectionedEndPointWeak {
        return ConnectionedEndPointWeak(Arc::downgrade(&self.0));
    }
}

impl Deref for ConnectionedEndPoint {
    type Target = ConnectionedEndPointInternal;

    fn deref(&self) -> &ConnectionedEndPointInternal {
        &(self.0).0
    }
}

impl PartialEq for ConnectionedEndPoint {
    fn eq(&self, other: &Self) -> bool {
        return Arc::ptr_eq(&self.0, &other.0);
    }
}

impl Eq for ConnectionedEndPoint {}

impl ConnectionedEndPoint {
    pub fn New(stype: i32) -> Self {
        let bep = BaseEndpoint::default();
        let ops = bep.SockOps();
        let sendBufferLimits = BufferSizeOption {
            Min: MINIMUM_BUFFER_SIZE,
            Default: DEFAULT_BUFFER_SIZE,
            Max: MAX_BUFFER_SIZE,
        };

        let receiveBufferLimits = BufferSizeOption {
            Min: MINIMUM_BUFFER_SIZE,
            Default: DEFAULT_BUFFER_SIZE,
            Max: MAX_BUFFER_SIZE,
        };

        ops.InitLimit(sendBufferLimits, receiveBufferLimits);
        ops.SetSendBufferSize(DEFAULT_BUFFER_SIZE as _, false);
        ops.SetReceiveBufferSize(DEFAULT_BUFFER_SIZE as _, false);

        let internal = ConnectionedEndPointInternal {
            baseEndpoint: bep,
            id: NewUID(),
            stype: stype,
            backlog: 0,
            acceptedChan: QMutex::new(None),
        };

        return Self(Arc::new((internal, QMutex::new(()))));
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&self.0);
    }

    pub fn State(&self) -> i32 {
        if self.Connected() {
            return SS_CONNECTED;
        }

        return SS_UNCONNECTED;
    }

    pub fn NewWithBaseEndpoint(baseEndpoint: BaseEndpoint, stype: i32) -> Self {
        let internal = ConnectionedEndPointInternal {
            baseEndpoint: baseEndpoint,
            id: NewUID(),
            stype: stype,
            backlog: 0,
            acceptedChan: QMutex::new(None),
        };

        return Self(Arc::new((internal, QMutex::new(()))));
    }

    pub fn NewPair(stype: i32) -> (Self, Self) {
        let a = Self::New(stype);
        let b = Self::New(stype);

        let aq = a.baseEndpoint.lock().queue.clone();
        let bq = b.baseEndpoint.lock().queue.clone();

        let q1 = MsgQueue::New(aq.clone(), bq.clone(), INITIAL_LIMIT);
        let q2 = MsgQueue::New(bq.clone(), aq.clone(), INITIAL_LIMIT);

        if stype == SockType::SOCK_STREAM {
            a.baseEndpoint.lock().receiver =
                Some(Arc::new(StreamQueueReceiver::New(q1.clone())));
            b.baseEndpoint.lock().receiver =
                Some(Arc::new(StreamQueueReceiver::New(q2.clone())));
        } else {
            a.baseEndpoint.lock().receiver = Some(Arc::new(QueueReceiver {
                readQueue: q1.clone(),
            }));
            b.baseEndpoint.lock().receiver = Some(Arc::new(QueueReceiver {
                readQueue: q2.clone(),
            }));
        }

        a.baseEndpoint.lock().connected = Some(Arc::new(UnixConnectedEndpoint {
            endpoint: Arc::new(b.clone()),
            writeQueue: q2,
        }));

        b.baseEndpoint.lock().connected = Some(Arc::new(UnixConnectedEndpoint {
            endpoint: Arc::new(a.clone()),
            writeQueue: q1,
        }));

        return (a, b);
    }

    // NewExternal creates a new externally backed Endpoint. It behaves like a
    // socketpair.
    pub fn NewExternal(
        stype: i32,
        queue: Queue,
        receiver: Arc<Receiver>,
        connected: Arc<ConnectedEndpoint>,
    ) -> Self {
        let baseEndpoint = BaseEndpoint::New(queue, receiver, connected);
        let ops = baseEndpoint.SockOps();
        let sendBufferLimits = BufferSizeOption {
            Min: MINIMUM_BUFFER_SIZE,
            Default: DEFAULT_BUFFER_SIZE,
            Max: MAX_BUFFER_SIZE,
        };

        let receiveBufferLimits = BufferSizeOption {
            Min: MINIMUM_BUFFER_SIZE,
            Default: DEFAULT_BUFFER_SIZE,
            Max: MAX_BUFFER_SIZE,
        };

        ops.InitLimit(sendBufferLimits, receiveBufferLimits);
        ops.SetSendBufferSize(DEFAULT_BUFFER_SIZE as _, false);
        ops.SetReceiveBufferSize(DEFAULT_BUFFER_SIZE as _, false);

        let internal = ConnectionedEndPointInternal {
            baseEndpoint: baseEndpoint,
            id: NewUID(),
            stype: stype,
            backlog: 0,
            acceptedChan: QMutex::new(None),
        };

        return Self(Arc::new((internal, QMutex::new(()))));
    }

    pub fn TryLock(&self) -> Option<QMutexGuard<()>> {
        return (self.0).1.try_lock();
    }

    // isBound returns true iff the connectionedEndpoint is bound (but not
    // listening).
    pub fn IsBound(&self) -> bool {
        self.TryLock();
        return (self.0).0.IsBound();
    }

    pub fn BidirectionalConnect<T: 'static + ConnectingEndpoint>(
        &self,
        task: &Task,
        ce: Arc<T>,
        returnConnect: impl Fn(Arc<Receiver>, Arc<ConnectedEndpoint>),
    ) -> Result<()> {
        if ce.Type() != self.stype {
            return Err(Error::SysError(SysErr::EPROTOTYPE));
        }

        // Check if ce is e to avoid a deadlock.
        match ce.as_any().downcast_ref::<ConnectionedEndPoint>() {
            None => (),
            Some(e) => {
                if e.clone() == *self {
                    return Err(Error::SysError(TcpipErr::ERR_INVALID_ENDPOINT_STATE.sysErr));
                }
            }
        }

        // Do a dance to safely acquire locks on both endpoints.
        let (lock1, lock2) = if self.ID() < ce.ID() {
            (self.Lock(), ce.Lock())
        } else {
            (ce.Lock(), self.Lock())
        };

        // Check connecting state.
        if ce.Connected() {
            return Err(Error::SysError(TcpipErr::ERR_ALREADY_CONNECTED.sysErr));
        }

        if ce.Listening() {
            return Err(Error::SysError(TcpipErr::ERR_INVALID_ENDPOINT_STATE.sysErr));
        }

        // Check bound state.
        if !self.Listening() {
            return Err(Error::SysError(TcpipErr::ERR_CONNECTION_REFUSED.sysErr));
        }

        // Create a newly bound connectionedEndpoint.
        let baseEndPoint = BaseEndpoint::default();
        baseEndPoint.lock().path = self.baseEndpoint.lock().path.to_string();
        let stype = self.stype;
        let ops = baseEndPoint.SockOps();
        let ne = ConnectionedEndPoint::NewWithBaseEndpoint(baseEndPoint, stype);

        let sendBufferLimits = BufferSizeOption {
            Min: MINIMUM_BUFFER_SIZE,
            Default: DEFAULT_BUFFER_SIZE,
            Max: MAX_BUFFER_SIZE,
        };

        let receiveBufferLimits = BufferSizeOption {
            Min: MINIMUM_BUFFER_SIZE,
            Default: DEFAULT_BUFFER_SIZE,
            Max: MAX_BUFFER_SIZE,
        };

        ops.InitLimit(sendBufferLimits, receiveBufferLimits);
        ops.SetSendBufferSize(DEFAULT_BUFFER_SIZE as _, false);
        ops.SetReceiveBufferSize(DEFAULT_BUFFER_SIZE as _, false);

        let readq = ce.WaiterQueue();
        let writeq = ne.baseEndpoint.lock().queue.clone();
        let readQueue = MsgQueue::New(readq.clone(), writeq.clone(), ops.GetSendBufferSize() as _);
        ne.baseEndpoint.lock().connected = Some(Arc::new(UnixConnectedEndpoint::New(
            ce.clone(),
            readQueue.clone(),
        )));

        let writeQueue = MsgQueue::New(writeq.clone(), readq.clone(), ops.GetReceiveBufferSize() as _);
        if self.stype == SockType::SOCK_STREAM {
            ne.baseEndpoint.lock().receiver =
                Some(Arc::new(StreamQueueReceiver::New(writeQueue.clone())))
        } else {
            let receiver = QueueReceiver::New(writeQueue.clone());
            ne.baseEndpoint.lock().receiver = Some(Arc::new(receiver));
        }

        let chan = self.acceptedChan.lock().clone().unwrap();
        match chan.Write(task, ne.clone()) {
            Err(e) => {
                //return Err(Error::SysError(SysErr::ECONNREFUSED))
                return Err(e)
            },
            Ok(()) => {
                let connected = UnixConnectedEndpoint::New(Arc::new(ne), writeQueue);
                if self.stype == SockType::SOCK_STREAM {
                    let receive = StreamQueueReceiver::New(readQueue.clone());
                    returnConnect(Arc::new(receive), Arc::new(connected));
                } else {
                    let receive = QueueReceiver::New(readQueue.clone());
                    returnConnect(Arc::new(receive), Arc::new(connected));
                }

                core::mem::drop(lock2);
                core::mem::drop(lock1);

                let q = self.baseEndpoint.lock().queue.clone();
                q.Notify(EVENT_IN);
                ce.WaiterQueue().Notify(WRITEABLE_EVENT);

                return Ok(());
            }
        }
    }

    // UnidirectionalConnect implements BoundEndpoint.UnidirectionalConnect.
    pub fn UnidirectionalConnect(&self) -> Result<UnixConnectedEndpoint> {
        return Err(Error::SysError(SysErr::ECONNREFUSED));
    }
}

impl ConnectingEndpoint for ConnectionedEndPoint {
    fn Lock(&self) -> QMutexGuard<()> {
        return (self.0).1.lock();
    }

    fn as_any(&self) -> &Any {
        return self;
    }

    fn ID(&self) -> u64 {
        return self.id;
    }

    fn Connected(&self) -> bool {
        return (self.0).0.Connected();
    }

    fn Listening(&self) -> bool {
        return (self.0).0.Listening();
    }

    fn WaiterQueue(&self) -> Queue {
        return self.baseEndpoint.lock().queue.clone();
    }
}

impl Passcred for ConnectionedEndPoint {
    fn Passcred(&self) -> bool {
        return self.baseEndpoint.Passcred();
    }
}

impl PartialEndPoint for ConnectionedEndPoint {
    fn Type(&self) -> i32 {
        return self.stype;
    }

    // GetLocalAddress returns the bound path.
    fn GetLocalAddress(&self) -> Result<SockAddrUnix> {
        return self.baseEndpoint.GetLocalAddress();
    }
}

impl Endpoint for ConnectionedEndPoint {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn Close(&self) {
        let mut c: Option<Arc<ConnectedEndpoint>> = None;
        let mut r: Option<Arc<Receiver>> = None;

        {
            self.TryLock();
            let e = self;

            if e.Connected() {
                let mut baseEndpoint = e.baseEndpoint.lock();
                c = baseEndpoint.connected.take();
                r = baseEndpoint.receiver.take();

                // Still have unread data? If yes, we set this into the write
                // end so that the peer can get ECONNRESET) when it does read.
                if r.as_ref().unwrap().RecvQueuedSize() > 0 {
                    c.as_ref().unwrap().CloseUnread();
                }

                c.as_ref().unwrap().CloseSend();
                r.as_ref().unwrap().CloseRecv();
            } else if e.IsBound() {
                let mut baseEndpoint = e.baseEndpoint.lock();
                baseEndpoint.path = "".to_string();
            } else if e.Listening() {
                {
                    let chan = e.acceptedChan.lock().take().unwrap();
                    chan.Close();
                    for n in &chan.lock().buf {
                        n.Close();
                    }
                }

                let mut baseEndpoint = e.baseEndpoint.lock();
                baseEndpoint.path = "".to_string();
            }
        }

        if c.is_some() {
            c.as_ref().unwrap().CloseNotify();
        }

        if r.is_some() {
            r.as_ref().unwrap().CloseNotify();
        }
    }

    fn RecvMsg(
        &self,
        data: &mut [IoVec],
        creds: bool,
        numRights: u64,
        peek: bool,
        addr: Option<&mut SockAddrUnix>,
    ) -> Result<(usize, usize, SCMControlMessages, bool)> {
        self.TryLock();
        return self
            .baseEndpoint
            .RecvMsg(data, creds, numRights, peek, addr);
    }

    // SendMsg writes data and a control message to the endpoint's peer.
    // This method does not block if the data cannot be written.
    fn SendMsg(
        &self,
        data: &[IoVec],
        c: &SCMControlMessages,
        to: &Option<BoundEndpoint>,
    ) -> Result<usize> {
        // Stream sockets do not support specifying the endpoint. Seqpacket
        // sockets ignore the passed endpoint.
        self.TryLock();
        let stype = self.stype;
        if stype == SockType::SOCK_STREAM && to.is_some() {
            return Err(Error::SysError(SysErr::EOPNOTSUPP));
        }

        let baseEndpoint = self.baseEndpoint.clone();
        return baseEndpoint.SendMsg(data, c, to);
    }

    fn Shutdown(&self, flags: ShutdownFlags) -> Result<()> {
        self.TryLock();
        return self.baseEndpoint.Shutdown(flags);
    }

    // Connect attempts to directly connect to another Endpoint.
    // Implements Endpoint.Connect.
    fn Connect(&self, task: &Task, server: &BoundEndpoint) -> Result<()> {
        let returnConnect = |r: Arc<Receiver>, ce: Arc<ConnectedEndpoint>| {
            self.baseEndpoint.lock().receiver = Some(r);
            self.baseEndpoint.lock().connected = Some(ce);
        };

        return server.BidirectionalConnect(task, Arc::new(self.clone()), returnConnect);
    }

    // Listen starts listening on the connection.
    fn Listen(&self, backlog: i32) -> Result<()> {
        self.TryLock();
        let e = self;

        if e.Listening() {
            // Adjust the size of the channel iff we can fix existing
            // pending connections into the new one.
            let newChan = BufChan::New(backlog as usize);

            {
                let origChan = e.acceptedChan.lock().clone().unwrap();
                let mut origChanLock = origChan.lock();
                if origChanLock.buf.len() > backlog as usize {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let mut newChanLock = newChan.lock();

                while origChanLock.buf.len() > 0 {
                    let ep = origChanLock.buf.pop_front().unwrap();
                    newChanLock.buf.push_back(ep);
                }
            }

            *e.acceptedChan.lock() = Some(newChan);
            return Ok(());
        }

        if !e.IsBound() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        *e.acceptedChan.lock() = Some(BufChan::New(backlog as usize));
        return Ok(());
    }

    // Accept accepts a new connection.
    fn Accept(&self) -> Result<ConnectionedEndPoint> {
        self.TryLock();
        let e = self;
        if !e.Listening() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        match e.acceptedChan.lock().as_ref().unwrap().TryRead()? {
            None => return Err(Error::SysError(SysErr::EWOULDBLOCK)),
            Some(ep) => Ok(ep),
        }
    }

    // Bind binds the connection.
    //
    // For Unix connectionedEndpoints, this _only sets the address associated with
    // the socket_. Work associated with sockets in the filesystem or finding those
    // sockets must be done by a higher level.
    //
    // Bind will fail only if the socket is connected, bound or the passed address
    // is invalid (the empty string).
    fn Bind(&self, addr: &SockAddrUnix) -> Result<()> {
        self.TryLock();
        let e = self;
        if e.IsBound() || e.Listening() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if addr.Path.len() == 0 {
            // The empty string is not permitted.
            return Err(Error::SysError(SysErr::EADDRNOTAVAIL));
        }

        e.baseEndpoint.lock().path = addr.Path.clone();
        return Ok(());
    }

    fn GetRemoteAddress(&self) -> Result<SockAddrUnix> {
        self.TryLock();
        return self.baseEndpoint.GetRemoteAddress();
    }
}

impl ConnectedPasscred for ConnectionedEndPoint {
    fn ConnectedPasscred(&self) -> bool {
        return self.baseEndpoint.ConnectedPasscred();
    }
}

impl Waitable for ConnectionedEndPoint {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        self.TryLock();
        let e = self;

        let mut ready = 0;
        if e.Connected() {
            if mask & READABLE_EVENT != 0 && e.baseEndpoint.lock().receiver.as_ref().unwrap().Readable() {
                ready |= READABLE_EVENT
            }
            if mask & WRITEABLE_EVENT != 0 && e.baseEndpoint.lock().connected.as_ref().unwrap().Writable()
            {
                ready |= WRITEABLE_EVENT
            }
        } else if e.Listening() {
            if mask & READABLE_EVENT != 0 && e.acceptedChan.lock().as_ref().unwrap().Len() > 0 {
                ready |= READABLE_EVENT
            }
        }

        return ready;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        self.TryLock();
        self.baseEndpoint.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        self.TryLock();
        self.baseEndpoint.EventUnregister(task, e)
    }
}
