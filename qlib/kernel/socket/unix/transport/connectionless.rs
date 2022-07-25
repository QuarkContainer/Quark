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
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::sync::Weak;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::super::common::*;
use super::super::super::super::super::linux::socket::*;
use super::super::super::super::super::linux_def::*;
use super::super::super::super::kernel::waiter::*;
use super::super::super::super::task::*;
use super::super::super::super::tcpip::tcpip::*;
use super::super::super::socketopts::*;
//use super::super::super::control::*;
use super::connectioned::*;
use super::queue::*;
use super::unix::*;

// connectionlessEndpoint is a unix endpoint for unix sockets that support operating in
// a conectionless fashon.
//
// Specifically, this means datagram unix sockets not created with
// socketpair(2).

#[derive(Clone)]
pub struct ConnectionLessEndPointWeak(Weak<QMutex<BaseEndpointInternal>>);

impl ConnectionLessEndPointWeak {
    pub fn Upgrade(&self) -> Option<ConnectionLessEndPoint> {
        let c = match self.0.upgrade() {
            None => return None,
            Some(c) => c,
        };

        return Some(ConnectionLessEndPoint(BaseEndpoint(c)));
    }
}

impl ConnectionLessEndPoint {
    pub fn Downgrade(&self) -> ConnectionLessEndPointWeak {
        return ConnectionLessEndPointWeak(Arc::downgrade(&(self.0).0));
    }
}

#[derive(Clone, PartialEq)]
pub struct ConnectionLessEndPoint(BaseEndpoint);

impl Deref for ConnectionLessEndPoint {
    type Target = BaseEndpoint;

    fn deref(&self) -> &BaseEndpoint {
        &self.0
    }
}

impl ConnectionLessEndPoint {
    pub fn New() -> Self {
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

        let queue = bep.lock().queue.clone();
        let queueReceiver =
            QueueReceiver::New(MsgQueue::New(queue, Queue::default(), DEFAULT_BUFFER_SIZE));
        bep.lock().receiver = Some(Arc::new(queueReceiver));
        return Self(bep);
    }

    pub fn IsBound(&self) -> bool {
        return self.0.IsBound();
    }

    pub fn State(&self) -> i32 {
        if self.IsBound() {
            return SS_UNCONNECTED;
        } else if self.lock().Connected() {
            return SS_CONNECTING;
        }

        return SS_DISCONNECTING;
    }

    pub fn RefCount(&self) -> usize {
        return Arc::strong_count(&self.0);
    }

    // BidirectionalConnect implements BoundEndpoint.BidirectionalConnect.
    pub fn BidirectionalConnect<T: 'static + ConnectingEndpoint>(
        &self,
        _task: &Task,
        _ce: Arc<T>,
        _returnConnect: impl Fn(Arc<Receiver>, Arc<ConnectedEndpoint>),
    ) -> Result<()> {
        return Err(Error::SysError(SysErr::ECONNREFUSED));
    }

    // UnidirectionalConnect implements BoundEndpoint.UnidirectionalConnect.
    pub fn UnidirectionalConnect(&self) -> Result<UnixConnectedEndpoint> {
        let r = self.lock().receiver.clone();

        let r = match r {
            None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
            Some(r) => r,
        };

        let q = match r.as_any().downcast_ref::<QueueReceiver>() {
            None => return Err(Error::SysError(SysErr::ECONNREFUSED)),
            Some(q) => q.readQueue.clone(),
        };

        return Ok(UnixConnectedEndpoint::New(Arc::new(self.clone()), q));
    }
}

impl Passcred for ConnectionLessEndPoint {
    fn Passcred(&self) -> bool {
        return self.0.Passcred();
    }
}

impl PartialEndPoint for ConnectionLessEndPoint {
    fn Type(&self) -> i32 {
        return SockType::SOCK_DGRAM;
    }

    // GetLocalAddress returns the bound path.
    fn GetLocalAddress(&self) -> Result<SockAddrUnix> {
        return self.0.GetLocalAddress();
    }
}

impl Endpoint for ConnectionLessEndPoint {
    fn as_any(&self) -> &Any {
        return self;
    }

    // Close puts the endpoint in a closed state and frees all resources associated
    // with it.
    //
    // The socket will be a fresh state after a call to close and may be reused.
    // That is, close may be used to "unbind" or "disconnect" the socket in error
    // paths.
    fn Close(&self) {
        let mut r = None;
        {
            let mut e = self.lock();

            if e.Connected() {
                e.receiver.as_ref().unwrap().CloseRecv();
                r = e.receiver.take();
                e.connected = None;
            }

            if e.path.len() != 0 {
                e.path = "".to_string();
            }
        }

        if let Some(r) = r {
            r.CloseNotify();
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
        return self.0.RecvMsg(data, creds, numRights, peek, addr);
    }

    // SendMsg writes data and a control message to the specified endpoint.
    // This method does not block if the data cannot be written.
    fn SendMsg(
        &self,
        data: &[IoVec],
        c: &SCMControlMessages,
        to: &Option<BoundEndpoint>,
    ) -> Result<usize> {
        let tmp = to.clone();
        let to = match tmp {
            None => return self.0.SendMsg(data, c, to),
            Some(to) => to.clone(),
        };

        let connected = match to.UnidirectionalConnect() {
            Err(_) => return Err(Error::SysError(SysErr::EINVAL)),
            Ok(c) => c,
        };

        let (n, notify) = connected.Send(data, c, &SockAddrUnix::New(&self.lock().path))?;

        if notify {
            connected.SendNotify();
        }

        return Ok(n);
    }

    // Connect attempts to connect directly to server.
    fn Connect(&self, _task: &Task, server: &BoundEndpoint) -> Result<()> {
        let connected = server.UnidirectionalConnect()?;

        self.lock().connected = Some(Arc::new(connected));
        return Ok(());
    }

    fn Shutdown(&self, flags: ShutdownFlags) -> Result<()> {
        return self.0.Shutdown(flags);
    }

    // Listen starts listening on the connection.
    fn Listen(&self, _: i32) -> Result<()> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    // Accept accepts a new connection.
    fn Accept(&self) -> Result<ConnectionedEndPoint> {
        return Err(Error::SysError(SysErr::EOPNOTSUPP));
    }

    // Bind binds the connection.
    //
    // For Unix endpoints, this _only sets the address associated with the socket_.
    // Work associated with sockets in the filesystem or finding those sockets must
    // be done by a higher level.
    //
    // Bind will fail only if the socket is connected, bound or the passed address
    // is invalid (the empty string).
    fn Bind(&self, addr: &SockAddrUnix) -> Result<()> {
        let mut e = self.lock();

        if e.path.len() != 0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        e.path = addr.Path.clone();
        return Ok(());
    }

    fn GetRemoteAddress(&self) -> Result<SockAddrUnix> {
        return self.0.GetRemoteAddress();
    }
}

impl ConnectedPasscred for ConnectionLessEndPoint {
    fn ConnectedPasscred(&self) -> bool {
        return self.0.ConnectedPasscred();
    }
}

impl Waitable for ConnectionLessEndPoint {
    fn Readiness(&self, _task: &Task, mask: EventMask) -> EventMask {
        let e = self.lock();

        let mut ready = 0;
        if mask & READABLE_EVENT != 0 && e.receiver.as_ref().unwrap().Readable() {
            ready |= READABLE_EVENT;
        }

        if e.Connected() {
            if mask & WRITEABLE_EVENT != 0 && e.connected.as_ref().unwrap().Writable() {
                ready |= WRITEABLE_EVENT;
            }
        }

        return ready;
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        self.0.EventRegister(task, e, mask)
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        self.0.EventUnregister(task, e)
    }
}
