// Copyright (c) 2021 QuarkSoft LLC
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
use alloc::string::String;
use alloc::string::ToString;
use spin::Mutex;
use core::ops::Deref;
use core::any::Any;

use super::super::super::super::kernel::waiter::*;
use super::super::super::super::tcpip::tcpip::*;
use super::super::super::super::qlib::common::*;
use super::super::super::super::qlib::linux_def::*;
use super::super::super::super::qlib::mem::seq::*;
use super::super::super::super::uid::*;
use super::super::super::super::task::*;
use super::super::super::buffer::view::*;
use super::super::super::control::*;
use super::queue::*;
use super::connectioned::*;
use super::connectionless::*;

pub const INITIAL_LIMIT: usize = 16 * 1024;

pub struct SockType;

impl SockType {
    // SockStream corresponds to syscall.SOCK_STREAM.
    pub const SOCK_STREAM: i32 = 1;
    // SockDgram corresponds to syscall.SOCK_DGRAM.
    pub const SOCK_DGRAM: i32 = 2;
    // SockRaw corresponds to syscall.SOCK_RAW.
    pub const SOCK_RAW: i32 = 3;
    // SockSeqpacket corresponds to syscall.SOCK_SEQPACKET.
    pub const SOCK_SEQPACKET: i32 = 5;
}

#[derive(Clone, Default)]
pub struct SCMControlMessages {
    // Rights is a control message containing FDs.
    pub Rights: Option<SCMRights>,

    // Credentials is a control message containing Unix credentials.
    pub Credentials: Option<ScmCredentials>,
}

impl SCMControlMessages {
    // Empty returns true iff the ControlMessages does not contain either
    // credentials or rights.
    pub fn Empty(&self) -> bool {
        return self.Rights.is_none() && self.Credentials.is_none()
    }

    pub fn Release(&mut self) {
        self.Rights = None;
        self.Credentials = None;
    }
}

pub trait ConnectedPasscred {
    // ConnectedPasscred returns whether or not the SO_PASSCRED socket option
    // is enabled on the connected end.
    fn ConnectedPasscred(&self) -> bool;
}

pub trait Passcred {
    // Passcred returns whether or not the SO_PASSCRED socket option is
    // enabled on this end.
    fn Passcred(&self) -> bool;
}

pub trait PartialEndPoint : Passcred + Sync + Send {
    // GetLocalAddress implements Endpoint.GetLocalAddress.
    fn GetLocalAddress(&self) -> Result<SockAddrUnix>;

    // Type implements Endpoint.Type.
    fn Type(&self) -> i32;
}

pub trait Endpoint : PartialEndPoint + Passcred + ConnectedPasscred + Waitable + Sync + Send {
    fn as_any(&self) -> &Any;

    // Close puts the endpoint in a closed state and frees all resources
    // associated with it.
    fn Close(&self);

    // RecvMsg reads data and a control message from the endpoint. This method
    // does not block if there is no data pending.
    //
    // creds indicates if credential control messages are requested by the
    // caller. This is useful for determining if control messages can be
    // coalesced. creds is a hint and can be safely ignored by the
    // implementation if no coalescing is possible. It is fine to return
    // credential control messages when none were requested or to not return
    // credential control messages when they were requested.
    //
    // numRights is the number of SCM_RIGHTS FDs requested by the caller. This
    // is useful if one must allocate a buffer to receive a SCM_RIGHTS message
    // or determine if control messages can be coalesced. numRights is a hint
    // and can be safely ignored by the implementation if the number of
    // available SCM_RIGHTS FDs is known and no coalescing is possible. It is
    // fine for the returned number of SCM_RIGHTS FDs to be either higher or
    // lower than the requested number.
    //
    // If peek is true, no data should be consumed from the Endpoint. Any and
    // all data returned from a peek should be available in the next call to
    // RecvMsg.
    //
    // recvLen is the number of bytes copied into data.
    //
    // msgLen is the length of the read message consumed for datagram Endpoints.
    // msgLen is always the same as recvLen for stream Endpoints.
    //
    // CMTruncated indicates that the numRights hint was used to receive fewer
    // than the total available SCM_RIGHTS FDs. Additional truncation may be
    // required by the caller.
    fn RecvMsg(&self, data: &mut [IoVec], creds: bool, numRights: u64, peek: bool, addr: Option<&mut SockAddrUnix>)
               -> Result<(usize, usize, SCMControlMessages, bool)>;

    // SendMsg writes data and a control message to the endpoint's peer.
    // This method does not block if the data cannot be written.
    //
    // SendMsg does not take ownership of any of its arguments on error.
    fn SendMsg(&self, data: &[IoVec], c: &SCMControlMessages, to: &Option<BoundEndpoint>) -> Result<usize>;

    // Connect connects this endpoint directly to another.
    //
    // This should be called on the client endpoint, and the (bound)
    // endpoint passed in as a parameter.
    //
    // The error codes are the same as Connect.
    fn Connect(&self, task: &Task, server: &BoundEndpoint) -> Result<()>;

    // Shutdown closes the read and/or write end of the endpoint connection
    // to its peer.
    fn Shutdown(&self, flags: ShutdownFlags) -> Result<()>;

    // Listen puts the endpoint in "listen" mode, which allows it to accept
    // new connections.
    fn Listen(&self, backlog: i32) -> Result<()>;

    // Accept returns a new endpoint if a peer has established a connection
    // to an endpoint previously set to listen mode. This method does not
    // block if no new connections are available.
    //
    // The returned Queue is the wait queue for the newly created endpoint.
    fn Accept(&self) -> Result<ConnectionedEndPoint>;

    // Bind binds the endpoint to a specific local address and port.
    // Specifying a NIC is optional.
    //
    // An optional commit function will be executed atomically with respect
    // to binding the endpoint. If this returns an error, the bind will not
    // occur and the error will be propagated back to the caller.
    fn Bind(&self, addr: &SockAddrUnix) -> Result<()>;

    // GetRemoteAddress returns the address to which the endpoint is
    // connected.
    fn GetRemoteAddress(&self) -> Result<SockAddrUnix>;

    // SetSockOpt sets a socket option. opt should be one of the tcpip.*Option
    // types.
    fn SetSockOpt(&self, opt: &SockOpt) -> Result<()>;

    // GetSockOpt gets a socket option. opt should be a pointer to one of the
    // tcpip.*Option types.
    fn GetSockOpt(&self, opt: &mut SockOpt) -> Result<()>;
}

#[derive(Clone)]
pub enum BoundEndpointWeak {
    Connected(ConnectionedEndPointWeak),
    ConnectLess(ConnectionLessEndPointWeak),
}

impl BoundEndpointWeak {
    pub fn Upgrade(&self) -> Option<BoundEndpoint> {
        match self {
            BoundEndpointWeak::Connected(ref c) => {
                match c.Upgrade() {
                    None => None,
                    Some(c) => Some(BoundEndpoint::Connected(c)),
                }
            }
            BoundEndpointWeak::ConnectLess(ref c) => {
                match c.Upgrade() {
                    None => None,
                    Some(c) => Some(BoundEndpoint::ConnectLess(c)),
                }
            }
        }
    }
}

#[derive(Clone)]
pub enum BoundEndpoint {
    Connected(ConnectionedEndPoint),
    ConnectLess(ConnectionLessEndPoint),
}

impl BoundEndpoint {
    pub fn RefCount(&self) -> usize {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.RefCount()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.RefCount()
            }
        }
    }

    pub fn Downgrade(&self) -> BoundEndpointWeak {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return BoundEndpointWeak::Connected(c.Downgrade());
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return BoundEndpointWeak::ConnectLess(c.Downgrade());
            }
        }
    }

    pub fn BidirectionalConnect<T: 'static + ConnectingEndpoint>(&self,
                                                                 task: &Task,
                                                                 ce: Arc<T>,
                                                                 returnConnect: impl Fn(Arc<Receiver>, Arc<ConnectedEndpoint>)) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.BidirectionalConnect(task, ce, returnConnect)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.BidirectionalConnect(task, ce, returnConnect)
            }
        }
    }

    pub fn UnidirectionalConnect(&self) -> Result<UnixConnectedEndpoint> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.UnidirectionalConnect()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.UnidirectionalConnect()
            }
        }
    }

    pub fn State(&self) -> i32 {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.State()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.State()
            }
        }
    }
}

impl Endpoint for BoundEndpoint {
    fn as_any(&self) -> &Any {
        return self
    }

    fn Close(&self) {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Close()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Close()
            }
        }
    }

    fn RecvMsg(&self, data: &mut [IoVec], creds: bool, numRights: u64, peek: bool, addr: Option<&mut SockAddrUnix>)
               -> Result<(usize, usize, SCMControlMessages, bool)> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.RecvMsg(data, creds, numRights, peek, addr)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.RecvMsg(data, creds, numRights, peek, addr)
            }
        }
    }

    fn SendMsg(&self, data: &[IoVec], ctrl: &SCMControlMessages, to: &Option<BoundEndpoint>) -> Result<usize> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.SendMsg(data, ctrl, to)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.SendMsg(data, ctrl, to)
            }
        }
    }

    fn Connect(&self, task: &Task, server: &BoundEndpoint) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Connect(task, server)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Connect(task, server)
            }
        }
    }

    fn Shutdown(&self, flags: ShutdownFlags) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Shutdown(flags)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Shutdown(flags)
            }
        }
    }

    fn Listen(&self, backlog: i32) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Listen(backlog)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Listen(backlog)
            }
        }
    }

    fn Accept(&self) -> Result<ConnectionedEndPoint> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Accept()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Accept()
            }
        }
    }

    fn Bind(&self, addr: &SockAddrUnix) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Bind(addr)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Bind(addr)
            }
        }
    }

    fn GetRemoteAddress(&self) -> Result<SockAddrUnix> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.GetRemoteAddress()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.GetRemoteAddress()
            }
        }
    }

    fn SetSockOpt(&self, opt: &SockOpt) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.SetSockOpt(opt)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.SetSockOpt(opt)
            }
        }
    }

    fn GetSockOpt(&self, opt: &mut SockOpt) -> Result<()> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.GetSockOpt(opt)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.GetSockOpt(opt)
            }
        }
    }
}

impl PartialEndPoint for BoundEndpoint {
    fn GetLocalAddress(&self) -> Result<SockAddrUnix> {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.GetLocalAddress()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.GetLocalAddress()
            }
        }
    }

    fn Type(&self) -> i32 {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Type()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Type()
            }
        }
    }
}

impl Passcred for BoundEndpoint {
    fn Passcred(&self) -> bool {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Passcred()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Passcred()
            }
        }
    }
}

impl ConnectedPasscred for BoundEndpoint {
    fn ConnectedPasscred(&self) -> bool {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.ConnectedPasscred()
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.ConnectedPasscred()
            }
        }
    }
}

impl Waitable for BoundEndpoint {
    fn Readiness(&self, task: &Task, mask: EventMask) -> EventMask {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.Readiness(task, mask)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.Readiness(task, mask)
            }
        }
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.EventRegister(task, e, mask)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.EventRegister(task, e, mask)
            }
        }
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        match self {
            BoundEndpoint::Connected(ref c) => {
                return c.EventUnregister(task, e)
            }
            BoundEndpoint::ConnectLess(ref c) => {
                return c.EventUnregister(task, e)
            }
        }
    }
}

// A Receiver can be used to receive Messages.
pub trait Receiver : Sync + Send {
    fn as_any(&self) -> &Any;

    // Recv receives a single message. This method does not block.
    //
    // See Endpoint.RecvMsg for documentation on shared arguments.
    //
    // notify indicates if RecvNotify should be called.
    fn Recv(&self, data: &mut [IoVec], _creds: bool, _numRights: u64, peek: bool)
            -> Result<(usize, usize, SCMControlMessages, bool, SockAddrUnix, bool)>;

    // RecvNotify notifies the Receiver of a successful Recv. This must not be
    // called while holding any endpoint locks.
    fn RecvNotify(&self);

    // CloseRecv prevents the receiving of additional Messages.
    //
    // After CloseRecv is called, CloseNotify must also be called.
    fn CloseRecv(&self);

    // CloseNotify notifies the Receiver of recv being closed. This must not be
    // called while holding any endpoint locks.
    fn CloseNotify(&self);

    // Readable returns if messages should be attempted to be received. This
    // includes when read has been shutdown.
    fn Readable(&self) -> bool;

    // RecvQueuedSize returns the total amount of data currently receivable.
    // RecvQueuedSize should return -1 if the operation isn't supported.
    fn RecvQueuedSize(&self) -> i64;

    // RecvMaxQueueSize returns maximum value for RecvQueuedSize.
    // RecvMaxQueueSize should return -1 if the operation isn't supported.
    fn RecvMaxQueueSize(&self) -> i64;

    // Release releases any resources owned by the Receiver. It should be
    // called before droping all references to a Receiver.
    //fn Release(&self);
}

pub trait ConnectedEndpoint : PartialEndPoint + Sync + Send {
    // Send sends a single message. This method does not block.
    //
    // notify indicates if SendNotify should be called.
    //
    // syserr.ErrWouldBlock can be returned along with a partial write if
    // the caller should block to send the rest of the data.
    fn Send(&self, data: &[IoVec], controlMessages: &SCMControlMessages, from: &SockAddrUnix) -> Result<(usize, bool)>;

    // SendNotify notifies the ConnectedEndpoint of a successful Send. This
    // must not be called while holding any endpoint locks.
    fn SendNotify(&self);

    // CloseSend prevents the sending of additional Messages.
    //
    // After CloseSend is call, CloseNotify must also be called.
    fn CloseSend(&self);

    // CloseNotify notifies the ConnectedEndpoint of send being closed. This
    // must not be called while holding any endpoint locks.
    fn CloseNotify(&self);

    // Writable returns if messages should be attempted to be sent. This
    // includes when write has been shutdown.
    fn Writable(&self) -> bool;

    // EventUpdate lets the ConnectedEndpoint know that event registrations
    // have changed.
    fn EventUpdate(&self);

    // SendQueuedSize returns the total amount of data currently queued for
    // sending. SendQueuedSize should return -1 if the operation isn't
    // supported.
    fn SendQueuedSize(&self) -> i64;

    // SendMaxQueueSize returns maximum value for SendQueuedSize.
    // SendMaxQueueSize should return -1 if the operation isn't supported.
    fn SendMaxQueueSize(&self) -> i64;

    // Release releases any resources owned by the ConnectedEndpoint. It should
    // be called before droping all references to a ConnectedEndpoint.
    //fn Release(&self);

    fn CloseUnread(&self);
}

pub struct QueueReceiver {
    pub readQueue: MsgQueue,
}

impl QueueReceiver{
    pub fn New(readQueue: MsgQueue) -> Self {
        return Self {
            readQueue
        }
    }
}

impl Receiver for QueueReceiver {
    fn as_any(&self) -> &Any {
        return self
    }

    //return (copied bytes, datalen, ControlMessages, bool?, FullAddr, notify)
    fn Recv(&self, data: &mut [IoVec], wantCreds: bool, numRights: u64, peek: bool)
            -> Result<(usize, usize, SCMControlMessages, bool, SockAddrUnix, bool)> {
        let (m, notify) = if peek {
            (self.readQueue.Peek()?, false)
        } else {
            self.readQueue.Dequeue()?
        };

        let msglen;
        let mut c;
        let addr;
        let copied;
        {
            let bs = BlockSeq::NewFromSlice(data);

            let msgLock = m.lock();
            msglen = msgLock.Data.len();
            c = msgLock.Control.clone();
            addr = msgLock.Address.clone();

            copied = bs.CopyOut(&msgLock.Data.0[..]);
        }

        if !wantCreds {
            c.Credentials = None;
        }

        let mut cmTruncated = false;
        if c.Rights.is_some() && numRights == 0 {
            c.Rights = None;
            cmTruncated = true;
        }

        return Ok((copied, msglen, c, cmTruncated, addr, notify))
    }

    fn RecvNotify(&self) {
        self.readQueue.lock().WriterQueue.Notify(EVENT_OUT)
    }

    fn CloseNotify(&self) {
        self.readQueue.lock().ReaderQueue.Notify(EVENT_IN);
        self.readQueue.lock().WriterQueue.Notify(EVENT_OUT);
    }

    fn CloseRecv(&self) {
        self.readQueue.Close();
    }

    fn Readable(&self) -> bool {
        return self.readQueue.IsReadable();
    }

    fn RecvQueuedSize(&self) -> i64 {
        return self.readQueue.QueuedSize()
    }

    fn RecvMaxQueueSize(&self) -> i64 {
        return self.readQueue.MaxQueueSize()
    }
}

pub struct StreamQueueReceiverInternal {
    pub readQueue: MsgQueue,
    pub buffer: View,
    pub control: SCMControlMessages,
    pub addr: SockAddrUnix,
}

#[derive(Clone)]
pub struct StreamQueueReceiver(Arc<Mutex<StreamQueueReceiverInternal>>);

impl Deref for StreamQueueReceiver {
    type Target = Arc<Mutex<StreamQueueReceiverInternal>>;

    fn deref(&self) -> &Arc<Mutex<StreamQueueReceiverInternal>> {
        &self.0
    }
}

//return (bytes copied, data, buf)
pub fn VecCopy<'a> (data: BlockSeq, buf: &mut View, peek: bool) -> (usize, BlockSeq) {
    let size = data.CopyOut(&buf.0);
    if !peek {
        //if not peek, consume the data
        buf.TrimFront(size);
    }
    return (size, data.DropFirst(size as u64))
}

impl StreamQueueReceiver {
    pub fn New(q: MsgQueue) -> Self {
        let internal = StreamQueueReceiverInternal {
            readQueue: q,
            buffer: View::default(),
            control: SCMControlMessages::default(),
            addr: SockAddrUnix::default(),
        };

        return Self(Arc::new(Mutex::new(internal)))
    }
}

impl Receiver for StreamQueueReceiver {
    fn as_any(&self) -> &Any {
        return self
    }

    //return (copied bytes, Message, bool?, notify)
    fn Recv(&self, data: &mut [IoVec], wantCreds: bool, numRights: u64, peek: bool)
            -> Result<(usize, usize, SCMControlMessages, bool, SockAddrUnix, bool)> {
        let mut q = self.lock();

        let data = BlockSeq::NewFromSlice(data);
        let mut notify = false;
        // If we have no data in the endpoint, we need to get some.
        if q.buffer.len() == 0 {
            // Load the next message into a buffer, even if we are peeking. Peeking
            // won't consume the message, so it will be still available to be read
            // the next time Recv() is called.
            let (m, n) = q.readQueue.Dequeue()?;
            notify = n;

            let mut mlock = m.lock();
            q.buffer = mlock.Data.Get();
            q.control = mlock.Control.clone();
            q.addr = mlock.Address.clone();
        }

        if peek {
            // Don't consume control message if we are peeking.
            let mut c = q.control.clone();

            // Don't consume data since we are peeking.
            let (copied, _) = VecCopy(data, &mut q.buffer, peek);
            if !wantCreds {
                c.Credentials = None;
            }

            let mut cmTruncated = false;
            if c.Rights.is_some() && numRights == 0 {
                c.Rights = None;
                cmTruncated = true;
            }

            return Ok((copied, copied, c, cmTruncated, q.addr.clone(), notify))
        }

        // Consume data and control message since we are not peeking.
        let (mut copied, mut data) = VecCopy(data, &mut q.buffer, false);

        // Save the original state of q.control.
        let mut c = q.control.clone();

        // Remove rights from q.control and leave behind just the creds.
        q.control.Rights = None;
        if !wantCreds {
            c.Credentials = None;
        }

        let mut cmTruncated = false;
        if c.Rights.is_some() && numRights == 0 {
            c.Rights = None;
            cmTruncated = true;
        }

        let mut hasRights = c.Rights.is_some();

        // If we have more capacity for data and haven't received any usable
        // rights.
        //
        // Linux never coalesces rights control messages.
        while !hasRights && data.Len() > 0 {
            // Get a message from the readQueue.
            let (m, n) = match q.readQueue.Dequeue() {
                Err(_) => {
                    // We already got some data, so ignore this error. This will
                    // manifest as a short read to the user, which is what Linux
                    // does.
                    break
                },
                Ok((m, n)) => (m, n)
            };

            notify |= n;
            let mut mlock = m.lock();
            q.buffer = mlock.Data.Get();
            q.control = mlock.Control.clone();
            q.addr = mlock.Address.clone();

            if wantCreds {
                if q.control.Credentials.is_none() != c.Credentials.is_none() {
                    // One message has credentials, the other does not.
                    break
                }

                if q.control.Credentials.is_some() && c.Credentials.is_some() && q.control.Credentials != c.Credentials {
                    // Both messages have credentials, but they don't match.
                    break
                }
            }

            if numRights != 0 && c.Rights.is_some() && q.control.Rights.is_some() {
                // Both messages have rights.
                break
            }

            let (cpd, datatmp) = VecCopy(data, &mut q.buffer, false);
            data = datatmp;
            copied += cpd;

            if cpd == 0 {
                // data was actually full.
                break;
            }

            if q.control.Rights.is_some() {
                // Consume rights.
                if numRights == 0 {
                    cmTruncated = true;
                } else {
                    c.Rights = q.control.Rights.clone();
                    hasRights = true;
                }

                q.control.Rights = None;
            }
        }

        return Ok((copied, copied, c, cmTruncated, q.addr.clone(), notify))
    }

    fn RecvNotify(&self) {
        let readQueue = self.lock().readQueue.clone();
        readQueue.lock().WriterQueue.Notify(EVENT_OUT);
    }

    fn CloseRecv(&self) {
        let readQueue = self.lock().readQueue.clone();
        readQueue.Close();
    }

    fn CloseNotify(&self) {
        let readQueue = self.lock().readQueue.clone();
        readQueue.lock().ReaderQueue.Notify(EVENT_IN);
        readQueue.lock().WriterQueue.Notify(EVENT_OUT);
    }

    fn Readable(&self) -> bool {
        let q = self.lock();
        let bl = q.buffer.len();
        let r = q.readQueue.IsReadable();

        // We're readable if we have data in our buffer or if the queue receiver is
        // readable.
        return bl > 0 || r;
    }

    fn RecvQueuedSize(&self) -> i64 {
        let q = self.lock();
        let bl = q.buffer.len();
        let qs = q.readQueue.QueuedSize() as usize;
        return (bl + qs) as i64;
    }

    fn RecvMaxQueueSize(&self) -> i64 {
        // The RecvMaxQueueSize() is the readQueue's MaxQueueSize() plus the largest
        // message we can buffer which is also the largest message we can receive.
        return 2 * self.lock().readQueue.MaxQueueSize()
    }
}

pub struct UnixConnectedEndpoint {
    // endpoint represents the subset of the Endpoint functionality needed by
    // the connectedEndpoint. It is implemented by both connectionedEndpoint
    // and connectionlessEndpoint and allows the use of types which don't
    // fully implement Endpoint.
    pub endpoint: Arc<PartialEndPoint>,

    pub writeQueue: MsgQueue,
}

impl UnixConnectedEndpoint {
    pub fn New<T: 'static + PartialEndPoint> (endpoint: Arc<T>, writeQueue: MsgQueue) -> Self {
        return Self {
            endpoint: endpoint,
            writeQueue: writeQueue,
        }
    }
}

impl Passcred for UnixConnectedEndpoint {
    // Passcred implements ConnectedEndpoint.Passcred.
    fn Passcred(&self) -> bool {
        return self.endpoint.Passcred();
    }
}

impl PartialEndPoint for UnixConnectedEndpoint {
    // GetLocalAddress implements ConnectedEndpoint.GetLocalAddress.
    fn GetLocalAddress(&self) -> Result<SockAddrUnix> {
        return self.endpoint.GetLocalAddress();
    }

    fn Type(&self) -> i32 {
        return self.endpoint.Type();
    }
}

impl ConnectedEndpoint for UnixConnectedEndpoint {
    // Send implements ConnectedEndpoint.Send.
    fn Send(&self, data: &[IoVec], controlMessage: &SCMControlMessages, from: &SockAddrUnix) -> Result<(usize, bool)> {
        let bs = BlockSeq::NewFromSlice(data);

        let l = bs.Len();
        let mut truncate = false;
        if self.endpoint.Type() == SockType::SOCK_STREAM {
            // Since stream sockets don't preserve message boundaries, we
            // can write only as much of the message as fits in the queue.
            truncate = true;

            // Discard empty stream packets. Since stream sockets don't
            // preserve message boundaries, sending zero bytes is a no-op.
            // In Linux, the receiver actually uses a zero-length receive
            // as an indication that the stream was closed.
            if l == 0 {
                return Ok((0, false))
            }
        }

        let v = bs.ToVec();
        let message = Message::New(v, controlMessage.clone(), from.clone());
        let (l, notify) = self.writeQueue.Enqueue(message, truncate)?;
        return Ok((l, notify))
    }

    // SendNotify implements ConnectedEndpoint.SendNotify.
    fn SendNotify(&self) {
        let readerQueue = self.writeQueue.lock().ReaderQueue.clone();
        readerQueue.Notify(EVENT_IN)
    }

    // CloseSend implements ConnectedEndpoint.CloseSend.
    fn CloseSend(&self) {
        self.writeQueue.Close();
    }

    // CloseNotify implements ConnectedEndpoint.CloseNotify.
    fn CloseNotify(&self) {
        let readerQueue = self.writeQueue.lock().ReaderQueue.clone();
        readerQueue.Notify(EVENT_IN);

        let writeQueue = self.writeQueue.lock().WriterQueue.clone();
        writeQueue.Notify(EVENT_OUT);
    }

    // Writable implements ConnectedEndpoint.Writable.
    fn Writable(&self) -> bool {
        return self.writeQueue.IsWritable();
    }

    // EventUpdate implements ConnectedEndpoint.EventUpdate.
    fn EventUpdate(&self) {}

    fn SendQueuedSize(&self) -> i64 {
        return self.writeQueue.QueuedSize();
    }

    fn SendMaxQueueSize(&self) -> i64 {
        return self.writeQueue.MaxQueueSize() as i64
    }

    fn CloseUnread(&self) {
        self.writeQueue.CloseUnread();
    }
}

// baseEndpoint is an embeddable unix endpoint base used in both the connected and connectionless
// unix domain socket Endpoint implementations.
//
// Not to be used on its own.
pub struct BaseEndpointInternal {
    pub id: u64,

    pub queue: Queue,

    // passcred specifies whether SCM_CREDENTIALS socket control messages are
    // enabled on this endpoint. Must be accessed atomically.
    pub passcred: i32,

    // receiver allows Messages to be received.
    pub receiver: Option<Arc<Receiver>>,

    // connected allows messages to be sent and state information about the
    // connected endpoint to be read.
    pub connected: Option<Arc<ConnectedEndpoint>>,

    // path is not empty if the endpoint has been bound,
    // or may be used if the endpoint is connected.
    pub path: String,

    // linger is used for SO_LINGER socket option.
    //pub linger: LingerOption,
}

impl Default for BaseEndpointInternal {
    fn default() -> Self {
        return Self {
            id: NewUID(),
            queue: Queue::default(),
            passcred: 0,
            receiver: None,
            connected: None,
            path: String::default(),
        }
    }
}

impl BaseEndpointInternal {
    pub fn Connected(&self) -> bool {
        return self.receiver.is_some() && self.connected.is_some();
    }
}

#[derive(Default)]
pub struct BaseEndpoint(Mutex<BaseEndpointInternal>);

impl Deref for BaseEndpoint {
    type Target = Mutex<BaseEndpointInternal>;

    fn deref(&self) -> &Mutex<BaseEndpointInternal> {
        &self.0
    }
}

impl Waitable for BaseEndpoint {
    fn Readiness(&self, task: &Task, mask: EventMask) -> EventMask {
        let q = self.lock().queue.clone();
        return q.Readiness(task, mask)
    }

    fn EventRegister(&self, task: &Task,e: &WaitEntry, mask: EventMask) {
        let b = self.lock();
        let queue = b.queue.clone();
        queue.EventRegister(task, e, mask);
        if b.connected.is_some() {
            b.connected.as_ref().unwrap().EventUpdate();
        }
    }

    fn EventUnregister(&self, task: &Task,e: &WaitEntry) {
        let b = self.lock();
        let queue = b.queue.clone();
        queue.EventUnregister(task, e);
        if b.connected.is_some() {
            b.connected.as_ref().unwrap().EventUpdate();
        }
    }
}

impl Passcred for BaseEndpoint {
    // Passcred implements Credentialer.Passcred.
    fn Passcred(&self) -> bool {
        return self.lock().passcred != 0;
    }
}

impl PartialEndPoint for BaseEndpoint {
    // GetLocalAddress implements Endpoint.GetLocalAddress.
    fn GetLocalAddress(&self) -> Result<SockAddrUnix> {
        panic!("no possible");
    }

    // Type implements Endpoint.Type.
    fn Type(&self) -> i32 {
        panic!("no possible");
    }
}

impl ConnectedPasscred for BaseEndpoint {
    // ConnectedPasscred implements Credentialer.ConnectedPasscred.
    fn ConnectedPasscred(&self) -> bool {
        let b = self.lock();
        return b.connected.is_some() && b.connected.as_ref().unwrap().Passcred();
    }
}

impl BaseEndpoint {
    pub fn New(queue: Queue, receiver: Arc<Receiver>, connected: Arc<ConnectedEndpoint>) -> Self {
        let internal = BaseEndpointInternal {
            queue: queue,
            passcred: 0,
            receiver: Some(receiver),
            connected: Some(connected),
            path: "".to_string(),
            ..Default::default()
        };

        return Self(Mutex::new(internal))
    }

    pub fn IsBound(&self) -> bool {
        return self.lock().path.as_str() != ""
    }

    pub fn Passcred(&self) -> bool {
        return self.lock().passcred != 0;
    }

    pub fn ConnectedPasscred(&self) -> bool {
        let e = self.lock();
        return e.connected.is_some() && e.connected.as_ref().unwrap().Passcred();
    }

    fn setPasscred(&self, pc: bool) {
        if pc {
            self.lock().passcred = 1;
        } else {
            self.lock().passcred = 0;
        }
    }

    pub fn Connected(&self) -> bool {
        let e = self.lock();
        return e.receiver.is_some() && e.connected.is_some();
    }

    // RecvMsg reads data and a control message from the endpoint.
    pub fn RecvMsg(&self, data: &mut [IoVec], creds: bool, numRights: u64, peek: bool, addr: Option<&mut SockAddrUnix>)
        -> Result<(usize, usize, SCMControlMessages, bool)> {
        let e = self.lock();

        if e.receiver.is_none() {
            return Err(Error::SysError(SysErr::ENOTCONN))
        }

        let receiver = e.receiver.as_ref().unwrap().clone();
        let (recvLen, msgLen, cms, cmt , a, notify) = receiver.Recv(data, creds, numRights, peek)?;

        if notify {
            receiver.RecvNotify();
        }

        if addr.is_some() {
            *(addr.unwrap()) = a;
        }

        return Ok((recvLen, msgLen, cms, cmt));
    }

    // SendMsg writes data and a control message to the endpoint's peer.
    // This method does not block if the data cannot be written.
    pub fn SendMsg(&self, data: &[IoVec], c: &SCMControlMessages, to: &Option<BoundEndpoint>) -> Result<usize> {
        let e = self.lock();

        if !e.Connected() {
            return Err(Error::SysError(SysErr::ENOTCONN))
        }

        if to.is_some() {
            return Err(Error::SysError(SysErr::EISCONN))
        }

        let addr = SockAddrUnix::New(&e.path);
        let (n, notify) = e.connected.as_ref().unwrap().Send(data, c, &addr)?;

        if notify {
            e.connected.as_ref().unwrap().SendNotify();
        }

        return Ok(n)
    }

    pub fn SetSockOpt(&self, opt: &SockOpt) -> Result<()> {
        match opt {
            SockOpt::PasscredOption(ref v) => {
                self.setPasscred(*v != 0);
                return Ok(())
            }
            _ => return Ok(())
        }
    }

    pub fn GetSockOpt(&self, opt: &mut SockOpt) -> Result<()> {
        match *opt {
            SockOpt::ErrorOption => {
                return Ok(())
            }
            SockOpt::SendQueueSizeOption(_) => {
                let qs = {
                    let e = self.lock();
                    if !e.Connected() {
                        return Err(Error::SysError(TcpipErr::ERR_NOT_CONNECTED.sysErr))
                    }

                    e.connected.as_ref().unwrap().SendQueuedSize() as i32
                };

                if qs < 0 {
                    return Err(Error::SysError(TcpipErr::ERR_QUEUE_SIZE_NOT_SUPPORTED.sysErr))
                }
                *opt = SockOpt::SendQueueSizeOption(qs);
                return Ok(())
            }
            SockOpt::ReceiveQueueSizeOption(_) => {
                let qs = {
                    let e = self.lock();
                    if !e.Connected() {
                        return Err(Error::SysError(TcpipErr::ERR_NOT_CONNECTED.sysErr))
                    }

                    e.receiver.as_ref().unwrap().RecvQueuedSize() as i32
                };

                if qs < 0 {
                    return Err(Error::SysError(TcpipErr::ERR_QUEUE_SIZE_NOT_SUPPORTED.sysErr))
                }
                *opt = SockOpt::ReceiveQueueSizeOption(qs);
                return Ok(())
            }
            SockOpt::PasscredOption(_) => {
                let val = if self.Passcred() {
                    1
                } else {
                    0
                };

                *opt = SockOpt::PasscredOption(val);
                return Ok(())
            }
            SockOpt::SendBufferSizeOption(_) => {
                let qs = {
                    let e = self.lock();
                    if !e.Connected() {
                        return Err(Error::SysError(TcpipErr::ERR_NOT_CONNECTED.sysErr))
                    }

                    e.connected.as_ref().unwrap().SendMaxQueueSize() as i32
                };

                if qs < 0 {
                    return Err(Error::SysError(TcpipErr::ERR_QUEUE_SIZE_NOT_SUPPORTED.sysErr))
                }

                *opt = SockOpt::SendBufferSizeOption(qs);
                return Ok(())
            }
            SockOpt::ReceiveBufferSizeOption(_) => {
                let qs = {
                    let e = self.lock();
                    if !e.Connected() {
                        return Err(Error::SysError(TcpipErr::ERR_NOT_CONNECTED.sysErr))
                    }

                    e.receiver.as_ref().unwrap().RecvMaxQueueSize() as i32
                };

                if qs < 0 {
                    return Err(Error::SysError(TcpipErr::ERR_QUEUE_SIZE_NOT_SUPPORTED.sysErr))
                }

                *opt = SockOpt::ReceiveBufferSizeOption(qs);
                return Ok(())
            }
            SockOpt::KeepaliveEnabledOption(_) => {
                *opt = SockOpt::KeepaliveEnabledOption(0);
                return Ok(());
            }
            _ => {
                return Err(Error::SysError(TcpipErr::ERR_UNKNOWN_PROTOCOL_OPTION.sysErr))
            }
        }
    }

    // Shutdown closes the read and/or write end of the endpoint connection to its
    // peer.
    pub fn Shutdown(&self, flags: ShutdownFlags) -> Result<()> {
        let (receiver, connected) = {
            let e = self.lock();
            if !e.Connected() {
                return Err(Error::SysError(TcpipErr::ERR_NOT_CONNECTED.sysErr))
            }

            let receiver = e.receiver.as_ref().unwrap().clone();
            let connected = e.connected.as_ref().unwrap().clone();

            if flags & SHUTDOWN_READ != 0 {
                receiver.CloseRecv();
            }

            if flags & SHUTDOWN_WRITE != 0 {
                connected.CloseSend();
            }

            (receiver, connected)
        };


        if flags & SHUTDOWN_READ != 0 {
            receiver.CloseNotify();
        }

        if flags & SHUTDOWN_WRITE != 0 {
            connected.CloseNotify();
        }

        return Ok(())
    }

    // GetLocalAddress returns the local address of the connected endpoint (if
    // available).
    pub fn GetLocalAddress(&self) -> Result<SockAddrUnix> {
        return Ok(SockAddrUnix::New(&self.lock().path))
    }

    // GetRemoteAddress returns the local address of the connected endpoint (if
    // available).
    pub fn GetRemoteAddress(&self) -> Result<SockAddrUnix> {
        let c = self.lock().connected.clone();

        if c.is_some() {
            return c.unwrap().GetLocalAddress();
        }

        return Err(Error::SysError(SysErr::ENOTCONN))
    }

    // Release implements BoundEndpoint.Release.
    pub fn Release(&self) {
        // Binding a baseEndpoint doesn't take a reference.
    }
}