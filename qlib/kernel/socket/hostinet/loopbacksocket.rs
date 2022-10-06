// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor& Authors.
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
use alloc::sync::Weak;
use core::ops::Deref;
use core::fmt;

use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::socket_buf::*;
use crate::qlib::bytestream::*;
use crate::qlib::kernel::kernel::waiter::Queue;
use crate::qlib::kernel::task::Task;

// input: client side queue, server side queue
// return: (client side socket, server side socket)
pub fn LoopbackSocketPair(clientQueue: Queue, serverQueue: Queue) -> (LoopbackSocket, LoopbackSocket) {
    let buf1 = ByteStream::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT);
    let buf2 = ByteStream::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT);

    let clientSockBuf = SocketBuff::New(buf1.clone(), buf2.clone());
    let serverSockBuf = SocketBuff::New(buf2, buf1);

    let clientSock = LoopbackSocket(Arc::new(LoopbackSocketInner {
        sockBuff: clientSockBuf.clone(),
        peer: serverSockBuf.Downgrade(),
        peerQueue: serverQueue,
    }));

    let serverSock = LoopbackSocket(Arc::new(LoopbackSocketInner {
        sockBuff: serverSockBuf.clone(),
        peer: clientSockBuf.Downgrade(),
        peerQueue: clientQueue,
    })); 

    return (clientSock, serverSock)
}

#[derive(Default)]
pub struct LoopbackSocketInner {
    pub sockBuff: SocketBuff,
    pub peer: SocketBuffWeak,
    pub peerQueue: Queue,
}

pub struct LoopbackSocketWeak(Weak<LoopbackSocketInner>);

impl LoopbackSocketWeak {
    pub fn Upgrade(&self) -> Option<LoopbackSocket> {
        let f = match self.0.upgrade() {
            None => return None,
            Some(f) => f,
        };

        return Some(LoopbackSocket(f));
    }
}

#[derive(Clone, Default)]
pub struct LoopbackSocket(Arc<LoopbackSocketInner>);

impl fmt::Debug for LoopbackSocket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return f.debug_struct("LoopbackSocket").finish();
    }
}

impl Deref for LoopbackSocket {
    type Target = Arc<LoopbackSocketInner>;

    fn deref(&self) -> &Arc<LoopbackSocketInner> {
        &self.0
    }
}

impl Drop for LoopbackSocket {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            self.SetRClosed();
            self.SetWClosed();
            self.peerQueue.Notify(EVENT_HUP);
        }
    }
}

impl LoopbackSocket {
    pub fn Downgrade(&self) -> LoopbackSocketWeak {
        return LoopbackSocketWeak(Arc::downgrade(&self.0));
    }

    pub fn Peer(&self) -> Option<SocketBuff> {
        return self.peer.Upgrade()
    }

    pub fn Writev(&self, task: &Task, iovs: &[IoVec]) -> Result<i64> {
        let (count, databuff) = self.sockBuff.Writev(task, iovs)?;
        match databuff {
            None => (),
            Some(_) => {
                self.peerQueue.Notify(READABLE_EVENT)
            }
        };

        return Ok(count as i64)
    }

    pub fn Readv(&self, task: &Task, iovs: &mut [IoVec], peek: bool) -> Result<i64> {
        let (trigger, count) = self.sockBuff.Readv(task, iovs, peek)?;
        if trigger {
            self.peerQueue.Notify(WRITEABLE_EVENT);
        }

        return Ok(count as i64)
    }

    pub fn SetWClosed(&self) {
        self.sockBuff.SetWClosed();
        match self.Peer() {
            Some(p) => p.SetRClosed(),
            _ => ()
        }
    }

    pub fn SetRClosed(&self) {
        self.sockBuff.SetRClosed();
        match self.Peer() {
            Some(p) => p.SetWClosed(),
            _ => ()
        }
    }

    pub fn Events(&self) -> EventMask {
        return self.sockBuff.Events();
    }
}
