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
use core::ops::Deref;
use alloc::collections::vec_deque::VecDeque;
use alloc::collections::btree_map::BTreeMap;
use spin::Mutex;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

use crate::qlib::kernel::fs::file::*;
use crate::qlib::kernel::kernel::waiter::queue::*;
use crate::qlib::common::*;
use crate::qlib::linux_def::*;
use crate::qlib::bytestream::*;
use crate::qlib::kernel::task::Task;
use crate::qlib::kernel::socket::socket::*;
use crate::qlib::kernel::socket::hostinet::uring_socket::*;
use super::user_msg::*;

lazy_static! {
    pub static ref RESILIENCE_CONN : ResilienceConn = ResilienceConn::New();
}

pub struct ResilienceConn {
    pub connection: FileOps,
    pub nextSessionId: AtomicU64,
    pub serviceSession: AtomicU64,
    pub sessions: Mutex<BTreeMap<u64, ResilienceSession>>,
    pub requests: Mutex<VecDeque<UserMsg>>
}

#[derive(Default)]
pub struct ResilienceSessionInner {
    pub sessionId: u64,
    pub queue: Queue,
    pub messages: VecDeque<UserMsg>
}

impl ResilienceSessionInner {
    pub fn New(sessionId: u64) -> Self {
        return Self {
            sessionId: sessionId,
            queue: Queue::default(),
            messages: VecDeque::new(),
        }
    }
    
    pub fn InsertMsg(&mut self, msg: UserMsg) {
        self.messages.push_back(msg);
        if self.messages.len() == 1 {
            self.queue.Notify(READABLE_EVENT);
        }
    }

}

#[derive(Clone)]
pub struct ResilienceSession(Arc<Mutex<ResilienceSessionInner>>);

impl Deref for ResilienceSession {
    type Target = Arc<Mutex<ResilienceSessionInner>>;

    fn deref(&self) -> &Arc<Mutex<ResilienceSessionInner>> {
        &self.0
    }
}


pub const ADDR: &str = "52.202.1.2:80";

impl ResilienceConn {
    pub fn New() -> Self {
        let task = Task::Current();
        let domain = AFType::AF_INET;
        let stype = SocketType::SOCK_STREAM;
        let protocol = 0;
        let socket = NewSocket(task, domain, stype, protocol).expect("ResilienceSocketOps new socket fail");
        let connection = socket.FileOp.clone();

        connection.Connect(task, ADDR.as_bytes(), false).expect("ResilienceSocketOps new socket fail");

        return Self {
            connection: connection,
            nextSessionId: AtomicU64::new(1),
            serviceSession: AtomicU64::new(0),
            sessions: Mutex::new(BTreeMap::new()),
            requests: Mutex::new(VecDeque::new()),
        }
    }

    pub fn NewSession(&self) -> ResilienceSession {
        let newSessionId = self.nextSessionId.fetch_add(1, Ordering::SeqCst);
        let inner = ResilienceSessionInner::New(newSessionId);
        return ResilienceSession(Arc::new(Mutex::new(inner)))
    }

    pub fn RemoveSession(&self, sessionId: u64) -> Result<()> {
        match self.sessions.lock().remove(&sessionId) {
            None => return Err(Error::SysError(SysErr::ENOENT)),
            Some(_) => return Ok(())
        }
    }

    pub fn ServiceSessionId(&self) -> u64 {
        return self.serviceSession.load(Ordering::SeqCst);
    }

    pub fn SetServiceSessionId(&self, sessionId: u64) {
        self.serviceSession.store(sessionId, Ordering::SeqCst);
    }

    pub fn Recv(&mut self, task: &Task) -> Result<()> {
        loop {
            let msg = self.ReadMsg(task)?;
            let sessionId = match &msg {
                UserMsg::UserFuncCall(_call) => {
                    let sessionId = self.ServiceSessionId();
                    if sessionId == 0 {
                        continue;
                    }

                    sessionId
                }
                UserMsg::UserFuncResp(resp) => {
                    resp.sessionId
                }
            };

            match self.sessions.lock().get(&sessionId) {
                None => {
                    info!("ResilienceConn::recv not exist session {:?}", sessionId)
                }
                Some(session) => {
                    session.lock().InsertMsg(msg);
                }
            }
        }
    }

    pub fn Send(&self, task: &Task) -> Result<()> {
        loop {
            let mut requests = self.requests.lock();
            let request = requests.pop_front();
            match request {
                None => return Ok(()),
                Some(req) => {
                    match self.WriteMsg(task, &req) {
                        Err(Error::SysError(SysErr::EAGAIN)) => {
                            requests.push_front(req);
                            return Err(Error::SysError(SysErr::EAGAIN));
                        }
                        Err(e) => return Err(e),
                        Ok(()) => ()
                    }
                }
            }
        }        
    }

    pub fn WriteUserRawMsg(&self, len: u32, sessionId: u64, payload: &[u8]) -> Result<()> {
        
    }

    pub fn ReadMsg(&self, task: &Task) -> Result<UserMsg> {
        match &self.connection {
            FileOps::UringSocketOperations(uringSock) => {
                let sockBufType = uringSock.socketType.lock().clone();
                match sockBufType {
                    UringSocketType::Uring(buf) => {
                        let bs = buf.writeBuf.clone();

                        let mut buf = SocketBufIovs::default();
                        uringSock.Consume(task, 0, &mut buf)?;

                        // there should enough space for message len
                        if buf.Count() < 4 {
                            bs.lock().SetWaitingRead();
                            return Err(Error::SysError(SysErr::EAGAIN))
                        }

                        let len = buf.ReadObj::<u32>()?;
                        if buf.Count() < len as usize {
                            bs.lock().SetWaitingRead();
                            return Err(Error::SysError(SysErr::EAGAIN))
                        }

                        let msg = UserMsg::Read(&mut buf).expect("UserMsg readmsg fail");
                        assert!(len as usize == msg.Size());

                        uringSock.Consume(task, len as usize + 4, &mut buf)?;

                        return Ok(msg)    
                    }
                    _ => {
                        return Err(Error::SysError(SysErr::EPIPE)); 
                    }
                }
            }
            _ => {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }
    }

    pub fn Call(&self, task: &Task, sessiondId: u64, funcName: &str, requestId: u64, payload: Vec<u8>) -> Result<()> {
        let call = UserFuncCall {

        }
    }

    pub fn WriteMsg(&self, task: &Task, msg: &UserMsg) -> Result<()> {
        match &self.connection {
            FileOps::UringSocketOperations(uringSock) => {
                let sockBufType = uringSock.socketType.lock().clone();
                match sockBufType {
                    UringSocketType::Uring(buf) => {
                        if buf.WClosed() {
                            return Err(Error::SysError(SysErr::EPIPE));
                        }

                        let bs = buf.writeBuf.clone();

                        let mut buf = SocketBufIovs::default();
                        uringSock.Produce(task, 0, &mut buf)?;
                        let size = msg.Size();
                        // there should enough space for msg len(4 bytes) and message
                        if size + 4 > buf.Count() {
                            bs.lock().SetWaitingWrite();
                            return Err(Error::SysError(SysErr::EAGAIN))
                        }

                        buf.WriteObj(&(size as u32)).expect("UserMsg writemsg fail");
                        msg.Write(&mut buf).expect("UserMsg writemsg fail");
                        uringSock.Produce(task, size + 4, &mut buf)?;
                        return Ok(())        
                    }
                    _ => {
                        return Err(Error::SysError(SysErr::EPIPE)); 
                    }
                }
            }
            _ => {
                return Err(Error::SysError(SysErr::EINVAL));
            }
        }
    }
}
