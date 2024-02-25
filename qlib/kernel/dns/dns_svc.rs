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

use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use core::ops::Deref;
use spin::Mutex;

use crate::qlib::common::*;
use crate::qlib::kernel::socket::hostinet::tsot_mgr::TsotSocketMgr;
use crate::qlib::linux_def::*;
use crate::qlib::kernel::tcpip::tcpip::SockAddrInet;
use crate::qlib::kernel::quring::uring_async::DNSSend;
use crate::qlib::kernel::IOURING;
use crate::qlib::tsot_msg::DnsResp;

use super::*;

#[derive(Debug)]
pub struct DnsReqContext {
    pub peerIp: [u8; 4],
    pub peerPort: u16,
    pub reqs: Vec<DnsQuestion>,
}

#[derive(Debug)]
pub struct DnsSvcInner {
    pub fd: i32,
    pub pendingReqs: BTreeMap<u16, DnsReqContext>,
    pub recvBuf: BytePacketBuffer,
    pub iov: IoVec,
    pub peerAddr: SockAddrInet,
    pub msg: MsgHdr,
}

#[derive(Debug, Clone)]
pub struct DnsSvc(pub Arc<Mutex<DnsSvcInner>>);

impl Deref for DnsSvc {
    type Target = Arc<Mutex<DnsSvcInner>>;

    fn deref(&self) -> &Arc<Mutex<DnsSvcInner>> {
        &self.0
    }
}

pub struct DnsAnswer {
    addr: Ipv4Addr,
    ttl: u32,
}

pub const DNS_SVC_ADDR: [u8; 4] = [127, 0, 0, 53];
pub const DNS_SVC_PORT: u16 = 53;

impl Default for DnsSvc {
    fn default() -> Self {
        return Self::New().unwrap();
    }
}

impl DnsSvc {
    pub fn New() -> Result<Self> {
        let sock = 0;

        let recvBuf = BytePacketBuffer::new();

        let inner = DnsSvcInner {
            fd: sock,
            pendingReqs: BTreeMap::new(),
            recvBuf: recvBuf,
            iov: IoVec::default(),
            peerAddr: SockAddrInet::default(),
            msg: MsgHdr::default(),
        };

        let svc = Self(Arc::new(Mutex::new(inner)));

        {
            let mut msg = MsgHdr::default();
            let mut lock = svc.lock();
            //let msg = &mut lock.msg;
            msg.msgName = &lock.peerAddr as * const _ as u64;
            msg.nameLen = lock.peerAddr.Len() as u32;
            lock.iov.start = &lock.recvBuf.buf[0] as * const _ as u64;
            lock.iov.len = lock.recvBuf.buf.len();
            msg.iov = &lock.iov as * const _ as u64;
            msg.iovLen  = 1;
            lock.msg = msg;
        }

        return Ok(svc)
    }

    pub fn ProcessDnsReq(&self, result: i32) -> Result<()> {
        if result < 0 {
            return Ok(())
        }

        let _len = result as usize;
        let mut inner = self.lock();
        inner.recvBuf.pos = 0;

        let p = match DnsPacket::from_buffer(&mut inner.recvBuf) {
            Err(e) => {
                error!("ProcessDnsReq fail with error {:?}", e);
                return Ok(())
            }
            Ok(p) => {
                p
            }
        };

        let mut domains = Vec::new();
        for req in &p.questions {
            if req.qtype == QueryType::A {
                domains.push(req.name.clone());
            } 
        }

        if domains.len() == 0 {
            let mut resp = DnsPacket::new();

            //let mut header = DnsHeader::new();
            resp.header.id = p.header.id;
            for req in p.questions {
                resp.questions.push(req);
            }

            let mut buf = BytePacketBuffer::new();
            resp.write(&mut buf)?;
            let peerAddr = SockAddrInet::New(inner.peerAddr.Ipv4Port(), &inner.peerAddr.Addr.clone());
            let mut dataBuf = DataBuff::New(buf.pos);
            for i in 0..buf.pos {
                dataBuf.buf[i] = buf.buf[i];
            };
    
            let dnsSend = DNSSend::New(inner.fd, dataBuf, peerAddr);
            IOURING.SendDns(dnsSend);
            return Ok(())
        }

        let context = DnsReqContext {
            peerIp: inner.peerAddr.Addr.clone(),
            peerPort: inner.peerAddr.Ipv4Port(),
            reqs: p.questions,
        };

        inner.pendingReqs.insert(p.header.id, context);

        TsotSocketMgr::DnsReq(p.header.id, &domains)?;
        return Ok(())
    }

    pub fn ProcessDnsResp(&self, resp: DnsResp) -> Result<()> {
        let context = match self.lock().pendingReqs.remove(&resp.reqId) {
            None => return Err(Error::Common(format!("DnsSvc::ProcessDnsResp get nonexisit reqid {}", resp.reqId))),
            Some(c) => c
        };

        assert!(context.reqs.len() == resp.count);

        let mut p = DnsPacket::new();

        //let mut header = DnsHeader::new();
        p.header.id = resp.reqId;

        for i in 0..resp.count {
            let addr = resp.ips[i];
            if addr != 0 {
                let answer = DnsRecord::A {
                    domain: context.reqs[i].name.clone(),
                    addr: Ipv4Addr::fromU32(addr),
                    ttl: 100,
                };
                p.answers.push(answer)
            }
        }

        for req in context.reqs {
            p.questions.push(req);
        }

        let mut buf = BytePacketBuffer::new();
        p.write(&mut buf)?;

        let peerAddr = SockAddrInet::New(context.peerPort, &context.peerIp);
        let mut dataBuf = DataBuff::New(buf.pos);
        for i in 0..buf.pos {
            dataBuf.buf[i] = buf.buf[i];
        };

        let dnsSend = DNSSend::New(self.lock().fd, dataBuf, peerAddr);
        IOURING.SendDns(dnsSend);
        
        return Ok(())
    }
}
