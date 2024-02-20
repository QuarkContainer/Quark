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

use std::net::SocketAddr;

use tokio::net::TcpListener;
use tokio::net::TcpStream;
use tokio::net::ToSocketAddrs;

use crate::qlib::common::*;

use super::message::Credential;
use super::message::MsgPayload;
use super::message::QMsg;
use super::msg_stream::*;

pub struct TestTCPClient {}

impl TestTCPClient {
    pub async fn Connect<A: ToSocketAddrs>(appId: u64, address: A) -> Result<MsgStream> {
        println!("Connect 1");
        let stream = TcpStream::connect(address).await?;
        println!("Connect 2");
        
        let stream = MsgStream::NewWithTcpStream(stream);

        let credential = Credential {
            appId: appId,
        };

        let payload = MsgPayload::Credential(credential);

        stream.WriteMsg(&QMsg {
            messageId: 0,
            payload: payload,
        }).await?;
        return Ok(stream);
    }
}

pub struct TestTCPServer {
    pub listener: TcpListener,
}

impl TestTCPServer {
    // addr: string like "127.0.0.1:8080"
    pub async fn New(addr: String) -> Result<Self> {
        let addr = addr.parse::<SocketAddr>().unwrap();
        let listener = TcpListener::bind(addr).await?;
        return Ok(Self{listener : listener})
    }

    pub async fn Accept(&self) -> Result<(Credential, MsgStream)> {
        let (stream, _addr) = self.listener.accept().await?;

        let stream = MsgStream::NewWithTcpStream(stream);
        let msg = stream.ReadMsg().await?;

        let credentail = match msg.payload {
            MsgPayload::Credential(c) => c,
            _ => panic!("TestTCPServer get wrong payload"),
        };

        return Ok((credentail, stream));
    }
}

