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

use std::collections::HashSet;
use std::collections::BTreeMap;
use hashbrown::HashMap;
use libc::NFT_REJECT_ICMPX_ADMIN_PROHIBITED;
use libc::user;
use alloc::sync::Arc;
use spin::Mutex;
use lazy_static::lazy_static;
use core::ops::Deref;

use tokio::sync::oneshot;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedReceiver;
use futures::future::select_all;
use futures::future::FutureExt;

use crate::qlib::common::*;
use crate::qlib::linux_def::*;

use super::agent::*;
use super::message::*;
use super::tcp_teststream::TestTCPServer;


pub async fn Execution() -> Result<()> {
    println!("Execution 1");
    let agentListener = TestTCPServer::New("0.0.0.0:8081".to_string()).await?;
    println!("Execution 2");
    let localListener = TestTCPServer::New("127.0.0.1:8080".to_string()).await?;
    println!("Execution 3");
    
    let funcAgent = FuncAgent::New();

    let mut futures = Vec::new();
    
    let agentFuture = async {
        loop {
            let (_credential, agentStream) = agentListener.Accept().await.unwrap();
            let _id = MSG_CENTER.AddChannel(agentStream).unwrap();
        }
    };

    let funcAgent1 = funcAgent.clone();
    let localFuture = async move {
        loop {
            println!("localFuture 1");
            let (credential, localStream) = localListener.Accept().await.unwrap();
            println!("localFuture 2 {}", credential.appId);
            let cid = MSG_CENTER.AddChannel(localStream).unwrap();
            funcAgent1.AddFuncInstance(credential.appId, cid);
        }
    };

    futures.push(localFuture.boxed());
    futures.push(agentFuture.boxed());
    
    loop {
        let tmp = futures.split_off(0);
        let (_, _, remains)  = select_all(tmp).await;
        futures = remains;
    }

}

pub struct FuncAgent(Arc<FuncAgentInner>);

impl Deref for FuncAgent {
    type Target = Arc<FuncAgentInner>;

    fn deref(&self) -> &Arc<FuncAgentInner> {
        &self.0
    }
}

impl FuncAgent {
    pub fn New() -> Self {
        let inner = FuncAgentInner::New();
        return Self(Arc::new(inner))
    }
}

pub struct FuncAgentInner {
    //pub agentListener: TestTCPServer,

    //pub localListener: TestTCPServer,

    // appId -> ChannelSet
    pub idleFuncInstances : Mutex<HashMap<u64, ChannelSet>>,

    pub idleAgents: Mutex<HashMap<u64, ChannelSet>>,

    // func Instance Channel id -> appId
    pub funcInstanceChannels: Mutex<HashMap<u64, u64>>,
}

impl FuncAgentInner {
    pub fn New() -> Self {
        //let agentListener = TestTCPServer::New(agentListenAddress).await?;
        //let localListener = TestTCPServer::New(localListenerAddr).await?;
        let ret = Self {
            //agentListener: agentListener,
            //localListener: localListener,
            idleFuncInstances: Mutex::new(HashMap::new()),
            idleAgents: Mutex::new(HashMap::new()),
            funcInstanceChannels: Mutex::new(HashMap::new()),
        };

        return ret;
    }

    pub fn AddFuncInstance(&self, appId: u64, channelId: u64) {
        let mut fcs = self.funcInstanceChannels.lock();
        fcs.insert(channelId, appId);


        let mut idlefi = self.idleFuncInstances.lock();
        match idlefi.get_mut(&appId) {
            None => {
                idlefi.insert(appId, ChannelSet::New(channelId, 1));
            } 
            Some(cs) => {
                cs.AddChannel(channelId, 1);
            }
        }
    }

    pub fn GetChannel(&self, channelId: u64) -> Result<MsgChannel> {
        return MSG_CENTER.GetChannel(channelId);
    }

    // ask for another agent to process function with appId and
    // add to the idleAgents
    pub async fn AskforAgent(&self, _targetAppId: u64) -> Result<u64> {
        todo!();
    }

    // based on target appId to get a remote FuncAgentId
    pub async fn GetAgent(&self, _targetAppId: u64) -> Result<u64> {
        todo!();
    }

    // based on target appId to get a local Function Instance Id
    pub async fn GetFunctionInstance(&self, _targetAppId: u64) -> Result<u64> {
        todo!();
    }

    // based on src AppId and target Function Name, return target appId
    pub async fn GetAppId(&self, _srcApp: u64, _funcName: &str) -> Result<u64> {
        todo!();
    }

    pub async fn GetChannelId(&self, _funcInstanceChannelId: u64) -> Result<u64> {
        todo!();
    }

    pub async fn UserCall(&self, funcInstanceChannelId: u64, funcName: String, payload: Vec<u8>) -> Result<FuncResp> {
        let srcAppId = self.GetChannelId(funcInstanceChannelId).await?;
        let targetAppId = self.GetAppId(srcAppId, &funcName).await?;
        let agentId = self.GetAgent(targetAppId).await?;
        if agentId == 0 { // local call
            /*let funcInstanceId = self.GetFunctionInstance(targetAppId).await?;
            let channel = self.GetChannel(funcInstanceId)?;
            return channel.UserCall(funcName, payload).await;*/
            return self.AgentCall(targetAppId, funcName, payload).await;
        } else {
            let channel = self.GetChannel(agentId)?;
            return channel.AgentCall(targetAppId, funcName, payload).await;
        };
    }

    pub async fn AgentCall(&self, targetAppId: u64, funcName: String, payload: Vec<u8>) -> Result<FuncResp> {
        let funcInstanceId = self.GetFunctionInstance(targetAppId).await?;
        let channel = self.GetChannel(funcInstanceId)?;
        return channel.UserCall(funcName, payload).await;
    }

    pub async fn HandleRequest(&self, channelId: u64, requestId: u64, payload: MsgPayload) {
        let response = match payload {
            MsgPayload::AgentFuncCall(agentFuncCall) => {
                self.AgentCall(agentFuncCall.appId, agentFuncCall.funcName, agentFuncCall.payload).await
            }
            MsgPayload::UserFuncCall(funcCall) => {                
                self.UserCall(channelId, funcCall.funcName, funcCall.payload).await
            }
            _ => {
                panic!("HandleRequest get repsonse")
            }
        };

        let response = match response {
            Err(_) => {
                FuncResp::NewErr(HTTP_INTERN_ERR)
            }
            Ok(resp) => resp,
        };

        let channel = match self.GetChannel(channelId) {
            Err(e) => {
                error!("RequestProcess get channel fail with error {:?}", e);
                return
            }
            Ok(channel)  => {
                channel
            }
        };

        channel.SendResp(requestId, response).await;
    }

    pub async fn RequestProcess(&self) {
        let mut reqRx = MSG_CENTER.reqRx.lock().take().unwrap();
        loop {
            let req = match reqRx.recv().await {
                None => break,
                Some(req) => req,
            };

            self.HandleRequest(req.channelId, req.msg.messageId, req.msg.payload).await;             
        }
    } 

    
}

#[derive(Clone, Default)]
pub struct ChannelSet {
    // mapping from channleId to weight, i.e. can handle how many funcs in parallel
    pub channels: Arc<Mutex<BTreeMap<u64, u32>>>
}

impl ChannelSet {
    pub fn New(channelId: u64, weight: u32) -> Self {
        let cs = Self::default();
        cs.channels.lock().insert(channelId, weight);
        return cs;
    }

    pub fn GetOneChannel(&self) -> Option<u64> {
        let mut channels = self.channels.lock();
        let (k, v) = match channels.pop_first() {
            None => return None,
            Some(p) => p
        };

        if v > 1 {
            channels.insert(k, v - 1);
        }

        return Some(k);
    }

    pub fn DecreaseChannel(&self, channelId: u64, count: u32) -> Result<()> {
        let mut channels = self.channels.lock();
        match channels.remove(&channelId) {
            Some(v) => {
                assert!(v>=count);
                if v > count {
                    channels.insert(channelId, v - count);
                }
                return Ok(())
            }
            None => {
                panic!("ChannelSet DecreaseChannel")
            }
        }
    }

    pub fn AddChannel(&self, channelId: u64, count: u32) {
        let mut channels = self.channels.lock();
        match channels.remove(&channelId) {
            Some(v) => {
                channels.insert(channelId, v + count);
            }
            None => {
                channels.insert(channelId, count);
            }
        }
    }
}

impl MsgChannel {
    pub async fn UserCall(&self, funcName: String, payload: Vec<u8>) -> Result<FuncResp> {
        let call = MsgPayload::NewUserFuncCall(funcName, payload);
        let (tx, rx) = oneshot::channel();
        self.SendRequest(tx, call).await?;
        match rx.await {
            Ok(resp) => return Ok(resp),
            Err(_) => return Ok(FuncResp::NewErr(HTTP_INTERN_ERR)),
        }
    }

    pub async fn AgentCall(&self, appId: u64, funcName: String, payload: Vec<u8>) -> Result<FuncResp> {
        let call = MsgPayload::NewAgentFuncCall(appId, funcName, payload);
        let (tx, rx) = oneshot::channel();
        self.SendRequest(tx, call).await?;
        match rx.await {
            Ok(resp) => return Ok(resp),
            Err(_) => return Ok(FuncResp::NewErr(HTTP_INTERN_ERR)),
        }
    }
}
