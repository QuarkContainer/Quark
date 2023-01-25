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
use libc::user;
use alloc::sync::Arc;
use spin::Mutex;
use lazy_static::lazy_static;

use tokio::sync::oneshot;

use crate::qlib::common::*;
use crate::qlib::linux_def::*;

use super::agent::*;
use super::message::*;

lazy_static! {
    pub static ref FUNC_AGENT: FuncAgent = FuncAgent::default();
}

#[derive(Default)]
pub struct FuncAgent {
    // appId -> ChannelSet
    pub idleFuncInstances : Mutex<HashMap<u64, ChannelSet>>,

    pub idleAgents: Mutex<HashMap<u64, ChannelSet>>,

    // func Instance Channel id -> appId
    pub funcInstanceChannels: Mutex<HashMap<u64, u64>>,
}

impl FuncAgent {
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
    // mapping from channleId to process power count, i.e. can handle how many funcs in parallel
    pub channels: Arc<Mutex<BTreeMap<u64, u32>>>
}

impl ChannelSet {
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
