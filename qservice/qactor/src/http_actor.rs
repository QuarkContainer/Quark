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
// limitations under

use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::sync::Mutex;
use core::ops::Deref;

use tokio::sync::oneshot;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TMutex;

use qshare::common::*;
use qshare::qactor;
use crate::actor_system::ACTOR_SYSTEM;

use axum::{
    response::IntoResponse,
    routing::post,
    http::StatusCode,
    Json, Router,
    extract::State,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptReq {
    pub prompt: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptResp {
    pub response: String,
}

#[derive(Debug)]
pub struct HttpActorInner {
    pub requests: Mutex<BTreeMap<u64, oneshot::Sender<PromptResp>>>,
    pub lastReqId: AtomicU64,

    pub inputTx: mpsc::Sender<qactor::TellReq>,
    pub inputRx: TMutex<mpsc::Receiver<qactor::TellReq>>,

    pub gatewayActorId: String,
    pub gatewayFunc: String,
    pub httpPort: u16,
}

#[derive(Debug, Clone)]
pub struct HttpActor(Arc<HttpActorInner>);

impl Deref for HttpActor {
    type Target = Arc<HttpActorInner>;

    fn deref(&self) -> &Arc<HttpActorInner> {
        &self.0
    }
}

impl HttpActor {
    pub fn New(gatewayActorId: &str, gatewayFunc: &str, httpPort: u16) -> Self {
        let (tx, rx) = mpsc::channel::<qactor::TellReq>(30);
        
        let inner = HttpActorInner {
            requests: Mutex::new(BTreeMap::new()),
            lastReqId: AtomicU64::new(1),
            inputTx: tx,
            inputRx: TMutex::new(rx),
            gatewayActorId: gatewayActorId.to_owned(),
            gatewayFunc: gatewayFunc.to_owned(),
            httpPort: httpPort,
        };

        return Self(Arc::new(inner))
    }

    pub async fn HttpServe(&self) -> Result<()> {
        let clone = self.clone();
        
        let app = Router::new()
            .route(
                "/prompt", 
                post({
                move |body| ProcessPrompt(body, State(clone))
            }));
    
        let addr = format!("0.0.0.0:{}", self.httpPort);
        let listener = match tokio::net::TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                println!("HttpServe 4.1 error {:?}", &e);
                return Err(e.into());
            }
        };
        axum::serve(listener, app).await?;
        
        return Ok(())
    }

    pub fn NewPrompt(&self, prompt: &str) -> oneshot::Receiver<PromptResp> {
        println!("NewPrompt 1 {}", prompt);
        let (tx, rx) = oneshot::channel();
        let reqId = self.lastReqId.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.requests.lock().unwrap().insert(reqId, tx);

        let tellReq = qactor::TellReq {
            actor_id: self.gatewayActorId.clone(),
            func: self.gatewayFunc.clone(),
            req_id: reqId,
            data: prompt.to_owned().into_bytes(),
        };

        ACTOR_SYSTEM.Send(tellReq).unwrap();

        return rx
    }

    pub fn Tell(&self, tell: qactor::TellReq) {
        self.inputTx.try_send(tell).unwrap();
    }

    pub async fn ProcessTell(&self) -> Result<()> {
        loop {
            match self.Recv().await {
                None => (),
                Some(tell) => {
                    self.HandleTell(tell).await?;
                }
            }
        }
    }

    pub async fn Process(&self) -> Result<()> {
        tokio::select! {
            e = self.ProcessTell() => {
                error!("GatewayActor::ProcessTell fail with {:?}", &e);
            }
            e = self.HttpServe() => {
                error!("GatewayActor::HttpServe fail with {:?}", e);
            } 
        }

        return Ok(())
    }

    pub async fn Recv(&self) -> Option<qactor::TellReq> {
        error!("Recv 1");
        let mut rx = self.inputRx.lock().await;
        error!("Recv 2");
        let req = rx.recv().await;
        error!("Recv 3");
        return req;
    }

    pub async fn HandleTell(&self, tell: qactor::TellReq) -> Result<()> {
        let reqId = tell.req_id;
        let tx = match self.requests.lock().unwrap().remove(&reqId) {
            None => return Err(Error::NotExist(format!("GatewayActor::HandleTell reqid {}", reqId))),
            Some(req) => req
        };

        let resp = PromptResp {
            response: String::from_utf8(tell.data).unwrap()
        };

        match tx.send(resp) {
            Err(e) => return Err(Error::NotExist(format!("GatewayActor::HandleTell send fail with error {:?}", e))),
            Ok(()) => (),
        }
        return Ok(())
    }
}

async fn ProcessPrompt(
    Json(payload): Json<PromptReq>,
    State(state): State<HttpActor>,
) -> impl IntoResponse {
    let rx = state.NewPrompt(&payload.prompt);
    let resp = rx.await.unwrap();
    
    (StatusCode::OK, Json(resp))
}
