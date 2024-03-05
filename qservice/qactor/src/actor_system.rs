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
use std::result::Result as SResult;
use std::sync::Arc;
use std::sync::Mutex;
use core::ops::Deref;
//use once_cell::sync::OnceCell;
use tonic::transport::Server;

use pyo3::prelude::*;

use qshare::common::*;
use qshare::qactor;
use qshare::qactor::TellReq;
use crate::actor::Actor;
use crate::actor::PyActor;
use crate::http_actor::HttpActor;

// pub static ACTOR_SYSTEM: OnceCell<ActorSystem> = OnceCell::new();

lazy_static::lazy_static! {
    pub static ref ACTOR_SYSTEM: ActorSystem = ActorSystem::NewLocal();
}


#[derive(Debug)]
pub struct ActorSystemInner {
    // pub pods: BTreeMap<u16, Pod>,
    pub actors: Mutex<BTreeMap<String, Actor>>,
    pub httpactors: Mutex<Option<HttpActor>>,
}


#[derive(Debug, Clone)]
pub struct ActorSystem(Arc<ActorSystemInner>);

impl Deref for ActorSystem {
    type Target = Arc<ActorSystemInner>;

    fn deref(&self) -> &Arc<ActorSystemInner> {
        &self.0
    }
}

impl ActorSystem {
    pub fn NewLocal() -> Self {
        let inner = ActorSystemInner {
            actors: Mutex::new(BTreeMap::new()),
            httpactors: Mutex::new(None)
        };

        return Self(Arc::new(inner));
    }

    pub async fn wait(&self) -> Result<()> {
        let addr = format!("0.0.0.0:6666");
        let actorServiceFuture = Server::builder()
            .add_service(qactor::actor_pod_service_server::ActorPodServiceServer::new(self.clone()))
            .serve(addr.parse().unwrap());

        let httpActor = self.httpactors.lock().unwrap().take();

        match httpActor {
            None => {
                actorServiceFuture.await?;
            }
            Some(httpactor) => {
                tokio::select! {
                    e = httpactor.Process() => {
                        return Ok(e?)
                    }
                    e = actorServiceFuture => {
                        return Ok(e?)
                    }
                }
            }
        }

        return Ok(())
    }

    pub fn NewPyActor(&self, id: &str, modName: &str, className: &str, queue: &PyAny) -> Result<()> {
        let actor = PyActor::New(id, modName, className, queue);
        self.NewActor(id, Actor::PyActor(actor))
    }

    pub fn NewHttpProxyActor(&self, proxyActorId: &str, gatewayActorId: &str, gatewayFunc: &str, httpPort: u16) -> Result<()> {
        let actor = HttpActor::New(gatewayActorId, gatewayFunc, httpPort);
        *self.httpactors.lock().unwrap() = Some(actor.clone());

        self.NewActor(proxyActorId, Actor::HttpActor(actor))
    }

    pub fn NewActor(&self, id: &str, actor: Actor) -> Result<()> {
        let mut actors = self.actors.lock().unwrap();
        if actors.contains_key(id) {
            return Err(Error::Exist(format!("ActorSystem::NewActor {}", id)));
        }

        actors.insert(id.to_owned(), actor.clone());
        return Ok(())
    }

    pub fn Send(&self, req: TellReq) -> Result<()> {
        let actor = match self.actors.lock().unwrap().get(&req.actor_id) {
            None => {
                return Err(Error::NotExist(format!("ActorSystem::Send ")))
            }
            Some(a) => a.clone()
        };

        actor.Tell(req);

        return Ok(())
    }

    // pub fn StartLocalActors(&self, actors: Vec<Actor)
}

pub struct Pod {
    pub podIp: u32,
}

#[tonic::async_trait]
impl qactor::actor_pod_service_server::ActorPodService for ActorSystem {
    async fn tell(
        &self,
        request: tonic::Request<qactor::TellReq>,
    ) -> SResult<tonic::Response<qactor::TellResp>, tonic::Status> {
        let req = request.into_inner();

        match self.Send(req) {
            Err(e) => {
                return Ok(tonic::Response::new(qactor::TellResp {
                    err: format!("{:?}", e),
                }))
            }
            Ok(()) => {
                return Ok(tonic::Response::new(qactor::TellResp {
                    err: String::new(),
                }))
            }
        }
    }
}