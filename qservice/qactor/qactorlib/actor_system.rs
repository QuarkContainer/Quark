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
use once_cell::sync::OnceCell;

use qshare::common::*;
use qshare::qactor;
use qshare::qactor::TellReq;
use crate::actor::Actor;

pub static ACTOR_SYSTEM: OnceCell<ActorSystem> = OnceCell::new();

pub struct ActorSystem {
    pub pods: BTreeMap<u16, Pod>,
    pub actors: BTreeMap<String, Actor>,
    // pub processes: BTreeMap<u16, Process>,
    pub localActors: Vec<String>,
    pub gatewayActorId: String,
    pub gatewayFunc: String,
}

impl ActorSystem {
    pub fn New(&self, _driverIp: [u8; 4]) -> Result<Self> {
        unimplemented!()
    }

    pub async fn Recv(&self, actorId: String) -> Result<Option<qactor::TellReq>> {
        let actor = match self.actors.get(&actorId) {
            None => {
                return Err(Error::NotExist(format!("ActorSystem::Recv actor {}", actorId)));
            }
            Some(a) => a.clone()
        };

        return Ok(actor.Recv().await)
    }

    pub fn Send(&self, req: TellReq) -> Result<()> {
        let actor = match self.actors.get(&req.actor_id) {
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