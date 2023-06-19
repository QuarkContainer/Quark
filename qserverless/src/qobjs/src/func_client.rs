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

use std::result::Result as SResult;
use std::sync::Arc;

use tonic::transport::channel::Channel;

use crate::func;
use crate::common::*;

pub struct FuncClient {
    pub client: func::func_agent_service_client::FuncAgentServiceClient<Channel>,
}

impl FuncClient {
    pub async fn Init(agentAddr: &str) -> Result<Self> {
        /*let client: func::func_agent_service_client::FuncAgentServiceClient<tonic::transport::Channel> = {
            let client;
            loop {
                match func::func_agent_service_client::FuncAgentServiceClient::connect(agentAddr.to_string()).await {
                    Ok(c) => {
                        client = c;
                        break;
                    }
                    Err(e) => {
                        error!("FuncClient can't connect to funcagent {}, {:?}", agentAddr, e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            }
            client
        };*/

        use tokio::net::UnixStream;
        use tonic::transport::{Endpoint, Uri};
        use tower::service_fn;

        let path = Arc::new(agentAddr.to_owned());

        let client = {
            let client;
            loop {
                let path = path.clone();
                // the uri is useless 
                let res = Endpoint::from_static("https://example.com")
                        .connect_with_connector(service_fn(move |_: Uri| {
                            let path = path.clone();
                            async move { UnixStream::connect(&*path).await }
                        }))
                .await;

                match res {
                    Err(e) => {
                        error!("can't connect to funcagent {}, {:?}", agentAddr, e);
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                    Ok(channel) => {
                        client = func::func_agent_service_client::FuncAgentServiceClient::new(channel);
                        break;
                    }
                }
            }
            client
        };
            
        return Ok(Self {
            client: client,
        })
    }

    pub async fn Call(
        &mut self, 
        namespace: &str, 
        packageName: &str,
        funcName: &str,
        parameters: &str,
        priority: usize,
    ) -> SResult<String, String> {
        let id = uuid::Uuid::new_v4().to_string();
        let req = func::FuncAgentCallReq {
            job_id: id.clone(),
            id: id,
            namespace: namespace.to_string(),
            package_name: packageName.to_string(),
            func_name: funcName.to_string(),
            parameters: parameters.to_string(),
            caller_func_id: String::new(), // direct call has no caller func
            priority: priority as u64,
        };

        let res = match self.client.func_call(req).await {
            Err(e) => return Err(format!("funcall fail with error {:?}", e)),
            Ok(res) => res,
        };

        let resp = res.into_inner();
        if resp.error.len() > 0 {
            return Err(resp.error);
        } else {
            return Ok(resp.resp);
        }
    }
}