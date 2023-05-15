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

use tonic::{Request, Response, Status};

use qobjs::selection_predicate::{ListOption, SelectionPredicate};
use qobjs::selector::Selector;
use qobjs::types::DataObject;

use qobjs::service_directory::*;

use crate::SVC_DIR;

#[derive(Default, Debug)]
pub struct EtcdSvc {}

impl EtcdSvc {
    // This is to verify the grpc server is working.
    // 1. go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
    // 2. Launch the grpc server
    // 3. grpcurl -plaintext -proto resilience_function/proto/service_directory.proto -d '{"client_name": "a client"}' [::]:50071 service_directory.ServiceDirectoryService/TestPing
    pub async fn test_ping(
        &self,
        request: Request<TestRequestMessage>,
    ) -> Result<Response<TestResponseMessage>, Status> {
        error!("Request from {:?}", request.remote_addr());

        let response = TestResponseMessage {
            server_name: "Server".to_owned(),
        };
        Ok(Response::new(response))
    }

    pub async fn put(
        &self,
        request: Request<PutRequestMessage>,
    ) -> Result<Response<PutResponseMessage>, Status> {
        error!("Request from {:?}", request.remote_addr());

        let response = PutResponseMessage { 
            error: String::new(),
            revision: 1 
        };
        Ok(Response::new(response))
    }

    pub async fn create(
        &self,
        request: Request<CreateRequestMessage>,
    ) -> Result<Response<CreateResponseMessage>, Status> {
        //info!("create Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(CreateResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                }))
            }
            Some(c) => c,
        };

        match &req.obj {
            None => {
                return Ok(Response::new(CreateResponseMessage {
                    error: format!("Invalid input: Empty obj"),
                    revision: 0,
                }))
            }
            Some(o) => {
                let dataObj = o.into();
                match cacher.Create(&dataObj).await {
                    Err(e) => {
                        return Ok(Response::new(CreateResponseMessage {
                            error: format!("Fail: {:?}", e),
                            revision: 0,
                        }))
                    }
                    Ok(obj) => {
                        return Ok(Response::new(CreateResponseMessage {
                            error: "".into(),
                            revision: obj.Revision(),
                        }))
                    }
                }
            }
        }
    }

    pub async fn get(
        &self,
        request: Request<GetRequestMessage>,
    ) -> Result<Response<GetResponseMessage>, Status> {
        //info!("get Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(GetResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    obj: None,
                }))
            }
            Some(c) => c,
        };

        match cacher.Get(&req.namespace, &req.name, req.revision).await {
            Err(e) => {
                return Ok(Response::new(GetResponseMessage {
                    error: format!("Fail: {:?}", e),
                    obj: None,
                }))
            }
            Ok(o) => {
                return Ok(Response::new(GetResponseMessage {
                    error: "".into(),
                    obj: match o {
                        None => None,
                        Some(o) => Some(o.Obj()),
                    },
                }))
            }
        }
    }

    pub async fn delete(
        &self,
        request: Request<DeleteRequestMessage>,
    ) -> Result<Response<DeleteResponseMessage>, Status> {
        //info!("Delete Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(DeleteResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                }))
            }
            Some(c) => c,
        };

        match cacher.Delete(&req.namespace, &req.name).await {
            Err(e) => {
                return Ok(Response::new(DeleteResponseMessage {
                    error: format!("Fail: {:?}", e),
                    revision: 0,
                }))
            }
            Ok(rev) => {
                return Ok(Response::new(DeleteResponseMessage {
                    error: "".into(),
                    revision: rev,
                }))
            }
        }
    }

    pub async fn update(
        &self,
        request: Request<UpdateRequestMessage>,
    ) -> Result<Response<UpdateResponseMessage>, Status> {
        //info!("create Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(UpdateResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    revision: 0,
                }))
            }
            Some(c) => c,
        };

        match &req.obj {
            None => {
                return Ok(Response::new(UpdateResponseMessage {
                    error: format!("Invalid input: Empty obj"),
                    revision: 0,
                }))
            }
            Some(o) => {
                let dataObj: DataObject = o.into();
                match cacher.Update(&dataObj).await {
                    Err(e) => {
                        return Ok(Response::new(UpdateResponseMessage {
                            error: format!("Fail: {:?}", e),
                            revision: 0,
                        }))
                    }
                    Ok(obj) => {
                        return Ok(Response::new(UpdateResponseMessage {
                            error: "".into(),
                            revision: obj.Revision(),
                        }))
                    }
                }
            }
        }
    }

    pub async fn list(
        &self,
        request: Request<ListRequestMessage>,
    ) -> Result<Response<ListResponseMessage>, Status> {
        //info!("create Request {:#?}", &request);

        let req = request.get_ref();
        let cacher = match SVC_DIR.GetCacher(&req.obj_type).await {
            None => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("doesn't support obj type {}", &req.obj_type),
                    ..Default::default()
                }))
            }
            Some(c) => c,
        };

        let labelSelector = match Selector::Parse(&req.label_selector) {
            Err(e) => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };
        let fieldSelector = match Selector::Parse(&req.field_selector) {
            Err(e) => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(s) => s,
        };

        let opts = ListOption {
            revision: req.revision,
            revisionMatch: qobjs::selection_predicate::RevisionMatch::Exact,
            predicate: SelectionPredicate {
                label: labelSelector,
                field: fieldSelector,
                limit: 00,
                continue_: None,
            },
        };

        match cacher.List(&req.namespace, &opts).await {
            Err(e) => {
                return Ok(Response::new(ListResponseMessage {
                    error: format!("Fail: {:?}", e),
                    ..Default::default()
                }))
            }
            Ok(resp) => {
                let mut objs = Vec::new();
                for o in resp.objs {
                    objs.push(o.Obj());
                }
                return Ok(Response::new(ListResponseMessage {
                    error: "".into(),
                    revision: resp.revision,
                    objs: objs,
                }));
            }
        }
    }
}
