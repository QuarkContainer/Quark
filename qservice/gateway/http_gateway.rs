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

use std::result::Result as TResult;
use axum::{
    body::Body,
    extract::{Request, State},
    http::uri::Uri,
    response::{IntoResponse, Response},
    routing::get,
    routing::post,
    routing::delete,
    Router,
    Json,
    extract::Path
};
use hyper::StatusCode;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;

//use hyper_util::{client::legacy::connect::HttpConnector, rt::TokioExecutor};

type Client = hyper_util::client::legacy::Client<HttpConnector, Body>;

use qshare::common::*;

use crate::func_mgr::FuncPackageSpec;
use crate::namespace_mgr::NamespaceSpec;
use crate::NAMESPACE_MGR;
use crate::NAMESPACE_STORE;

pub struct HttpGateway {

}

impl HttpGateway {
    pub async fn HttpServe(&self) -> Result<()> {
        tokio::spawn(server());

        let client: Client =
            hyper_util::client::legacy::Client::<(), ()>::builder(TokioExecutor::new())
                .build(HttpConnector::new());
    
        let app = Router::new()
            .route("/func/*request", get(ReqHandler))
            .route("/namespaces/", post(PostNamespace))
            .route("/funcpackages/", post(PostFuncPackage))
            .route("/funcpackages/:namespace/:name", delete(DropFuncPackage))
            .route("/funcpackages/:namespace/:name", get(GetFuncPackage))
            .with_state(client);
        
        let listener = tokio::net::TcpListener::bind("127.0.0.1:4000")
            .await
            .unwrap();
        println!("listening on {}", listener.local_addr().unwrap());
        axum::serve(listener, app).await.unwrap();

        return Ok(())
    }
}

async fn PostNamespace(
    Json(payload): Json<NamespaceSpec>
) -> impl IntoResponse {
    error!("postnamespace {:?}", &payload);
    if NAMESPACE_MGR.get().unwrap().ContainersNamespace(&payload.namespace) {
        match NAMESPACE_STORE.get().unwrap().UpdateNamespace(&payload).await {
            Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
            Ok(()) => (StatusCode::OK, Json(format!("ok"))),
        }
    } else {
        match NAMESPACE_STORE.get().unwrap().CreateNamespace(&payload).await {
            Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
            Ok(()) => (StatusCode::OK, Json(format!("ok"))),
        }
    }
}

async fn PostFuncPackage(
    Json(payload): Json<FuncPackageSpec>
) -> impl IntoResponse {
    match NAMESPACE_MGR.get().unwrap().ContainersFuncPackage(&payload.namespace, &payload.name) {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
        Ok(containers) => {
            if containers {
                match NAMESPACE_STORE.get().unwrap().UpdateFuncPackage(&payload).await {
                    Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
                    Ok(()) => (StatusCode::OK, Json(format!("ok"))),
                } 
            } else {
                match NAMESPACE_STORE.get().unwrap().CreateFuncPackage(&payload).await {
                    Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
                    Ok(()) => (StatusCode::OK, Json(format!("ok"))),
                } 
            }
        }
    }
}

async fn DropFuncPackage(
    Path((namespace, name)): Path<(String, String)>
) -> impl IntoResponse {
    error!("DropFuncPackage 1 {:?}/{}", &namespace, &name);
    match NAMESPACE_MGR.get().unwrap().GetFuncPackage(&namespace, &name) {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
        Ok(funcPackage) => {
            let revision = funcPackage.lock().unwrap().spec.revision;
            error!("DropFuncPackage 2 {:?}/{}/{}", &namespace, &name, revision);
            match NAMESPACE_STORE.get().unwrap().DropFuncPackage(&namespace, &name, revision).await {
                Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
                Ok(()) => (StatusCode::OK, Json(format!("ok"))),
            } 
        }
    }
}

async fn GetFuncPackage(
    Path((namespace, name)): Path<(String, String)>
) -> impl IntoResponse {
    error!("GetFuncPackage1 {:?}/{}", &namespace, &name);
    match NAMESPACE_MGR.get().unwrap().GetFuncPackage(&namespace, &name) {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}",e))),
        Ok(funcPackage) => {
            let spec = funcPackage.lock().unwrap().spec.ToJson();
            error!("GetFuncPackage {:?}", &spec);
            (StatusCode::OK, Json(spec))
        }
    }
}

async fn ReqHandler(State(client): State<Client>, mut req: Request) -> TResult<Response, StatusCode> {
    let path = req.uri().path();
    let path_query = req
        .uri()
        .path_and_query()
        .map(|v| v.as_str())
        .unwrap_or(path);

    let uri = req.uri();
    println!("path is {:?}",uri.query());
    println!("path is {:?}",req.uri().path_and_query());

    let uri = format!("http://127.0.0.1:3000{}", path_query);

    *req.uri_mut() = Uri::try_from(uri).unwrap();

    Ok(client
        .request(req)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .into_response())
}

async fn server() {
    let app = Router::new().route("/", get(|| async { "Hello, world!" }));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    println!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}