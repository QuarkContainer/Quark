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

use axum::{
    body::Body, extract::Path, response::IntoResponse, routing::delete, routing::get,
    routing::post, Json, Router,
};
use hyper::StatusCode;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use serde::{Deserialize, Serialize};

//use hyper_util::{client::legacy::connect::HttpConnector, rt::TokioExecutor};

type Client = hyper_util::client::legacy::Client<HttpConnector, Body>;

use qshare::common::*;

use crate::namespace_mgr::NamespaceSpec;
use crate::NAMESPACE_MGR;
use crate::NAMESPACE_STORE;
use crate::{func_mgr::FuncPackageSpec, func_worker::FUNCAGENT_MGR};

pub const FUNCPOD_TYPE: &str = "funcpod_type.qservice.io";
pub const FUNCPOD_FUNCNAME: &str = "fun_name.qservice.io";
pub const FUNCPOD_PROMPT: &str = "prompt";

pub struct HttpGateway {}

impl HttpGateway {
    pub async fn HttpServe(&self) -> Result<()> {
        let client: Client =
            hyper_util::client::legacy::Client::<(), ()>::builder(TokioExecutor::new())
                .build(HttpConnector::new());

        let app = Router::new()
            .route("/namespaces/", post(PostNamespace))
            .route("/funcpackages/", post(PostFuncPackage))
            .route(
                "/funcpackages/:tenant/:namespace/:name",
                delete(DropFuncPackage),
            )
            .route(
                "/funcpackages/:tenant/:namespace/:name",
                get(GetFuncPackage),
            )
            .route("/funcpackages/:tenant/:namespace", get(GetFuncPackages))
            .route("/funcpods/:tenant/:namespace/:name", get(GetFuncPods))
            .route("/funccall/", post(PostFuncCall))
            .with_state(client);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:4000")
            .await
            .unwrap();
        println!("listening on {}", listener.local_addr().unwrap());
        axum::serve(listener, app).await.unwrap();

        return Ok(());
    }
}

async fn PostNamespace(Json(spec): Json<NamespaceSpec>) -> impl IntoResponse {
    if NAMESPACE_MGR
        .get()
        .unwrap()
        .ContainsNamespace(&spec.tenant, &spec.namespace)
    {
        match NAMESPACE_STORE.get().unwrap().UpdateNamespace(&spec).await {
            Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
            Ok(()) => (StatusCode::OK, Json(format!("ok"))),
        }
    } else {
        match NAMESPACE_STORE.get().unwrap().CreateNamespace(&spec).await {
            Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
            Ok(()) => (StatusCode::OK, Json(format!("ok"))),
        }
    }
}

async fn GetFuncPods(
    Path((tenant, namespace, funcName)): Path<(String, String, String)>,
) -> impl IntoResponse {
    match NAMESPACE_MGR
        .get()
        .unwrap()
        .GetFuncPods(&tenant, &namespace, &funcName)
    {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
        Ok(pods) => {
            let pods = serde_json::to_string_pretty(&pods).unwrap();
            (StatusCode::OK, Json(pods))
        }
    }
}

async fn PostFuncPackage(Json(spec): Json<FuncPackageSpec>) -> impl IntoResponse {
    match NAMESPACE_MGR.get().unwrap().ContainsFuncPackage(
        &spec.tenant,
        &spec.namespace,
        &spec.name,
    ) {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
        Ok(contains) => {
            if contains {
                match NAMESPACE_STORE
                    .get()
                    .unwrap()
                    .UpdateFuncPackage(&spec)
                    .await
                {
                    Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
                    Ok(()) => (StatusCode::OK, Json(format!("ok"))),
                }
            } else {
                match NAMESPACE_STORE
                    .get()
                    .unwrap()
                    .CreateFuncPackage(&spec)
                    .await
                {
                    Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
                    Ok(()) => (StatusCode::OK, Json(format!("ok"))),
                }
            }
        }
    }
}

async fn DropFuncPackage(
    Path((tenant, namespace, name)): Path<(String, String, String)>,
) -> impl IntoResponse {
    match NAMESPACE_MGR
        .get()
        .unwrap()
        .GetFuncPackage(&tenant, &namespace, &name)
    {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
        Ok(funcPackage) => {
            let revision = funcPackage.spec.revision;
            match NAMESPACE_STORE
                .get()
                .unwrap()
                .DropFuncPackage(&namespace, &name, revision)
                .await
            {
                Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
                Ok(()) => (StatusCode::OK, Json(format!("ok"))),
            }
        }
    }
}

async fn GetFuncPackage(
    Path((tenant, namespace, name)): Path<(String, String, String)>,
) -> impl IntoResponse {
    match NAMESPACE_MGR
        .get()
        .unwrap()
        .GetFuncPackage(&tenant, &namespace, &name)
    {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
        Ok(funcPackage) => {
            let spec = funcPackage.spec.ToJson();
            (StatusCode::OK, Json(spec))
        }
    }
}

async fn GetFuncPackages(Path((tenant, namespace)): Path<(String, String)>) -> impl IntoResponse {
    match NAMESPACE_MGR
        .get()
        .unwrap()
        .GetFuncPackages(&tenant, &namespace)
    {
        Err(e) => (StatusCode::BAD_REQUEST, Json(format!("{:?}", e))),
        Ok(funcPackages) => {
            let str = serde_json::to_string(&funcPackages).unwrap(); // format!("{:#?}", funcPackages);
            (StatusCode::OK, Json(str))
        }
    }
}

async fn PostFuncCall(Json(req): Json<PromptReq>) -> impl IntoResponse {
    match NAMESPACE_MGR
        .get()
        .unwrap()
        .GetFuncPackage(&req.tenant, &req.namespace, &req.func)
    {
        Err(e) => {
            return (StatusCode::BAD_REQUEST, Json(format!("{:?}", e)));
        }
        Ok(funcPackage) => {
            let resp = FUNCAGENT_MGR.Call(&funcPackage, req).await;

            return (resp.status, Json(resp.response));
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct PromptReq {
    pub tenant: String,
    pub namespace: String,
    pub func: String,
    pub prompt: String,
}
