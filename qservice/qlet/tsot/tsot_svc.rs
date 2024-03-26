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

use std::fs;
use std::path::Path;
use std::result::Result as SResult;
use std::sync::atomic::*;
use std::sync::Arc;

use tokio::net::UnixListener;
use tokio::sync::Notify;
use tonic::transport::Server;

use qshare::common::*;
use qshare::tsot_cni;

use super::pod_broker::*;
use qshare::tsot_msg::*;

use crate::pod_mgr::NAMESPACE_MGR;
use crate::tsot::conn_svc::ConnectionSvc;
use crate::tsot::dns_proxy::DNS_PROXY;
use crate::QLET_CONFIG;

pub struct TsotSvc {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    // listen for pod connection
    pub listener: UnixListener,

    // listen for gateway connection
    pub hostListener: UnixListener,
}

impl TsotSvc {
    pub fn New() -> Result<Self> {
        let socket = Path::new(TSOT_SOCKET_PATH);
        let hostSocket = Path::new(TSOT_HOST_SOCKET_PATH);

        // create the parent folder if it doesn't exist
        let path = socket.parent().unwrap();
        fs::create_dir_all(path).ok();
        let hostpath = hostSocket.parent().unwrap();
        fs::create_dir_all(hostpath).ok();

        // Delete old socket if necessary
        if socket.exists() {
            std::fs::remove_file(&socket).unwrap();
        }

        // Delete old socket if necessary
        if hostSocket.exists() {
            std::fs::remove_file(&hostSocket).unwrap();
        }

        let listener = UnixListener::bind(socket)?;
        let hostListener = UnixListener::bind(hostSocket)?;

        return Ok(Self {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            listener: listener,
            hostListener: hostListener,
        });
    }

    pub fn Close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }

    pub async fn Process(&self) -> Result<()> {
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                res = self.listener.accept() => {
                    match res {
                        Err(e) => {
                            error!("TsotSvc accept get error {:?}", e);
                            continue;
                        }
                        Ok((stream, _addr)) => {
                            tokio::spawn(async move {
                                let podBroker = PodBroker::New(stream);
                                podBroker.Process(false).await;
                            });
                        }
                    }
                }
                res = self.hostListener.accept() => {
                    match res {
                        Err(e) => {
                            error!("TsotSvc accept get error {:?}", e);
                            continue;
                        }
                        Ok((stream, _addr)) => {
                            tokio::spawn(async move {
                                let podBroker = PodBroker::New(stream);
                                podBroker.Process(true).await;
                            });
                        }
                    }
                }
            }
        }

        return Ok(());
    }
}

pub struct TostCniSvc {}

impl TostCniSvc {
    pub fn GetPodSandboxAddr(&self, _namespace: &str, uid: &str) -> Result<IpAddress> {
        return NAMESPACE_MGR.GetPodSandboxAddr(uid);
    }

    pub fn RemovePodSandbox(&self, _namespace: &str, uid: &str) -> Result<()> {
        return NAMESPACE_MGR.RemovePodSandbox(uid);
    }
}

#[tonic::async_trait]
impl tsot_cni::tsot_cni_service_server::TsotCniService for TostCniSvc {
    async fn get_pod_sandbox_addr(
        &self,
        request: tonic::Request<tsot_cni::GetPodSandboxAddrReq>,
    ) -> SResult<tonic::Response<tsot_cni::GetPodSandboxAddrResp>, tonic::Status> {
        let req = request.into_inner();
        match self.GetPodSandboxAddr(&req.namespace, &req.pod_uid) {
            Ok(addr) => {
                return Ok(tonic::Response::new(tsot_cni::GetPodSandboxAddrResp {
                    error: "".to_owned(),
                    ip_addr: addr.0,
                }))
            }
            Err(e) => {
                return Ok(tonic::Response::new(tsot_cni::GetPodSandboxAddrResp {
                    error: format!("fail: {:?}", e),
                    ip_addr: 0,
                }))
            }
        }
    }
    async fn remove_pod_sandbox(
        &self,
        request: tonic::Request<tsot_cni::RemovePodSandboxReq>,
    ) -> SResult<tonic::Response<tsot_cni::RemovePodSandboxResp>, tonic::Status> {
        let req = request.into_inner();
        match self.RemovePodSandbox(&req.namespace, &req.pod_uid) {
            Ok(()) => {
                return Ok(tonic::Response::new(tsot_cni::RemovePodSandboxResp {
                    error: "".to_owned(),
                }))
            }
            Err(e) => {
                return Ok(tonic::Response::new(tsot_cni::RemovePodSandboxResp {
                    error: format!("fail: {:?}", e),
                }))
            }
        }
    }
}

pub async fn TsotSvc() -> Result<()> {
    info!("Tsot service start ...");
    let tsotSvc = TsotSvc::New()?;
    let tsotSvcFuture = tsotSvc.Process();

    let tsotCniSvc = TostCniSvc {};
    let cniAddr = format!("127.0.0.1:{}", QLET_CONFIG.tsotCniPort);

    let tostCniSvcFuture = Server::builder()
        .add_service(tsot_cni::tsot_cni_service_server::TsotCniServiceServer::new(tsotCniSvc))
        .serve(cniAddr.parse().unwrap());

    let connectionSvcFuture = tokio::spawn(async move {
        let connectionSvc = ConnectionSvc::New(QLET_CONFIG.tsotSvcPort);
        connectionSvc.Process().await.unwrap();
    });

    let dnsProxyFuture = DNS_PROXY.Process();

    tokio::select! {
        _ = tsotSvcFuture => {},
        _ = tostCniSvcFuture => {},
        _ = connectionSvcFuture => {},
        _ = dnsProxyFuture => {}
    }
    info!("Tsot service finish ...");
    return Ok(());
}
