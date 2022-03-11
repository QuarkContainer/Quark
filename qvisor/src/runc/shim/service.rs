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

use std::sync::{Arc};

use containerd_shim::ExitSignal;
use containerd_shim::api;
use containerd_shim::api::*;
use containerd_shim::util::*;
use containerd_shim::TtrpcResult;
use containerd_shim::TtrpcContext;
use containerd_shim::Task;
use containerd_shim::Error;
use containerd_shim::spawn;
use containerd_shim::StartOpts;
use containerd_shim::Config;
use containerd_shim::RemotePublisher;
use containerd_shim::Shim;
use containerd_shim::Result;

//use super::super::super::qlib::common;
//use super::super::super::qlib::common::*;

use super::shim_task::*;

//pub type ShimResult<T> = std::result::Result<T, containerd_shim::Error>;

const GROUP_LABELS: [&str; 2] = [
    "io.containerd.runc.v2.group",
    "io.kubernetes.cri.sandbox-id",
];

#[derive(Default)]
struct Service {
    exit: Arc<ExitSignal>,
    id: String,
    namespace: String,
}

impl Shim for Service {
    type T = ShimTask;

    fn new(
        _runtime_id: &str,
        id: &str,
        namespace: &str,
        _publisher: RemotePublisher,
        _config: &mut Config,
    ) -> Self {
        Service {
            exit: Arc::new(ExitSignal::default()),
            id: id.to_string(),
            namespace: namespace.to_string(),
        }
    }

    fn start_shim(&mut self, opts: StartOpts) -> Result<String> {
        let mut grouping = opts.id.clone();
        let spec = read_spec_from_file("")?;
        match spec.annotations() {
            Some(annotations) => {
                for label in GROUP_LABELS.iter() {
                    if let Some(value) = annotations.get(*label) {
                        grouping = value.to_string();
                        break;
                    }
                }
            }
            None => {}
        }

        let address = spawn(opts, &grouping, Vec::new())?;
        write_address(&address)?;
        Ok(address.to_string())
    }

    fn delete_shim(&mut self) -> Result<DeleteResponse> {
        Err(Error::Unimplemented("delete shim".to_string()))
    }

    fn wait(&mut self) {
        self.exit.wait();
    }

    fn create_task_service(&self) -> Self::T {
        ShimTask::default()
    }
}

impl Task for Service {
    fn create(
        &self,
        _ctx: &TtrpcContext,
        _req: api::CreateTaskRequest,
    ) -> TtrpcResult<api::CreateTaskResponse> {
        // New task nere...
        Ok(api::CreateTaskResponse::default())
    }

    fn connect(
        &self,
        _ctx: &TtrpcContext,
        _req: api::ConnectRequest,
    ) -> TtrpcResult<api::ConnectResponse> {
        info!("Connect request");
        Ok(api::ConnectResponse {
            version: String::from("example"),
            ..Default::default()
        })
    }

    fn shutdown(&self, _ctx: &TtrpcContext, _req: api::ShutdownRequest) -> TtrpcResult<api::Empty> {
        info!("Shutdown request");
        self.exit.signal(); // Signal to shutdown shim server
        Ok(api::Empty::default())
    }
}