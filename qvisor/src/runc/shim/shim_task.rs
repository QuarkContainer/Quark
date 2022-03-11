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
use std::path::Path;
use std::path::PathBuf;
use nix::unistd::{mkdir, Pid};
use nix::sys::stat::Mode;

use containerd_shim::protos::protobuf::{CodedInputStream, Message, RepeatedField};
use containerd_shim::TtrpcResult;
use containerd_shim::api;
use containerd_shim::api::*;
use containerd_shim::TtrpcContext;
use containerd_shim::ExitSignal;
use containerd_shim::Task;
use containerd_shim::mount::*;
use containerd_shim::util::*;
//use containerd_shim::protos::containerd_shim_protos::ttrpc;

use shim_proto::ttrpc;

use super::super::cmd::config::*;

//use super::super::super::qlib::common;
//use containerd_shim::api::*;

#[derive(Default)]
pub struct ShimTask {
    namespace: String,
    exit: Arc<ExitSignal>,
}

impl Task for ShimTask {
    fn create(
        &self,
        _ctx: &TtrpcContext,
        req: api::CreateTaskRequest,
    ) -> TtrpcResult<api::CreateTaskResponse> {
        let bundle = req.bundle.as_str();
        let mut opts = Options::new();
        if let Some(any) = req.options.as_ref() {
            let mut input = CodedInputStream::from_bytes(any.value.as_ref());
            opts.merge_from(&mut input).map_err(|e|ttrpc::Error::Others(format!("ttrpc error is {:?}", e)))?;
        }
        if opts.compute_size() > 0 {
            debug!("create options: {:?}", &opts);
        }
        let mut runtime = opts.binary_name.as_str();
        write_options(bundle, &opts)?;
        write_runtime(bundle, runtime)?;

        let rootfs_vec = req.get_rootfs().to_vec();
        let rootfs = if !rootfs_vec.is_empty() {
            let tmp_rootfs = Path::new(bundle).join("rootfs");
            if !tmp_rootfs.as_path().exists() {
                mkdir(tmp_rootfs.as_path(), Mode::from_bits(0o711).unwrap()).map_err(|e|ttrpc::Error::Others(format!("ttrpc error is {:?}", e)))?;
            }
            tmp_rootfs
        } else {
            PathBuf::new()
        };
        let rootfs = rootfs
            .as_path()
            .to_str()
            .ok_or_else(|| ttrpc::Error::Others(format!("failed to convert rootfs to str")))?;
        for m in rootfs_vec {
            let mount_type = m.field_type.as_str().none_if(|&x| x.is_empty());
            let source = m.source.as_str().none_if(|&x| x.is_empty());
            mount_rootfs(mount_type, source, &m.options.to_vec(), rootfs)?;
        }

        let root = Path::new(opts.root.as_str()).join(&self.namespace);
        let log_buf = Path::new(bundle).join("log.json");


        let config = GlobalConfig {
            RootDir: root.into_os_string().into_string().unwrap(),
            DebugLevel: DebugLevel::Info,
            DebugLog: log_buf.into_os_string().into_string().unwrap(),
            FileAccess: FileAccessType::default(),
            Network: NetworkType::default()
        };

        let id = self.req.get_id();


        /*let runc = {
            if runtime.is_empty() {
                runtime = DEFAULT_COMMAND;
            }
            let root = opts.root.as_str();
            let root = Path::new(if root.is_empty() {
                DEFAULT_RUNC_ROOT
            } else {
                root
            })
                .join(ns);
            let log_buf = Path::new(bundle).join("log.json");
            GlobalOpts::default()
                .command(runtime)
                .root(root)
                .log(log_buf)
                .systemd_cgroup(opts.get_systemd_cgroup())
                .log_json()
                .build()
                .map_err(other_error!(e, "unable to create runc instance"))?
        };

        let id = req.get_id();
        let stdio = Stdio {
            stdin: req.get_stdin().to_string(),
            stdout: req.get_stdout().to_string(),
            stderr: req.get_stderr().to_string(),
            terminal: req.get_terminal(),
        };

        let mut init = InitProcess::new(id, bundle, runc, stdio);
        init.rootfs = rootfs.to_string();
        let work_dir = Path::new(bundle).join("work");
        let work_dir = work_dir
            .as_path()
            .to_str()
            .ok_or_else(|| other!("failed to get work_dir str"))?;
        init.work_dir = work_dir.to_string();
        init.io_uid = opts.get_io_uid();
        init.io_gid = opts.get_io_gid();
        init.no_pivot_root = opts.get_no_pivot_root();
        init.no_new_key_ring = opts.get_no_new_keyring();
        init.criu_work_path = if opts.get_criu_path().is_empty() {
            work_dir.to_string()
        } else {
            opts.get_criu_path().to_string()
        };

        let config = CreateConfig::default();
        init.create(&config)?;
        let container = RuncContainer {
            common: CommonContainer {
                id: id.to_string(),
                bundle: bundle.to_string(),
                init,
                processes: Default::default(),
            },
        };
        Ok(container) */

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