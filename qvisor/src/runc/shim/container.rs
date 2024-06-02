/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc::sync_channel;
use std::sync::mpsc::Receiver;

use containerd_shim::api::*;
use containerd_shim::mount::*;
use containerd_shim::protos::cgroups::metrics::Metrics;
use containerd_shim::protos::protobuf::well_known_types::Timestamp;
use containerd_shim::protos::protobuf::{CodedInputStream, Message};
use containerd_shim::util::read_spec_from_file;
use containerd_shim::util::*;
use nix::sys::stat::Mode;
use nix::unistd::mkdir;
use oci_spec::runtime::LinuxNamespaceType;
use time::OffsetDateTime;

use super::super::super::qlib::common::*;
use super::super::super::runc::oci::LinuxResources;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::container_io::*;
use super::process::*;

#[derive(Clone, Default)]
pub struct ContainerFactory {}

impl ContainerFactory {
    pub fn Create(ns: &str, req: &CreateTaskRequest) -> Result<CommonContainer> {
        let mut bundle = req.bundle.clone();
        if crate::QUARK_CONFIG.lock().Sandboxed {
            bundle = format!("/{}", req.id);
        }

        let mut opts = Options::new();
        if let Some(any) = req.options.as_ref() {
            let mut input = CodedInputStream::from_bytes(any.value.as_ref());
            opts.merge_from(&mut input)
                .map_err(|e| Error::Common(format!("ttrpc error is {:?}", e)))?;
        }
        if opts.compute_size() > 0 {
            debug!("create options: {:?}", &opts);
        }
        let runtime = opts.binary_name.as_str();
        write_options(&bundle, &opts)
            .map_err(|e| Error::Common(format!("ContainerFactory {:?}", e)))?;
        write_runtime(&bundle, runtime)
            .map_err(|e| Error::Common(format!("ContainerFactory {:?}", e)))?;

        let rootfs_vec = req.get_rootfs().to_vec();
        let rootfs = if !rootfs_vec.is_empty() {
            let tmp_rootfs = Path::new(&bundle).join("rootfs");
            if !tmp_rootfs.as_path().exists() {
                mkdir(tmp_rootfs.as_path(), Mode::from_bits(0o711).unwrap())
                    .map_err(|e| Error::Common(format!("ttrpc error is {:?}", e)))?;
            }
            tmp_rootfs
        } else {
            PathBuf::new()
        };
        let rootfs = rootfs
            .as_path()
            .to_str()
            .ok_or_else(|| Error::Common(format!("failed to convert rootfs to str")))?;
        for m in rootfs_vec {
            let mount_type = m.field_type.as_str().none_if(|&x| x.is_empty());
            let source = m.source.as_str().none_if(|&x| x.is_empty());

            let (chdir, options) = if true
                || (mount_type == Some("overlay") && options_size(&m.options) >= 4096 - 512)
            {
                LowerdirCompactor::new(&m.options).compact()
            } else {
                (None, m.options.to_vec())
            };

            error!(
                "mount_rootfs type is {:?} source is {:?}\n target is {:?} options is {:#?} len is {}",
                mount_type, &source, rootfs, &options, options_size(&m.options)
            );

            if let Some(workdir) = chdir {
                let currDir = std::env::current_dir().unwrap();
                std::env::set_current_dir(Path::new(&workdir)).unwrap_or_else(|_| {
                    panic!("ContainerFactory::Create fail when change workdir");
                });
                mount_rootfs(mount_type, source, &options, rootfs)
                    .map_err(|e| Error::Common(format!("ttrpc error is {:?}", e)))?;

                std::env::set_current_dir(currDir).unwrap_or_else(|_| {
                    panic!("ContainerFactory::Create fail when change back workdir");
                });
            } else {
                mount_rootfs(mount_type, source, &options, rootfs)
                    .map_err(|e| Error::Common(format!("ttrpc error is {:?}", e)))?;
            }

        }

        let root = Path::new(opts.root.as_str()).join(ns);
        let log_buf = Path::new(&bundle).join("log.json");

        let id = req.get_id();
        let stdio = ContainerStdio {
            stdin: req.get_stdin().to_string(),
            stdout: req.get_stdout().to_string(),
            stderr: req.get_stderr().to_string(),
            terminal: req.get_terminal(),
        };

        let mut init = InitProcess::New(id, &bundle, stdio);
        init.rootfs = rootfs.to_string();
        let work_dir = Path::new(&bundle).join("work");
        let work_dir = work_dir
            .as_path()
            .to_str()
            .ok_or_else(|| Error::Common(format!("failed to get work_dir str")))?;
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

        let config = GlobalConfig {
            RootDir: root.into_os_string().into_string().unwrap(),
            DebugLevel: DebugLevel::Info,
            DebugLog: log_buf.into_os_string().into_string().unwrap(),
            FileAccess: FileAccessType::default(),
            Network: NetworkType::default(),
            SystemdCgroup: false,
        };

        let container = init
            .Create(&config)
            .map_err(|e| Error::Common(format!("ttrpc error is {:?}", e)))?;
        init.common.pid = container.SandboxPid();
        let container = CommonContainer {
            id: id.to_string(),
            container: container,
            bundle: bundle.to_string(),
            init,
            processes: Default::default(),
        };

        Ok(container)
    }
}

/******************************copy from crate rust-extension **************************************************** */

// need to upgrade the crate when fix the dependency error

fn options_size(options: &[String]) -> usize {
    options.iter().fold(0, |sum, x| sum + x.len())
}

const OVERLAY_LOWERDIR_PREFIX: &str = "lowerdir=";

fn longest_common_prefix(dirs: &[String]) -> &str {
    if dirs.is_empty() {
        return "";
    }

    let first_dir = &dirs[0];

    for (i, byte) in first_dir.as_bytes().iter().enumerate() {
        for dir in dirs {
            if dir.as_bytes().get(i) != Some(byte) {
                let mut end = i;
                // guaranteed not to underflow since is_char_boundary(0) is always true
                while !first_dir.is_char_boundary(end) {
                    end -= 1;
                }

                return &first_dir[0..end];
            }
        }
    }

    first_dir
}

// NOTE: the snapshot id is based on digits.
// in order to avoid to get snapshots/x, should be back to parent dir.
// however, there is assumption that the common dir is ${root}/io.containerd.v1.overlayfs/snapshots.
fn trim_flawed_dir(s: &str) -> String {
    s[0..s.rfind('/').unwrap_or(0) + 1].to_owned()
}

#[derive(Default)]
struct LowerdirCompactor {
    options: Vec<String>,
    lowerdirs: Option<Vec<String>>,
    lowerdir_prefix: Option<String>,
}

impl LowerdirCompactor {
    fn new(options: &[String]) -> Self {
        Self {
            options: options.to_vec(),
            ..Self::default()
        }
    }

    fn lowerdirs(&mut self) -> &mut Self {
        self.lowerdirs = Some(
            self.options
                .iter()
                .filter(|x| x.starts_with(OVERLAY_LOWERDIR_PREFIX))
                .map(|x| x.strip_prefix(OVERLAY_LOWERDIR_PREFIX).unwrap_or(x))
                .flat_map(|x| x.split(':'))
                .map(str::to_string)
                .collect(),
        );
        self
    }

    fn lowerdir_prefix(&mut self) -> &mut Self {
        self.lowerdir_prefix = self
            .lowerdirs
            .as_ref()
            .filter(|x| x.len() > 1)
            .map(|x| longest_common_prefix(x))
            .map(trim_flawed_dir)
            .filter(|x| !x.is_empty() && x != "/");
        self
    }

    fn compact(&mut self) -> (Option<String>, Vec<String>) {
        self.lowerdirs().lowerdir_prefix();
        if let Some(chdir) = &self.lowerdir_prefix {
            let lowerdir_str = self
                .lowerdirs
                .as_ref()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|x| x.strip_prefix(chdir).unwrap_or(x))
                .collect::<Vec<&str>>()
                .join(":");
            let replace = |x: &str| -> String {
                if x.starts_with(OVERLAY_LOWERDIR_PREFIX) {
                    format!("{}{}", OVERLAY_LOWERDIR_PREFIX, lowerdir_str)
                } else {
                    x.to_string()
                }
            };
            (
                self.lowerdir_prefix.clone(),
                self.options
                    .iter()
                    .map(|x| replace(x))
                    .collect::<Vec<String>>(),
            )
        } else {
            (None, self.options.to_vec())
        }
    }
}

/****************************** end copy from crate rust-extension **************************************************** */

pub struct CommonContainer {
    pub id: String,
    pub container: Container,
    pub bundle: String,
    pub init: InitProcess,
    pub processes: HashMap<String, ExecProcess>,
}

impl CommonContainer {
    pub fn should_kill_all_on_exit(&mut self, bundle_path: &str) -> bool {
        match read_spec_from_file(bundle_path) {
            Ok(spec) => match spec.linux() {
                None => return true,
                Some(linux) => match linux.namespaces() {
                    None => true,
                    Some(namespaces) => {
                        for ns in namespaces {
                            if ns.typ() == LinuxNamespaceType::Pid && ns.path().is_none() {
                                return false;
                            }
                        }
                        return true;
                    }
                },
            },
            Err(e) => {
                error!("should_kill_all_on_exit: {}", e);
                return false;
            }
        }
    }

    pub fn SandboxId(&self) -> String {
        return self.container.Sandbox.as_ref().unwrap().ID.clone();
    }

    pub fn Pid(&self) -> i32 {
        return self.init.common.pid();
    }

    pub fn state(&self, exec_id: Option<&str>) -> Result<StateResponse> {
        let mut resp = match exec_id {
            Some(exec_id) => {
                let process = self
                    .processes
                    .get(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;
                process.state()
            }
            None => self.init.common.state(),
        };

        resp.bundle = self.bundle.to_string();
        debug!("container state: {:?}", resp);
        Ok(resp)
    }

    pub fn exec(&mut self, req: ExecProcessRequest) -> Result<()> {
        let exec_id = req.exec_id.to_string();
        let mut exec_process =
            ExecProcess::try_from(req).map_err(|e| Error::Common(format!("{:?}", e)))?;

        let stdio = exec_process.common.stdio.CreateIO()?;
        exec_process.common.containerIO = stdio;

        self.processes.insert(exec_id, exec_process);
        Ok(())
    }

    pub fn wait_channel(&mut self, exec_id: Option<&str>) -> Result<Receiver<i8>> {
        let process = match exec_id {
            Some(exec_id) => {
                let p = self
                    .processes
                    .get_mut(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;
                &mut p.common
            }
            None => &mut self.init.common,
        };

        let (tx, rx) = sync_channel::<i8>(0);
        if process.exited_at() == None {
            process.add_wait(tx);
        }
        Ok(rx)
    }

    pub fn wait(&mut self, exec_id: Option<&str>) -> Result<u32> {
        let pid = match exec_id {
            Some(exec_id) => {
                let process = self
                    .processes
                    .get(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;

                process.pid()
            }
            None => self.init.common.pid(),
        };

        return self.container.WaitPid(pid, true); // todo: how to clear this?
    }

    pub fn get_exit_info(
        &self,
        exec_id: Option<&str>,
    ) -> Result<(i32, i32, Option<OffsetDateTime>)> {
        match exec_id {
            Some(exec_id) => {
                let process = self
                    .processes
                    .get(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;

                Ok((process.pid(), process.exit_code(), process.exited_at()))
            }
            None => Ok((
                self.init.common.pid(),
                self.init.common.exit_code(),
                self.init.common.exited_at(),
            )),
        }
    }

    pub fn resize_pty(&mut self, exec_id: Option<&str>, height: u32, width: u32) -> Result<()> {
        match exec_id {
            Some(exec_id) => {
                let process = self
                    .processes
                    .get_mut(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;
                process
                    .resize_pty(height, width)
                    .map_err(|e| Error::Common(format!("{:?}", e)))?;
                Ok(())
            }
            None => {
                self.init
                    .common
                    .resize_pty(height, width)
                    .map_err(|e| Error::Common(format!("{:?}", e)))?;
                Ok(())
            }
        }
    }

    pub fn kill(&mut self, exec_id: Option<&str>, signal: u32, all: bool) -> Result<()> {
        match exec_id {
            Some(exec_id) => {
                let p = self
                    .processes
                    .get(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;
                self.container.SignalProcess(signal as i32, p.pid())?;
            }
            None => {
                self.container.SignalContainer(signal as i32, all)?;
            }
        }

        return Ok(());
    }

    pub fn delete(&mut self, exec_id_opt: Option<&str>) -> Result<(i32, u32, Timestamp)> {
        let (pid, code, exit_at) = self.get_exit_info(exec_id_opt)?;
        match exec_id_opt {
            Some(exec_id) => {
                self.processes.remove(exec_id);
            }
            None => {
                // never return Error for failed destroy
                self.container.Destroy().unwrap();
            }
        };

        let mut time_stamp = Timestamp::new();
        if let Some(exit_at) = exit_at {
            time_stamp.set_seconds(exit_at.unix_timestamp());
            time_stamp.set_nanos(exit_at.nanosecond() as i32);
        } else {
            return Err(Error::Common(
                "failed to get exit status from container".to_string(),
            ));
        }

        Ok((pid, code as u32, time_stamp))
    }

    pub fn pids(&self) -> Result<PidsResponse> {
        /*let processInfos = self.container.Processes();

        for pi in &processInfos {
            let pid = pi.PID;

        }*/

        return Err(Error::Unimplemented("CommonContainer::pids".to_string()));
    }

    pub fn stats(&self) -> Result<Metrics> {
        return Err(Error::Unimplemented("CommonContainer::pids".to_string()));
        /*let mut metrics = Metrics::new();
        // get container main process cgroup
        let path = get_cgroups_relative_paths_by_pid(self.common.init.pid() as u32)?;
        let cgroup = Cgroup::load_with_relative_paths(hierarchies::auto(), Path::new("."), path);

        // to make it easy, fill the necessary metrics only.
        for sub_system in Cgroup::subsystems(&cgroup) {
            match sub_system {
                Subsystem::CpuAcct(cpuacct_ctr) => {
                    let mut cpu_usage = CPUUsage::new();
                    cpu_usage.set_total(cpuacct_ctr.cpuacct().usage);
                    let mut cpu_stat = CPUStat::new();
                    cpu_stat.set_usage(cpu_usage);
                    metrics.set_cpu(cpu_stat);
                }
                Subsystem::Mem(mem_ctr) => {
                    let mem = mem_ctr.memory_stat();
                    let mut mem_entry = MemoryEntry::new();
                    mem_entry.set_usage(mem.usage_in_bytes);
                    let mut mem_stat = MemoryStat::new();
                    mem_stat.set_usage(mem_entry);
                    mem_stat.set_total_inactive_file(mem.stat.total_inactive_file);
                    metrics.set_memory(mem_stat);
                }
                _ => {}
            }
        }
        Ok(metrics)*/
    }

    pub fn update(&mut self, _resources: &LinuxResources) -> Result<()> {
        return Err(Error::Unimplemented("CommonContainer::pids".to_string()));
        /*// get container main process cgroup
        let path = get_cgroups_relative_paths_by_pid(self.common.init.pid() as u32)?;
        let cgroup = Cgroup::load_with_relative_paths(hierarchies::auto(), Path::new("."), path);

        for sub_system in Cgroup::subsystems(&cgroup) {
            match sub_system {
                Subsystem::Pid(pid_ctr) => {
                    // set maximum number of PIDs
                    if let Some(pids) = resources.pids() {
                        pid_ctr
                            .set_pid_max(MaxValue::Value(pids.limit()))
                            .map_err(other_error!(e, "set pid max"))?;
                    }
                }
                Subsystem::Mem(mem_ctr) => {
                    if let Some(memory) = resources.memory() {
                        // set memory limit in bytes
                        if let Some(limit) = memory.limit() {
                            mem_ctr
                                .set_limit(limit)
                                .map_err(other_error!(e, "set mem limit"))?;
                        }

                        // set memory swap limit in bytes
                        if let Some(swap) = memory.swap() {
                            mem_ctr
                                .set_memswap_limit(swap)
                                .map_err(other_error!(e, "set memsw limit"))?;
                        }
                    }
                }
                Subsystem::CpuSet(cpuset_ctr) => {
                    if let Some(cpu) = resources.cpu() {
                        // set CPUs to use within the cpuset
                        if let Some(cpus) = cpu.cpus() {
                            cpuset_ctr
                                .set_cpus(cpus)
                                .map_err(other_error!(e, "set CPU sets"))?;
                        }

                        // set list of memory nodes in the cpuset
                        if let Some(mems) = cpu.mems() {
                            cpuset_ctr
                                .set_mems(mems)
                                .map_err(other_error!(e, "set CPU memes"))?;
                        }
                    }
                }
                Subsystem::Cpu(cpu_ctr) => {
                    if let Some(cpu) = resources.cpu() {
                        // set CPU shares
                        if let Some(shares) = cpu.shares() {
                            cpu_ctr
                                .set_shares(shares)
                                .map_err(other_error!(e, "set CPU share"))?;
                        }

                        // set CPU hardcap limit
                        if let Some(quota) = cpu.quota() {
                            cpu_ctr
                                .set_cfs_quota(quota)
                                .map_err(other_error!(e, "set CPU quota"))?;
                        }

                        // set CPU hardcap period
                        if let Some(period) = cpu.period() {
                            cpu_ctr
                                .set_cfs_period(period)
                                .map_err(other_error!(e, "set CPU period"))?;
                        }
                    }
                }
                Subsystem::HugeTlb(ht_ctr) => {
                    // set the limit if "pagesize" hugetlb usage
                    if let Some(hp_limits) = resources.hugepage_limits() {
                        for limit in hp_limits {
                            ht_ctr
                                .set_limit_in_bytes(
                                    limit.page_size().as_str(),
                                    limit.limit() as u64,
                                )
                                .map_err(other_error!(e, "set huge page limit"))?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())*/
    }

    pub fn start(&mut self, exec_id: Option<&str>) -> Result<i32> {
        match exec_id {
            Some(exec_id) => {
                let process = self
                    .processes
                    .get_mut(exec_id)
                    .ok_or_else(|| Error::Common("can not find the exec by id".to_string()))?;

                let fds = process.common.containerIO.StdioFds()?;

                let pid = self.container.Sandbox.as_ref().unwrap().Exec1(
                    &self.id,
                    exec_id,
                    &process.spec,
                    &fds,
                )?;
                process.common.pid = pid;
                let sandboxId = self.container.Sandbox.as_ref().unwrap().ID.clone();
                process
                    .common
                    .CopyIO(&*self.id, pid, sandboxId)
                    .map_err(|e| Error::Common(format!("{:?}", e)))?;
                return Ok(pid);
            }
            None => {
                self.container.Start()?;
                self.init.common.set_status(Status::RUNNING);
                Ok(self.init.common.pid())
            }
        }
    }

    pub fn pid(&self) -> i32 {
        self.init.pid()
    }
}
