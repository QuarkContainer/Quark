// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");

use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::{thread, time};

use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::path::*;
use super::cgroup_v2::{Cpu2, CpuSet2, Memory2, SUBTREE_CONTROL};
use super::super::oci::*;

pub const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const CONTROLLERS_FILE: &str = "cgroup.controllers";
const V2_CONTROLLERS: &[&str] = &["cpu", "memory", "cpuset"];
/// Leaf name for the sandbox process when the OCI path is a delegation parent (CRI pod cgroup).
const PROCESS_LEAF: &str = "init";

/// Quark requires the unified cgroup v2 hierarchy (Linux 5.8+ / modern K8s nodes).
pub fn cgroup_v2_available() -> bool {
    Path::new(&Join(CGROUP_ROOT, CONTROLLERS_FILE)).exists()
}

pub fn require_cgroup_v2() -> Result<()> {
    if cgroup_v2_available() {
        return Ok(());
    }
    Err(Error::Common(
        "Quark requires cgroup v2 (unified hierarchy at /sys/fs/cgroup)".to_string(),
    ))
}

pub fn unified_cgroup_path(name: &str) -> String {
    Join(CGROUP_ROOT, name.trim_start_matches('/'))
}

fn self_cgroup_v2_path() -> Result<String> {
    let f = File::open("/proc/self/cgroup").map_err(|e| Error::IOError(format!("{:?}", e)))?;
    for line in BufReader::new(f).lines() {
        let l = line.map_err(|e| Error::IOError(format!("{:?}", e)))?;
        let parts: Vec<&str> = l.splitn(3, ':').collect();
        if parts.len() == 3 && parts[0] == "0" {
            return Ok(unified_cgroup_path(parts[2]));
        }
    }
    Err(Error::Common(
        "cgroup v2 path not found in /proc/self/cgroup".to_string(),
    ))
}

pub fn SetOptionalValueInt(path: &str, name: &str, val: Option<i64>) -> Result<()> {
    let val = match val {
        None => return Ok(()),
        Some(v) => {
            if v == 0 {
                return Ok(());
            }
            v
        }
    };
    SetValue(path, name, &format!("{}", val))
}

pub fn SetOptionalValueUint(path: &str, name: &str, val: Option<u64>) -> Result<()> {
    let val = match val {
        None => return Ok(()),
        Some(v) => {
            if v == 0 {
                return Ok(());
            }
            v
        }
    };
    SetValue(path, name, &format!("{}", val))
}

pub fn SetValue(path: &str, name: &str, data: &str) -> Result<()> {
    WriteFile(&Join(path, name), data)
}

pub fn WriteFile(path: &str, data: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| Error::IOError(format!("WriteFile {:?} io::error is {:?}", path, e)))?;
    file.write_all(data.as_bytes())
        .map_err(|e| Error::IOError(format!("SetValue {:?} io::error is {:?}", path, e)))?;
    Ok(())
}

pub fn GetValue(path: &str, name: &str) -> Result<String> {
    let fullpath = Join(path, name);
    fs::read_to_string(&fullpath).map_err(|e| {
        Error::IOError(format!(
            "GetValue fail when read file {} with error {:?}",
            &fullpath, e
        ))
    })
}

/// If `cgroup_path/<file>` is empty, copy from the nearest ancestor (explicit
/// value first, then `*.effective` — cgroup v2 leaves `cpuset.cpus` empty when
/// inheriting the full mask).
pub fn inherit_cgroup_file(cgroup_path: &str, file: &str) -> Result<()> {
    let file_path = Join(cgroup_path, file);
    let current = fs::read_to_string(&file_path).map_err(|e| {
        Error::IOError(format!("inherit_cgroup_file read {}: {:?}", file_path, e))
    })?;
    if !current.trim().is_empty() {
        return Ok(());
    }

    let effective_file = match file {
        "cpuset.cpus" => "cpuset.cpus.effective",
        "cpuset.mems" => "cpuset.mems.effective",
        _ => file,
    };

    let mut parent = Dir(cgroup_path);
    loop {
        let explicit = Join(&parent, file);
        if let Ok(v) = fs::read_to_string(&explicit) {
            if !v.trim().is_empty() {
                return WriteFile(&file_path, v.trim());
            }
        }
        if effective_file != file {
            let eff_path = Join(&parent, effective_file);
            if let Ok(v) = fs::read_to_string(&eff_path) {
                if !v.trim().is_empty() {
                    return WriteFile(&file_path, v.trim());
                }
            }
        }
        if parent.len() <= CGROUP_ROOT.len() {
            break;
        }
        parent = Dir(&parent);
    }

    Err(Error::Common(format!(
        "cgroup file {}/{} has no inheritable value",
        cgroup_path, file
    )))
}

/// True when `path` has delegated controllers and cannot hold processes (cgroup v2).
pub fn subtree_control_active(path: &str) -> Result<bool> {
    let control = Join(path, SUBTREE_CONTROL);
    if !Path::new(&control).exists() {
        return Ok(false);
    }
    let s = fs::read_to_string(&control)
        .map_err(|e| Error::IOError(format!("read {}: {:?}", control, e)))?;
    Ok(!s.trim().is_empty())
}

/// Enable cgroup v2 controllers on `parent` so a child cgroup may use them.
fn enable_subtree_controllers(parent: &str, controllers: &[&str]) -> Result<()> {
    let control = Join(parent, SUBTREE_CONTROL);
    if !Path::new(&control).exists() {
        return Ok(());
    }
    for ctrl in controllers {
        let _ = SetValue(parent, SUBTREE_CONTROL, &format!("+{}", ctrl));
    }
    Ok(())
}

/// Create each level of a unified cgroup path, delegating controllers on parents only.
pub fn create_unified_hierarchy(leaf_path: &str) -> Result<()> {
    let rel = leaf_path
        .strip_prefix(CGROUP_ROOT)
        .unwrap_or(leaf_path)
        .trim_start_matches('/');
    if rel.is_empty() {
        return Ok(());
    }

    let parts: Vec<&str> = rel.split('/').filter(|p| !p.is_empty()).collect();
    let mut current = CGROUP_ROOT.to_string();
    for (i, part) in parts.iter().enumerate() {
        let parent = current.clone();
        current = Join(&current, part);
        if Path::new(&current).exists() {
            continue;
        }
        enable_subtree_controllers(&parent, V2_CONTROLLERS)?;
        fs::create_dir(&current).map_err(|e| {
            Error::IOError(format!("create cgroup dir {}: {:?}", current, e))
        })?;
        // Leaf cgroups hold processes; enabling subtree_control on them returns EBUSY on join.
        if i + 1 < parts.len() {
            enable_subtree_controllers(&current, V2_CONTROLLERS)?;
        }
    }
    Ok(())
}

// countCpuset returns the number of CPUs in a string like "0-2,7,12-14".
pub fn count_cpuset(cpuset: &str) -> Result<usize> {
    let mut count: usize = 0;
    for p in cpuset.split(',') {
        let interval: Vec<&str> = p.split('-').collect();
        match interval.len() {
            1 => {
                interval[0]
                    .parse::<usize>()
                    .map_err(|_| Error::Common(format!("invalid cpuset: {}", p)))?;
                count += 1;
            }
            2 => {
                let start = interval[0]
                    .parse::<usize>()
                    .map_err(|_| Error::Common(format!("invalid cpuset: {}", p)))?;
                let end = interval[1]
                    .parse::<usize>()
                    .map_err(|_| Error::Common(format!("invalid cpuset: {}", p)))?;
                if start > end {
                    return Err(Error::Common(format!("invalid cpuset: {}", p)));
                }
                count += end - start + 1;
            }
            _ => return Err(Error::Common(format!("invalid cpuset: {}", p))),
        }
    }
    Ok(count)
}

pub struct CgroupCleanup<'a> {
    pub cgroup: &'a mut Cgroup,
    pub enable: bool,
}

impl<'a> Drop for CgroupCleanup<'a> {
    fn drop(&mut self) {
        if self.enable {
            self.cgroup.Uninstall();
        }
    }
}

/// Host cgroup v2 handle for a sandbox or container (OCI `linux.cgroupsPath`).
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Cgroup {
    pub Name: String,
    /// Legacy field from cgroup v1 era; ignored on v2 paths. Kept for serde compat.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub Parents: BTreeMap<String, String>,
    pub Own: bool,
    /// When `Name` is a delegation parent (CRI pod cgroup), processes join this leaf.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ProcessPath: Option<String>,
    #[serde(default)]
    pub OwnProcess: bool,
}

impl Cgroup {
    pub fn New(spec: &Spec) -> Result<Option<Self>> {
        if spec.linux.is_none() || spec.linux.as_ref().unwrap().cgroups_path.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            Name: spec.linux.as_ref().unwrap().cgroups_path.to_string(),
            Parents: BTreeMap::new(),
            Own: false,
            ProcessPath: None,
            OwnProcess: false,
        }))
    }

    pub fn path(&self) -> String {
        unified_cgroup_path(&self.Name)
    }

    /// Directory where this process should be joined (leaf under delegation parents).
    pub fn process_path(&self) -> String {
        self.ProcessPath
            .as_ref()
            .map(|p| unified_cgroup_path(p))
            .unwrap_or_else(|| self.path())
    }

    fn apply_resources(&self, res: &Option<LinuxResources>, path: &str) -> Result<()> {
        Cpu2 {}.Set(res, path)?;
        CpuSet2 {}.Set(res, path)?;
        Memory2 {}.Set(res, path)?;
        Ok(())
    }

    /// Ensure a leaf cgroup exists when `Name` points at a delegation parent.
    fn ensure_process_cgroup(&mut self, res: &Option<LinuxResources>) -> Result<()> {
        let parent = self.path();
        if !subtree_control_active(&parent)? {
            CpuSet2 {}.Set(res, &parent)?;
            return Ok(());
        }

        let leaf_name = Join(self.Name.trim_start_matches('/'), PROCESS_LEAF);
        let leaf_path = unified_cgroup_path(&leaf_name);
        self.ProcessPath = Some(leaf_name);

        if !Path::new(&leaf_path).exists() {
            create_unified_hierarchy(&leaf_path)?;
            self.OwnProcess = true;
        }
        self.apply_resources(res, &leaf_path)?;
        Ok(())
    }

    pub fn Install(&mut self, res: &Option<LinuxResources>) -> Result<()> {
        require_cgroup_v2()?;
        let path = self.path();
        if Path::new(&Join(&path, "cgroup.procs")).exists() {
            info!("Using pre-created cgroup (v2) {}", &self.Name);
            return self.ensure_process_cgroup(res);
        }

        info!("Creating cgroup (v2) {}", &self.Name);
        self.Own = true;

        if let Err(e) = self.install_owned(res) {
            self.Uninstall();
            return Err(e);
        }
        Ok(())
    }

    fn install_owned(&mut self, res: &Option<LinuxResources>) -> Result<()> {
        create_unified_hierarchy(&self.path())?;
        self.apply_resources(res, &self.path())?;
        self.ensure_process_cgroup(res)
    }

    pub fn Uninstall(&self) {
        if self.OwnProcess {
            if let Some(ref leaf) = self.ProcessPath {
                let path = unified_cgroup_path(leaf);
                info!("Deleting process cgroup (v2) {}", leaf);
                for i in 0..7 {
                    match fs::remove_dir(&path) {
                        Ok(()) => break,
                        Err(e) => {
                            if e.raw_os_error() == Some(SysErr::ENOENT) {
                                break;
                            }
                            error!("can't uninstall ({:?}) failed: {:?}", path, e);
                        }
                    }
                    thread::sleep(time::Duration::from_millis(100 << i));
                }
            }
        }
        if !self.Own {
            return;
        }
        let path = self.path();
        info!("Deleting cgroup (v2) {}", &self.Name);
        for i in 0..7 {
            match fs::remove_dir(&path) {
                Ok(()) => return,
                Err(e) => {
                    if e.raw_os_error() == Some(SysErr::ENOENT) {
                        return;
                    }
                    error!("can't uninstall ({:?}) failed: {:?}", path, e);
                }
            }
            thread::sleep(time::Duration::from_millis(100 << i));
        }
    }

    pub fn Join(&self) -> Result<std::boxed::Box<dyn Fn()>> {
        require_cgroup_v2()?;
        let path = self.process_path();
        let undo_path = self_cgroup_v2_path()?;

        if undo_path == path {
            return Ok(std::boxed::Box::new(|| {}));
        }

        let undo = move || {
            info!("Restoring cgroup {}", &undo_path);
            if let Err(e) = SetValue(&undo_path, "cgroup.procs", "0") {
                info!("Error restoring cgroup {}: {:?}", &undo_path, e);
            }
        };

        let pid = format!("{}", std::process::id());
        info!("Joining cgroup (v2) {}", &path);
        SetValue(&path, "cgroup.procs", &pid)?;

        Ok(std::boxed::Box::new(undo))
    }

    pub fn NumCPU(&self) -> Result<usize> {
        let cpuset = GetValue(&self.process_path(), "cpuset.cpus")?;
        count_cpuset(cpuset.trim())
    }

    pub fn MemoryLimit(&self) -> Result<u64> {
        let lim_str = GetValue(&self.process_path(), "memory.max")?;
        let lim_str = lim_str.trim();
        if lim_str == "max" {
            return Ok(0);
        }
        lim_str
            .parse::<u64>()
            .map_err(|e| Error::IOError(format!("MemoryLimit parse {:?}: {:?}", lim_str, e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_cgroup_path_trims_leading_slash() {
        assert_eq!(
            unified_cgroup_path("/k8s.io/foo"),
            "/sys/fs/cgroup/k8s.io/foo"
        );
        assert_eq!(
            unified_cgroup_path("k8s.io/foo"),
            "/sys/fs/cgroup/k8s.io/foo"
        );
    }

    #[test]
    fn count_cpuset_parses_ranges() {
        assert_eq!(count_cpuset("0-2,7").unwrap(), 4);
        assert_eq!(count_cpuset("1").unwrap(), 1);
        assert!(count_cpuset("3-1").is_err());
    }

    #[test]
    fn require_cgroup_v2_errors_when_unavailable() {
        if cgroup_v2_available() {
            assert!(require_cgroup_v2().is_ok());
        } else {
            assert!(require_cgroup_v2().is_err());
        }
    }

    #[test]
    fn subtree_control_active_reads_delegation_parent() {
        if std::env::consts::OS != "linux" || !cgroup_v2_available() {
            return;
        }
        let k8s = unified_cgroup_path("k8s.io");
        if Path::new(&k8s).exists() {
            let _ = subtree_control_active(&k8s);
        }
    }

    /// Skips when not Linux, cgroup v2 unavailable, or quark log dir not writable.
    #[test]
    fn cgroup_v2_install_join_roundtrip() {
        if std::env::consts::OS != "linux" || !cgroup_v2_available() {
            return;
        }
        if std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(crate::print::LOG_FILE_DEFAULT)
            .is_err()
        {
            return;
        }

        let pid = std::process::id();
        let rel = self_cgroup_v2_path()
            .ok()
            .map(|p| {
                p.strip_prefix(CGROUP_ROOT)
                    .unwrap_or("")
                    .trim_start_matches('/')
                    .to_string()
            })
            .unwrap_or_default();
        let name = if rel.is_empty() {
            format!("quark-cgroup-test-{}", pid)
        } else {
            format!("{}/quark-cgroup-test-{}", rel, pid)
        };

        let mut cg = Cgroup {
            Name: name,
            Parents: BTreeMap::new(),
            Own: false,
            ProcessPath: None,
            OwnProcess: false,
        };

        let mut memory = LinuxMemory::default();
        memory.limit = Some(64 * 1024 * 1024);
        let mut cpu = LinuxCPU::default();
        cpu.shares = Some(512);
        let res = LinuxResources {
            memory: Some(memory),
            cpu: Some(cpu),
            ..Default::default()
        };

        if cg.Install(&Some(res)).is_err() {
            return;
        }

        let path = cg.path();
        let mem_max = fs::read_to_string(format!("{}/memory.max", path)).unwrap_or_default();
        assert!(
            mem_max.trim() == "67108864",
            "unexpected memory.max: {:?}",
            mem_max
        );

        let _restore = match cg.Join() {
            Ok(r) => r,
            Err(_) => {
                cg.Uninstall();
                return;
            }
        };

        let metrics = super::super::stats::MetricsFromCgroup(&cg).expect("stats from cgroup");
        assert!(metrics.get_memory().has_usage());

        cg.Uninstall();
        assert!(!Path::new(&path).exists());
    }
}
