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
// limitations under the License.

use alloc::string::String;
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::path::Path;
use std::{thread, time};

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::path::*;
use super::oci::*;
use super::specutils::specutils::MkdirAll;

pub const CONTROLLERS : [(&str, fn(spec: &LinuxResources, path: &str) -> Result<()>) ; 11] = [
    ("blkio", BlockIO),
    ("cpu", CPU),
    ("cpuset", CpuSet),
    ("memory", Memory),
    ("net_cls", NetworkClass),
    ("net_prio", NetworkPrio),

    // These controllers either don't have anything in the OCI spec or is
    // irrevalant for a sandbox, e.g. pids.
    ("devices", Noop),
    ("freezer", Noop),
    ("perf_event", Noop),
    ("pids", Noop),
    ("systemd", Noop),
];



pub const CGROUP_ROOT : &str = "/sys/fs/cgroup";

pub fn SetOptionalValueInt(path: &str, name: &str, val: Option<i64>) -> Result<()> {
    let val = match val {
        None => return Ok(()),
        Some(v) => {
            if v == 0 {
                return Ok(())
            }
            v
        }
    };

    let str = format!("{}", val);
    return SetValue(path, name, &str);
}

pub fn SetOptionalValueUint(path: &str, name: &str, val: Option<u64>) -> Result<()> {
    let val = match val {
        None => return Ok(()),
        Some(v) => {
            if v == 0 {
                return Ok(())
            }
            v
        }
    };

    let str = format!("{}", val);
    return SetValue(path, name, &str);
}

pub fn SetOptionalValueU32(path: &str, name: &str, val: Option<u32>) -> Result<()> {
    let val = match val {
        None => return Ok(()),
        Some(v) => {
            if v == 0 {
                return Ok(())
            }
            v
        }
    };

    let str = format!("{}", val);
    return SetValue(path, name, &str);
}

pub fn SetOptionalValueU16(path: &str, name: &str, val: Option<u16>) -> Result<()> {
    let val = match val {
        None => return Ok(()),
        Some(v) => {
            if v == 0 {
                return Ok(())
            }
            v
        }
    };

    let str = format!("{}", val);
    return SetValue(path, name, &str);
}

pub fn SetValue(path: &str, name: &str, data: &str) -> Result<()> {
    let fullpath = Join(path, name);

    return WriteFile(&fullpath, data);
}

pub fn WriteFile(path: &str, data: &str) -> Result<()> {
    let mut options = OpenOptions::new();
    let mut file = options.write(true).create(true).truncate(true).open(path).map_err(|e| Error::IOError(format!("WriteFile {:?} io::error is {:?}", path, e)))?;
    file.write_all(data.as_bytes()).map_err(|e| Error::IOError(format!("SetValue {:?} io::error is {:?}", path, e)))?;
    return Ok(())
}

pub fn GetValue(path: &str, name: &str) -> Result<String> {
    let fullpath = Join(path, name);

    let contents = fs::read_to_string(&fullpath)
        .map_err(|e| Error::IOError(format!("GetValue fail when read file {} with error {:?}", &fullpath, e)))?;

    return Ok(contents)
}

// fillFromAncestor sets the value of a cgroup file from the first ancestor
// that has content. It does nothing if the file in 'path' has already been set.
pub fn FillFromAncestor(path: &str) -> Result<String> {
    let out = fs::read_to_string(&path)
        .map_err(|e| Error::IOError(format!("FileFromAncestor fail when read file {} with error {:?}", path, e)))?;

    let val = out.trim();
    if val.len() != 0 {
        // File is set, stop here.
        return Ok(val.to_string())
    }

    // File is not set, recurse to parent and then  set here.
    let name = Base(path);
    let parent = Dir(&Dir(path));

    let val = FillFromAncestor(&Join(&parent, &name))?;
    WriteFile(path, &val)?;

    return Ok(val)
}

pub fn LoadPaths(pid: &str) -> Result<BTreeMap<String, String>> {
    // Open the file in read-only mode (ignoring errors).
    let file = File::open(format!("/proc/{}/cgroup", pid)).map_err(|e| Error::IOError(format!("LoadPath:: io::error is {:?}", e)))?;
    let reader = BufReader::new(file);

    let mut paths = BTreeMap::new();
    // Read the file line by line using the lines() iterator from std::io::BufRead.
    for (_index, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| Error::IOError(format!("LoadPath:: io::error is {:?}", e)))?; // Ignore errors.
        // Show the line and its number.
        let tokens : Vec<&str> = line.split(':').collect();

        if tokens.len() != 3 {
            return Err(Error::Common(format!("invalid cgroups file, line: {}", &line)));
        }

        let tokens1 = tokens[1].to_string();
        let ctrlrs : Vec<&str> = tokens1.split(',').collect();
        for ctrlr in ctrlrs {
            paths.insert(ctrlr.to_string(), tokens[2].to_string());
        }
    }

    return Ok(paths);
}

// countCpuset returns the number of CPU in a string formatted like:
// 		"0-2,7,12-14  # bits 0, 1, 2, 7, 12, 13, and 14 set" - man 7 cpuset
pub fn CountCpuset(cpuset: &str) -> Result<usize> {
    let mut count : usize = 0;

    let arr : Vec<&str> = cpuset.split(',').collect();
    for p in arr {
        let interval : Vec<&str> = p.split('-').collect();
        match interval.len() {
            1 => {
                match interval[0].parse::<usize>() {
                    Ok(_i) => count += 1,
                    Err(_e) => {
                        return Err(Error::Common(format!("invalid cpuset: {}", p)));
                    }
                };
            }
            2 => {
                let start = match interval[0].parse::<usize>() {
                    Ok(i) => i,
                    Err(_e) => {
                        return Err(Error::Common(format!("invalid cpuset: {}", p)));
                    }
                };

                let end = match interval[0].parse::<usize>() {
                    Ok(i) => i,
                    Err(_e) => {
                        return Err(Error::Common(format!("invalid cpuset: {}", p)));
                    }
                };

                if start > end {
                    return Err(Error::Common(format!("invalid cpuset: {}", p)));
                }

                count += end - start + 1;
            }
            _ => {
                return Err(Error::Common(format!("invalid cpuset: {}", p)));
            }
        }
    }

    return Ok(count)
}


pub struct CgroupCleanup <'a> {
    pub cgroup: &'a mut Cgroup,
    pub enable: bool,
}

impl <'a> Drop for CgroupCleanup <'a> {
    fn drop(&mut self) {
        if self.enable {
            self.cgroup.Uninstall();
        }
    }
}

// Cgroup represents a group inside all controllers. For example: Name='/foo/bar'
// maps to /sys/fs/cgroup/<controller>/foo/bar on all controllers.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Cgroup {
    pub Name: String,
    pub Parents: BTreeMap<String, String>,
    pub Own: bool,
}

impl Cgroup {
    pub fn New(spec: &Spec) -> Result<Option<Self>> {
        if spec.linux.is_none() || spec.linux.as_ref().unwrap().cgroups_path.len() == 0 {
            return Ok(None)
        }

        let cgroupsPath = spec.linux.as_ref().unwrap().cgroups_path.to_string();

        let parents = if !IsAbs(&cgroupsPath) {
            LoadPaths("self")?
        } else {
            BTreeMap::new()
        };

        return Ok(Some(Self {
            Name: cgroupsPath,
            Parents: parents,
            Own: false,
        }))
    }

    // Install creates and configures cgroups according to 'res'. If cgroup path
    // already exists, it means that the caller has already provided a
    // pre-configured cgroups, and 'res' is ignored.
    pub fn Install(&mut self, res: &Option<LinuxResources>) -> Result<()> {
        if Path::new(&self.MakePath("memory")).exists() {
            info!("Using pre-created cgroup {}", &self.Name);
            return Ok(())
        }

        info!("Creating cgroup {}", &self.Name);
        self.Own = true;

        let mut cgroupCleanup = CgroupCleanup {
            cgroup: self,
            enable: true,
        };

        for controller in &CONTROLLERS {
            let path = cgroupCleanup.cgroup.MakePath(&controller.0);
            MkdirAll(&path)?;
            match res {
                None => (),
                Some(ref res) => {
                    controller.1(res, &path)?;
                }
            }
        }

        // The Cleanup object cleans up partially created cgroups when an error occurs.
        // Errors occuring during cleanup itself are ignored.
        cgroupCleanup.enable = false;

        return Ok(())
    }

    pub fn Uninstall(&self) {
        if !self.Own {
            return
        }

        info!("Deleting cgroup {}", &self.Name);
        for c in &CONTROLLERS {
            let path = self.MakePath(c.0);
            info!("Removing cgroup controller for key={} path={}", &c.0, &path);

            // If we try to remove the cgroup too soon after killing the
            // sandbox we might get EBUSY, so we retry for a few seconds
            // until it succeeds.
            for i in 0..7 {
                match fs::remove_dir(&path) {
                    Ok(()) => break,
                    Err(e) => if let Some(errno) = e.raw_os_error() {
                        if errno == SysErr::ENOENT {
                            continue
                        }

                        error!("can't uninstall ({:?}) failed: {:?}", path, e);
                        break;
                    }
                }

                //sleep 2^i * 100 ms
                let millies = time::Duration::from_millis(100 << i);
                thread::sleep(millies);
            }
        }
    }

    // Join adds the current process to the all controllers. Returns function that
    // restores cgroup to the original state.
    pub fn Join(&self) -> Result<impl Fn()> {
        let paths = match LoadPaths("self") {
            Ok(p) => p,
            Err(e) => {
                return Err(e)
            }
        };

        let mut undoPaths = Vec::new();
        //'outer:
        for (ctrlr, path) in &paths {
            for c in &CONTROLLERS {
                if ctrlr == c.0 {
                    let fullpath = Join(&Join(CGROUP_ROOT, ctrlr), path);
                    undoPaths.push(fullpath);

                    //break 'outer;
                }
            }
        }

        // Replace empty undo with the real thing before changes are made to cgroups.
        let undo = move || {
            for path in &undoPaths {
                info!("Restoring cgroup {}", &path);
                match SetValue(&path, "cgroup.procs", "0") {
                    Ok(()) => (),
                    Err(e) => info!("Error restoring cgroup {}: {:?}", &path, e),
                }
            }
        };

        // Now join the cgroups.
        for c in &CONTROLLERS {
            let path = self.MakePath(&c.0);
            info!("Joining cgroup {}", &path);

            match SetValue(&path, "cgroup.procs", "0") {
                Ok(()) => (),
                Err(e) => {
                    info!("Error set cgroup {}: {:?}", &path, e);
                    return Err(e);
                },
            }
        }

        return Ok(undo)
    }

    // NumCPU returns the number of CPUs configured in 'cpuset/cpuset.cpus'.
    pub fn NumCPU(&self) -> Result<usize> {
        let path = self.MakePath("cpuset");
        let cpuset = GetValue(&path, "cpuset.cpus")?;
        return CountCpuset(&cpuset)
    }

    // MemoryLimit returns the memory limit.
    pub fn MemoryLimit(&self) -> Result<u64> {
        let path = self.MakePath("memory");
        let limStr = GetValue(&path, "memory.limit_in_bytes")?;
        let limStr = limStr.trim();
        return Ok(limStr.parse::<u64>().expect(&format!("MemoryLimit: can't parse limStr as u64 {}", &limStr)));
    }

    pub fn MakePath(&self, controllerName: &str) -> String {
        let mut path = self.Name.to_string();
        match self.Parents.get(controllerName) {
            None => (),
            Some(parent) => {
                path = Join(parent, &self.Name);
            }
        }

        return Join(CGROUP_ROOT, &Join(controllerName, &path))
    }
}

fn Noop(_spec: &LinuxResources, _path: &str) -> Result<()> {
    return Ok(())
}

fn Memory(spec: &LinuxResources, path: &str) -> Result<()> {
    match spec.memory {
        None => return Ok(()),
        Some(ref m) => {
            SetOptionalValueInt(path, "memory.limit_in_bytes", m.limit)?;
            SetOptionalValueInt(path, "memory.soft_limit_in_bytes", m.reservation)?;
            SetOptionalValueInt(path, "memory.memsw.limit_in_bytes", m.swap)?;
            SetOptionalValueInt(path, "memory.kmem.limit_in_bytes", m.kernel)?;
            SetOptionalValueInt(path, "memory.kmem.tcp.limit_in_bytes", m.kernel_tcp)?;
            SetOptionalValueUint(path, "memory.swappiness", m.swappiness)?;

            if m.disableOOMKiller.is_some() && *m.disableOOMKiller.as_ref().unwrap() {
                SetValue(path, "memory.oom_control", "1")?;
            }

            return Ok(())
        }
    }
}

fn CPU(spec: &LinuxResources, path: &str) -> Result<()> {
    match spec.cpu {
        None => return Ok(()),
        Some(ref c) => {
            SetOptionalValueUint(path, "cpu.shares", c.shares)?;
            SetOptionalValueInt(path, "cpu.cfs_quota_us", c.quota)?;
            SetOptionalValueUint(path, "cpu.cfs_period_us", c.period)?;

            return Ok(())
        }
    }
}

fn CpuSet(spec: &LinuxResources, path: &str) -> Result<()> {
    if spec.cpu.is_none() || spec.cpu.as_ref().unwrap().cpus.len() == 0 {
        FillFromAncestor(&Join(path, "cpuset.cpus"))?;
    } else {
        SetValue(path, "cpuset.cpus", &spec.cpu.as_ref().unwrap().cpus)?;
    }

    if spec.cpu.is_none() || spec.cpu.as_ref().unwrap().mems.len() == 0 {
        FillFromAncestor(&Join(path, "cpuset.mems"))?;
    } else {
        SetValue(path, "cpuset.mems", &spec.cpu.as_ref().unwrap().mems)?;
    }

    return Ok(())
}

fn BlockIO(spec: &LinuxResources, path: &str) -> Result<()> {
    match spec.block_io {
        None => return Ok(()),
        Some(ref b) => {
            SetOptionalValueU16(path, "blkio.weight", b.weight)?;
            SetOptionalValueU16(path, "blkio.leaf_weight", b.leaf_weight)?;

            for dev in &b.weight_device {
                let val = format!("{}:{} {}", dev.major, dev.minor, dev.weight.expect("expect weight is not none"));
                SetValue(path, "blkio.weight_device", &val)?;

                let val = format!("{}:{} {}", dev.major, dev.minor, dev.leaf_weight.expect("expect leaf_weight is not none"));
                SetValue(path, "blkio.leaf_weight_device", &val)?;
            }

            SetThrottle(path, "blkio.throttle.read_bps_device", &b.throttle_read_bps_device)?;
            SetThrottle(path, "blkio.throttle.write_bps_device", &b.throttle_write_bps_device)?;
            SetThrottle(path, "blkio.throttle.read_iops_device", &b.throttle_read_iops_device)?;
            SetThrottle(path, "blkio.throttle.write_iops_device", &b.throttle_write_iops_device)?;

            return Ok(())
        }
    }
}

pub fn SetThrottle(path: &str, name: &str, devs: &[LinuxThrottleDevice]) -> Result<()> {
    for dev in devs {
        let val = format!("{}:{} {}", dev.major, dev.minor, dev.rate);
        SetValue(path, name, &val)?;
    }

    return Ok(())
}

fn NetworkClass(spec: &LinuxResources, path: &str) -> Result<()> {
    match spec.network {
        None => return Ok(()),
        Some(ref n) => {
            SetOptionalValueU32(path, "net_cls.classid", n.class_id)?;

            return Ok(())
        }
    }
}

fn NetworkPrio(spec: &LinuxResources, path: &str) -> Result<()> {
    match spec.network {
        None => return Ok(()),
        Some(ref n) => {
            for prio in &n.priorities {
                let val = format!("{} {}", prio.name, prio.priority);
                SetValue(path, "net_prio.ifpriomap", &val)?;
            }
        }
    }

    return Ok(())
}