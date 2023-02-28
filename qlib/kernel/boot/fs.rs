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

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;

use super::super::super::auth::*;
use super::super::super::common::*;
use super::super::super::linux_def::{FileMode, FilePermissions, SysErr};
use super::super::super::path::*;
use super::super::fs::dirent::*;
use super::super::fs::filesystems::*;
use super::super::fs::host::fs::*;
use super::super::fs::host::util::*;
use super::super::fs::inode::*;
use super::super::fs::mount::*;
use super::super::fs::overlay::*;
use super::super::fs::ramfs::tree::*;
use super::super::task::*;

use super::*;

// ChildContainersDir is the directory where child container root
// filesystems are mounted.
const CHILD_CONTAINERS_DIR: &str = "/__runsc_containers__";

// Filesystems that runsc supports.
const DEVPTS: &str = "devpts";
const DEVTMPFS: &str = "devtmpfs";
const PROCFS: &str = "proc";
const SYSFS: &str = "sysfs";
const TMPFS: &str = "tmpfs";
const NONEFS: &str = "none";

fn CreateRootMount(
    task: &Task,
    spec: &oci::Spec,
    config: &config::Config,
    mounts: &Vec<oci::Mount>,
) -> Result<Inode> {
    let mf = MountSourceFlags {
        ReadOnly: spec.root.readonly,
        ..Default::default()
    };

    let rootStr = &config.RootDir;
    let (fd, writeable, fstat) = TryOpenAt(-100, rootStr)?;

    let ms = MountSource::NewHostMountSource(
        &rootStr,
        &ROOT_OWNER,
        &WhitelistFileSystem::New(),
        &mf,
        false,
    );
    let hostRoot = Inode::NewHostInode(
        task,
        &Arc::new(QMutex::new(ms)),
        fd,
        &fstat,
        writeable,
        false,
    )?;

    let submounts = SubTargets(&"/".to_string(), mounts);
    //submounts.append(&mut vec!["/dev1".to_string(), "/sys".to_string(), "/proc".to_string(), "/tmp".to_string()]);

    let rootInode = AddSubmountOverlay(task, &hostRoot, &submounts)?;

    return Ok(rootInode);
}

pub fn AddSubmountOverlay(task: &Task, inode: &Inode, submounts: &Vec<String>) -> Result<Inode> {
    let msrc = Arc::new(QMutex::new(MountSource::NewPseudoMountSource()));
    let mountTree = MakeDirectoryTree(task, &msrc, submounts)?;

    let overlayInode = NewOverlayRoot(task, inode, &mountTree, &MountSourceFlags::default())?;
    return Ok(overlayInode);
}

fn SubTargets(root: &str, mnts: &Vec<oci::Mount>) -> Vec<String> {
    let mut targets = Vec::new();

    for mnt in mnts {
        let (relPath, isSubpath) = IsSubpath(&mnt.destination, root);
        if isSubpath {
            targets.push(relPath)
        }
    }

    return targets;
}

fn GetMountNameAndOptions(_conf: &config::Config, m: &oci::Mount) -> Result<(String, Vec<String>)> {
    let fsName;
    let mut opts = Vec::new();

    match m.typ.as_str() {
        DEVPTS | DEVTMPFS | PROCFS | SYSFS => {
            fsName = m.typ.to_string();
        }
        NONEFS => {
            fsName = SYSFS.to_string();
        }
        TMPFS => {
            fsName = m.typ.to_string();
            opts = ParseAndFilterOptions(&m.options, &vec!["mode", "uid", "gid"])?;
        }
        _ => {
            info!("ignoring unknown filesystem type {}", m.typ);
            return Err(Error::Common(format!(
                "ignoring unknown filesystem type {}",
                m.typ
            )));
        }
    }

    return Ok((fsName, opts));
}

pub fn InitTestSpec() -> oci::Spec {
    return oci::Spec {
        version: "".to_string(),
        platform: None,
        //process:...,
        root: oci::Root {
            path: "".to_string(),
            readonly: false,
        },
        hostname: "".to_string(),
        mounts: vec![],
        hooks: None,
        annotations: BTreeMap::new(),
        linux: None,
        solaris: None,
        windows: None,
    };
}

pub fn InitRootFs(task: &mut Task, root: &str) -> Result<MountNs> {
    let config = config::Config {
        RootDir: root.to_string(),
        Debug: true,
    };

    debug!("init rootfs under {} for container", root);

    return SetupContainerFS(task, &InitTestSpec(), &config);
}

// This function will be used by both root container and subcontainer
pub fn SetupContainerFS(
    task: &mut Task,
    spec: &oci::Spec,
    conf: &config::Config,
) -> Result<MountNs> {
    let mounts = CompileMounts(spec);

    //error!("SetupRootContainerFS 1.0 mounts[0].destination is {:?}", &mounts[0].destination);

    let rootInode = CreateRootMount(task, spec, conf, &mounts)?;
    let mns = MountNs::New(task, &rootInode);

    let root = mns.Root();

    MountSubmounts(task, conf, &mns, &root, &mounts)?;
    return Ok(mns);
}

fn CompileMounts(spec: &oci::Spec) -> Vec<oci::Mount> {
    let mut _procMounted = false;
    let mut _sysMounted = false;
    let mut mounts = Vec::new();

    mounts.push(oci::Mount {
        //destination: "/dev1".to_string(),
        destination: "/dev".to_string(),
        typ: DEVTMPFS.to_string(),
        source: "".to_string(),
        options: Vec::new(),
    });

    mounts.push(oci::Mount {
        destination: "/dev/pts".to_string(),
        typ: DEVPTS.to_string(),
        source: "".to_string(),
        options: Vec::new(),
    });

    mounts.push(oci::Mount {
        destination: "/proc".to_string(),
        typ: PROCFS.to_string(),
        source: "".to_string(),
        options: Vec::new(),
    });

    mounts.push(oci::Mount {
        destination: "/sys".to_string(),
        typ: SYSFS.to_string(),
        source: "".to_string(),
        options: Vec::new(),
    });

    /*mounts.push(oci::Mount {
        destination: "/tmp".to_string(),
        typ: TMPFS.to_string(),
        source: "".to_string(),
        options: Vec::new(),
    });*/

    for m in &spec.mounts {
        if !specutils::IsSupportedDevMount(m) {
            info!("ignoring dev mount at {}", m.destination);
            continue;
        }

        mounts.push(m.clone());
        match Clean(&m.destination).as_str() {
            "/proc" => _procMounted = true,
            "/sys" => _sysMounted = true,
            _ => (),
        }
    }

    let mut mandatoryMounts = Vec::new();
    /*if !procMounted {
        mandatoryMounts.push(oci::Mount {
            destination: "/proc".to_string(),
            typ: PROCFS.to_string(),
            source: "".to_string(),
            options: Vec::new(),
        })
    }

    if !sysMounted {
        mandatoryMounts.push(oci::Mount {
            destination: "/sys".to_string(),
            typ: SYSFS.to_string(),
            source: "".to_string(),
            options: Vec::new(),
        })
    }*/

    mandatoryMounts.append(&mut mounts);

    return mandatoryMounts;
}

fn MountSubmounts(
    task: &Task,
    config: &config::Config,
    mns: &MountNs,
    root: &Dirent,
    mounts: &Vec<oci::Mount>,
) -> Result<()> {
    for m in mounts {
        debug!("mounting submounts {:?}", m);
        MountSubmount(task, config, mns, root, m, mounts)?;
    }

    //todo: mount tmp
    return Ok(());
}

/// This function ensures the mount point for submount to mount exist
fn MakeMountPoint(task: &Task, mns: &MountNs, root: &Dirent, path: &str) -> Result<()> {
    let rootInode = root.Inode();
    if !rootInode.StableAttr().IsDir() {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    let mut remainingTraversals = 0;
    let res = mns.FindDirent(
        task,
        root,
        Some(root.clone()),
        path,
        &mut remainingTraversals,
        true,
    );

    match res {
        Ok(_) => {
            // The mount point exists already, we are done
            return Ok(());
        }
        _ => {
            let perms = FilePermissions::FromMode(FileMode(
                u16::from_str_radix("011101110111", 2).unwrap(),
            ));

            // remove leading / as CreateDirectory only takes relative path
            let path = String::from(path);
            let trimmedPath = path.trim_start_matches('/');
            root.CreateDirectory(task, root, trimmedPath, &perms)?;
        }
    }
    return Ok(());
}

fn MountSubmount(
    task: &Task,
    config: &config::Config,
    mns: &MountNs,
    root: &Dirent,
    m: &oci::Mount,
    mounts: &Vec<oci::Mount>,
) -> Result<()> {
    let (fsName, opts) = GetMountNameAndOptions(config, m)?;

    if fsName.as_str() == "" {
        return Ok(());
    }

    let filesystem = MustFindFilesystem(&fsName);
    let mf = mountFlags(&m.options);

    let mut inode = filesystem
        .lock()
        .Mount(task, &"none".to_string(), &mf, &opts.join(","))?;
    let submounts = SubTargets(&m.destination, mounts);
    if submounts.len() > 0 {
        info!("adding submount overlay over {}", m.destination);
        inode = AddSubmountOverlay(task, &inode, &submounts)?;
    }
    MakeMountPoint(task, mns, root, &m.destination)?;
    let mut maxTraversals = 0;
    let dirent = mns.FindDirent(
        task,
        root,
        Some(root.clone()),
        &m.destination,
        &mut maxTraversals,
        true,
    )?;
    mns.Mount(&dirent, &inode)?;

    info!("Mounted {} to {} type {}", m.source, m.destination, m.typ);
    return Ok(());
}

fn mountFlags(opts: &Vec<String>) -> MountSourceFlags {
    let mut mf = MountSourceFlags::default();

    for o in opts {
        match o.as_str() {
            "rw" => mf.ReadOnly = false,
            "ro" => mf.ReadOnly = true,
            "noatime" => mf.NoAtime = true,
            "noexec" => mf.NoExec = true,
            _ => info!("ignoring unknown mount option {}", o),
        }
    }

    return mf;
}

fn MustFindFilesystem(name: &str) -> Arc<QMutex<Filesystem>> {
    return FindFilesystem(name).expect(format!("could not find filesystem {}", name).as_str());
}

fn ParseAndFilterOptions(opts: &Vec<String>, allowedKeys: &Vec<&str>) -> Result<Vec<String>> {
    let mut res = Vec::new();

    for o in opts {
        let kv: Vec<&str> = o.split('=').collect();

        match kv.len() {
            1 => {
                if specutils::ContainsStr(allowedKeys, o) {
                    res.push(o.to_string());
                    continue;
                }

                info!("ignoring unsupported key {}", o)
            }
            2 => {
                if specutils::ContainsStr(allowedKeys, &kv[0]) {
                    res.push(o.to_string());
                    continue;
                }
                info!("ignoring unsupported key {}", kv[0])
            }
            _ => return Err(Error::Common(format!("invalid option {}", o))),
        }
    }

    return Ok(res);
}
