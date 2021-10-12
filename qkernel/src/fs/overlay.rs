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
use alloc::sync::Arc;
use ::qlib::mutex::*;
use spin::*;

use super::super::qlib::linux_def::*;
use super::super::qlib::common::*;
use super::super::qlib::path::*;
use super::super::task::*;
use super::inode::*;
use super::mount::*;
use super::mount_overlay::*;
use super::filesystems::*;
use super::attr::*;

pub const XATTR_OVERLAY_PREFIX: &str = "trusted.overlay.";
pub const XATTR_OVERLAY_WHITEOUT_PREFIX: &str = "trusted.overlay.whiteout.";

pub fn XattrOverlayWhiteout(name: &str) -> String {
    return XATTR_OVERLAY_PREFIX.to_string() + name;
}

pub fn IsXattrOverlay(name: &str) -> bool {
    return HasPrefix(name, &XATTR_OVERLAY_PREFIX.to_string());
}

pub fn NewOverlayRoot(task: &Task, upper: &Inode, lower: &Inode, flags: &MountSourceFlags) -> Result<Inode> {
    if !upper.StableAttr().IsDir() {
        return Err(Error::Common(format!("upper Inode is a {:?}, not a directory", upper.StableAttr().Type)))
    }

    if !lower.StableAttr().IsDir() {
        return Err(Error::Common(format!("lower Inode is a {:?}, not a directory", upper.StableAttr().Type)))
    }

    if upper.lock().Overlay.is_some() {
        return Err(Error::Common(format!("cannot nest overlay in upper file of another overlay")));
    }

    let msrc = NewOverlayMountSource(&upper.lock().MountSource.clone(), &lower.lock().MountSource.clone(), flags);
    let overlay = OverlayEntry::New(task, Some(upper.clone()), Some(lower.clone()), true)?;
    return Ok(NewOverlayInode(task, overlay, &msrc))
}

pub fn NewOverlayInode(_task: &Task, o: OverlayEntry, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let inode = o.Inode();
    let inodeOptions = inode.lock().InodeOp.clone(); //useless, just avoid to have empty inodeoperations
    let inodeInternal = InodeIntern {
        InodeOp: inodeOptions,
        StableAttr: inode.StableAttr(),
        LockCtx: LockCtx::default(),
        MountSource: msrc.clone(),
        Overlay: Some(Arc::new(RwLock::new(o))),
        ..Default::default()
    };

    return Inode(Arc::new(QMutex::new(inodeInternal)))
}

pub fn overlayUpperMountSource(overlayMountSource: &Arc<QMutex<MountSource>>) -> Arc<QMutex<MountSource>> {
    let currentOps = overlayMountSource.lock().MountSourceOperations.clone();
    let upper = currentOps.lock().as_any().downcast_ref::<OverlayMountSourceOperations>().expect("OverlayMountSourceOperations convert fail").upper.clone();
    return upper;
}

pub struct OverlayEntry {
    pub LowerExists: bool,
    pub lower: Option<Inode>,
    pub upper: Option<Inode>,
}

impl OverlayEntry {
    pub fn New(_task: &Task, upper: Option<Inode>, lower: Option<Inode>, lowerExists: bool) -> Result<Self> {
        match &lower {
            Some(ref l) => {
                let t = l.StableAttr().Type;
                if !(t == InodeType::RegularFile || t == InodeType::Directory || t == InodeType::Symlink || t == InodeType::Socket) {
                    info!("{:?} not supported in lower filesystem", t);
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }
            None => ()
        }

        return Ok(Self {
            LowerExists: lowerExists,
            lower: lower,
            upper: upper,
        })
    }

    pub fn Inode(&self) -> Inode {
        match &self.upper {
            Some(ref u) => return u.clone(),
            None => (),
        }

        match &self.lower {
            Some(ref l) => return l.clone(),
            _ => {
                panic!("OverlayEntry: both upper and lower are None")
            }
        }
    }
}