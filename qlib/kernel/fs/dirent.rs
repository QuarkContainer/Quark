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
use alloc::sync::Arc;
use alloc::sync::Weak;
use spin::*;
use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::ops::Deref;
use core::cmp::Eq;
use core::cmp::PartialEq;

use super::super::uid::*;
use super::super::super::common::*;
use super::super::super::singleton::*;
use super::super::task::*;
use super::super::super::linux_def::*;
use super::super::socket::unix::transport::unix::*;
use super::inode::*;
use super::flags::*;
use super::file::*;
use super::dentry::*;
use super::mount::*;

pub static RENAME : Singleton<RwLock<()>> = Singleton::<RwLock<()>>::New();
pub unsafe fn InitSingleton() {
    RENAME.Init(RwLock::new(()));
}

#[derive(Clone)]
pub struct Dirent(pub Arc<(QMutex<InterDirent>, u64)>);

impl Default for Dirent {
    fn default() -> Self {
        return Self(Arc::new((QMutex::new(InterDirent::default()), NewUID())))
    }
}

impl Deref for Dirent {
    type Target = Arc<(QMutex<InterDirent>, u64)>;

    fn deref(&self) -> &Arc<(QMutex<InterDirent>, u64)> {
        &self.0
    }
}

impl PartialEq for Dirent {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Dirent {}

impl Drop for Dirent {
    fn drop(&mut self) {
        if Arc::strong_count(&self.0) == 1 {
            let parent = (self.0).0.lock().Parent.take();
            match parent {
                None => (),
                Some(parent) => {
                    let name = (self.0).0.lock().Name.clone();
                    parent.RemoveChild(&name);
                }
            }
        }
    }
}

impl Dirent {
    pub fn New(inode: &Inode, name: &str) -> Self {
        return Self(Arc::new((QMutex::new(InterDirent {
            Inode: inode.clone(),
            Name: name.to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            frozen: false,
            mounted: false,
        }), NewUID())))
    }

    pub fn NewTransient(inode: &Inode) -> Self {
        let iDirent = QMutex::new(InterDirent {
            Inode: inode.clone(),
            Name: "transient".to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            frozen: false,
            mounted: false,
        });
        return Self(Arc::new((iDirent, NewUID())))
    }

    pub fn ID(&self) -> u64 {
        return self.1
    }

    pub fn IsRoot(&self) -> bool {
        match &(self.0).0.lock().Parent {
            None => return true,
            _ => return false,
        }
    }

    pub fn Parent(&self) -> Option<Dirent> {
        match &(self.0).0.lock().Parent {
            None => None,
            Some(ref d) => Some(d.clone())
        }
    }

    pub fn Inode(&self) -> Inode {
        return (self.0).0.lock().Inode.clone();
    }

    pub fn MyFullName(&self) -> String {
        let _a = RENAME.read();

        return self.myFullName();
    }

    fn myFullName(&self) -> String {
        let name = (self.0).0.lock().Name.clone();
        let parent = match &(self.0).0.lock().Parent {
            None => {
                return name
            },
            Some(ref p) => p.clone(),
        };

        let pName = parent.myFullName();

        if pName == "/".to_string() {
            return "/".to_string() + &(self.0).0.lock().Name;
        }

        return pName + &"/".to_string() + &(self.0).0.lock().Name;
    }

    pub fn ChildDenAttrs(&self, task: &Task) -> Result<BTreeMap<String, DentAttr>> {
        let dirName = self.MyFullName();

        let inode = self.Inode();
        let dir = match inode.GetFile(task, &self, &FileFlags {
            Read: true,
            ..Default::default()
        }) {
            Err(err) => {
                info!("failed to open directory {}", &dirName);
                return Err(err)
            }
            Ok(dir) => dir,
        };


        let mut serializer = CollectEntriesSerilizer::New();
        dir.ReadDir(task, &mut serializer)?;

        serializer.Entries.remove(".");
        serializer.Entries.remove("..");
        return Ok(serializer.Entries);
    }

    pub fn FullName(&self, root: &Dirent) -> (String, bool) {
        let _a = RENAME.read();

        return self.fullName(root)
    }

    //ret: (fulname, whether root node is reachable from d)
    fn fullName(&self, root: &Dirent) -> (String, bool) {
        if self == root {
            return ("/".to_string(), true)
        }

        if (self.0).0.lock().IsRoot() {
            return ((self.0).0.lock().Name.clone(), false)
        }

        let parent = (self.0).0.lock().Parent.as_ref().unwrap().clone();
        let (pName, reachable) = parent.fullName(root);

        if pName == "/".to_string() {
            return (pName + &(self.0).0.lock().Name, reachable);
        }

        let ret = pName + &"/".to_string() + &(self.0).0.lock().Name;
        return (ret, reachable)
    }

    pub fn MountRoot(&self) -> Self {
        let _a = RENAME.write();

        let mut mountRoot = self.clone();
        loop {
            if (mountRoot.0).0.lock().mounted {
                return mountRoot
            }

            let parent = (mountRoot.0).0.lock().Parent.clone();
            match parent {
                None => return mountRoot,
                Some(p) => mountRoot = p
            }
        }
    }

    pub fn Freeze(&self) {
        {
            let mut d = (self.0).0.lock();

            if d.frozen {
                return;
            }

            d.frozen = true;

            //add extra refence to avoid free
            let msrc = d.Inode.lock().MountSource.clone();
            msrc.lock().Froze(self);

            //self.Froze();

            for (_, w) in &d.Children {
                match w.upgrade() {
                    None => {
                        //println!("in freeze: weak pointer");
                    }
                    Some(d) => {
                        Dirent(d).Freeze()
                    }
                }
            }
        }

        self.flush();
    }

    pub fn DescendantOf(&self, p: &Dirent) -> bool {
        let mut d = self.clone();
        let p = p.clone();

        loop {
            if d == p {
                return true;
            }

            let parent = match &(d.0).0.lock().Parent {
                None => return false,
                Some(ref parent) => parent.clone()
            };

            d = parent;
        }
    }

    fn getChild(&self, name: &str) -> Option<Dirent> {
        let remove = match (self.0).0.lock().Children.get(name) {
            Some(subD) => {
                match subD.upgrade() {
                    Some(cd) => {
                        return Some(Dirent(cd.clone()))
                    }
                    None => {
                        true
                    }
                }
            }

            None => {
                false
            }
        };

        if remove {
            (self.0).0.lock().Children.remove(name);
        }

        return None
    }

    fn GetCacheChild(&self, name: &str) -> Option<Arc<(QMutex<InterDirent>, u64)>> {
        match (self.0).0.lock().Children.get(name) {
            Some(subD) => {
                match subD.upgrade() {
                    Some(cd) => {
                        return Some(cd.clone())
                    }
                    None => ()
                }
            }

            None => ()
        };

        return None
    }

    fn walk(&self, task: &Task, root: &Dirent, name: &str) -> Result<Dirent> {
        let inode = self.Inode();
        if !inode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::ENOTDIR))
        }

        if name == "" || name == "." {
            return Ok(self.clone())
        } else if name == ".." {
            if self.clone() == root.clone() {
                return Ok(self.clone());
            }

            match &(self.0).0.lock().Parent {
                None => return Ok(self.clone()),
                Some(ref p) => return Ok(p.clone()),
            }
        }

        let child = self.GetCacheChild(name);
        let remove = match child {
            Some(cd) => {
                //error!("walk 1");
                let subInode = cd.0.lock().Inode.clone();
                let mountSource = subInode.lock().MountSource.clone();
                let mountsourceOpations = mountSource.lock().MountSourceOperations.clone();
                let mounted = cd.0.lock().mounted;
                //error!("walk 2");
                let revalidate = mountsourceOpations.lock().Revalidate(name, &inode, &subInode);
                //error!("walk 3");
                if mounted || !revalidate {
                    return Ok(Dirent(cd.clone()))
                }

                false
            }
            None => {
                true
            }
        };

        if remove {
            (self.0).0.lock().Children.remove(name);
        }

        let frozen = (self.0).0.lock().frozen;
        if frozen && !inode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let c = inode.Lookup(task, name)?;

        assert!(&(c.0).0.lock().Name == name, "lookup get mismatch name");

        self.AddChild(&c);

        return Ok(c)
    }

    pub fn Walk(&self, task: &Task, root: &Dirent, name: &str) -> Result<Dirent> {
        //error!("Walk 1 {}", name);
        //defer!(error!("Walk 2 {}", name));
        let _a = RENAME.read();

        return self.walk(task, root, name)
    }

    pub fn RemoveChild(&self, name: &String) {
        (self.0).0.lock().Children.remove(name);
    }

    pub fn AddChild(&self, child: &Arc<(QMutex<InterDirent>, u64)>) -> Option<Weak<(QMutex<InterDirent>, u64)>> {
        assert!(child.0.lock().IsRoot(), "Add child request the child has no parent");
        child.0.lock().Parent = Some(self.clone());
        child.0.lock().frozen = (self.0).0.lock().frozen;

        return self.addChild(child)
    }

    pub fn addChild(&self, child: &Arc<(QMutex<InterDirent>, u64)>) -> Option<Weak<(QMutex<InterDirent>, u64)>> {
        assert!(Dirent(child.clone()).Parent().unwrap() == self.clone(), "Dirent addChild assumes the child already belongs to the parent");

        let name = child.0.lock().Name.clone();
        //println!("addChild the name is {}", name);
        return (self.0).0.lock().Children.insert(name, Arc::downgrade(child))
    }

    fn exists(&self, task: &Task, root: &Dirent, name: &str) -> bool {
        match self.walk(task, root, name) {
            Err(_) => false,
            _ => true,
        }
    }

    pub fn Create(&self, task: &Task, root: &Dirent, name: &str, flags: &FileFlags, perms: &FilePermissions) -> Result<File> {
        let _a = RENAME.write();

        if self.exists(task, root, name) {
            return Err(Error::SysError(SysErr::EEXIST))
        }

        let frozen = (self.0).0.lock().frozen;

        let mut inode = self.Inode();
        if frozen && inode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let file = inode.Create(task, self, name, flags, perms)?;

        let child = file.Dirent.clone();

        self.AddChild(&child);
        child.ExtendReference();

        return Ok(file)
    }

    fn genericCreate(&self, task: &Task, root: &Dirent, name: &str, create: &mut FnMut() -> Result<()>) -> Result<()> {
        let _a = RENAME.write();

        if self.exists(task, root, name) {
            return Err(Error::SysError(SysErr::EEXIST))
        }

        let fronzon = (self.0).0.lock().frozen;

        let inode = self.Inode();
        if fronzon && inode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let remove = match (self.0).0.lock().Children.get(name) {
            Some(_) => {
                true
            }

            None => {
                false
            }
        };

        if remove {
            (self.0).0.lock().Children.remove(name);
        }

        return create()
    }

    pub fn CreateLink(&self, task: &Task, root: &Dirent, oldname: &str, newname: &str) -> Result<()> {
        return self.genericCreate(task, root, newname, &mut || -> Result<()> {
            let mut inode = self.Inode();
            return inode.CreateLink(task, self, oldname, newname)
        });
    }

    pub fn CreateHardLink(&self, task: &Task, root: &Dirent, target: &Dirent, name: &str) -> Result<()> {
        let mut inode = self.Inode();
        let targetInode = target.Inode();
        if !Arc::ptr_eq(&inode.lock().MountSource, &targetInode.lock().MountSource) {
            return Err(Error::SysError(SysErr::EXDEV))
        }

        if targetInode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::EPERM))
        }

        return self.genericCreate(task, root, name, &mut || -> Result<()> {
            return inode.CreateHardLink(task, self, &target, name)
        });
    }

    pub fn CreateDirectory(&self, task: &Task, root: &Dirent, name: &str, perms: &FilePermissions) -> Result<()> {
        return self.genericCreate(task, root, name, &mut || -> Result<()> {
            let mut inode = self.Inode();
            let ret = inode.CreateDirectory(task, self, name, perms);
            return ret;
        });
    }

    pub fn Bind(&self, task: &Task, root: &Dirent, name: &str, data: &BoundEndpoint, perms: &FilePermissions) -> Result<Dirent> {
        let result = self.genericCreate(task, root, name, &mut || -> Result<()> {
            let inode = self.Inode();
            let childDir = inode.Bind(task, name, data, perms)?;
            self.AddChild(&childDir);
            childDir.ExtendReference();
            return Ok(())
        });

        match result {
            Err(Error::SysError(SysErr::EEXIST)) => return Err(Error::SysError(SysErr::EADDRINUSE)),
            Err(e) => return Err(e),
            _ => (),
        };

        let inode = self.Inode();
        let childDir = inode.Lookup(task, name)?;

        return Ok(childDir)
    }

    pub fn CreateFifo(&self, task: &Task, root: &Dirent, name: &str, perms: &FilePermissions) -> Result<()> {
        return self.genericCreate(task, root, name, &mut || -> Result<()> {
            let mut inode = self.Inode();
            return inode.CreateFifo(task, self, name, perms)
        });
    }

    pub fn GetDotAttrs(&self, root: &Dirent) -> (DentAttr, DentAttr) {
        let inode = self.Inode();
        let dot = inode.StableAttr().DentAttr();
        if !self.IsRoot() && self.DescendantOf(root) {
            let parent = self.Parent();
            let pInode = parent.unwrap().Inode();
            let dotdot = pInode.StableAttr().DentAttr();
            return (dot, dotdot)
        }

        return (dot, dot)
    }

    fn readdirFrozen(&self, task: &Task, root: &Dirent, offset: i64, dirCtx: &mut DirCtx) -> (i64, Result<i64>) {
        let mut map = BTreeMap::new();

        let (dot, dotdot) = self.GetDotAttrs(root);

        map.insert(".".to_string(), dot);
        map.insert("..".to_string(), dotdot);

        for (name, d) in &(self.0).0.lock().Children {
            match d.upgrade() {
                Some(subd) => {
                    let inode = subd.0.lock().Inode.clone();
                    let dentAttr = inode.StableAttr().DentAttr();
                    map.insert(name.clone(), dentAttr);
                }
                None => ()
            }
        }

        let mut i = 0;
        for (name, dent) in &map {
            if i >= offset {
                match dirCtx.DirEmit(task, name, dent) {
                    Err(e) => return (i, Err(e)),
                    Ok(()) => ()
                }
            }

            i += 1;
        }

        return (i, Ok(0));
    }

    pub fn flush(&self) {
        let mut expired = Vec::new();
        let mut d = (self.0).0.lock();

        for (n, w) in &d.Children {
            match w.upgrade() {
                None => expired.push(n.clone()),
                Some(cd) => {
                    let dirent = Dirent(cd.clone());
                    dirent.flush();
                    dirent.DropExtendedReference();
                }
            }
        }

        for n in &expired {
            d.Children.remove(n);
        }
    }

    pub fn IsMountPoint(&self) -> bool {
        let d = (self.0).0.lock();

        return d.mounted || d.IsRoot();
    }

    pub fn Mount(&self, inode: &Inode) -> Result<Dirent> {
        if inode.lock().StableAttr().IsSymlink() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let parent = self.Parent().unwrap();
        let parentInode = parent.Inode();
        if (parent.0).0.lock().frozen && !parentInode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let replacement = Arc::new((QMutex::new(InterDirent::New(inode.clone(), &(self.0).0.lock().Name)), NewUID()));
        replacement.0.lock().mounted = true;

        parent.AddChild(&replacement);

        return Ok(Dirent(replacement))
    }

    pub fn UnMount(&self, replace: &Dirent) -> Result<()> {
        let parent = self.Parent().expect("unmount required the parent is not none");
        let parentInode = parent.Inode();
        if (parent.0).0.lock().frozen && !parentInode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let old = parent.addChild(&replace.0);

        match old {
            None => panic!("mount must mount over an existing dirent"),
            Some(_) => (),
        }

        (self.0).0.lock().mounted = false;

        return Ok(())
    }

    pub fn Remove(&self, task: &Task, root: &Dirent, name: &str, dirPath: bool) -> Result<()> {
        let mut inode = self.Inode();

        if (self.0).0.lock().frozen && !inode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        let child = self.Walk(task, root, name)?;
        let childInode = child.Inode();
        if childInode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::EISDIR))
        } else if dirPath {
            return Err(Error::SysError(SysErr::EISDIR))
        }

        if child.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        inode.Remove(task, self, &child)?;

        (self.0).0.lock().Children.remove(name);
        child.DropExtendedReference();

        return Ok(())
    }

    pub fn RemoveDirectory(&self, task: &Task, root: &Dirent, name: &str) -> Result<()> {
        let mut inode = self.Inode();
        if (self.0).0.lock().frozen && !inode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        if name == "." {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if name == ".." {
            return Err(Error::SysError(SysErr::ENOTEMPTY))
        }

        let child = self.Walk(task, root, name)?;
        let childInode = child.Inode();

        if !childInode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::ENOTDIR))
        }

        if child.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        inode.Remove(task, self, &child)?;

        (self.0).0.lock().Children.remove(name);

        child.DropExtendedReference();

        return Ok(())
    }

    pub fn Rename(task: &Task, root: &Dirent, oldParent: &Dirent, oldName: &str, newParent: &Dirent, newName: &str) -> Result<()> {
        let _a = RENAME.write();

        if Arc::ptr_eq(oldParent, newParent) {
            if oldName == newName {
                return Ok(())
            }

            return Self::renameOfOneDirent(task, root, oldParent, oldName, newName)
        }

        let mut child = newParent.clone();

        loop {
            let p = match &(child.0).0.lock().Parent {
                None => break,
                Some(ref dirent) => dirent.clone(),
            };

            if Arc::ptr_eq(&oldParent.0, &p.0) {
                if &(child.0).0.lock().Name == oldName {
                    return Err(Error::SysError(SysErr::EINVAL))
                }
            }

            child = p;
        }

        let oldInode = oldParent.Inode();
        let newInode = newParent.Inode();

        if (oldParent.0).0.lock().frozen && !oldInode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        if (newParent.0).0.lock().frozen && !newInode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        oldInode.CheckPermission(task, &PermMask { write: true, execute: true, read: false })?;
        newInode.CheckPermission(task, &PermMask { write: true, execute: true, read: false })?;

        let renamed = oldParent.walk(task, root, oldName)?;
        oldParent.mayDelete(task, &renamed)?;

        if renamed.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        if newParent.DescendantOf(&renamed) {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let renamedInode = renamed.Inode();
        if renamedInode.StableAttr().IsDir() {
            renamedInode.CheckPermission(task, &PermMask { write: true, execute: false, read: false })?;
        }

        let exist;
        match newParent.walk(task, root, newName) {
            Ok(replaced) => {
                newParent.mayDelete(task, &replaced)?;
                if replaced.IsMountPoint() {
                    return Err(Error::SysError(SysErr::EBUSY))
                }

                let oldIsDir = renamedInode.StableAttr().IsDir();
                let replacedInode = replaced.Inode();
                let newIsDir = replacedInode.StableAttr().IsDir();

                if !newIsDir && oldIsDir {
                    return Err(Error::SysError(SysErr::ENOTDIR))
                }

                if newIsDir && !oldIsDir {
                    return Err(Error::SysError(SysErr::EISDIR))
                }

                replaced.DropExtendedReference();
                replaced.flush();

                exist = true;
            }
            Err(Error::SysError(SysErr::ENOENT)) => {
                exist = false;
            }
            Err(e) => {
                return Err(e)
            }
        }

        let mut newInode = renamed.Inode();
        newInode.Rename(task, oldParent, &renamed, newParent, newName, exist)?;
        (renamed.0).0.lock().Name = newName.to_string();

        (newParent.0).0.lock().Children.remove(newName);
        (oldParent.0).0.lock().Children.remove(oldName);

        (newParent.0).0.lock().Children.insert(newName.to_string(), Arc::downgrade(&renamed));

        renamed.DropExtendedReference();
        renamed.flush();

        return Ok(())
    }

    fn renameOfOneDirent(task: &Task, root: &Dirent, parent: &Dirent, oldName: &str, newName: &str) -> Result<()> {
        let inode = parent.Inode();

        if (parent.0).0.lock().frozen && !inode.IsVirtual() {
            return Err(Error::SysError(SysErr::ENOENT))
        }

        inode.CheckPermission(task, &PermMask { write: true, execute: true, read: false })?;

        let renamed = parent.walk(task, root, oldName)?;

        parent.mayDelete(task, &renamed)?;

        if renamed.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        let renamedInode = renamed.Inode();
        if renamedInode.StableAttr().IsDir() {
            renamedInode.CheckPermission(task, &PermMask { write: true, execute: false, read: false })?;
        }

        let exist;
        match parent.walk(task, root, newName) {
            Ok(replaced) => {
                parent.mayDelete(task, &replaced)?;
                if replaced.IsMountPoint() {
                    return Err(Error::SysError(SysErr::EBUSY))
                }

                let oldIsDir = renamedInode.StableAttr().IsDir();
                let replacedInode = replaced.Inode();
                let newIsDir = replacedInode.StableAttr().IsDir();

                if !newIsDir && oldIsDir {
                    return Err(Error::SysError(SysErr::ENOTDIR))
                }

                if newIsDir && !oldIsDir {
                    return Err(Error::SysError(SysErr::EISDIR))
                }

                replaced.DropExtendedReference();
                replaced.flush();

                exist = true;
            }
            Err(Error::SysError(SysErr::ENOENT)) => {
                exist = false;
            }
            Err(e) => {
                return Err(e)
            }
        }

        let mut newInode = renamed.Inode();
        newInode.Rename(task, parent, &renamed, parent, newName, exist)?;

        (renamed.0).0.lock().Name = newName.to_string();

        let mut p = (parent.0).0.lock();
        p.Children.remove(oldName);
        p.Children.insert(newName.to_string(), Arc::downgrade(&renamed.0));

        renamed.DropExtendedReference();
        renamed.flush();

        return Ok(())
    }

    pub fn MayDelete(&self, task: &Task, root: &Dirent, name: &str) -> Result<()> {
        let inode = self.Inode();

        inode.CheckPermission(task, &PermMask { write: true, execute: true, ..Default::default() })?;

        let victim = self.Walk(task, root, name)?;

        return self.mayDelete(task, &victim);
    }

    fn mayDelete(&self, task: &Task, victim: &Dirent) -> Result<()> {
        self.checkSticky(task, victim)?;

        if victim.IsRoot() {
            return Err(Error::SysError(SysErr::EBUSY))
        }

        return Ok(())
    }

    fn checkSticky(&self, task: &Task, victim: &Dirent) -> Result<()> {
        let inode = self.Inode();
        let uattr = match inode.UnstableAttr(task) {
            Err(_) => return Err(Error::SysError(SysErr::EPERM)),
            Ok(a) => a,
        };

        if !uattr.Perms.Sticky {
            return Ok(())
        }

        let creds = task.creds.clone();
        if uattr.Owner.UID == creds.lock().EffectiveKUID {
            return Ok(())
        }

        let inode = victim.Inode();
        let vuattr = match inode.UnstableAttr(task) {
            Err(_) => return Err(Error::SysError(SysErr::EPERM)),
            Ok(a) => a,
        };

        if vuattr.Owner.UID == creds.lock().EffectiveKUID {
            return Ok(())
        }

        if inode.CheckCapability(task, Capability::CAP_FOWNER) {
            return Ok(())
        }

        return Err(Error::SysError(SysErr::EPERM))
    }

    pub fn ExtendReference(&self) {
        let msrc = self.Inode().lock().MountSource.clone();
        let keep = msrc.lock().Keep(self);
        if keep {
            msrc.lock().ExtendReference(self);
        }
    }

    pub fn DropExtendedReference(&self) {
        let msrc = self.Inode().lock().MountSource.clone();
        msrc.lock().DropExtendReference(self);
    }

    pub fn Froze(&self) {
        let msrc = self.Inode().lock().MountSource.clone();
        msrc.lock().Froze(self);
    }
}

pub fn DirentReadDir(task: &Task, d: &Dirent, it: &FileOperations, root: &Dirent, dirCtx: &mut DirCtx, offset: i64) -> Result<i64> {
    let (offset, err) = direntReadDir(task, d, it, root, dirCtx, offset);

    if dirCtx.Serializer.Written() > 0 {
        return Ok(offset)
    }

    return err
}

fn direntReadDir(task: &Task, d: &Dirent, it: &FileOperations, root: &Dirent, dirCtx: &mut DirCtx, offset: i64) -> (i64, Result<i64>) {
    let mut offset = offset;

    let inode = d.Inode();
    if !inode.StableAttr().IsDir() {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)))
    }

    if offset == FILE_MAX_OFFSET {
        return (offset, Ok(0))
    }

    if (d.0).0.lock().frozen {
        return d.readdirFrozen(task, root, offset, dirCtx);
    }

    let (dot, dotdot) = d.GetDotAttrs(root);

    if offset == 0 {
        match dirCtx.DirEmit(task, &".".to_string(), &dot) {
            Err(e) => return (offset, Err(e)),
            Ok(_) => ()
        }

        offset += 1;
    }

    if offset == 1 {
        match dirCtx.DirEmit(task, &"..".to_string(), &dotdot) {
            Err(e) => return (offset, Err(e)),
            Ok(_) => ()
        }

        offset += 1;
    }

    offset -= 2;

    let (mut newOffset, err) = it.IterateDir(task, d, dirCtx, offset as i32);

    if (newOffset as i64) < offset {
        //let msg = format!("node.Readdir returned offset {} less than input offset {}", newOffset, offset);
        panic!("node.Readdir fail");
    }

    newOffset += 2;
    return (newOffset as i64, err)
}

#[derive(Clone)]
pub struct InterDirent {
    pub Inode: Inode,
    pub Name: String,
    pub Parent: Option<Dirent>,
    pub Children: BTreeMap<String, Weak<(QMutex<InterDirent>, u64)>>,

    pub frozen: bool,
    pub mounted: bool,
}

impl Default for InterDirent {
    fn default() -> Self {
        return Self {
            Inode: Inode::default(),
            Name: "".to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            frozen: false,
            mounted: false,
        }
    }
}

impl InterDirent {
    pub fn New(inode: Inode, name: &str) -> Self {
        return Self {
            Inode: inode.clone(),
            Name: name.to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            frozen: false,
            mounted: false,
        }
    }

    pub fn NewTransient(inode: &Inode) -> Self {
        return Self {
            Inode: inode.clone(),
            Name: "transient".to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            frozen: false,
            mounted: false,
        }
    }

    pub fn IsRoot(&self) -> bool {
        match &self.Parent {
            None => return true,
            _ => return false,
        }
    }
}
