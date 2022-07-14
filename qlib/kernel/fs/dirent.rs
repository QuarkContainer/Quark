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
use alloc::sync::Weak;
use alloc::vec::Vec;
use core::cmp::Eq;
use core::cmp::PartialEq;
use core::ops::Deref;
use spin::*;
use lazy_static::lazy_static;

use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::singleton::*;
use super::super::socket::unix::transport::unix::*;
use super::super::task::*;
use super::super::SHARESPACE;
use super::super::uid::*;
use super::dentry::*;
use super::file::*;
use super::flags::*;
use super::inode::*;
use super::mount::*;
use super::inotify::*;

lazy_static! {
    pub static ref NEGATIVE_DIRENT: Dirent = Dirent::default();
    pub static ref NEGATIVE_DIRENT1: Dirent = Dirent::default();
}

pub static RENAME: Singleton<RwLock<()>> = Singleton::<RwLock<()>>::New();
pub unsafe fn InitSingleton() {
    RENAME.Init(RwLock::new(()));
}

pub fn DirentReadDir(
    task: &Task,
    d: &Dirent,
    it: &FileOperations,
    root: &Dirent,
    dirCtx: &mut DirCtx,
    offset: i64,
) -> Result<i64> {
    let (offset, err) = direntReadDir(task, d, it, root, dirCtx, offset);

    if dirCtx.Serializer.Written() > 0 {
        return Ok(offset);
    }

    return err;
}

fn direntReadDir(
    task: &Task,
    d: &Dirent,
    it: &FileOperations,
    root: &Dirent,
    dirCtx: &mut DirCtx,
    offset: i64,
) -> (i64, Result<i64>) {
    let mut offset = offset;

    let inode = d.Inode();
    if !inode.StableAttr().IsDir() {
        return (0, Err(Error::SysError(SysErr::ENOTDIR)));
    }

    if offset == FILE_MAX_OFFSET {
        return (offset, Ok(0));
    }

    let (dot, dotdot) = d.GetDotAttrs(root);

    if offset == 0 {
        match dirCtx.DirEmit(task, &".".to_string(), &dot) {
            Err(e) => return (offset, Err(e)),
            Ok(_) => (),
        }

        offset += 1;
    }

    if offset == 1 {
        match dirCtx.DirEmit(task, &"..".to_string(), &dotdot) {
            Err(e) => return (offset, Err(e)),
            Ok(_) => (),
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
    return (newOffset as i64, err);
}

#[derive(Clone)]
pub struct DirentWeak(pub Weak<DirentInternal>);

impl DirentWeak {
    pub fn Upgrade(&self) -> Option<Dirent> {
        let d = match self.0.upgrade() {
            None => return None,
            Some(d) => d,
        };

        return Some(Dirent(d));
    }
}

#[derive(Clone, Default)]
pub struct Dirent (Arc<DirentInternal>);

impl Dirent {
    pub fn Downgrade(&self) -> DirentWeak {
        return DirentWeak(Arc::downgrade(&self.0));
    }
}

impl Deref for Dirent {
    type Target = Arc<DirentInternal>;

    fn deref(&self) -> &Arc<DirentInternal> {
        &self.0
    }
}

impl Ord for Dirent {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl PartialOrd for Dirent {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
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
            let parent = self.main.lock().Parent.take();
            match parent {
                None => (),
                Some(parent) => {
                    let name = self.main.lock().Name.clone();
                    parent.RemoveChild(&name);
                }
            }

            if SHARESPACE.config.read().EnableInotify {
                let watches = self.Watches();

                // If this inode is being destroyed because it was unlinked, queue a
                // deletion event. This may not be the case for inodes being revalidated.
                let unlinked = watches.read().unlinked;
                if unlinked {
                    watches.Notify("",
                                   InotifyEvent::IN_DELETE_SELF,
                                   0,
                                   EventType::InodeEvent,
                                   false);
                }

                // Remove references from the watch owners to the watches on this inode,
                // since the watches are about to be GCed. Note that we don't need to worry
                // about the watch pins since if there were any active pins, this inode
                // wouldn't be in the destructor.
                watches.TargetDestroyed();
            }
        }
    }
}

impl Dirent {
    pub fn New(inode: &Inode, name: &str) -> Self {
        let main = DirentMain {
            Inode: inode.clone(),
            Name: name.to_string(),
            Parent: None,
            mounted: false,
        };

        let intern = DirentInternal {
            id: NewUID(),
            inode: inode.clone(),
            watches: Watches::default(),
            main: QMutex::new(main),
            dirMutex: QRwLock::new(()),
            children: QMutex::new(BTreeMap::new()),
        };

        return Self(Arc::new(intern));
    }

    pub fn Watches(&self) -> Watches {
        return self.watches.clone();
    }

    pub fn NewTransient(inode: &Inode) -> Self {
        return Self::New(inode, "transient");
    }

    pub fn ID(&self) -> u64 {
        return self.id;
    }

    pub fn Parent(&self) -> Option<Dirent> {
        match &self.main.lock().Parent {
            None => None,
            Some(ref d) => Some(d.clone()),
        }
    }

    pub fn Inode(&self) -> Inode {
        return self.inode.clone();
    }


    pub fn MyFullName(&self) -> String {
        let _a = RENAME.read();

        return self.myFullName();
    }

    fn myFullName(&self) -> String {
        let name = self.main.lock().Name.clone();
        let parent = match &self.main.lock().Parent {
            None => return name,
            Some(ref p) => p.clone(),
        };

        let pName = parent.myFullName();

        if pName == "/".to_string() {
            return "/".to_string() + &self.main.lock().Name;
        }

        return pName + &"/".to_string() + &self.main.lock().Name;
    }

    pub fn ChildDenAttrs(&self, task: &Task) -> Result<BTreeMap<String, DentAttr>> {
        let dirName = self.MyFullName();

        let inode = self.Inode();
        let dir = match inode.GetFile(
            task,
            &self,
            &FileFlags {
                Read: true,
                ..Default::default()
            },
        ) {
            Err(err) => {
                info!("failed to open directory {}", &dirName);
                return Err(err);
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

        return self.fullName(root);
    }

    //ret: (fulname, whether root node is reachable from d)
    fn fullName(&self, root: &Dirent) -> (String, bool) {
        if self == root {
            return ("/".to_string(), true);
        }

        if self.main.lock().IsRoot() {
            return (self.main.lock().Name.clone(), false);
        }

        let parent = self.main.lock().Parent.as_ref().unwrap().clone();
        let (pName, reachable) = parent.fullName(root);

        if pName == "/".to_string() {
            return (pName + &self.main.lock().Name, reachable);
        }

        let ret = pName + &"/".to_string() + &self.main.lock().Name;
        return (ret, reachable);
    }

    pub fn MountRoot(&self) -> Self {
        let _a = RENAME.write();

        let mut mountRoot = self.clone();
        loop {
            if mountRoot.main.lock().mounted {
                return mountRoot;
            }

            let parent = mountRoot.main.lock().Parent.clone();
            match parent {
                None => return mountRoot,
                Some(p) => mountRoot = p,
            }
        }
    }

    pub fn DescendantOf(&self, p: &Dirent) -> bool {
        let mut d = self.clone();
        let p = p.clone();

        loop {
            if d == p {
                return true;
            }

            let parent = match &d.main.lock().Parent {
                None => return false,
                Some(ref parent) => parent.clone(),
            };

            d = parent;
        }
    }

    fn getChild(&self, name: &str) -> Option<Dirent> {
        let remove = match self.children.lock().get(name) {
            Some(subD) => match subD.Upgrade() {
                Some(cd) => return Some(cd),
                None => true,
            },

            None => false,
        };

        if remove {
            self.children.lock().remove(name);
        }

        return None;
    }

    fn GetCacheChild(&self, name: &str) -> Option<Dirent> {
        match self.children.lock().get(name) {
            Some(subD) => match subD.Upgrade() {
                Some(cd) => return Some(cd),
                None => (),
            },

            None => (),
        };

        return None;
    }

    pub fn IsNegative(&self) -> bool {
        return Arc::ptr_eq(&self.0, &NEGATIVE_DIRENT1.0)
    }

    fn walk(&self, task: &Task, root: &Dirent, name: &str) -> Result<Dirent> {
        let inode = self.Inode();
        if !inode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::ENOTDIR));
        }

        if name == "" || name == "." {
            return Ok(self.clone());
        } else if name == ".." {
            if self.clone() == root.clone() {
                return Ok(self.clone());
            }

            match &self.main.lock().Parent {
                None => return Ok(self.clone()),
                Some(ref p) => return Ok(p.clone()),
            }
        }

        let child = self.GetCacheChild(name);
        let remove = match child {
            Some(dirent) => {
                if dirent.IsNegative() {
                    return Err(Error::SysError(SysErr::ENOENT))
                }

                let mounted = dirent.main.lock().mounted;
                /*let subInode = cd.0.lock().Inode.clone();
                let mountSource = subInode.lock().MountSource.clone();
                let mountsourceOpations = mountSource.lock().MountSourceOperations.clone();*/

                // very likely there is a deadlock in the Revalidate(..). As there is no revalidate=true case
                // work around now. and todo: fix it
                let revalidate = false; //mountsourceOpations.lock().Revalidate(name, &inode, &subInode);
                if mounted || !revalidate {
                    return Ok(dirent);
                }

                dirent.Watches().Unpin(&dirent);

                false
            }
            None => true,
        };

        if remove {
            self.children.lock().remove(name);
        }

        let c = match inode.Lookup(task, name) {
            Err(Error::SysError(SysErr::ENOENT)) => {

                // why the negative doesn't work? todo: fix this
                //let negative = Arc::downgrade(&(NEGATIVE_DIRENT.0));
                //(self.0).0.lock().Children.insert(String::from(name), negative);

                return Err(Error::SysError(SysErr::ENOENT))
            }
            Err(e) => return Err(e),
            Ok(c) => c,
        };

        assert!(c.main.lock().Name == name, "lookup get mismatch name");

        self.AddChild(String::from(name), &c);

        return Ok(c);
    }

    pub fn Walk(&self, task: &Task, root: &Dirent, name: &str) -> Result<Dirent> {
        //error!("Walk 1 {}", name);
        //defer!(error!("Walk 2 {}", name));
        let _a = RENAME.read();

        return self.walk(task, root, name);
    }

    pub fn RemoveChild(&self, name: &String) {
        self.children.lock().remove(name);
    }

    pub fn IsRoot(&self) -> bool {
        return self.main.lock().IsRoot()
    }

    pub fn AddChild(
        &self,
        name: String,
        child: &Dirent,
    ) -> Option<Dirent> {
        assert!(
        child.IsRoot(),
        "Add child request the child has no parent"
        );
        child.main.lock().Parent = Some(self.clone());

        return self.addChild(name, child);
    }

    pub fn Name(&self) -> String {
        return self.main.lock().Name.clone()
    }

    pub fn addChild(
        &self,
        name: String,
        child: &Dirent,
    ) -> Option<Dirent> {
        assert!(
        child.Parent().unwrap() == self.clone(),
        "Dirent addChild assumes the child already belongs to the parent"
        );

        //let name = child.0.lock().Name.clone();
        //println!("addChild the name is {}", name);
        match self
            .children
            .lock()
            .insert(name, child.Downgrade()) {
            None => return None,
            Some(c) => return c.Upgrade(),
        }
    }

    fn exists(&self, task: &Task, root: &Dirent, name: &str) -> bool {
        match self.walk(task, root, name) {
            Err(_) => false,
            _ => true,
        }
    }

    pub fn Create(
        &self,
        task: &Task,
        root: &Dirent,
        name: &str,
        flags: &FileFlags,
        perms: &FilePermissions,
    ) -> Result<File> {
        let _a = RENAME.read();

        if self.exists(task, root, name) {
            return Err(Error::SysError(SysErr::EEXIST));
        }

        let mut inode = self.Inode();
        let file = inode.Create(task, self, name, flags, perms)?;

        let child = file.Dirent.clone();

        self.AddChild(String::from(name), &child);
        child.ExtendReference();
        if SHARESPACE.config.read().EnableInotify {
            self.Watches().Notify(name,
                                   InotifyEvent::IN_CREATE,
                                   0,
                                   EventType::InodeEvent,
                                   false);
        }

        return Ok(file);
    }

    fn genericCreate(
        &self,
        task: &Task,
        root: &Dirent,
        name: &str,
        create: &mut FnMut() -> Result<()>,
    ) -> Result<()> {
        let _a = RENAME.read();

        if self.exists(task, root, name) {
            return Err(Error::SysError(SysErr::EEXIST));
        }

        self.children.lock().remove(name);
        return create();
    }

    pub fn CreateLink(
        &self,
        task: &Task,
        root: &Dirent,
        oldname: &str,
        newname: &str,
    ) -> Result<()> {
        return self.genericCreate(task, root, newname, &mut || -> Result<()> {
            let mut inode = self.Inode();
            inode.CreateLink(task, self, oldname, newname)?;
            if SHARESPACE.config.read().EnableInotify {
                self.Watches().Notify(newname,
                                       InotifyEvent::IN_CREATE,
                                       0,
                                       EventType::PathEvent,
                                       false);
            }
            self.children.lock().remove(oldname);
            self.children.lock().remove(newname);
            return Ok(())
        });
    }

    pub fn CreateHardLink(
        &self,
        task: &Task,
        root: &Dirent,
        target: &Dirent,
        name: &str,
    ) -> Result<()> {
        let mut inode = self.Inode();
        let targetInode = target.Inode();
        if !Arc::ptr_eq(&inode.lock().MountSource, &targetInode.lock().MountSource) {
            return Err(Error::SysError(SysErr::EXDEV));
        }

        if targetInode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::EPERM));
        }

        return self.genericCreate(task, root, name, &mut || -> Result<()> {
            inode.CreateHardLink(task, self, &target, name)?;
            self.children.lock().remove(name);

            if SHARESPACE.config.read().EnableInotify {
                target.Watches().Notify("",
                                             InotifyEvent::IN_ATTRIB,
                                             0,
                                             EventType::InodeEvent,
                                             false);
                self.Watches().Notify(name,
                                       InotifyEvent::IN_CREATE,
                                       0,
                                       EventType::InodeEvent,
                                       false);
            }
            return Ok(())
        });
    }

    pub fn CreateDirectory(
        &self,
        task: &Task,
        root: &Dirent,
        name: &str,
        perms: &FilePermissions,
    ) -> Result<()> {
        return self.genericCreate(task, root, name, &mut || -> Result<()> {
            let mut inode = self.Inode();
            let ret = inode.CreateDirectory(task, self, name, perms);
            if SHARESPACE.config.read().EnableInotify {
                self.Watches().Notify(name,
                                       InotifyEvent::IN_ISDIR | InotifyEvent::IN_CREATE,
                                       0,
                                       EventType::PathEvent,
                                       false);
            }

            self.children.lock().remove(name);
            return ret;
        });
    }

    pub fn Bind(
        &self,
        task: &Task,
        root: &Dirent,
        name: &str,
        data: &BoundEndpoint,
        perms: &FilePermissions,
    ) -> Result<Dirent> {
        let result = self.genericCreate(task, root, name, &mut || -> Result<()> {
            let inode = self.Inode();
            let childDir = inode.Bind(task, self, name, data, perms)?;
            self.AddChild(String::from(name), &childDir);
            childDir.ExtendReference();
            return Ok(());
        });

        match result {
            Err(Error::SysError(SysErr::EEXIST)) => {
                return Err(Error::SysError(SysErr::EADDRINUSE))
            }
            Err(e) => return Err(e),
            _ => (),
        };

        let inode = self.Inode();
        let childDir = inode.Lookup(task, name)?;
        if SHARESPACE.config.read().EnableInotify {
            self.Watches().Notify(name,
                                   InotifyEvent::IN_CREATE,
                                   0,
                                   EventType::InodeEvent,
                                   false);
        }

        return Ok(childDir);
    }

    pub fn CreateFifo(
        &self,
        task: &Task,
        root: &Dirent,
        name: &str,
        perms: &FilePermissions,
    ) -> Result<()> {
        return self.genericCreate(task, root, name, &mut || -> Result<()> {
            let mut inode = self.Inode();
            inode.CreateFifo(task, self, name, perms)?;
            self.children.lock().remove(name);
            if SHARESPACE.config.read().EnableInotify {
                self.Watches().Notify(name,
                                       InotifyEvent::IN_CREATE,
                                       0,
                                       EventType::InodeEvent,
                                       false);
            }
            return Ok(())
        });
    }

    pub fn GetDotAttrs(&self, root: &Dirent) -> (DentAttr, DentAttr) {
        let inode = self.Inode();
        let dot = inode.StableAttr().DentAttr();
        if !self.IsRoot() && self.DescendantOf(root) {
            let parent = self.Parent();
            let pInode = parent.unwrap().Inode();
            let dotdot = pInode.StableAttr().DentAttr();
            return (dot, dotdot);
        }

        return (dot, dot);
    }

    pub fn flush(&self) {
        let mut expired = Vec::new();
        let mut current = Vec::new();
        let mut children = self.children.lock();

        for (n, w) in children.iter() {
            match w.Upgrade() {
                None => expired.push(n.clone()),
                Some(cd) => {
                    let dirent = cd.clone();
                    dirent.flush();
                    dirent.DropExtendedReference();
                    current.push(dirent);
                }
            }
        }

        for n in &expired {
            children.remove(n);
        }

        drop(children);
        drop(current);
        drop(expired);
    }

    pub fn IsMountPoint(&self) -> bool {
        let mounted = self.main.lock().mounted;
        return mounted || self.IsRoot();
    }

    pub fn Mount(&self, inode: &Inode) -> Result<Dirent> {
        if inode.lock().StableAttr().IsSymlink() {
            return Err(Error::SysError(SysErr::ENOENT));
        }

        let parent = self.Parent().unwrap();
        let replacement = Dirent::New(inode, &self.Name());
        replacement.main.lock().mounted = true;

        let name = self.Name();
        parent.AddChild(name, &replacement);

        return Ok(replacement);
    }

    pub fn UnMount(&self, replace: &Dirent) -> Result<()> {
        let parent = self
            .Parent()
            .expect("unmount required the parent is not none");
        let old = parent.AddChild(replace.Name(), replace);

        match old {
            None => panic!("mount must mount over an existing dirent"),
            Some(_) => (),
        }

        self.main.lock().mounted = false;
        return Ok(());
    }

    pub fn Remove(&self, task: &Task, root: &Dirent, name: &str, dirPath: bool) -> Result<()> {
        let mut inode = self.Inode();

        let child = self.Walk(task, root, name)?;
        let childInode = child.Inode();
        if childInode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::EISDIR));
        } else if dirPath {
            return Err(Error::SysError(SysErr::EISDIR));
        }

        if child.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY));
        }

        inode.Remove(task, self, &child)?;

        // Link count changed, this only applies to non-directory nodes.
        if SHARESPACE.config.read().EnableInotify {
            child.Watches().Notify("",
                                        InotifyEvent::IN_ATTRIB,
                                        0,
                                        EventType::InodeEvent,
                                        false);
        }

        self.children.lock().remove(name);
        child.DropExtendedReference();

        // Finally, let inotify know the child is being unlinked. Drop any extra
        // refs from inotify to this child dirent. This doesn't necessarily mean the
        // watches on the underlying inode will be destroyed, since the underlying
        // inode may have other links. If this was the last link, the events for the
        // watch removal will be queued by the inode destructor.
        if SHARESPACE.config.read().EnableInotify {
            child.Watches().MarkUnlinked();
            child.Watches().Unpin(&child);
        }

        // trigger inode destroy
        drop(child);
        drop(childInode);

        if SHARESPACE.config.read().EnableInotify {
            self.Watches().Notify(name,
                                   InotifyEvent::IN_DELETE,
                                   0,
                                   EventType::InodeEvent,
                                   false);
        }

        return Ok(());
    }

    pub fn RemoveDirectory(&self, task: &Task, root: &Dirent, name: &str) -> Result<()> {
        let mut inode = self.Inode();
        if name == "." {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if name == ".." {
            return Err(Error::SysError(SysErr::ENOTEMPTY));
        }

        let child = self.Walk(task, root, name)?;
        let childInode = child.Inode();

        if !childInode.StableAttr().IsDir() {
            return Err(Error::SysError(SysErr::ENOTDIR));
        }

        if child.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY));
        }

        inode.Remove(task, self, &child)?;

        self.children.lock().remove(name);

        child.DropExtendedReference();

        // Finally, let inotify know the child is being unlinked. Drop any extra
        // refs from inotify to this child dirent.
        if SHARESPACE.config.read().EnableInotify {
            child.Watches().MarkUnlinked();
            child.Watches().Unpin(&child);
            self.Watches().Notify(name,
                                   InotifyEvent::IN_ISDIR | InotifyEvent::IN_DELETE,
                                   0,
                                   EventType::PathEvent,
                                   false);
        }

        return Ok(());
    }

    pub fn Rename(
        task: &Task,
        root: &Dirent,
        oldParent: &Dirent,
        oldName: &str,
        newParent: &Dirent,
        newName: &str,
    ) -> Result<()> {
        let _a = RENAME.write();

        if Arc::ptr_eq(oldParent, newParent) {
            if oldName == newName {
                return Ok(());
            }

            return Self::renameOfOneDirent(task, root, oldParent, oldName, newName);
        }

        let mut child = newParent.clone();

        loop {
            let p = match child.Parent() {
                None => break,
                Some(dirent) => dirent,
            };

            if Arc::ptr_eq(&oldParent.0, &p.0) {
                if child.main.lock().Name == oldName {
                    return Err(Error::SysError(SysErr::EINVAL));
                }
            }

            child = p;
        }

        let oldInode = oldParent.Inode();
        let newInode = newParent.Inode();

        oldInode.CheckPermission(
            task,
            &PermMask {
                write: true,
                execute: true,
                read: false,
            },
        )?;
        newInode.CheckPermission(
            task,
            &PermMask {
                write: true,
                execute: true,
                read: false,
            },
        )?;

        let renamed = oldParent.walk(task, root, oldName)?;
        oldParent.mayDelete(task, &renamed)?;

        if renamed.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY));
        }

        if newParent.DescendantOf(&renamed) {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        let renamedInode = renamed.Inode();
        if renamedInode.StableAttr().IsDir() {
            renamedInode.CheckPermission(
                task,
                &PermMask {
                    write: true,
                    execute: false,
                    read: false,
                },
            )?;
        }

        let exist;
        match newParent.walk(task, root, newName) {
            Ok(replaced) => {
                newParent.mayDelete(task, &replaced)?;
                if replaced.IsMountPoint() {
                    return Err(Error::SysError(SysErr::EBUSY));
                }

                let oldIsDir = renamedInode.StableAttr().IsDir();
                let replacedInode = replaced.Inode();
                let newIsDir = replacedInode.StableAttr().IsDir();

                if !newIsDir && oldIsDir {
                    return Err(Error::SysError(SysErr::ENOTDIR));
                }

                if newIsDir && !oldIsDir {
                    return Err(Error::SysError(SysErr::EISDIR));
                }

                replaced.DropExtendedReference();
                replaced.flush();

                exist = true;
            }
            Err(Error::SysError(SysErr::ENOENT)) => {
                exist = false;
            }
            Err(e) => return Err(e),
        }

        let mut newInode = renamed.Inode();
        newInode.Rename(task, oldParent, &renamed, newParent, newName, exist)?;
        renamed.main.lock().Name = newName.to_string();

        newParent.children.lock().remove(newName);
        oldParent.children.lock().remove(oldName);

        newParent
            .children
            .lock()
            .insert(newName.to_string(), renamed.Downgrade());

        // Queue inotify events for the rename.
        let mut ev : u32 = 0;
        if newInode.StableAttr().IsDir() {
            ev |=  InotifyEvent::IN_ISDIR;
        }

        if SHARESPACE.config.read().EnableInotify {
            let cookie = NewInotifyCookie();
            oldParent.Watches().Notify(
                oldName,
                ev | InotifyEvent::IN_MOVED_FROM,
                cookie,
                EventType::InodeEvent,
                false);
            newParent.Watches().Notify(
                newName,
                ev | InotifyEvent::IN_MOVED_TO,
                cookie,
                EventType::InodeEvent,
                false);

            // Somewhat surprisingly, self move events do not have a cookie.
            renamed.Watches().Notify(
                "",
                InotifyEvent::IN_MOVE_SELF,
                0,
                EventType::InodeEvent,
                false);
        }

        renamed.DropExtendedReference();
        renamed.Watches().Unpin(&renamed);
        renamed.flush();

        return Ok(());
    }

    fn renameOfOneDirent(
        task: &Task,
        root: &Dirent,
        parent: &Dirent,
        oldName: &str,
        newName: &str,
    ) -> Result<()> {
        let inode = parent.Inode();

        inode.CheckPermission(
            task,
            &PermMask {
                write: true,
                execute: true,
                read: false,
            },
        )?;

        let renamed = parent.walk(task, root, oldName)?;

        parent.mayDelete(task, &renamed)?;

        if renamed.IsMountPoint() {
            return Err(Error::SysError(SysErr::EBUSY));
        }

        let renamedInode = renamed.Inode();
        if renamedInode.StableAttr().IsDir() {
            renamedInode.CheckPermission(
                task,
                &PermMask {
                    write: true,
                    execute: false,
                    read: false,
                },
            )?;
        }

        let exist;
        match parent.walk(task, root, newName) {
            Ok(replaced) => {
                parent.mayDelete(task, &replaced)?;
                if replaced.IsMountPoint() {
                    return Err(Error::SysError(SysErr::EBUSY));
                }

                let oldIsDir = renamedInode.StableAttr().IsDir();
                let replacedInode = replaced.Inode();
                let newIsDir = replacedInode.StableAttr().IsDir();

                if !newIsDir && oldIsDir {
                    return Err(Error::SysError(SysErr::ENOTDIR));
                }

                if newIsDir && !oldIsDir {
                    return Err(Error::SysError(SysErr::EISDIR));
                }

                replaced.DropExtendedReference();
                replaced.flush();

                exist = true;
            }
            Err(Error::SysError(SysErr::ENOENT)) => {
                exist = false;
            }
            Err(e) => return Err(e),
        }

        let mut newInode = renamed.Inode();
        newInode.Rename(task, parent, &renamed, parent, newName, exist)?;

        renamed.main.lock().Name = newName.to_string();

        {
            let mut p = parent.children.lock();
            p.remove(oldName);
            p.insert(newName.to_string(), renamed.Downgrade());
        }

        // Queue inotify events for the rename.
        let mut ev : u32 = 0;
        if newInode.StableAttr().IsDir() {
            ev |=  InotifyEvent::IN_ISDIR;
        }

        if SHARESPACE.config.read().EnableInotify {
            let cookie = NewInotifyCookie();

            parent.Watches().Notify(
                oldName,
                ev | InotifyEvent::IN_MOVED_FROM,
                cookie,
                EventType::InodeEvent,
                false);
            parent.Watches().Notify(
                newName,
                ev | InotifyEvent::IN_MOVED_TO,
                cookie,
                EventType::InodeEvent,
                false);

            // Somewhat surprisingly, self move events do not have a cookie.
            renamed.Watches().Notify(
                "",
                InotifyEvent::IN_MOVE_SELF,
                0,
                EventType::InodeEvent,
                false);
        }

        renamed.DropExtendedReference();
        renamed.flush();

        return Ok(());
    }

    pub fn MayDelete(&self, task: &Task, root: &Dirent, name: &str) -> Result<()> {
        let inode = self.Inode();

        inode.CheckPermission(
            task,
            &PermMask {
                write: true,
                execute: true,
                ..Default::default()
            },
        )?;

        let victim = self.Walk(task, root, name)?;

        return self.mayDelete(task, &victim);
    }

    fn mayDelete(&self, task: &Task, victim: &Dirent) -> Result<()> {
        self.checkSticky(task, victim)?;

        if victim.IsRoot() {
            return Err(Error::SysError(SysErr::EBUSY));
        }

        return Ok(());
    }

    fn checkSticky(&self, task: &Task, victim: &Dirent) -> Result<()> {
        let inode = self.Inode();
        let uattr = match inode.UnstableAttr(task) {
            Err(_) => return Err(Error::SysError(SysErr::EPERM)),
            Ok(a) => a,
        };

        if !uattr.Perms.Sticky {
            return Ok(());
        }

        let creds = task.creds.clone();
        if uattr.Owner.UID == creds.lock().EffectiveKUID {
            return Ok(());
        }

        let inode = victim.Inode();
        let vuattr = match inode.UnstableAttr(task) {
            Err(_) => return Err(Error::SysError(SysErr::EPERM)),
            Ok(a) => a,
        };

        if vuattr.Owner.UID == creds.lock().EffectiveKUID {
            return Ok(());
        }

        if inode.CheckCapability(task, Capability::CAP_FOWNER) {
            return Ok(());
        }

        return Err(Error::SysError(SysErr::EPERM));
    }

    // InotifyEvent notifies all watches on the inode for this dirent and its parent
    // of potential events. The events may not actually propagate up to the user,
    // depending on the event masks. InotifyEvent automatically provides the name of
    // the current dirent as the subject of the event as required, and adds the
    // IN_ISDIR flag for dirents that refer to directories.
    pub fn InotifyEvent(&self, event: u32, cookie: u32, et: EventType) {
        if SHARESPACE.config.read().EnableInotify {
            let _ = RENAME.read();

            let mut event = event;

            let inode = self.Inode();
            if inode.StableAttr().IsDir() {
                event |= InotifyEvent::IN_ISDIR;
            }

            // The ordering below is important, Linux always notifies the parent first.
            let parent = self.Parent();
            match parent {
                None => (),
                Some(p) => {
                    let name = self.Name();
                    p.Watches().Notify(&name,
                                               event,
                                               cookie,
                                               et,
                                               false);
                }
            }

            self.Watches().Notify("",
                                   event,
                                   cookie,
                                   et,
                                   false);
        }
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

pub struct DirentInternal {
    pub id: u64,
    pub inode: Inode,
    pub watches: Watches,
    pub main: QMutex<DirentMain>,
    pub dirMutex: QRwLock<()>,
    pub children: QMutex<BTreeMap<String, DirentWeak>>,
}

impl Default for DirentInternal {
    fn default() -> Self {
        return Self {
            id: NewUID(),
            inode: Inode::default(),
            watches: Watches::default(),
            main: Default::default(),
            dirMutex: Default::default(),
            children: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct DirentMain {
    pub Inode: Inode,
    pub Name: String,
    pub Parent: Option<Dirent>,
    pub mounted: bool,
}

impl Default for DirentMain {
    fn default() -> Self {
        return Self {
            Inode: Inode::default(),
            Name: "".to_string(),
            Parent: None,
            mounted: false,
        };
    }
}

impl DirentMain {
    pub fn New(inode: Inode, name: &str) -> Self {
        return Self {
            Inode: inode.clone(),
            Name: name.to_string(),
            Parent: None,
            mounted: false,
        };
    }

    pub fn NewTransient(inode: &Inode) -> Self {
        return Self {
            Inode: inode.clone(),
            Name: "transient".to_string(),
            Parent: None,
            mounted: false,
        };
    }

    pub fn IsRoot(&self) -> bool {
        match &self.Parent {
            None => return true,
            _ => return false,
        }
    }
}

#[derive(Clone)]
pub struct DirentMain1 {
    pub Inode: Inode,
    pub Name: String,
    pub Parent: Option<Dirent>,
    pub Children: BTreeMap<String, Weak<(QMutex<DirentMain1>, u64)>>,

    pub mounted: bool,
}

impl Default for DirentMain1 {
    fn default() -> Self {
        return Self {
            Inode: Inode::default(),
            Name: "".to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            mounted: false,
        };
    }
}

impl DirentMain1 {
    pub fn New(inode: Inode, name: &str) -> Self {
        return Self {
            Inode: inode.clone(),
            Name: name.to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            mounted: false,
        };
    }

    pub fn NewTransient(inode: &Inode) -> Self {
        return Self {
            Inode: inode.clone(),
            Name: "transient".to_string(),
            Parent: None,
            Children: BTreeMap::new(),
            mounted: false,
        };
    }

    pub fn IsRoot(&self) -> bool {
        match &self.Parent {
            None => return true,
            _ => return false,
        }
    }
}

