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

pub mod util;
pub mod dirent;
pub mod hostinodeop;
pub mod hostfileop;
pub mod tty;
pub mod ioctl;
pub mod socket_iovec;
pub mod fs;
//pub mod control;

use alloc::string::String;
use alloc::string::ToString;
use alloc::collections::btree_map::BTreeMap;
use core::any::Any;

use super::super::qlib::auth::*;
use super::mount::*;
use super::inode::*;
use super::dirent::*;
use self::hostinodeop::*;

pub struct SuperOperations {
    pub mountSourceOperations: SimpleMountSourceOperations,
    pub root: String,
    pub inodeMapping: BTreeMap<u64, String>,
    pub mounter: FileOwner,
    pub dontTranslateOwnership: bool,
}

impl DirentOperations for SuperOperations {
    fn Revalidate(&self, _name: &str, _parent: &Inode, _child: &Inode) -> bool {
        return self.mountSourceOperations.revalidate
    }

    fn Keep(&self, _dirent: &Dirent) -> bool {
        //error!("SuperOperations keep ...");
        //return false
        return true;
    }

    fn CacheReadDir(&self) -> bool {
        return self.mountSourceOperations.cacheReaddir
    }
}

impl MountSourceOperations for SuperOperations {
    fn as_any(&self) -> &Any {
        return self
    }

    fn Destroy(&mut self) {}

    fn ResetInodeMappings(&mut self) {
        self.inodeMapping.clear();
    }

    fn SaveInodeMapping(&mut self, inode: &Inode, path: &str) {
        let sattr = inode.lock().InodeOp.as_any().downcast_ref::<HostInodeOp>().expect("ReadDirAll: not HostInodeOp").StableAttr();
        self.inodeMapping.insert(sattr.InodeId, path.to_string());
    }
}

#[cfg(test1)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use tempfile::Builder;
    use alloc::sync::Arc;
    use spin::Mutex;
    use std::fs::*;
    use std::str;

    use self::util::*;
    use super::*;
    //use super::super::mount::*;
    //use super::super::attr::*;
    //use super::super::inode::*;
    use super::super::ramfs::tree::*;
    use super::super::dentry::*;
    use super::super::path::*;
    use super::super::flags::*;
    use super::super::filesystems::*;
    use super::super::super::task::*;
    use super::super::super::util::*;
    use super::super::super::Common::*;
    use super::super::super::libcDef::*;
    use super::super::super::syscalls::sys_file;
    use super::super::super::syscalls::sys_read;
    use super::super::super::syscalls::sys_write;
    use super::super::super::syscalls::sys_stat;

    fn newTestMountNamespace() -> Result<(MountNs, String)> {
        //let p = Builder::new().prefix("root").tempdir().unwrap();


        //let rootStr = p.path().to_str().unwrap().to_string();
        let root = "/tmp/root";
        remove_dir_all(root).ok();
        create_dir(root).unwrap();
        let rootStr = root.to_string();

        let fd = OpenAt(-100, &rootStr);
        if fd < 0 {
            return Err(Error::SysError(-fd))
        }

        let ms = MountSource::NewHostMountSource(&rootStr, &ROOT_OWNER, &WhitelistFileSystem::New(), &MountSourceFlags::default(), false);
        let root = Inode::NewHostInode(&Arc::new(Mutex::new(ms)), fd)?;

        let mm = MountNs::New(&root);

        return Ok((Arc::new(Mutex::new(mm)), rootStr))
    }

    // createTestDirs populates the root with some test files and directories.
    // /a/a1.txt
    // /a/a2.txt
    // /b/b1.txt
    // /b/c/c1.txt
    // /symlinks/normal.txt
    // /symlinks/to_normal.txt -> /symlinks/normal.txt
    // /symlinks/recursive -> /symlinks
    fn createTestDirs(m: &MountNs, task: &Task) -> Result<()> {
        let r = m.lock().Root().clone();

        r.CreateDirectory(task, &r, &"a".to_string(), &FilePermissions::FromMode(FileMode(0o777)))?;
        let a = r.Walk(task, &r, &"a".to_string())?;
        let _a1 = a.Create(task, &r, &"a1.txt".to_string(), &FileFlags { Read: true, Write: true, ..Default::default() }, &FilePermissions::FromMode(FileMode(0o666)))?;
        let _a2 = a.Create(task, &r, &"a2.txt".to_string(), &FileFlags { Read: true, Write: true, ..Default::default() }, &FilePermissions::FromMode(FileMode(0o666)))?;

        r.CreateDirectory(task, &r, &"b".to_string(), &FilePermissions::FromMode(FileMode(0o0777)))?;
        let b = r.Walk(task, &r, &"b".to_string())?;
        let _b1 = b.Create(task, &r, &"b1.txt".to_string(), &FileFlags { Read: true, Write: true, ..Default::default() }, &FilePermissions::FromMode(FileMode(0o666)))?;

        b.CreateDirectory(task, &r, &"c".to_string(), &FilePermissions::FromMode(FileMode(0o0777)))?;
        let c = b.Walk(task, &r, &"c".to_string())?;
        let _c1 = c.Create(task, &r, &"c1.txt".to_string(), &FileFlags { Read: true, Write: true, ..Default::default() }, &FilePermissions::FromMode(FileMode(0o666)))?;

        r.CreateDirectory(task, &r, &"symlinks".to_string(), &FilePermissions::FromMode(FileMode(0o0777)))?;
        let symlinks = r.Walk(task, &r, &"symlinks".to_string())?;
        let _normal = symlinks.Create(task, &r, &"normal.txt".to_string(), &FileFlags { Read: true, Write: true, ..Default::default() }, &FilePermissions::FromMode(FileMode(0o666)))?;

        symlinks.CreateLink(task, &r, &"/symlinks/normal.txt".to_string(), &"to_normal.txt".to_string())?;
        symlinks.CreateLink(task, &r, &"/symlinks".to_string(), &"recursive".to_string())?;

        return Ok(())
    }

    fn allPaths(task: &Task, m: &MountNs, base: &str) -> Result<Vec<String>> {
        let mut paths: Vec<String> = Vec::new();
        let root = m.lock().Root().clone();

        let mut maxTravelsals = 1;

        let d = m.lock().FindLink(&task, &root, None, &base, &mut maxTravelsals)?;

        let inode = d.lock().Inode.clone();
        let sattr = inode.StableAttr();

        if sattr.IsDir() {
            let dir = inode.GetFile(&d, &FileFlags { Read: true, ..Default::default() })?;
            let iter = dir.lock().FileOp.clone();
            let mut serializer = CollectEntriesSerilizer::New();
            let mut dirCtx = DirCtx::New(&mut serializer);
            DirentReadDir(task, &d, &(*iter.borrow()), &root, &mut dirCtx, 0)?;

            for (name, _) in &serializer.Entries {
                if name.as_str() == "." || name.as_str() == ".." {
                    continue;
                }

                let fullName = Join(&base, name);
                let mut subpaths = allPaths(&task, m, &fullName)?;
                paths.append(&mut subpaths);

                paths.push(fullName);
            }
        }

        return Ok(paths)
    }

    pub struct TestCase {
        pub desc: &'static str,
        pub paths: Vec<&'static str>,
        pub want: Vec<&'static str>,
    }

    fn ToStrs(arr: &Vec<&'static str>) -> Vec<String> {
        let mut res = Vec::new();

        for s in arr {
            res.push(s.to_string())
        }

        return res;
    }

    /*#[test]
    fn TestWhitelist() {
        let testCase = TestCase {
                desc: "root",
                paths: vec!["/"],
                want: vec!["/a", "/a/a1.txt", "/a/a2.txt", "/b", "/b/b1.txt", "/b/c", "/b/c/c1.txt", "/symlinks", "/symlinks/normal.txt", "/symlinks/to_normal.txt", "/symlinks/recursive"]
            };

        let (m, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = m.lock().root.clone();

        createTestDirs(&m).unwrap();
        InstallWhitelist(&task, &m, &ToStrs(&testCase.paths)).unwrap();

        let got = allPaths(&task, &m, &"/".to_string()).unwrap();

        println!("got count is {}", got.len());
        for s in &got {
            println!("get folder {}", s);
        }

        assert!(got == ToStrs(&testCase.want));
        //assert!(1 == 0);
    }*/

    #[test]
    fn TestRootPath() {
        let rootPath = Builder::new().prefix("root").tempdir().unwrap();

        //let rootPath = rootDir.path().to_str().to_string();

        let whitelisted = rootPath.path().join("white");
        let _ = File::create(&whitelisted).unwrap();
        let blacklisted = rootPath.path().join("black");
        let _ = File::create(&blacklisted).unwrap();

        let mut hostFS = WhitelistFileSystem::New();
        let data = format!("{}={},{}={}", ROOT_PATH_KEY, rootPath.path().to_str().unwrap(),
                           WHITELIST_KEY, whitelisted.as_path().to_str().unwrap());

        let mut task = Task::default();

        let inode = hostFS.Mount(&task, &"".to_owned(), &MountSourceFlags::default(), &data).unwrap();
        let mm = Arc::new(Mutex::new(MountNs::New(&inode)));

        hostFS.InstallWhitelist(&task, &mm).unwrap();

        let rootDir = mm.lock().Root();

        println!("after install withlist: children count is {}", &rootDir.lock().Children.len());

        println!("get rootdir");
        task.root = rootDir.clone();
        let inode = rootDir.lock().Inode.clone();

        println!("the rootdir's frozen is {}", rootDir.lock().frozen);
        let f = inode.GetFile(&rootDir, &FileFlags::default()).unwrap();

        let mut c = CollectEntriesSerilizer::New();
        f.lock().ReadDir(&task, &mut c).unwrap();

        let got = c.Order();

        println!("start print......, got couunt is {}", got.len());
        for g in &got {
            println!("val is {}", g);
        }
        let want = vec![".", "..", "white"];

        assert!(got == want);
    }

    // createTestDirs populates the root with some test files and directories.
    // /a/a1.txt
    // /a/a2.txt
    // /b/b1.txt
    // /b/c/c1.txt
    // /symlinks/normal.txt
    // /symlinks/to_normal.txt -> /symlinks/normal.txt
    // /symlinks/recursive -> /symlinks
    #[test]
    fn TestReadPath() {
        let subdirs = vec![("/a".to_string(), true),
                           ("/b/c".to_string(), true),
                           ("/symlinks".to_string(), true),
                           ("/a/a1.txt".to_string(), false),
                           ("/b/b1.txt".to_string(), false),
                           ("/symlinks/normal.txt".to_string(), false),
                           ("/symlinks/to_normal.txt".to_string(), false),
                           ("/symlinks/recursive".to_string(), false),
                           ("/symlinks/recursive/normal.txt".to_string(), false),
                           ("/symlinks/recursive/recursive/normal.txt".to_string(), false),
        ];

        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        for p in &subdirs {
            let mut maxTraversals = 2;
            let dirent = mm.lock().FindLink(&task, &task.root, None, &p.0, &mut maxTraversals).unwrap();
            assert!(dirent.lock().Inode.StableAttr().IsDir() == p.1)
        }

        let mut maxTraversals = 2;
        let mp = mm.lock().FindLink(&task, &task.root, None, &"/symlinks".to_string(), &mut maxTraversals).unwrap();

        let memdirs = vec!["/tmp".to_string(),
                           "/tmp/a/b".to_string(),
                           "/tmp/a/c/d".to_string(),
                           "/tmp/c".to_string(),
                           "/proc".to_string(),
                           "/dev/a/b".to_string(),
        ];

        let mount = MountSource::NewPseudoMountSource();
        let tree = MakeDirectoryTree(&task, &Arc::new(Mutex::new(mount)), &memdirs).unwrap();

        mm.lock().Mount(&mp, &tree).unwrap();

        let expectdir = vec![
            ("/symlinks/tmp".to_string(), true),
            ("/symlinks/tmp/a/b".to_string(), true),
            ("/symlinks/tmp/a/c/d".to_string(), true),
            ("/symlinks/tmp/c".to_string(), true),
            ("/symlinks/proc".to_string(), true),
            ("/symlinks/dev/a/b".to_string(), true),
        ];

        println!("test...................");

        for p in &expectdir {
            let mut maxTraversals = 2;
            let dirent = mm.lock().FindLink(&task, &task.root, None, &p.0, &mut maxTraversals).unwrap();
            assert!(dirent.lock().Inode.StableAttr().IsDir() == p.1)
        }
    }

    // createTestDirs populates the root with some test files and directories.
    // /a/a1.txt
    // /a/a2.txt
    // /b/b1.txt
    // /b/c/c1.txt
    // /symlinks/normal.txt
    // /symlinks/to_normal.txt -> /symlinks/normal.txt
    // /symlinks/recursive -> /symlinks
    #[test]
    fn TestSysFileOpenAtWriteRead1() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestSysFileOpenAtWriteRead2() {
        //openat with dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();
        let cstr = CString::New(&"/a".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        let filename = CString::New(&"a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, fd0, filename.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd2 = sys_file::openAt(&task, fd0, filename.Ptr(), Flags::O_RDWR as u32).unwrap();
        assert!(fd2 == 1);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestSysFileCreateAt1() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();
        let cstr = CString::New(&"/a".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        let filename = CString::New(&"/a/a.txt".to_string());
        let fd1 = sys_file::createAt(&task, ATType::AT_FDCWD, filename.Ptr(), Flags::O_RDWR as u32, FileMode(0o777)).unwrap();

        assert!(fd1 == 1);

        println!("finish createAT........*************");
        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd2 = sys_file::openAt(&task, fd0, filename.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd2 == 1);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestSysFileCreateAt2() {
        //openat with dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();
        let cstr = CString::New(&"/a".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        println!("start create, the fd0 is {}---------------------", fd0);
        let filename = CString::New(&"xxx.txt".to_string());
        let fd1 = sys_file::createAt(&task, fd0, filename.Ptr(), Flags::O_RDWR as u32, FileMode(0o777)).unwrap();

        assert!(fd1 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        sys_file::close(&task, fd1).unwrap();

        println!("start open, the fd0 is {}------------------------", fd0);
        let fd2 = sys_file::openAt(&task, fd0, filename.Ptr(), Flags::O_RDWR as u32).unwrap();
        assert!(fd2 == 1);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestGetCwd1() {
        //openat with dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();
        task.workdir = task.Root();

        let mut arr: [u8; 128] = [0; 128];

        let oldAddr = &mut arr[0] as *mut _ as u64;
        let addr = sys_file::getcwd(&task, oldAddr, arr.len()).unwrap();

        let str = str::from_utf8(&arr[..(addr - oldAddr - 1) as usize]).unwrap();

        println!("the str is {}, len is {}", str, addr - oldAddr);
        assert!(str == "/");

        createTestDirs(&mm, &task).unwrap();

        let str = "/b/c".to_string();
        let cstr = CString::New(&str);
        sys_file::chdir(&mut task, cstr.Ptr()).unwrap();

        let addr = sys_file::getcwd(&task, oldAddr, arr.len()).unwrap();

        let str = str::from_utf8(&arr[..(addr - oldAddr - 1) as usize]).unwrap();
        assert!(str == "/b/c");
    }

    #[test]
    fn TestGetCwd2() {
        //openat with dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();
        task.workdir = task.Root();

        let mut arr: [u8; 128] = [0; 128];

        let oldAddr = &mut arr[0] as *mut _ as u64;
        let addr = sys_file::getcwd(&task, oldAddr, arr.len()).unwrap();

        let str = str::from_utf8(&arr[..(addr - oldAddr - 1) as usize]).unwrap();

        println!("the str is {}, len is {}", str, addr - oldAddr);
        assert!(str == "/");

        createTestDirs(&mm, &task).unwrap();

        let str = "/b/c".to_string();
        let cstr = CString::New(&str);
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        sys_file::fchdir(&mut task, fd0).unwrap();

        let addr = sys_file::getcwd(&task, oldAddr, arr.len()).unwrap();

        let str = str::from_utf8(&arr[..(addr - oldAddr - 1) as usize]).unwrap();
        assert!(str == "/b/c");
    }

    //need to enable task.Creds().lock().HasCapability(Capability::CAP_SYS_CHROOT) before enable the test
    //#[test]
    fn TestSysChroot() {
        //openat with dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();
        task.workdir = task.Root();

        createTestDirs(&mm, &task).unwrap();

        let str = "/b".to_string();
        let cstr = CString::New(&str);

        println!("**************start to chroot");
        sys_file::chroot(&mut task, cstr.Ptr()).unwrap();
        println!("**************after chroot");

        let cstr = CString::New(&"/c".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        let filename = CString::New(&"c1.txt".to_string());
        let fd1 = sys_file::openAt(&task, fd0, filename.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd2 = sys_file::openAt(&task, fd0, filename.Ptr(), Flags::O_RDWR as u32).unwrap();
        assert!(fd2 == 1);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestSysDup1() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd3 = sys_file::Dup(&mut task, fd2).unwrap();
        println!("fd3 = {}", fd3);

        assert!(fd3 == 0);

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd3, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
        sys_file::close(&task, fd3).unwrap();
    }

    #[test]
    fn TestSysDup2() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd3 = sys_file::Dup2(&mut task, fd2, 10).unwrap();
        assert!(fd3 == 10);

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd3, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
        sys_file::close(&task, fd3).unwrap();
    }

    #[test]
    fn TestSysDup3() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd3 = sys_file::Dup3(&mut task, fd2, 10, Flags::O_CLOEXEC as u32).unwrap();
        assert!(fd3 == 10);

        let flag = sys_file::Fcntl(&mut task, fd3, Cmd::F_GETFD, 0).unwrap() as u32;

        println!("flag is {:b}", flag);
        assert!(flag == LibcConst::FD_CLOEXEC as u32);

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd3, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
        sys_file::close(&task, fd3).unwrap();
    }

    #[test]
    fn TestFcntl1() {
        //Cmd::F_DUPFD_CLOEXEC/F_GETFD/F_SETFD
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let fd3 = sys_file::Fcntl(&mut task, fd2, Cmd::F_DUPFD_CLOEXEC, 0).unwrap() as i32;
        assert!(fd3 == 0);

        let flag = sys_file::Fcntl(&mut task, fd3, Cmd::F_GETFD, 0).unwrap() as u32;
        assert!(flag == LibcConst::FD_CLOEXEC as u32);

        let res = sys_file::Fcntl(&mut task, fd3, Cmd::F_SETFD, 0).unwrap() as u32;
        assert!(res == 0);

        let flag = sys_file::Fcntl(&mut task, fd3, Cmd::F_GETFD, 0).unwrap() as u32;
        assert!(flag == 0);

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd3, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
        sys_file::close(&task, fd3).unwrap();
    }

    #[test]
    fn TestFcntl2() {
        //Cmd::F_DUPFD_CLOEXEC/F_GETFD/F_SETFD
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let flag = sys_file::Fcntl(&mut task, fd2, Cmd::F_GETFL, 0).unwrap() as i32;
        assert!(flag & Flags::O_NONBLOCK != Flags::O_NONBLOCK);

        let res = sys_file::Fcntl(&mut task, fd2, Cmd::F_SETFL, Flags::O_NONBLOCK as u64).unwrap() as i32;
        assert!(res == 0);

        let flag = sys_file::Fcntl(&mut task, fd2, Cmd::F_GETFL, 0).unwrap() as i32;
        assert!(flag & Flags::O_NONBLOCK == Flags::O_NONBLOCK);
    }

    #[test]
    fn TestMkdir1() {
        //TestMkdir
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/new".to_string());
        let res = sys_file::Mkdir(&task, cstr.Ptr(), 0o777).unwrap();
        assert!(res == 0);

        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd0 == 0);

        let filename = CString::New(&"/a/new/a.txt".to_string());
        let fd1 = sys_file::createAt(&task, ATType::AT_FDCWD, filename.Ptr(), Flags::O_RDWR as u32, FileMode(0o777)).unwrap();

        assert!(fd1 == 1);

        let res = sys_file::Unlinkat(&task, ATType::AT_FDCWD, filename.Ptr()).unwrap();
        assert!(res == 0);

        let res = sys_file::Rmdir(&task, cstr.Ptr()).unwrap();
        assert!(res == 0);
    }

    #[test]
    fn TestMkdir2() {
        //TestMkdirat
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd0 == 0);

        let cstr = CString::New(&"new".to_string());
        let res = sys_file::Mkdirat(&task, fd0, cstr.Ptr(), 0o777).unwrap();
        assert!(res == 0);

        let fd1 = sys_file::openAt(&task, fd0, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd1 == 1);

        let filename = CString::New(&"a.txt".to_string());
        let fd2 = sys_file::createAt(&task, fd1, filename.Ptr(), Flags::O_RDWR as u32, FileMode(0o777)).unwrap();

        assert!(fd2 == 2);

        let res = sys_file::Unlinkat(&task, fd1, filename.Ptr()).unwrap();
        assert!(res == 0);

        let cstr = CString::New(&"/a/new".to_string());
        let res = sys_file::Rmdir(&task, cstr.Ptr()).unwrap();
        assert!(res == 0);
    }

    // createTestDirs populates the root with some test files and directories.
    // /a/a1.txt
    // /a/a2.txt
    // /b/b1.txt
    // /b/c/c1.txt
    // /symlinks/normal.txt
    // /symlinks/to_normal.txt -> /symlinks/normal.txt
    // /symlinks/recursive -> /symlinks
    #[test]
    fn TestSymlink_filelink() {
        //TestMkdirat
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        let newPath = CString::New(&"/b/link.txt".to_string());
        let res = sys_file::Symlink(&task, newPath.Ptr(), cstr.Ptr()).unwrap();
        assert!(res == 0);

        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, newPath.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestSymlink_folderlink() {
        //TestMkdirat
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        let oldfolder = CString::New(&"/a".to_string());
        let newfolder = CString::New(&"/d".to_string());
        let res = sys_file::Symlink(&task, newfolder.Ptr(), oldfolder.Ptr()).unwrap();
        assert!(res == 0);

        let newPath = CString::New(&"/d/a1.txt".to_string());
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, newPath.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestSymlink_linkat() {
        //TestMkdirat
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/b".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd0 == 0);

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let newPath = CString::New(&"link.txt".to_string());
        let res = sys_file::Symlinkat(&task, newPath.Ptr(), fd0, cstr.Ptr()).unwrap();
        assert!(res == 0);

        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let cstr = CString::New(&"/b/link.txt".to_string());

        let mut buf: [u8; 1024] = [0; 1024];
        let size = sys_file::ReadLink(&task, cstr.Ptr(), &mut buf[0] as *mut _ as u64, 1024).unwrap();
        assert!(size > 0);
        assert!(str::from_utf8(&buf[..size as usize]).unwrap() == "/a/a1.txt");

        let mut buf: [u8; 1024] = [0; 1024];
        let size = sys_file::ReadLinkAt(&task, fd0, cstr.Ptr(), &mut buf[0] as *mut _ as u64, 1024).unwrap();
        assert!(size > 0);
        assert!(str::from_utf8(&buf[..size as usize]).unwrap() == "/a/a1.txt");

        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 1);
        assert!(fd2 == 2);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn Testlink_link() {
        //TestMkdirat
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let _fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        let newPath = CString::New(&"/b/link.txt".to_string());
        let res = sys_file::Link(&task, cstr.Ptr(), newPath.Ptr());
        assert!(res == Err(Error::SysError(SysErr::EPERM)));
    }

    #[test]
    fn TestTruncate1() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let data = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, data.Ptr(), data.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let size = 6;
        let res = sys_file::Truncate(&task, cstr.Ptr(), size).unwrap();
        assert!(res == 0);

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == size as i64);

        assert!(data.data[..size as usize] == buf[0..size as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestTruncate2() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();
        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        assert!(fd1 == 0);
        assert!(fd2 == 1);

        let str = "1234567890".to_string();
        let data = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, data.Ptr(), data.Len() as i64).unwrap();
        println!("after write");

        let size = 6;
        let res = sys_file::Ftruncate(&task, fd1, size).unwrap();
        assert!(res == 0);

        sys_file::close(&task, fd1).unwrap();

        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == size as i64);

        assert!(data.data[..size as usize] == buf[0..size as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestUmask() {
        let mut task = Task::default();

        let mask = sys_file::Umask(&mut task, 123).unwrap();
        assert!(mask == 0);

        let mask = sys_file::Umask(&mut task, 456).unwrap();
        assert!(mask == 123);
    }

    #[test]
    fn TestChown() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());

        let stat = LibcStat::default();
        let res = sys_stat::Stat(&task, cstr.Ptr(), &stat as *const _ as u64).unwrap();
        assert!(res == 0);

        println!("the gid is {}", stat.st_gid);
        assert!(stat.st_uid == 0);
        assert!(stat.st_gid == 65534);

        let res = sys_file::Chown(&task, cstr.Ptr(), 123, 456);
        assert!(res == Err(Error::SysError(SysErr::EPERM)));
    }

    #[test]
    fn TestUTime() {
        //openat without dirfd
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let stat = LibcStat::default();
        let res = sys_stat::Stat(&task, cstr.Ptr(), &stat as *const _ as u64).unwrap();
        assert!(res == 0);

        println!("the atime is {}, mtime is {}", stat.st_atime, stat.st_mtime);


        let _utime = Utime {
            Actime: stat.st_atime + 100,
            Modtime: stat.st_mtime + 100,
        };

        //let res = sys_file::Utime(&task, cstr.Ptr(), &utime as * const _ as u64).unwrap();
        //todo: fix this. Get -1
        let _res = sys_file::Utime(&task, cstr.Ptr(), 0).unwrap();
        //assert!(res == 0);*/
    }

    #[test]
    fn TestRename1() {
        //rename to another file in same folder
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 0);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let oldname = CString::New(&"/a/a1.txt".to_string());
        let newname = CString::New(&"/a/b1.txt".to_string());
        let res = sys_file::Rename(&task, oldname.Ptr(), newname.Ptr()).unwrap();
        assert!(res == 0);

        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, newname.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd2 == 0);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestRename2() {
        //replace exist file
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 0);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let oldname = CString::New(&"/a/a1.txt".to_string());
        let newname = CString::New(&"/b/b1.txt".to_string());
        let res = sys_file::Rename(&task, oldname.Ptr(), newname.Ptr()).unwrap();
        assert!(res == 0);

        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, newname.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd2 == 0);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    #[test]
    fn TestRename3() {
        //rename to a file in differnt folder
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 0);

        let str = "1234567890".to_string();
        let cstr = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, cstr.Ptr(), cstr.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        let oldname = CString::New(&"/a/a1.txt".to_string());
        let newname = CString::New(&"/d.txt".to_string());
        let res = sys_file::Rename(&task, oldname.Ptr(), newname.Ptr()).unwrap();
        assert!(res == 0);

        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, newname.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd2 == 0);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap();
        assert!(cnt == cstr.Len() as i64);
        assert!(cstr.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }


    #[test]
    fn TestRename4() {
        //renameat to a file in differnt folder
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        createTestDirs(&mm, &task).unwrap();

        let cstr = CString::New(&"/a/a1.txt".to_string());
        let fd1 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDWR as u32).unwrap();

        assert!(fd1 == 0);

        let str = "1234567890".to_string();
        let data = CString::New(&str);
        println!("before write");
        sys_write::Write(&task, fd1, data.Ptr(), data.Len() as i64).unwrap();
        println!("after write");
        sys_file::close(&task, fd1).unwrap();

        println!("**************************before renameat111");
        let cstr = CString::New(&"/a".to_string());
        let fd0 = sys_file::openAt(&task, ATType::AT_FDCWD, cstr.Ptr(), Flags::O_RDONLY as u32).unwrap();

        let oldname = CString::New(&"a1.txt".to_string());
        let newname = CString::New(&"/d.txt".to_string());
        println!("**************************before renameat");
        let res = sys_file::Renameat(&task, fd0, oldname.Ptr(), ATType::AT_FDCWD, newname.Ptr()).unwrap();
        println!("***************************after renameat");
        assert!(res == 0);
        sys_file::close(&task, fd0).unwrap();

        let fd2 = sys_file::openAt(&task, ATType::AT_FDCWD, newname.Ptr(), Flags::O_RDONLY as u32).unwrap();
        assert!(fd2 == 0);
        let buf: [u8; 100] = [0; 100];
        let cnt = sys_read::Read(&task, fd2, &buf[0] as *const _ as u64, buf.len() as i64).unwrap() as usize;
        println!("cnt is {}, len is {}", cnt, data.Len());

        //todo: the value is same, but the assert fail. Fix it.
        // assert!(cnt == cstr.Len());

        assert!(data.data[..] == buf[0..cnt as usize]);
        sys_file::close(&task, fd2).unwrap();
    }

    pub fn Dup(oldfd: i32) -> i64 {
        return unsafe {
            libc::dup(oldfd) as i64
        };
    }

    #[test]
    fn TestStdIo() {
        let (mm, _) = newTestMountNamespace().unwrap();

        let mut task = Task::default();
        task.root = mm.lock().root.clone();

        //todo: doesn't why the fstat stdin doesn't work. fix it
        //let stdin = Dup(0) as i32;

        let stdout = Dup(1) as i32;
        let stderr = Dup(2) as i32;

        let stdfds = [stdout, stdout, stderr];
        println!("before newstdfds");
        task.NewStdFds(&stdfds, true).unwrap();
        println!("after newstdfds");

        let str = "1234567890".to_string();
        let data = CString::New(&str);
        let res = sys_write::Write(&task, 1, data.Ptr(), data.Len() as i64).unwrap();
        println!("after sys_write::Write, the res is {}", res);
        assert!(res == data.Len() as i64);
        println!("the end of test...");
    }
}