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

use std::io::prelude::*;
use std::fs::{OpenOptions, create_dir_all};
use std::string::String;
use std::ffi::CString;
use libc::*;

use super::vmspace::syscall::*;
use super::qlib::SysCallID;

pub fn GetRet(ret: i32) -> i32 {
    if ret == -1 {
        return -errno::errno().0
    }

    return ret
}

pub struct Util {}

impl Util {
    fn WriteFile(filename: &str, str: &str) {
        //let mut file = File::open(filename).expect("Open file fail");
        let mut file = OpenOptions::new().write(true).truncate(true).open(&filename).expect("Open file fail");
        file.write_all(str.as_bytes()).expect("write all fail");
    }

    pub fn Mount(src: &str, target: &str, fstype: &str, flags: u64, data: &str) -> i32 {
        let src = CString::new(src.clone()).expect("CString::new src failed");
        let target = CString::new(target.clone()).expect("CString::new target failed");
        let fstype = CString::new(fstype.clone()).expect("CString::new fstype failed");
        let data = CString::new(data.clone()).expect("CString::new fstype failed");

        let srcAddr = if src.as_bytes().len() ==0  {
            0 as *const c_char
        } else {
            src.as_ptr()
        };

        let targetAddr = if target.as_bytes().len() ==0  {
            0 as *const c_char
        } else {
            target.as_ptr()
        };

        let fstypeAddr = if fstype.as_bytes().len() ==0  {
            0 as *const c_char
        } else {
            fstype.as_ptr()
        };

        let dataAddr = if data.as_bytes().len() ==0  {
            0 as *const c_void
        } else {
            data.as_ptr() as *const c_void
        };

        return unsafe {
            GetRet(mount(srcAddr, targetAddr, fstypeAddr, flags, dataAddr as *const c_void))
        }
    }

    pub fn Umount2(target: &str, flags: i32) -> i32 {
        let target = CString::new(target.clone()).expect("CString::new target failed");

        return unsafe {
            GetRet(umount2(target.as_ptr(), flags))
        }
    }

    pub fn Chdir(dir: &str) -> i32 {
        let dir = CString::new(dir.clone()).expect("CString::new src failed");

        return unsafe {
            GetRet(chdir(dir.as_ptr()))
        }
    }

    pub fn Mkdir(path: &str, mode: u32) -> i32 {
        let path = CString::new(path.clone()).expect("CString::new src failed");

        return unsafe {
            GetRet(mkdir(path.as_ptr(), mode))
        }
    }

    pub fn PivotRoot(newRoot: &str, putOld: &str) -> i32 {
        let newRoot = CString::new(newRoot.clone()).expect("CString::new src failed");
        let putOld = CString::new(putOld.clone()).expect("CString::new target failed");

        let nr = SysCallID::sys_pivot_root as usize;
        unsafe {
            return GetRet(syscall2(nr, newRoot.as_ptr() as usize, putOld.as_ptr() as usize) as i32);
        }
    }
}


pub struct MountNs {
    pub rootfs: String,
}

impl MountNs {
    pub fn New(rootfs: String) -> Self {
        return Self {
            rootfs: rootfs
        }
    }

    /* 
        this mounts the host's root path to a /old_root directory when pivoting
        this is an adhoc change to make qvisor able to mount further container fs images to the sandboxRootDir 
        and made these available to current sandbox once get a "StartSubContainer" ucall (for k8s integration). 
        Otherwise, these path won't be available to the mountNS.
        Notice this might be a big security problem, 
    */
    pub fn PivotRoot2(&self) {
        if Util::Chdir(&self.rootfs) < 0 {
            panic!("chdir fail for rootfs {}", &self.rootfs)
        }

        match create_dir_all("old_root") {
            Ok(()) => (),
            Err(_e) => panic!("failed to create dir to put old root")
        };

        let errno = Util::PivotRoot(".", ".");
        if errno != 0 {
            panic!("pivot fail with errno = {}", errno)
        }

        if Util::Chdir("/") < 0 {
            panic!("chdir fail")
        }

        // https://man.archlinux.org/man/pivot_root.2.en#pivot_root(&quot;.&quot;,_&quot;.&quot;)
        if Util::Umount2("/", MNT_DETACH) < 0 {
            panic!("UMount2 fail")
        }
    }

    pub fn PivotRoot(&self) {
        /*let flags = MS_REC | MS_SLAVE;

        if Util::Mount("","/", "", flags, "") < 0 {
            panic!("mount root fail")
        }*/
        
        if Util::Chdir(&self.rootfs) < 0 {
            panic!("chdir fail for rootfs {}", &self.rootfs)
        }

        let errno = Util::PivotRoot(".", ".");
        if errno != 0 {
            panic!("pivot fail with errno = {}", errno)
        }

        if Util::Chdir("/") < 0 {
            panic!("chdir fail")
        }

        // https://man.archlinux.org/man/pivot_root.2.en#pivot_root(&quot;.&quot;,_&quot;.&quot;)
        if Util::Umount2("/", MNT_DETACH) < 0 {
            panic!("UMount2 fail")
        }
    }

    pub fn PivotRoot1(&self) {
        let flags = MS_REC | MS_SLAVE;

        if Util::Mount("","/", "", flags, "") < 0 {
            panic!("mount root fail")
        }

        if Util::Mount(&self.rootfs, &self.rootfs, "", MS_REC | MS_BIND, "") < 0 {
            panic!("mount rootfs fail")
        }

        if Util::Chdir(&self.rootfs) == -1 {
            panic!("chdir fail")
        }

        let errno = Util::PivotRoot(".", ".");
        if errno != 0 {
            panic!("pivot fail with errno = {}", errno)
        }

        if Util::Chdir("/") == -1 {
            panic!("chdir fail")
        }

        if Util::Umount2("/", MNT_DETACH) < 0 {
            panic!("UMount2 fail")
        }
    }

    pub fn PrepareProcfs(&self) {
        let proc = "/proc".to_string();
        if Util::Mkdir(&proc, 0o555) == -1 && errno::errno().0 != EEXIST {
            panic!("mkdir put_old fail")
        }

        if Util::Mount(&"proc".to_string(), &proc, &"proc".to_string(), 0, &"".to_string()) < 0 {
            panic!("mount rootfs fail")
        }
    }
}

pub struct UserNs {
    pub pid: i32,
    pub uid: i32,
    pub gid: i32,
}

impl UserNs {
    pub fn New(pid: i32, uid: i32) -> Self {
        return Self {
            pid: pid,
            uid: uid,
            gid: uid,
        }
    }

    pub fn Set(&self) {
        let path = format!("/proc/{}/uid_map", self.pid);
        let line = format!("0 {} 1\n", self.uid);
        Util::WriteFile(&path, &line);

        let path = format!("/proc/{}/setgroups", self.pid);
        let line = format!("deny");
        Util::WriteFile(&path, &line);

        let path = format!("/proc/{}/gid_map", self.pid);
        let line = format!("0 {} 1\n", self.gid);
        Util::WriteFile(&path, &line);
    }

    // Map nobody in the new namespace to nobody in the parent namespace.
    //
    // A sandbox process will construct an empty
    // root for itself, so it has to have the CAP_SYS_ADMIN
    // capability.
    //
    pub fn SetNobody(&self) {
        let nobody = 65534;

        let path = format!("/proc/{}/uid_map", self.pid);
        let line = format!("0 {} 1\n", nobody - 1);
        let line = line + &format!("{} {} 1\n", nobody, nobody);
        Util::WriteFile(&path, &line);

        let path = format!("/proc/{}/gid_map", self.pid);
        let line = format!("{} {} 1\n", nobody, nobody);
        Util::WriteFile(&path, &line);
    }
}