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
use alloc::vec::Vec;

use super::super::super::addr::*;
use super::super::super::auxv::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::super::path::*;
use super::super::super::range::*;
use super::super::fs::dirent::*;
use super::super::fs::file::*;
use super::super::fs::flags::*;
use super::super::fs::inotify::*;
use super::super::kernel::timer::*;
use super::super::kernel_util::*;
use super::super::memmgr::*;
use super::super::stack::*;
use super::super::task::*;
use super::elf::*;
//use super::super::memmgr::mm::*;
use super::interpreter::*;

// maxLoaderAttempts is the maximum number of attempts to try to load
// an interpreter scripts, to prevent loops. 6 (initial + 5 changes) is
// what the Linux kernel allows (fs/exec.c:search_binary_handler).
pub const MAX_LOADER_ATTEMPTS: usize = 6;

pub fn SliceCompare(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }

    for i in 0..left.len() {
        if left[i] != right[i] {
            return false;
        }
    }

    return true;
}

pub fn LoadVDSO(task: &mut Task) -> Result<u64> {
    let vAddr = task
        .mm
        .FindAvailableSeg(task, 0, 3 * MemoryDef::PAGE_SIZE)?;

    let vdsoParamPageAddr = GetVDSOParamPageAddr();
    let paramVAddr = MapVDSOParamPage(task, vAddr, vdsoParamPageAddr)?;
    assert!(paramVAddr == vAddr, "LoadVDSO paramVAddr doesn't match");
    let vdsoVAddr = MapVDSOPage(
        task,
        paramVAddr + MemoryDef::PAGE_SIZE,
        vdsoParamPageAddr + MemoryDef::PAGE_SIZE,
    )?;

    //info!("vdsoParamPageAddr is {:x}, phyaddr is {:x}", vdsoParamPageAddr, task.VirtualToPhy(paramVAddr)?);
    //info!("paramVAddr is {:x}, phyaddr is {:x}", paramVAddr, task.VirtualToPhy(paramVAddr)?);
    //info!("vdsoVAddr is {:x}, phyaddr is {:x}", vdsoVAddr, task.VirtualToPhy(vdsoVAddr)?);
    //info!("paramVAddr is {:x}, vdsoVAddr is {:x}", paramVAddr, vdsoVAddr);

    return Ok(vdsoVAddr);
}

pub fn MapVDSOParamPage(task: &mut Task, virtualAddr: u64, vdsoParamPageAddr: u64) -> Result<u64> {
    let mut moptions = MMapOpts::NewAnonOptions("[vvar]".to_string())?;
    moptions.Length = MemoryDef::PAGE_SIZE;
    moptions.Addr = virtualAddr;
    moptions.Fixed = true;
    moptions.Perms = AccessType::ReadOnly();
    moptions.MaxPerms = AccessType::ReadOnly();
    moptions.Private = true;
    moptions.VDSO = true;
    moptions.Kernel = false;
    moptions.Offset = vdsoParamPageAddr; //use offset to store the phyaddress

    let addr = task.mm.MMap(task, &mut moptions)?;
    return Ok(addr);
}

pub fn MapVDSOPage(task: &mut Task, virtualAddr: u64, vdsoAddr: u64) -> Result<u64> {
    let mut moptions = MMapOpts::NewAnonOptions("[vdso]".to_string())?;
    moptions.Length = 2 * MemoryDef::PAGE_SIZE;
    moptions.Addr = virtualAddr;
    moptions.Fixed = true;
    moptions.Perms = AccessType::Executable();
    moptions.MaxPerms = AccessType::Executable();
    moptions.Private = false;
    moptions.VDSO = true;
    moptions.Kernel = false;
    moptions.Offset = vdsoAddr; //use offset to store the phyaddress

    let addr = task.mm.MMap(task, &mut moptions)?;
    return Ok(addr);
}

pub fn OpenPath(task: &mut Task, filename: &str, maxTraversals: u32) -> Result<(File, Dirent)> {
    let fscontex = task.fsContext.clone();
    let cwd = fscontex.lock().cwd.clone();
    let root = fscontex.lock().root.clone();
    let mut remainingTraversals = maxTraversals;

    let d = task.mountNS.FindDirent(
        task,
        &root,
        Some(cwd),
        filename,
        &mut remainingTraversals,
        true,
    )?;

    let perms = PermMask {
        read: true,
        execute: true,
        ..Default::default()
    };

    let inode = d.Inode();
    inode.CheckPermission(task, &perms)?;

    let len = filename.len();
    // If they claim it's a directory, then make sure.
    //
    // N.B. we reject directories below, but we must first reject
    // non-directories passed as directories.
    if len > 0 && filename.as_bytes()[len - 1] == '/' as u8 && inode.StableAttr().IsDir() {
        return Err(Error::SysError(SysErr::ENOTDIR));
    }

    let file = inode.GetFile(
        task,
        &d,
        &FileFlags {
            Read: true,
            ..Default::default()
        },
    )?;

    file.Dirent
        .InotifyEvent(InotifyEvent::IN_OPEN, 0, EventType::InodeEvent);

    return Ok((file, d));
}

// loadPath resolves filename to a binary and loads it.
pub fn LoadExecutable(
    task: &mut Task,
    filename: &str,
    argv: &mut Vec<String>,
) -> Result<(LoadedElf, Dirent, Vec<String>)> {
    let mut filename = filename.to_string();

    let mut tmp = Vec::new();
    tmp.append(argv);
    let mut argv = tmp;

    for _i in 0..MAX_LOADER_ATTEMPTS {
        let (file, executable) = OpenPath(task, &filename, 40)?;
        defer!(file
            .Dirent
            .InotifyEvent(InotifyEvent::IN_CLOSE_NOWRITE, 0, EventType::InodeEvent));
        let mut hdr: [u8; 4] = [0; 4];

        match ReadAll(task, &file, &mut hdr, 0) {
            Err(e) => {
                print!("Error loading ELF {:?}", e);
                return Err(Error::SysError(SysErr::ENOEXEC));
            }
            Ok(n) => {
                file.Dirent
                    .InotifyEvent(InotifyEvent::IN_ACCESS, 0, EventType::InodeEvent);
                if n < 4 {
                    print!(
                        "Error loading ELF, there is less than 4 bytes data, cnt is {}",
                        n
                    );
                    return Err(Error::SysError(SysErr::ENOEXEC));
                }
            }
        }

        if SliceCompare(&hdr, ELF_MAGIC.as_bytes()) {
            let loaded = LoadElf(task, &file)?;
            return Ok((loaded, executable, argv));
        } else if SliceCompare(&hdr[..2], INTERPRETER_SCRIPT_MAGIC.as_bytes()) {
            info!("start to load script {}", filename);
            let (newpath, newargv) = match ParseInterpreterScript(task, &filename, &file, argv) {
                Err(e) => {
                    info!("Error loading interpreter script: {:?}", e);
                    return Err(e);
                }
                Ok((p, a)) => (p, a),
            };

            //error!("script is {} {:?}", &newpath, &newargv);

            filename = newpath;
            argv = newargv;

            //info!("load script filename is {} argv is {:?}", &filename, &argv);
        } else {
            // it is possible there is shell scrip without "#!" header
            // this work around todo: check how linux handle this?
            info!("unknow magic: {:?}", hdr);
            let mut newargv = Vec::new();
            // Build the new argument list:
            //
            // 1. The interpreter.
            newargv.push("/bin/bash".to_string());

            // 3. The original arguments. The original argv[0] is replaced with the
            // full script filename.
            if argv.len() > 0 {
                argv[0] = filename.to_string();
            } else {
                argv.push(filename.to_string());
            }

            newargv.append(&mut argv);

            filename = "bin/bash".to_string();
            argv = newargv;

            error!("the interrupt is forced to /bin/sh, argv is {:?}", &argv);

            //return Err(Error::SysError(SysErr::ENOEXEC));
        }
    }

    return Err(Error::SysError(SysErr::ENOEXEC));
}

pub const DEFAULT_STACK_SOFT_LIMIT: u64 = 8 * 1024 * 1024;

pub fn CreateStack(task: &Task) -> Result<Range> {
    let stackSize = DEFAULT_STACK_SOFT_LIMIT;

    let stackEnd = task.mm.MapStackAddr();
    let stackStart = stackEnd - stackSize;

    let mut moptions = MMapOpts::NewAnonOptions("[stack]".to_string())?;
    moptions.Length = stackSize;
    moptions.Addr = stackStart;
    moptions.Fixed = true;
    moptions.Perms = AccessType::ReadWrite();
    moptions.MaxPerms = AccessType::ReadWrite();
    moptions.Private = true;
    moptions.GrowsDown = true;

    let addr = task.mm.MMap(task, &mut moptions)?;
    assert!(addr == stackStart);

    return Ok(Range::New(stackStart, stackSize));
}

pub const TASK_COMM_LEN: usize = 16;

// Load loads file with filename into memory.
//return (entry: u64, usersp: u64, kernelsp: u64)
pub fn Load(
    task: &mut Task,
    filename: &str,
    argv: &mut Vec<String>,
    envv: &[String],
    extraAuxv: &[AuxEntry],
) -> Result<(u64, u64, u64)> {
    let vdsoAddr = LoadVDSO(task)?;
    task.mm.SetUserVDSOBase(vdsoAddr);

    let (loaded, executable, tmpArgv) = LoadExecutable(task, filename, argv)?;
    let argv = tmpArgv;

    let e = Addr(loaded.end).RoundUp()?.0;

    task.mm.BrkSetup(e);
    task.mm.SetExecutable(&executable);

    let mut name = Base(&filename);
    if name.len() > TASK_COMM_LEN - 1 {
        name = &name[0..TASK_COMM_LEN - 1];
    }

    task.thread.as_ref().unwrap().lock().name = name.to_string();

    let stackRange = CreateStack(task)?;

    let mut stack = Stack::New(stackRange.End());

    let usersp = SetupUserStack(
        task, &mut stack, &loaded, filename, &argv, envv, extraAuxv, vdsoAddr,
    )?;

    let kernelsp = Task::TaskAddr() + MemoryDef::DEFAULT_STACK_SIZE - 0x10;
    let entry = loaded.entry;

    return Ok((entry, usersp, kernelsp));
}

//return: user stack sp
pub fn SetupUserStack(
    task: &Task,
    stack: &mut Stack,
    loaded: &LoadedElf,
    _filename: &str,
    argv: &[String],
    envv: &[String],
    extraAuxv: &[AuxEntry],
    vdsoAddr: u64,
) -> Result<u64> {
    /* auxv dagta */
    #[cfg(target_arch = "x86_64")]
    let arch = stack.PushStr(task, "x86_64")?;
    #[cfg(target_arch = "aarch64")]
    let arch = stack.PushStr(task, "aarch64")?;

    /* random */
    let (rand1, rand2) = RandU128().unwrap();
    stack.PushU64(task, rand1)?;
    let randAddr = stack.PushU64(task, rand2)?;

    let execfn = stack.PushStr(task, argv[0].as_str())?;

    /*auxv vector*/
    let mut auxv = Vec::new();
    auxv.push(AuxEntry {
        Key: AuxVec::AT_NULL,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_PLATFORM,
        Val: arch,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_EXECFN,
        Val: execfn,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_HWCAP2,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_RANDOM,
        Val: randAddr,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_SECURE,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_EGID,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_GID,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_EUID,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_UID,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_FLAGS,
        Val: 0,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_CLKTCK,
        Val: 100,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_PAGESZ,
        Val: 4096,
    });
    #[cfg(target_arch = "x86_64")]
    auxv.push(AuxEntry {
        Key: AuxVec::AT_HWCAP,
        Val: 0xbfebfbff,
    });
    auxv.push(AuxEntry {
        Key: AuxVec::AT_SYSINFO_EHDR,
        Val: vdsoAddr,
    });

    for e in &loaded.auxv {
        auxv.push(*e)
    }

    for e in extraAuxv {
        auxv.push(*e)
    }

    let l = stack.LoadEnv(task, envv, argv, &auxv)?;

    task.mm.SetupStack(&l, &auxv);

    return Ok(stack.sp);
}
