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

use super::kvm_vcpu::KVMVcpu;
use super::qlib::common::*;
use super::qlib::kernel::*;
use super::qlib::qmsg::*;
use super::qlib::range::*;
use super::qlib::ShareSpace;
use super::*;
// use crate::qlib::proxy::*;
// use std::time::{Duration, Instant};
// use std::collections::BTreeMap;
// use std::sync::Mutex;

// lazy_static! {
//     static ref COUNTER: Mutex<BTreeMap<ProxyCommand, Duration>>  = Mutex::new(BTreeMap::new());
// }
pub fn AQHostCall(msg: HostOutputMsg, _shareSpace: &ShareSpace) {
    let _l = super::GLOCK.lock();
    match msg {
        HostOutputMsg::Default => {
            panic!("AQHostCall Process get Default msg...");
        }
        HostOutputMsg::QCall(_addr) => {
            panic!("AQHostCall Process get Qcall msg...");
        }
        HostOutputMsg::EventfdWriteAsync(msg) => {
            let ret = super::VMSpace::EventfdWrite(msg.fd);
            if ret < 0 {
                panic!("Eventfd write fail with error {}", ret)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QcallRet {
    Normal,
    //the normal return
    Block,
}

impl KVMVcpu {
    //return : true(push the result back), false(block wait)
    pub fn qCall(msg: &'static Msg) -> u64 {
        let mut ret = 0;

        match msg {
            Msg::LoadProcessKernel(msg) => {
                ret = super::VMS.lock().LoadProcessKernel(msg.processAddr) as u64;
            }
            Msg::GetStdfds(msg) => {
                ret = super::VMSpace::GetStdfds(msg.addr) as u64;
            }
            Msg::CreateMemfd(msg) => {
                ret = super::VMSpace::CreateMemfd(msg.len, msg.flags) as u64;
            }
            //Syscall
            Msg::Fallocate(msg) => {
                ret = super::VMSpace::Fallocate(msg.fd, msg.mode, msg.offset, msg.len) as u64;
            }
            Msg::RenameAt(msg) => {
                ret = super::VMSpace::RenameAt(msg.olddirfd, msg.oldpath, msg.newdirfd, msg.newpath)
                    as u64;
            }
            Msg::Ftruncate(msg) => {
                ret = super::VMSpace::Ftruncate(msg.fd, msg.len) as u64;
            }
            Msg::Seek(msg) => {
                ret = super::VMSpace::Seek(msg.fd, msg.offset, msg.whence) as u64;
            }
            Msg::FSetXattr(msg) => {
                ret = super::VMSpace::FSetXattr(msg.fd, msg.name, msg.value, msg.size, msg.flags)
                    as u64;
            }
            Msg::FGetXattr(msg) => {
                ret = super::VMSpace::FGetXattr(msg.fd, msg.name, msg.value, msg.size) as u64;
            }
            Msg::FRemoveXattr(msg) => {
                ret = super::VMSpace::FRemoveXattr(msg.fd, msg.name) as u64;
            }
            Msg::FListXattr(msg) => {
                ret = super::VMSpace::FListXattr(msg.fd, msg.list, msg.size) as u64;
            }
            Msg::ReadLinkAt(msg) => {
                ret = super::VMSpace::ReadLinkAt(msg.dirfd, msg.path, msg.buf, msg.bufsize) as u64;
            }
            Msg::GetTimeOfDay(msg) => {
                ret = super::VMSpace::GetTimeOfDay(msg.tv, msg.tz) as u64;
            }
            Msg::IoCtl(msg) => {
                ret = super::VMSpace::IoCtl(msg.fd, msg.cmd, msg.argp) as u64;
            }
            Msg::Fcntl(msg) => {
                ret = super::VMSpace::Fcntl(msg.fd, msg.cmd, msg.arg) as u64;
            }
            Msg::Close(msg) => {
                ret = super::VMSpace::Close(msg.fd) as u64;
            }
            Msg::Fstat(msg) => {
                ret = super::VMSpace::Fstat(msg.fd, msg.buff) as u64;
            }
            Msg::Fstatat(msg) => {
                ret = super::VMSpace::Fstatat(msg.dirfd, msg.pathname, msg.buff, msg.flags) as u64;
            }
            Msg::Fstatfs(msg) => {
                ret = super::VMSpace::Fstatfs(msg.fd, msg.buf) as u64;
            }
            Msg::TryOpenWrite(msg) => {
                ret = super::VMSpace::TryOpenWrite(msg.dirfd, msg.oldfd, msg.name) as u64;
            }
            Msg::TryOpenAt(msg) => {
                ret = super::VMSpace::TryOpenAt(msg.dirfd, msg.name, msg.addr, msg.skiprw) as u64;
            }
            Msg::OpenAt(msg) => {
                ret = super::VMSpace::OpenAt(msg.dirfd, msg.name, msg.flags, msg.addr) as u64;
            }
            Msg::RemapGuestMemRanges(msg) => {
                ret = super::VMSpace::RemapGuestMemRanges(msg.len, msg.addr, msg.count) as u64;
            }
            Msg::UnmapGuestMemRange(msg) => {
                ret = super::VMSpace::UnmapGuestMemRange(msg.start, msg.len) as u64;
            }
            Msg::HostUnixConnect(msg) => {
                ret = super::VMSpace::HostUnixConnect(msg.type_, msg.addr, msg.len) as u64;
            }
            Msg::HostUnixRecvMsg(msg) => {
                ret = super::VMSpace::HostUnixRecvMsg(msg.fd, msg.msghdr, msg.flags) as u64;
            }
            Msg::TsotRecvMsg(msg) => {
                ret = super::VMSpace::TsotRecvMsg(msg.msgAddr) as u64;
            }
            Msg::TsotSendMsg(msg) => {
                ret = super::VMSpace::TsotSendMsg(msg.msgAddr) as u64;
            }
            Msg::OpenDevFile(msg) => {
                ret = super::VMSpace::OpenDevFile(msg.dirfd, msg.name, msg.flags) as u64;
            }
            Msg::CreateAt(msg) => {
                ret = super::VMSpace::CreateAt(
                    msg.dirfd,
                    msg.pathName,
                    msg.flags,
                    msg.mode,
                    msg.uid,
                    msg.gid,
                    msg.fstatAddr,
                ) as u64;
            }
            Msg::Unlinkat(msg) => {
                ret = super::VMSpace::Unlinkat(msg.dirfd, msg.pathname, msg.flags) as u64;
            }
            Msg::Mkdirat(msg) => {
                ret = super::VMSpace::Mkdirat(msg.dirfd, msg.pathname, msg.mode_, msg.uid, msg.gid)
                    as u64;
            }
            Msg::Mkfifoat(msg) => {
                ret = super::VMSpace::Mkfifoat(msg.dirfd, msg.name, msg.mode, msg.uid, msg.gid)
                    as u64;
            }
            Msg::SysSync(_msg) => {
                ret = super::VMSpace::SysSync() as u64;
            }
            Msg::SyncFs(msg) => {
                ret = super::VMSpace::SyncFs(msg.fd) as u64;
            }
            Msg::SyncFileRange(msg) => {
                ret =
                    super::VMSpace::SyncFileRange(msg.fd, msg.offset, msg.nbytes, msg.flags) as u64;
            }
            Msg::FSync(msg) => {
                ret = super::VMSpace::FSync(msg.fd) as u64;
            }
            Msg::MSync(msg) => {
                ret = super::VMSpace::MSync(msg.addr, msg.len, msg.flags) as u64;
            }
            Msg::MAdvise(msg) => {
                ret = super::VMSpace::MAdvise(msg.addr, msg.len, msg.advise) as u64;
            }
            Msg::FDataSync(msg) => {
                ret = super::VMSpace::FDataSync(msg.fd) as u64;
            }
            Msg::FAccessAt(msg) => {
                ret =
                    super::VMSpace::FAccessAt(msg.dirfd, msg.pathname, msg.mode, msg.flags) as u64;
            }
            Msg::Socket(msg) => {
                ret = super::VMSpace::Socket(msg.domain, msg.type_, msg.protocol) as u64;
            }
            Msg::GetPeerName(msg) => {
                ret = super::VMSpace::GetPeerName(msg.sockfd, msg.addr, msg.addrlen) as u64;
            }
            Msg::GetSockName(msg) => {
                ret = super::VMSpace::GetSockName(msg.sockfd, msg.addr, msg.addrlen) as u64;
            }
            Msg::GetSockOpt(msg) => {
                ret = super::VMSpace::GetSockOpt(
                    msg.sockfd,
                    msg.level,
                    msg.optname,
                    msg.optval,
                    msg.optlen,
                ) as u64;
            }
            Msg::SetSockOpt(msg) => {
                ret = super::VMSpace::SetSockOpt(
                    msg.sockfd,
                    msg.level,
                    msg.optname,
                    msg.optval,
                    msg.optlen,
                ) as u64;
            }
            Msg::IOBind(msg) => {
                ret = super::VMSpace::Bind(msg.sockfd, msg.addr, msg.addrlen, msg.umask) as u64;
            }
            Msg::RDMAListen(msg) => {
                ret = super::VMSpace::RDMAListen(
                    msg.sockfd,
                    msg.backlog,
                    msg.block,
                    msg.acceptQueue.clone(),
                ) as u64;
                // panic!("RDMAListen qcall not implemented")
            }
            Msg::RDMANotify(msg) => {
                ret = super::VMSpace::RDMANotify(msg.sockfd, msg.typ) as u64;
                // panic!("RDMANotify qcall not implemented")
            }
            Msg::IOListen(msg) => {
                ret = super::VMSpace::Listen(msg.sockfd, msg.backlog, msg.block) as u64;
            }
            Msg::IOShutdown(msg) => ret = super::VMSpace::Shutdown(msg.sockfd, msg.how) as u64,
            Msg::SchedGetAffinity(msg) => {
                ret = super::VMSpace::SchedGetAffinity(msg.pid, msg.cpuSetSize, msg.mask) as u64;
            }
            Msg::GetRandom(msg) => {
                ret = super::VMS.lock().GetRandom(msg.buf, msg.len, msg.flags) as u64;
            }
            Msg::Fchdir(msg) => {
                ret = super::VMSpace::Fchdir(msg.fd) as u64;
            }
            Msg::Fadvise(msg) => {
                ret = super::VMSpace::Fadvise(msg.fd, msg.offset, msg.len, msg.advice) as u64;
            }
            Msg::Mlock2(msg) => {
                ret = super::VMSpace::Mlock2(msg.addr, msg.len, msg.flags) as u64;
            }
            Msg::MUnlock(msg) => {
                ret = super::VMSpace::MUnlock(msg.addr, msg.len) as u64;
            }
            Msg::Chown(msg) => {
                ret = super::VMSpace::Chown(msg.pathname, msg.owner, msg.group) as u64;
            }
            Msg::FChown(msg) => {
                ret = super::VMSpace::FChown(msg.fd, msg.owner, msg.group) as u64;
            }
            Msg::Chmod(_msg) => {
                panic!("Panic not implemented")
            }
            Msg::Fchmod(msg) => {
                ret = super::VMSpace::Fchmod(msg.fd, msg.mode) as u64;
            }
            Msg::SwapInPage(msg) => {
                ret = super::VMSpace::SwapInPage(msg.addr) as u64;
            }
            Msg::SwapOut(_msg) => {
                //error!("qcall.rs swapout");
                #[cfg(feature = "cuda")]
                super::VMS.lock().SwapOutGPUPage();

                #[cfg(not(feature = "cuda"))]
                #[cfg(not(feature = "cc"))]
                {
                    let (heapStart, heapEnd) = GLOBAL_ALLOCATOR.HeapRange();
                    SHARE_SPACE
                        .hiberMgr
                        .SwapOut(heapStart, heapEnd - heapStart)
                        .unwrap();
                }

                #[cfg(feature = "cc")]
                {
                    let (heapStart, heapEnd) = GLOBAL_ALLOCATOR.HeapRangeAll();
                    SHARE_SPACE
                        .hiberMgr
                        .SwapOut(heapStart, heapEnd - heapStart)
                        .unwrap();
                }
                ret = 0;
            }
            Msg::SwapIn(_msg) => {
                #[cfg(not(feature = "cuda"))]
                SHARE_SPACE.hiberMgr.ReapSwapIn().unwrap();
                //error!("qcall.rs swapin");

                #[cfg(feature = "cuda")]
                super::VMS.lock().SwapInGPUPage();
                ret = 0;
            }
            Msg::Proxy(msg) => {
                // let start = Instant::now();
                ret = super::VMS.lock().Proxy(&msg.cmd, &msg.parameters) as u64;
                // let duration = start.elapsed();
                // COUNTER.lock().unwrap().entry(msg.cmd.clone()).and_modify(|time| *time += duration).or_insert(duration);
                // if msg.cmd.clone() == ProxyCommand::CudaUnregisterFatBinary {
                //     error!("counter is: {:#?}", &COUNTER.lock().unwrap());
                // }
            }
            Msg::SymLinkAt(msg) => {
                ret = super::VMSpace::SymLinkAt(msg.oldpath, msg.newdirfd, msg.newpath) as u64;
            }
            Msg::LinkAt(msg) => {
                ret = super::VMSpace::LinkAt(
                    msg.olddirfd,
                    msg.oldpath,
                    msg.newdirfd,
                    msg.newpath,
                    msg.flags,
                ) as u64;
            }
            Msg::Futimens(msg) => {
                ret = super::VMSpace::Futimens(msg.fd, msg.times) as u64;
            }

            Msg::IORead(msg) => {
                ret = super::VMSpace::IORead(msg.fd, msg.iovs, msg.iovcnt) as u64;
            }
            Msg::IOTTYRead(msg) => {
                ret = super::VMSpace::IOTTYRead(msg.fd, msg.iovs, msg.iovcnt) as u64;
            }
            Msg::IOWrite(msg) => {
                ret = super::VMSpace::IOWrite(msg.fd, msg.iovs, msg.iovcnt) as u64;
            }
            Msg::IOReadAt(msg) => {
                ret = super::VMSpace::IOReadAt(msg.fd, msg.iovs, msg.iovcnt, msg.offset) as u64;
            }
            Msg::IOWriteAt(msg) => {
                ret = super::VMSpace::IOWriteAt(msg.fd, msg.iovs, msg.iovcnt, msg.offset) as u64;
            }
            Msg::IOAppend(msg) => {
                ret =
                    super::VMSpace::IOAppend(msg.fd, msg.iovs, msg.iovcnt, msg.fileLenAddr) as u64;
            }
            Msg::IOAccept(msg) => {
                ret = super::VMSpace::IOAccept(msg.fd, msg.addr, msg.addrlen) as u64;
            }
            Msg::IOConnect(msg) => {
                ret = super::VMSpace::IOConnect(msg.fd, msg.addr, msg.addrlen) as u64;
            }
            Msg::IORecvMsg(msg) => {
                ret = super::VMSpace::IORecvMsg(msg.fd, msg.msghdr, msg.flags) as u64;
            }
            Msg::IORecvfrom(msg) => {
                ret = super::VMSpace::IORecvfrom(
                    msg.fd, msg.buf, msg.size, msg.flags, msg.addr, msg.len,
                ) as u64;
            }
            Msg::IOSendMsg(msg) => {
                ret = super::VMSpace::IOSendMsg(msg.fd, msg.msghdr, msg.flags) as u64;
            }
            Msg::IOSendto(msg) => {
                ret = super::VMSpace::IOSendto(
                    msg.fd, msg.buf, msg.size, msg.flags, msg.addr, msg.len,
                ) as u64;
            }
            Msg::MMapFile(msg) => {
                ret = match super::PMA_KEEPER.MapFile(msg.len, msg.prot, msg.fd, msg.offset) {
                    Err(Error::SysError(e)) => -e as u64,
                    Ok(phyAddr) => phyAddr,
                    Err(err) => panic!("MMapFile: unexpected error {:?}", err),
                }
            }
            Msg::MUnmap(msg) => match super::PMA_KEEPER.Unmap(&Range::New(msg.addr, msg.len)) {
                Ok(()) => {}
                Err(err) => panic!("MUnmap: unexpected error {:?}", err),
            },
            Msg::NonBlockingPoll(msg) => {
                ret = super::VMSpace::NonBlockingPoll(msg.fd, msg.mask) as u64;
            }
            Msg::NewTmpfsFile(msg) => {
                ret = super::VMSpace::NewTmpfsFile(msg.typ, msg.addr) as u64;
            }
            Msg::Statm(msg) => {
                ret = super::VMSpace::Statm(msg.buf) as u64;
            }
            Msg::NewSocket(msg) => {
                ret = super::VMSpace::NewSocket(msg.fd) as u64;
            }
            Msg::HostEpollWaitProcess(_) => {
                ret = super::VMSpace::HostEpollWaitProcess() as u64;
            }
            Msg::EventfdWrite(msg) => {
                ret = super::VMSpace::EventfdWrite(msg.fd) as u64;
            }
            Msg::ReadControlMsg(msg) => {
                ret = super::VMSpace::ReadControlMsg(msg.fd, msg.addr) as u64;
            }
            Msg::WriteControlMsgResp(msg) => {
                ret = super::VMSpace::WriteControlMsgResp(msg.fd, msg.addr, msg.len, msg.close)
                    as u64;
            }
            Msg::UpdateWaitInfo(msg) => {
                ret = super::VMSpace::UpdateWaitInfo(msg.fd, msg.waitinfo.clone()) as u64;
            }
            Msg::Sysinfo(msg) => {
                ret = super::VMSpace::Sysinfo(msg.addr) as u64;
            }
            Msg::ReadDir(msg) => {
                ret = super::VMSpace::ReadDir(msg.dirfd, msg.addr, msg.len, msg.reset) as u64;
            }
            Msg::Rdtsc(_msg) => {
                ret = TSC.Rdtsc() as u64;
            }
            Msg::SetTscOffset(msg) => {
                TSC.SetOffset(msg.offset);
                VcpuFreqInit();
                ret = 0;
            }
            Msg::TlbShootdown(msg) => {
                ret = SHARE_SPACE.TlbShootdown(msg.vcpuMask);
            }
            Msg::HostMemoryBarrier(_) => {
                VMSpace::HostMemoryBarrier();
            }
        };

        return ret;
    }
}
