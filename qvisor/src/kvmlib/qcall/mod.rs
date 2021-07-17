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

use core::str;
use core::slice;

use super::qlib::{ShareSpace};
use super::qlib::common::*;
use super::qlib::qmsg::*;
use super::qlib::range::*;
use super::VMSpace;
use super::syncmgr::*;
use super::*;

pub fn AQHostCall(msg: HostOutputMsg, shareSpace: &ShareSpace) {
    let _l = super::GLOCK.lock();
    match msg {
        HostOutputMsg::QCall(_addr) => {
            panic!("AQHostCall Process get Qcall msg...");
        }
        HostOutputMsg::WaitFD(msg) => {
            let ret = super::VMSpace::WaitFD(msg.fd, msg.mask);
            if ret < 0 {
                panic!("WaitFD fail err is {}", ret);
            }
        }
        HostOutputMsg::Close(msg) => {
            super::VMSpace::Close(0, msg.fd);
        }
        HostOutputMsg::MUnmap(msg) => {
            match super::PMA_KEEPER.lock().Unmap(&Range::New(msg.addr, msg.len)) {
                Ok(()) => (),
                Err(err) => panic!("MUnmap: unexpected error {:?}", err),
            }
        }
        HostOutputMsg::IOBufWrite(msg) => {
            let _ret = VMSpace::IOBufWrite(msg.fd, msg.addr, msg.len, msg.offset);
            //error!("HostOutputMsg::IOBufWrite ret is {}", ret);
            /*shareSpace.AQHostInputCall(HostInputMsg::IOBufWriteResp(IOBufWriteResp{
                fd: msg.fd,
                addr: msg.addr,
                len: msg.len,
                ret: ret,
            }));*/
        }
        HostOutputMsg::PrintStr(msg) => {
            let ptr = msg.addr as *const u8;
            let slice = unsafe { slice::from_raw_parts(ptr, msg.len) };
            info!("{}", str::from_utf8(slice).expect("PrintStr handling fail"));
            shareSpace.AQHostInputCall(&HostInputMsg::PrintStrResp(PrintStrResp{
                addr: msg.addr,
                len: msg.len,
            }));
        }
        HostOutputMsg::WakeVCPU(msg) => {
            let vcpuId = msg.vcpuId as usize;
            SyncMgr::WakeVcpu(vcpuId);
        }
    }
}

impl<'a> ShareSpace {
    pub fn AQHostInputCall(&self, item: &HostInputMsg) {
        loop {
            if self.QInput.IsFull() {
                continue;
            }

            self.QInput.Push(&item).unwrap();
            break;
        }
        //SyncMgr::WakeVcpu(self, TaskIdQ::default());

        //SyncMgr::WakeVcpu(self, TaskIdQ::New(1<<12, 0));
        KERNEL_IO_THREAD.Wakeup(self);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QcallRet {
    Normal,
    //the normal return
    Block,
}

//return : true(push the result back), false(block wait)
pub fn qCall(eventAddr: u64, event: &'static mut Event) -> QcallRet {
    let _l = super::GLOCK.lock();
    //error!("qcall event is {:x?}", event);
    //defer!(error!("qcall2"));
    match event {
        Event { taskId: _taskId, interrupted: _, ret: _, msg: Msg::Print(_level, str) } => {
            info!("{}", str);
            //print!("{}", str);
        }
        Event { taskId: _taskId, interrupted: _, ret: _, msg: Msg::PrintInt(msg) } => {
            error!("PrintInt: {}", msg.val);
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::MMapAnon(MMapAnon) } => {
            *ret = match super::PMA_KEEPER.lock().MapAnon(MMapAnon.len, MMapAnon.prot) {
                Err(Error::SysError(e)) => -e as u64,
                Ok(phyAddr) => phyAddr,
                Err(err) => panic!("MapAnon: unexpected error {:?}", err),
            }
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::MMapFile(MMapFile) } => {
            *ret = match super::PMA_KEEPER.lock().MapFile(MMapFile.len, MMapFile.prot, MMapFile.fd, MMapFile.offset) {
                Err(Error::SysError(e)) => -e as u64,
                Ok(phyAddr) => phyAddr,
                Err(err) => panic!("MMapFile: unexpected error {:?}", err),
            };
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::LoadProcessKernel(msg) } => {
            *ret = super::VMS.lock().LoadProcessKernel(taskId.Addr(), msg.processAddr, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ControlMsgRet(msg) } => {
            *ret = super::VMS.lock().ControlMsgRet(taskId.Addr(), msg.msgId, msg.addr, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ControlMsgCall(msg) } => {
            let retAddr = ret as * const _ as u64;
            let ret = super::VMS.lock().ControlMsgCall(*taskId, msg.addr, msg.len, retAddr);
            return ret;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::RenameAt(msg) } => {
            *ret = super::VMSpace::RenameAt(taskId.Addr(), msg.olddirfd, msg.oldpath, msg.newdirfd, msg.newpath) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fallocate(msg) } => {
            *ret = super::VMSpace::Fallocate(taskId.Addr(), msg.fd, msg.mode, msg.offset, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Truncate(msg) } => {
            *ret = super::VMSpace::Truncate(taskId.Addr(), msg.path, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Ftruncate(msg) } => {
            *ret = super::VMSpace::Ftruncate(taskId.Addr(), msg.fd, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Eventfd(msg) } => {
            *ret = super::VMSpace::Eventfd(taskId.Addr(), msg.initval, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Seek(msg) } => {
            *ret = super::VMSpace::Seek(taskId.Addr(), msg.fd, msg.offset, msg.whence) as u64;
        }

        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ReadLink(msg) } => {
            *ret = super::VMSpace::ReadLink(taskId.Addr(), msg.path, msg.buf, msg.bufsize) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ReadLinkAt(msg) } => {
            *ret = super::VMSpace::ReadLinkAt(taskId.Addr(), msg.dirfd, msg.path, msg.buf, msg.bufsize) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetTimeOfDay(msg) } => {
            //info!("start of GetTimeOfDay");
            *ret = super::VMSpace::GetTimeOfDay(taskId.Addr(), msg.tv, msg.tz) as u64;
            //info!("end of GetTimeOfDay");
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ClockGetRes(msg) } => {
            *ret = super::VMSpace::ClockGetRes(taskId.Addr(), msg.clkId, msg.ts) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ClockGetTime(msg) } => {
            *ret = super::VMSpace::ClockGetTime(taskId.Addr(), msg.clkId, msg.ts) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::ClockSetTime(msg) } => {
            *ret = super::VMSpace::ClockSetTime(taskId.Addr(), msg.clkId, msg.ts) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Times(msg) } => {
            *ret = super::VMSpace::Times(taskId.Addr(), msg.tms) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IoCtl(msg) } => {
            *ret = super::VMSpace::IoCtl(taskId.Addr(), msg.fd, msg.cmd, msg.argp) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fcntl(msg) } => {
            *ret = super::VMSpace::Fcntl(taskId.Addr(), msg.fd, msg.cmd, msg.arg) as u64;
        }
       Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::Time(msg) } => {
            *ret = super::VMSpace::Time(msg.tloc) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::PRLimit(msg) } => {
            *ret = super::VMSpace::PRLimit(msg.pid, msg.resource, msg.newLimit, msg.oldLimit) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::GetRLimit(msg) } => {
            *ret = super::VMSpace::GetRLimit(msg.resource, msg.rlimit) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::SetRLimit(msg) } => {
            *ret = super::VMSpace::SetRLimit(msg.resource, msg.rlimit) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::Stat(msg) } => {
            *ret = super::VMSpace::Stat(msg.pathName, msg.statBuff) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fstat(msg) } => {
            *ret = super::VMSpace::Fstat(taskId.Addr(), msg.fd, msg.buff) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fstatat(msg) } => {
            *ret = super::VMSpace::Fstatat(taskId.Addr(), msg.dirfd, msg.pathname, msg.buff, msg.flags) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::Statfs(msg) } => {
            *ret = super::VMSpace::Statfs(msg.path, msg.buf) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fstatfs(msg) } => {
            *ret = super::VMSpace::Fstatfs(taskId.Addr(), msg.fd, msg.buf) as u64;
        }
        /*Event {taskId: _, interrupted: _, ref mut ret, msg: Msg::ForkFdTbl(msg)} => {
            *ret = super::VMSpace::ForkFdTbl(msg.pTgid, msg.tgid) as u64;
        }*/
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::TryOpenAt(msg) } => {
            *ret = super::VMSpace::TryOpenAt(taskId.Addr(), msg.dirfd, msg.name, msg.addr) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::CreateAt(msg) } => {
            *ret = super::VMSpace::CreateAt(taskId.Addr(), msg.dirfd, msg.pathName, msg.flags, msg.mode, msg.uid, msg.gid, msg.fstatAddr) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SysSync(_msg) } => {
            *ret = super::VMSpace::SysSync(taskId.Addr()) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SyncFs(msg) } => {
            *ret = super::VMSpace::SyncFs(taskId.Addr(), msg.fd) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SyncFileRange(msg) } => {
            *ret = super::VMSpace::SyncFileRange(taskId.Addr(), msg.fd, msg.offset, msg.nbytes, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::FSync(msg) } => {
            *ret = super::VMSpace::FSync(taskId.Addr(), msg.fd) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::MSync(msg) } => {
            *ret = super::VMSpace::MSync(taskId.Addr(), msg.addr, msg.len, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::MAdvise(msg) } => {
            *ret = super::VMSpace::MAdvise(taskId.Addr(), msg.addr, msg.len, msg.advise) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::FDataSync(msg) } => {
            *ret = super::VMSpace::FDataSync(taskId.Addr(), msg.fd) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Uname(msg) } => {
            *ret = super::VMSpace::Uname(taskId.Addr(), msg.buff) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Umask(msg) } => {
            *ret = super::VMSpace::Umask(taskId.Addr(), msg.mask) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Unlinkat(msg) } => {
            *ret = super::VMSpace::Unlinkat(taskId.Addr(), msg.dirfd, msg.pathname, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Mkdirat(msg) } => {
            *ret = super::VMSpace::Mkdirat(taskId.Addr(), msg.dirfd, msg.pathname, msg.mode_, msg.uid, msg.gid) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Access(msg) } => {
            *ret = super::VMSpace::Access(taskId.Addr(), msg.pathName, msg.mode) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::FAccessAt(msg) } => {
            *ret = super::VMSpace::FAccessAt(taskId.Addr(), msg.dirfd, msg.pathname, msg.mode, msg.flags) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::Pause(_msg) } => {
            *ret = 0;
            return super::VMSpace::Pause();
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Socket(msg) } => {
            *ret = super::VMSpace::Socket(taskId.Addr(), msg.domain, msg.type_, msg.protocol) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SocketPair(msg) } => {
            *ret = super::VMSpace::SocketPair(taskId.Addr(), msg.domain, msg.type_, msg.protocol, msg.socketVect) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetSockName(msg) } => {
            *ret = super::VMSpace::GetSockName(taskId.Addr(), msg.sockfd, msg.addr, msg.addrlen) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetPeerName(msg) } => {
            *ret = super::VMSpace::GetPeerName(taskId.Addr(), msg.sockfd, msg.addr, msg.addrlen) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetSockOpt(msg) } => {
            *ret = super::VMSpace::GetSockOpt(taskId.Addr(), msg.sockfd, msg.level, msg.optname, msg.optval, msg.optlen) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SetSockOpt(msg) } => {
            *ret = super::VMSpace::SetSockOpt(taskId.Addr(), msg.sockfd, msg.level, msg.optname, msg.optval, msg.optlen) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Bind(msg) } => {
            *ret = super::VMSpace::Bind(taskId.Addr(), msg.sockfd, msg.addr, msg.addrlen, msg.umask) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Listen(msg) } => {
            *ret = super::VMSpace::Listen(taskId.Addr(), msg.sockfd, msg.backlog) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Shutdown(msg) } => {
            *ret = super::VMSpace::Shutdown(taskId.Addr(), msg.sockfd, msg.how) as u64
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::MinCore(msg) } => {
            *ret = super::VMSpace::MinCore(taskId.Addr(), msg.addr, msg.len, msg.vec) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetDents(msg) } => {
            *ret = super::VMSpace::GetDents(taskId.Addr(), msg.fd, msg.dirp, msg.count) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetDents64(msg) } => {
            *ret = super::VMSpace::GetDents64(taskId.Addr(), msg.fd, msg.dirp, msg.count) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetUid(_msg) } => {
            *ret = super::VMSpace::GetUid(taskId.Addr()) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetEUid(_msg) } => {
            *ret = super::VMSpace::GetEUid(taskId.Addr()) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetGid(_msg) } => {
            *ret = super::VMSpace::GetGid(taskId.Addr()) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SetGid(msg) } => {
            *ret = super::VMSpace::SetGid(taskId.Addr(), msg.gid) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetEGid(_msg) } => {
            *ret = super::VMSpace::GetEGid(taskId.Addr()) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetGroups(msg) } => {
            *ret = super::VMSpace::GetGroups(taskId.Addr(), msg.size, msg.list) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SetGroups(msg) } => {
            *ret = super::VMSpace::SetGroups(taskId.Addr(), msg.size, msg.list) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Sysinfo(msg) } => {
            *ret = super::VMSpace::Sysinfo(taskId.Addr(), msg.info) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetCwd(msg) } => {
            *ret = super::VMSpace::GetCwd(taskId.Addr(), msg.buf, msg.size) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::CreateMemfd(msg) } => {
            *ret = super::VMSpace::CreateMemfd(taskId.Addr(), msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Pipe2(msg) } => {
            *ret = super::VMSpace::Pipe2(taskId.Addr(), msg.fds, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SchedGetAffinity(msg) } => {
            *ret = super::VMSpace::SchedGetAffinity(taskId.Addr(), msg.pid, msg.cpuSetSize, msg.mask) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Getxattr(msg) } => {
            *ret = super::VMSpace::Getxattr(taskId.Addr(), msg.path, msg.name, msg.value, msg.size) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Lgetxattr(msg) } => {
            *ret = super::VMSpace::Lgetxattr(taskId.Addr(), msg.path, msg.name, msg.value, msg.size) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fgetxattr(msg) } => {
            *ret = super::VMSpace::Fgetxattr(taskId.Addr(), msg.fd, msg.name, msg.value, msg.size) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetRandom(msg) } => {
            *ret = super::VMS.lock().GetRandom(taskId.Addr(), msg.buf, msg.len, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Chdir(msg) } => {
            *ret = super::VMSpace::Chdir(taskId.Addr(), msg.path) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fchdir(msg) } => {
            *ret = super::VMSpace::Fchdir(taskId.Addr(), msg.fd) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fadvise(msg) } => {
            *ret = super::VMSpace::Fadvise(taskId.Addr(), msg.fd, msg.offset, msg.len, msg.advice) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Prctl(msg) } => {
            *ret = super::VMSpace::Prctl(taskId.Addr(), msg.option, msg.arg2, msg.arg3, msg.arg4, msg.arg5) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Mlock2(msg) } => {
            *ret = super::VMSpace::Mlock2(taskId.Addr(), msg.addr, msg.len, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::MUnlock(msg) } => {
            *ret = super::VMSpace::MUnlock(taskId.Addr(), msg.addr, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOSetup(msg) } => {
            *ret = super::VMSpace::IOSetup(taskId.Addr(), msg.nr_events, msg.ctx_idp) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOSubmit(msg) } => {
            *ret = super::VMSpace::IOSubmit(taskId.Addr(), msg.ctx_id, msg.nr, msg.iocbpp) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Rename(msg) } => {
            *ret = super::VMSpace::Rename(taskId.Addr(), msg.oldpath, msg.newpath) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Chown(msg) } => {
            *ret = super::VMSpace::Chown(taskId.Addr(), msg.pathname, msg.owner, msg.group) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::FChown(msg) } => {
            *ret = super::VMSpace::FChown(taskId.Addr(), msg.fd, msg.owner, msg.group) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::TimerFdCreate(msg) } => {
            *ret = super::VMSpace::TimerFdCreate(taskId.Addr(), msg.clockId, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::TimerFdSetTime(msg) } => {
            *ret = super::VMSpace::TimerFdSetTime(taskId.Addr(), msg.fd, msg.flags, msg.newValue, msg.oldValue) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::TimerFdGetTime(msg) } => {
            *ret = super::VMSpace::TimerFdGetTime(taskId.Addr(), msg.fd, msg.currVal) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SetHostName(msg) } => {
            *ret = super::VMSpace::SetHostName(taskId.Addr(), msg.name, msg.len) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Chmod(msg) } => {
            *ret = super::VMSpace::Chmod(taskId.Addr(), msg.pathname, msg.mode) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Fchmod(msg) } => {
            *ret = super::VMSpace::Fchmod(taskId.Addr(), msg.fd, msg.mode) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::SymLinkAt(msg) } => {
            *ret = super::VMSpace::SymLinkAt(taskId.Addr(), msg.oldpath, msg.newdirfd, msg.newpath) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Futimens(msg) } => {
            *ret = super::VMSpace::Futimens(taskId.Addr(), msg.fd, msg.times) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::GetStdfds(msg) } => {
            *ret = super::VMSpace::GetStdfds(taskId.Addr(), msg.addr) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::HostCPUInfo(msg) } => {
            *ret = super::VMSpace::HostCPUInfo(taskId.Addr(), msg.axArg, msg.cxArg, msg.addr) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::NonBlockingPoll(msg) } => {
            *ret = super::VMSpace::NonBlockingPoll(taskId.Addr(), msg.fd, msg.mask) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IORead(msg) } => {
            *ret = super::VMSpace::IORead(taskId.Addr(), msg.fd, msg.iovs, msg.iovcnt) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOWrite(msg) } => {
            *ret = super::VMSpace::IOWrite(taskId.Addr(), msg.fd, msg.iovs, msg.iovcnt) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOReadAt(msg) } => {
            *ret = super::VMSpace::IOReadAt(taskId.Addr(), msg.fd, msg.iovs, msg.iovcnt, msg.offset) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOWriteAt(msg) } => {
            *ret = super::VMSpace::IOWriteAt(taskId.Addr(), msg.fd, msg.iovs, msg.iovcnt, msg.offset) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOAppend(msg) } => {
            *ret = super::VMSpace::IOAppend(taskId.Addr(), msg.fd, msg.iovs, msg.iovcnt, msg.fileLenAddr) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOAccept(msg) } => {
            *ret = super::VMSpace::IOAccept(taskId.Addr(), msg.fd, msg.addr, msg.addrlen, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOConnect(msg) } => {
            *ret = super::VMSpace::IOConnect(taskId.Addr(), msg.fd, msg.addr, msg.addrlen) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IORecvMsg(msg) } => {
            *ret = super::VMSpace::IORecvMsg(taskId.Addr(), msg.fd, msg.msghdr, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::IOSendMsg(msg) } => {
            *ret = super::VMSpace::IOSendMsg(taskId.Addr(), msg.fd, msg.msghdr, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::NewTmpfsFile(msg) } => {
            *ret = super::VMSpace::NewTmpfsFile(taskId.Addr(), msg.typ, msg.addr) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::OpenFifo(msg) } => {
            *ret = super::VMSpace::OpenFifo(taskId.Addr(), msg.UID, msg.flags) as u64;
        }
        Event { taskId, interrupted: _, ref mut ret, msg: Msg::Statm(msg) } => {
            *ret = super::VMSpace::Statm(taskId.Addr(), msg.buf) as u64;
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::IoUringSetup(msg) } => {
            *ret = match URING_MGR.lock().Setup(msg.submission, msg.completion) {
                Ok(v) => v as u64,
                Err(Error::SysError(v)) => -v as i64 as u64,
                _ => panic!("UringMgr setup fail")
            }
        }
        Event { taskId: _, interrupted: _, ref mut ret, msg: Msg::IoUringEnter(msg) } => {
            *ret = match URING_MGR.lock().Enter(msg.toSubmit, msg.minComplete, msg.flags) {
                Ok(v) => v as u64,
                Err(Error::SysError(v)) => -v as i64 as u64,
                _ => panic!("UringMgr Enter fail")
            }
        }
        _ => {
            error!("unsupported qcall {:x}", eventAddr);
            panic!("unsupported qcall {:x}, event is {:?}", eventAddr, event);
        }
        //_  => info!("get nothing")
    }
    return QcallRet::Normal
}
