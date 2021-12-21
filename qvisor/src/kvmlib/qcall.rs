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

use core::sync::atomic::Ordering;

use super::qlib::{ShareSpace};
use super::qlib::common::*;
use super::qlib::qmsg::*;
use super::qlib::range::*;
use super::*;
use super::kvm_vcpu::KVMVcpu;

pub fn AQHostCall(msg: HostOutputMsg, _shareSpace: &ShareSpace) {
    let _l = super::GLOCK.lock();
    match msg {
        HostOutputMsg::Default => {
            panic!("AQHostCall Process get Default msg...");
        }
        HostOutputMsg::QCall(_addr) => {
            panic!("AQHostCall Process get Qcall msg...");
        }
        HostOutputMsg::WaitFDAsync(msg) => {
            let ret = super::VMSpace::WaitFD(msg.fd, msg.op, msg.mask);
            if ret < 0 {
                // ignore -9 EBADF, when change the Close to HCall, the waitfd is still async call,
                // there is chance that the WaitFd fired before close
                if ret != -9 {
                    error!("WaitFD fail err is {}, fd is {}, errorno is {}",
                        ret, msg.fd, ret);
                }
            }
        }
        HostOutputMsg::EventfdWriteAsync(msg) => {
            let ret = super::VMSpace::EventfdWrite(msg.fd);
            if ret < 0 {
                panic!("Eventfd write fail with error {}", ret)
            }
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

    pub fn LogFlush(&self, partial: bool) {
        let lock = self.logLock.try_lock();
        if lock.is_none() {
            return;
        }

        let logfd = self.logfd.load(Ordering::Relaxed);

        let mut cnt = 0;
        if partial {
            let (addr, len) = self.ConsumeAndGetAvailableWriteBuf(cnt);
            if len == 0 {
                return
            }

            /*if len > 16 * 1024 {
                len = 16 * 1024
            };*/

            let ret = unsafe {
                libc::write(logfd, addr as _, len)
            };
            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            cnt = ret as usize;
            self.ConsumeAndGetAvailableWriteBuf(cnt);
            return
        }

        loop {
            let (addr, len) = self.ConsumeAndGetAvailableWriteBuf(cnt);
            if len == 0 {
                return
            }

            let ret = unsafe {
                libc::write(logfd, addr as _, len)
            };
            if ret < 0 {
                panic!("log flush fail {}", ret);
            }

            cnt = ret as usize;
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
    pub fn qCall(&self, msg: &'static Msg) -> u64 {
        let mut ret = 0;

        match msg {
            Msg::LoadProcessKernel(msg) => {
                ret = super::VMS.lock().LoadProcessKernel(msg.processAddr, msg.len) as u64;
            },
            Msg::GetStdfds(msg) => {
                ret = super::VMSpace::GetStdfds(msg.addr) as u64;
            },
            Msg::CreateMemfd(msg) => {
                ret = super::VMSpace::CreateMemfd(msg.len) as u64;
            },
            //Syscall
            Msg::Fallocate(msg) => {
                ret = super::VMSpace::Fallocate(msg.fd, msg.mode, msg.offset, msg.len) as u64;
            },
            Msg::RenameAt(msg) => {
                ret = super::VMSpace::RenameAt(msg.olddirfd, msg.oldpath, msg.newdirfd, msg.newpath) as u64;
            },
            Msg::Ftruncate(msg) => {
                ret = super::VMSpace::Ftruncate(msg.fd, msg.len) as u64;
            },
            Msg::Seek(msg) => {
                ret = super::VMSpace::Seek(msg.fd, msg.offset, msg.whence) as u64;
            },
            Msg::ReadLinkAt(msg) => {
                ret = super::VMSpace::ReadLinkAt(msg.dirfd, msg.path, msg.buf, msg.bufsize) as u64;
            },
            Msg::GetTimeOfDay(msg) => {
                ret = super::VMSpace::GetTimeOfDay(msg.tv, msg.tz) as u64;
            },
            Msg::IoCtl(msg) => {
                ret = super::VMSpace::IoCtl(msg.fd, msg.cmd, msg.argp) as u64;
            },
            Msg::Fcntl(msg) => {
                ret = super::VMSpace::Fcntl(msg.fd, msg.cmd, msg.arg) as u64;
            },
            Msg::Close(msg) => {
                ret = super::VMSpace::Close(msg.fd) as u64;
            },
            Msg::Getxattr(msg) => {
                ret = super::VMSpace::Getxattr(msg.path, msg.name, msg.value, msg.size) as u64;
            },
            Msg::Lgetxattr(msg) => {
                ret = super::VMSpace::Lgetxattr(msg.path, msg.name, msg.value, msg.size) as u64;
            },
            Msg::Fgetxattr(msg) => {
                ret = super::VMSpace::Fgetxattr(msg.fd, msg.name, msg.value, msg.size) as u64;
            },
            Msg::Fstat(msg) => {
                ret = super::VMSpace::Fstat(msg.fd, msg.buff) as u64;
            },
            Msg::BatchFstatat(msg) => {
                ret = super::VMSpace::BatchFstatat(msg.addr, msg.count) as u64;
            },
            Msg::Fstatat(msg) => {
                ret = super::VMSpace::Fstatat(msg.dirfd, msg.pathname, msg.buff, msg.flags) as u64;
            },
            Msg::Fstatfs(msg) => {
                ret = super::VMSpace::Fstatfs(msg.fd, msg.buf) as u64;
            },
            Msg::GetDents64(msg) => {
                ret = super::VMSpace::GetDents64(msg.fd, msg.dirp, msg.count) as u64;
            },

            Msg::TryOpenAt(msg) => {
                ret = super::VMSpace::TryOpenAt(msg.dirfd, msg.name, msg.addr) as u64;
            },
            Msg::CreateAt(msg) => {
                ret = super::VMSpace::CreateAt(msg.dirfd, msg.pathName, msg.flags, msg.mode, msg.uid, msg.gid, msg.fstatAddr) as u64;
            },
            Msg::Unlinkat(msg) => {
                ret = super::VMSpace::Unlinkat(msg.dirfd, msg.pathname, msg.flags) as u64;
            },
            Msg::Mkdirat(msg) => {
                ret = super::VMSpace::Mkdirat(msg.dirfd, msg.pathname, msg.mode_, msg.uid, msg.gid) as u64;
            },
            Msg::SysSync(_msg) => {
                ret = super::VMSpace::SysSync() as u64;
            },
            Msg::SyncFs(msg) => {
                ret = super::VMSpace::SyncFs(msg.fd) as u64;
            },
            Msg::SyncFileRange(msg) => {
                ret = super::VMSpace::SyncFileRange(msg.fd, msg.offset, msg.nbytes, msg.flags) as u64;
            },
            Msg::FSync(msg) => {
                ret = super::VMSpace::FSync(msg.fd) as u64;
            },
            Msg::MSync(msg) => {
                ret = super::VMSpace::MSync(msg.addr, msg.len, msg.flags) as u64;
            },
            Msg::MAdvise(msg) => {
                ret = super::VMSpace::MAdvise(msg.addr, msg.len, msg.advise) as u64;
            },
            Msg::FDataSync(msg) => {
                ret = super::VMSpace::FDataSync(msg.fd) as u64;
            },
            Msg::FAccessAt(msg) => {
                ret = super::VMSpace::FAccessAt(msg.dirfd, msg.pathname, msg.mode, msg.flags) as u64;
            },
            Msg::Socket(msg) => {
                ret = super::VMSpace::Socket(msg.domain, msg.type_, msg.protocol) as u64;
            },
            Msg::SocketPair(msg) => {
                ret = super::VMSpace::SocketPair(msg.domain, msg.type_, msg.protocol, msg.socketVect) as u64;
            },
            Msg::GetPeerName(msg) => {
                ret = super::VMSpace::GetPeerName(msg.sockfd, msg.addr, msg.addrlen) as u64;
            },
            Msg::GetSockName(msg) => {
                ret = super::VMSpace::GetSockName(msg.sockfd, msg.addr, msg.addrlen) as u64;
            },
            Msg::GetSockOpt(msg) => {
                ret = super::VMSpace::GetSockOpt(msg.sockfd, msg.level, msg.optname, msg.optval, msg.optlen) as u64;
            },
            Msg::SetSockOpt(msg) => {
                ret = super::VMSpace::SetSockOpt(msg.sockfd, msg.level, msg.optname, msg.optval, msg.optlen) as u64;
            },
            Msg::Bind(msg) => {
                ret = super::VMSpace::Bind(msg.sockfd, msg.addr, msg.addrlen, msg.umask) as u64;
            },
            Msg::Listen(msg) => {
                ret = super::VMSpace::Listen(msg.sockfd, msg.backlog, msg.block) as u64;
            },
            Msg::Shutdown(msg) => {
                ret = super::VMSpace::Shutdown(msg.sockfd, msg.how) as u64
            },
            Msg::SchedGetAffinity(msg) => {
                ret = super::VMSpace::SchedGetAffinity(msg.pid, msg.cpuSetSize, msg.mask) as u64;
            },
            Msg::GetRandom(msg) => {
                ret = super::VMS.lock().GetRandom(msg.buf, msg.len, msg.flags) as u64;
            },
            Msg::Fchdir(msg) => {
                ret = super::VMSpace::Fchdir(msg.fd) as u64;
            },
            Msg::Fadvise(msg) => {
                ret = super::VMSpace::Fadvise(msg.fd, msg.offset, msg.len, msg.advice) as u64;
            },
            Msg::Mlock2(msg) => {
                ret = super::VMSpace::Mlock2(msg.addr, msg.len, msg.flags) as u64;
            },
            Msg::MUnlock(msg) => {
                ret = super::VMSpace::MUnlock(msg.addr, msg.len) as u64;
            },
            Msg::Chown(msg) => {
                ret = super::VMSpace::Chown(msg.pathname, msg.owner, msg.group) as u64;
            },
            Msg::FChown(msg) => {
                ret = super::VMSpace::FChown(msg.fd, msg.owner, msg.group) as u64;
            },
            Msg::Chmod(_msg) => {
                panic!("Panic not implemented")
            },
            Msg::Fchmod(msg) => {
                ret = super::VMSpace::Fchmod(msg.fd, msg.mode) as u64;
            },
            Msg::SymLinkAt(msg) => {
                ret = super::VMSpace::SymLinkAt(msg.oldpath, msg.newdirfd, msg.newpath) as u64;
            },
            Msg::Futimens(msg) => {
                ret = super::VMSpace::Futimens(msg.fd, msg.times) as u64;
            },

            Msg::IORead(msg) => {
                ret = super::VMSpace::IORead(msg.fd, msg.iovs, msg.iovcnt) as u64;
            },
            Msg::IOTTYRead(msg) => {
                ret = super::VMSpace::IOTTYRead(msg.fd, msg.iovs, msg.iovcnt) as u64;
            },
            Msg::IOWrite(msg) => {
                ret = super::VMSpace::IOWrite(msg.fd, msg.iovs, msg.iovcnt) as u64;
            },
            Msg::IOReadAt(msg) => {
                ret = super::VMSpace::IOReadAt(msg.fd, msg.iovs, msg.iovcnt, msg.offset) as u64;
            },
            Msg::IOWriteAt(msg) => {
                ret = super::VMSpace::IOWriteAt(msg.fd, msg.iovs, msg.iovcnt, msg.offset) as u64;
            },
            Msg::IOAppend(msg) => {
                ret = super::VMSpace::IOAppend(msg.fd, msg.iovs, msg.iovcnt, msg.fileLenAddr) as u64;
            },
            Msg::IOAccept(msg) => {
                ret = super::VMSpace::IOAccept(msg.fd, msg.addr, msg.addrlen, msg.flags) as u64;
            },
            Msg::IOConnect(msg) => {
                ret = super::VMSpace::IOConnect(msg.fd, msg.addr, msg.addrlen) as u64;
            },
            Msg::IORecvMsg(msg) => {
                ret = super::VMSpace::IORecvMsg(msg.fd, msg.msghdr, msg.flags) as u64;
            },
            Msg::IOSendMsg(msg) => {
                ret = super::VMSpace::IOSendMsg(msg.fd, msg.msghdr, msg.flags) as u64;
            },
            Msg::MMapFile(msg) => {
                ret = match super::PMA_KEEPER.MapFile(msg.len, msg.prot, msg.fd, msg.offset) {
                    Err(Error::SysError(e)) => -e as u64,
                    Ok(phyAddr) => phyAddr,
                    Err(err) => panic!("MMapFile: unexpected error {:?}", err),
                }
            },
            Msg::MUnmap(msg) => {
                match super::PMA_KEEPER.Unmap(&Range::New(msg.addr, msg.len)) {
                    Ok(()) => {},
                    Err(err) => panic!("MUnmap: unexpected error {:?}", err),
                }
            },
            Msg::NonBlockingPoll(msg) => {
                ret = super::VMSpace::NonBlockingPoll(msg.fd, msg.mask) as u64;
            },
            Msg::NewTmpfsFile(msg) => {
                ret = super::VMSpace::NewTmpfsFile(msg.typ, msg.addr) as u64;
            },
            Msg::IoUringSetup(msg) => {
                ret = match URING_MGR.lock().Setup(msg.idx, msg.submission, msg.completion) {
                    Ok(v) => v as u64,
                    Err(Error::SysError(v)) => -v as i64 as u64,
                    _ => panic!("UringMgr setup fail")
                }
            },
            Msg::IoUringEnter(msg) => {
                ret = match URING_MGR.lock().Enter(msg.idx, msg.toSubmit, msg.minComplete, msg.flags) {
                    Ok(v) => v as u64,
                    Err(Error::SysError(v)) => -v as i64 as u64,
                    _ => panic!("UringMgr Enter fail")
                }
            },
            Msg::Statm(msg) => {
                ret = super::VMSpace::Statm(msg.buf) as u64;
            },
            Msg::NewFd(msg) => {
                ret = super::VMSpace::NewFd(msg.fd) as u64;
            },
            Msg::HostEpollWaitProcess(msg) => {
                ret = super::VMSpace::HostEpollWaitProcess(msg.addr, msg.count) as u64;
            },
            Msg::VcpuWait(msg) => {
                ret = self.VcpuWait(msg.addr, msg.count) as u64;
            },
            Msg::EventfdWrite(msg) => {
                ret = super::VMSpace::EventfdWrite(msg.fd) as u64;
            },
            Msg::ReadControlMsg(msg) => {
                ret = super::VMSpace::ReadControlMsg(msg.fd, msg.addr, msg.len) as u64;
            },
            Msg::WriteControlMsgResp(msg) => {
                ret = super::VMSpace::WriteControlMsgResp(msg.fd, msg.addr, msg.len) as u64;
            },
        };

        return ret
    }

}

