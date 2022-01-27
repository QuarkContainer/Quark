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

use alloc::sync::Arc;

use super::super::*;
use super::super::common::*;
use super::super::qmsg;
use super::super::config::*;
use super::super::qmsg::*;
use super::super::linux_def::*;
use super::super::socket_buf::*;
use super::guestfdnotifier::*;
use super::task::*;
use super::asm::*;
use super::taskMgr;

extern "C" {
    pub fn rdtsc() -> i64;
}

pub struct HostSpace {}

impl HostSpace {
    pub fn WakeupVcpu(vcpuId: u64) {
        HyperCall64(HYPERCALL_WAKEUP_VCPU, vcpuId, 0, 0);
    }

    pub fn IOWait() {
        HyperCall64(HYPERCALL_IOWAIT, 0, 0, 0);
    }

    pub fn Hlt() {
        HyperCall64(HYPERCALL_HLT, 0, 0, 0);
    }

    pub fn UringWake(idx: usize, minCompleted: u64) {
        HyperCall64(HYPERCALL_URING_WAKE, idx as u64, minCompleted, 0);
    }

    pub fn LoadProcessKernel(processAddr: u64, len: usize) -> i64 {
        let mut msg = Msg::LoadProcessKernel(LoadProcessKernel {
            processAddr: processAddr,
            len: len,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn CreateMemfd(len: i64) -> i64 {
        let mut msg = Msg::CreateMemfd(CreateMemfd {
            len: len,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fallocate(fd: i32, mode: i32, offset: i64, len: i64) -> i64 {
        let mut msg = Msg::Fallocate(Fallocate {
            fd,
            mode,
            offset,
            len,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn EventfdWrite(fd: i32) -> i64 {
        let mut msg = Msg::EventfdWrite(EventfdWrite {
            fd,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn RenameAt(olddirfd: i32, oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let mut msg = Msg::RenameAt(RenameAt {
            olddirfd,
            oldpath,
            newdirfd,
            newpath,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Ftruncate(fd: i32, len: i64) -> i64 {
        let mut msg = Msg::Ftruncate(Ftruncate {
            fd,
            len,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Rdtsc() -> i64 {
        let mut msg = Msg::Rdtsc(Rdtsc {});

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SetTscOffset(offset: i64) -> i64 {
        let mut msg = Msg::SetTscOffset(SetTscOffset {
            offset: offset
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn IORead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let mut msg = Msg::IORead(IORead {
            fd,
            iovs,
            iovcnt,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IOTTYRead(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let mut msg = Msg::IOTTYRead(IOTTYRead {
            fd,
            iovs,
            iovcnt,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IOWrite(fd: i32, iovs: u64, iovcnt: i32) -> i64 {
        let mut msg = Msg::IOWrite(IOWrite {
            fd,
            iovs,
            iovcnt,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IOReadAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let mut msg = Msg::IOReadAt(IOReadAt {
            fd,
            iovs,
            iovcnt,
            offset,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IOWriteAt(fd: i32, iovs: u64, iovcnt: i32, offset: u64) -> i64 {
        let mut msg = Msg::IOWriteAt(IOWriteAt {
            fd,
            iovs,
            iovcnt,
            offset,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IOAppend(fd: i32, iovs: u64, iovcnt: i32) -> (i64, i64) {
        let mut fileLen : i64 = 0;
        let mut msg = Msg::IOAppend(IOAppend {
            fd,
            iovs,
            iovcnt,
            fileLenAddr : &mut fileLen as * mut _ as u64,
        });

        let ret = HostSpace::Call(&mut msg, false) as i64;
        if ret < 0 {
            return (ret, 0)
        }

        return (ret, fileLen)
    }

    pub fn IOAccept(fd: i32, addr: u64, addrlen: u64) -> i64 {
        let mut msg = Msg::IOAccept(IOAccept {
            fd,
            addr,
            addrlen,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn IOConnect(fd: i32, addr: u64, addrlen: u32) -> i64 {
        let mut msg = Msg::IOConnect(IOConnect {
            fd,
            addr,
            addrlen,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn PostRDMAConnect(task: &Task, fd: i32, socketBuf: Arc<SocketBuff>) -> i64 {
        let mut msg = PostRDMAConnect {
            fd,
            socketBuf,
            taskId: task.GetTaskId(),
            ret: 0
        };

        let addr = &mut msg as * mut _ as u64;
        let om = HostOutputMsg::PostRDMAConnect(addr);

        super::SHARESPACE.AQCall(&om);
        taskMgr::Wait();
        return msg.ret;
    }

    pub fn IORecvMsg(fd: i32, msghdr: u64, flags: i32, blocking: bool) -> i64 {
        let mut msg = Msg::IORecvMsg(IORecvMsg {
            fd,
            msghdr,
            flags,
            blocking,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IOSendMsg(fd: i32, msghdr: u64, flags: i32, blocking: bool) -> i64 {
        let mut msg = Msg::IOSendMsg(IOSendMsg {
            fd,
            msghdr,
            flags,
            blocking,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetTimeOfDay(tv: u64, tz: u64) -> i64 {
        let mut msg = Msg::GetTimeOfDay(GetTimeOfDay {
            tv,
            tz,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn ReadLinkAt(dirfd: i32, path: u64, buf: u64, bufsize: u64) -> i64 {
        let mut msg = Msg::ReadLinkAt(ReadLinkAt {
            dirfd,
            path,
            buf,
            bufsize,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fcntl(fd: i32, cmd: i32, arg: u64) -> i64 {
        let mut msg = Msg::Fcntl(Fcntl {
            fd,
            cmd,
            arg,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IoCtl(fd: i32, cmd: u64, argp: u64) -> i64 {
        let mut msg = Msg::IoCtl(IoCtl {
            fd,
            cmd,
            argp,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fstatfs(fd: i32, buf: u64) -> i64 {
        let mut msg = Msg::Fstatfs(Fstatfs {
            fd,
            buf,
        });

        return HostSpace::Call(&mut msg, false) as i64
    }

    pub fn NewSocket(fd: i32) -> i64 {
        let mut msg = Msg::NewSocket(NewSocket {
            fd
        });

        return HostSpace::HCall(&mut msg, true) as i64
    }

    pub fn FAccessAt(dirfd: i32, pathname: u64, mode: i32, flags: i32) -> i64 {
        let mut msg = Msg::FAccessAt(FAccessAt {
            dirfd,
            pathname,
            mode,
            flags
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fstat(fd: i32, buff: u64) -> i64 {
        let mut msg = Msg::Fstat(Fstat {
            fd,
            buff,
        });

        return Self::HCall(&mut msg, false) as i64;
        //return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn BatchFstatat(filetypes: &mut [FileType]) -> i64 {
        let addr = &filetypes[0] as * const _ as u64;
        let count = filetypes.len();
        let mut msg = Msg::BatchFstatat(BatchFstatat {
            addr,
            count
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Fstatat(dirfd: i32, pathname: u64, buff: u64, flags: i32) -> i64 {
        let mut msg = Msg::Fstatat(Fstatat {
            dirfd,
            pathname,
            buff,
            flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Getxattr(path: u64, name: u64, value: u64, size: u64) -> i64 {
        let mut msg = Msg::Getxattr(Getxattr {
            path,
            name,
            value,
            size,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Lgetxattr(path: u64, name: u64, value: u64, size: u64) -> i64 {
        let mut msg = Msg::Lgetxattr(Lgetxattr {
            path,
            name,
            value,
            size,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fgetxattr(fd: i32, name: u64, value: u64, size: u64) -> i64 {
        let mut msg = Msg::Fgetxattr(Fgetxattr {
            fd,
            name,
            value,
            size,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Unlinkat(dirfd: i32, pathname: u64, flags: i32) -> i64 {
        let mut msg = Msg::Unlinkat(Unlinkat {
            dirfd,
            pathname,
            flags
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Mkdirat(dirfd: i32, pathname: u64, mode_: u32, uid: u32, gid: u32) -> i64 {
        let mut msg = Msg::Mkdirat(Mkdirat {
            dirfd,
            pathname,
            mode_,
            uid,
            gid,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SysSync() -> i64 {
        let mut msg = Msg::SysSync(SysSync {});

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn SyncFs(fd: i32) -> i64 {
        let mut msg = Msg::SyncFs(SyncFs {
            fd,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn SyncFileRange(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i64 {
        let mut msg = Msg::SyncFileRange(SyncFileRange {
            fd,
            offset,
            nbytes,
            flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn FSync(fd: i32) -> i64 {
        let mut msg = Msg::FSync(FSync {
            fd,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn MSync(addr: u64, len: usize, flags: i32) -> i64 {
        let mut msg = Msg::MSync(MSync {
            addr,
            len,
            flags,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn Madvise(addr: u64, len: usize, advise: i32) -> i64 {
        let mut msg = Msg::MAdvise(MAdvise {
            addr,
            len,
            advise,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn FDataSync(fd: i32) -> i64 {
        let mut msg = Msg::FDataSync(FDataSync {
            fd,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Seek(fd: i32, offset: i64, whence: i32) -> i64 {
        let mut msg = Msg::Seek(Seek {
            fd,
            offset,
            whence,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetDents64(fd: i32, dirp: u64, count: u32) -> i64 {
        let mut msg = Msg::GetDents64(GetDents64 {
            fd,
            dirp,
            count,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn GetRandom(buf: u64, len: u64, flags: u32) -> i64 {
        let mut msg = Msg::GetRandom(GetRandom {
            buf,
            len,
            flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Statm(statm: &mut StatmInfo) -> i64 {
        let mut msg = Msg::Statm(Statm {
            buf: statm as * const _ as u64
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Socket(domain: i32, type_: i32, protocol: i32) -> i64 {
        let mut msg = Msg::Socket(Socket {
            domain,
            type_,
            protocol,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetSockName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let mut msg = Msg::GetSockName(GetSockName {
            sockfd,
            addr,
            addrlen,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetPeerName(sockfd: i32, addr: u64, addrlen: u64) -> i64 {
        let mut msg = Msg::GetPeerName(GetPeerName {
            sockfd,
            addr,
            addrlen,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u64) -> i64 {
        let mut msg = Msg::GetSockOpt(GetSockOpt {
            sockfd,
            level,
            optname,
            optval,
            optlen,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn SetSockOpt(sockfd: i32, level: i32, optname: i32, optval: u64, optlen: u32) -> i64 {
        let mut msg = Msg::SetSockOpt(SetSockOpt {
            sockfd,
            level,
            optname,
            optval,
            optlen,
        });

        //return Self::HCall(&mut msg) as i64;
        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Bind(sockfd: i32, addr: u64, addrlen: u32, umask: u32) -> i64 {
        let mut msg = Msg::IOBind(IOBind {
            sockfd,
            addr,
            addrlen,
            umask,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Listen(sockfd: i32, backlog: i32, block: bool) -> i64 {
        let mut msg = Msg::IOListen(IOListen {
            sockfd,
            backlog,
            block,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn RDMAListen(sockfd: i32, backlog: i32, block: bool, acceptQueue: AcceptQueue) -> i64 {
        let mut msg = Msg::RDMAListen(RDMAListen {
            sockfd,
            backlog,
            block,
            acceptQueue,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn RDMANotify(sockfd: i32, typ: RDMANotifyType) -> i64 {
        let mut msg = Msg::RDMANotify(RDMANotify {
            sockfd,
            typ
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Shutdown(sockfd: i32, how: i32) -> i64 {
        let mut msg = Msg::IOShutdown(IOShutdown {
            sockfd,
            how,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn ExitVM(exitCode: i32) {
        HyperCall64(HYPERCALL_EXIT_VM, exitCode as u64, 0, 0);
        //Self::AQCall(qmsg::HostOutputMsg::ExitVM(exitCode));
    }

    pub fn Panic(str: &str) {
        let msg = Print {
            level: DebugLevel::Error,
            str: str,
        };

        HyperCall64(HYPERCALL_PANIC, &msg as *const _ as u64, 0, 0);
    }

    pub fn TryOpenAt(dirfd: i32, name: u64, addr: u64) -> i64 {
        let mut msg = Msg::TryOpenAt(TryOpenAt {
            dirfd: dirfd,
            name: name,
            addr: addr,
        });

        let ret = Self::HCall(&mut msg, false) as i64;
        return ret;
        //return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn CreateAt(dirfd: i32, pathName: u64, flags: i32, mode: i32, uid: u32, gid: u32, fstatAddr: u64) -> i64 {
        let mut msg = Msg::CreateAt(CreateAt {
            dirfd,
            pathName,
            flags,
            mode,
            uid,
            gid,
            fstatAddr
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn SchedGetAffinity(pid: i32, cpuSetSize: u64, mask: u64) -> i64 {
        let mut msg = Msg::SchedGetAffinity(SchedGetAffinity {
            pid,
            cpuSetSize,
            mask,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fchdir(fd: i32) -> i64 {
        let mut msg = Msg::Fchdir(Fchdir {
            fd,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fadvise(fd: i32, offset: u64, len: u64, advice: i32) -> i64 {
        let mut msg = Msg::Fadvise(Fadvise {
            fd,
            offset,
            len,
            advice,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Mlock2(addr: u64, len: u64, flags: u32) -> i64 {
        let mut msg = Msg::Mlock2(Mlock2 {
            addr,
            len,
            flags,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn MUnlock(addr: u64, len: u64) -> i64 {
        let mut msg = Msg::MUnlock(MUnlock {
            addr,
            len,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn NonBlockingPoll(fd: i32, mask: EventMask) -> i64 {
        let mut msg = Msg::NonBlockingPoll(NonBlockingPoll {
            fd,
            mask,
        });

        //return HostSpace::Call(&mut msg, false) as i64;
        let ret = Self::HCall(&mut msg, false) as i64;
        //error!("NonBlockingPoll2 fd is {} ret is {}", fd, ret);

        return ret;
    }

    pub fn HostEpollWaitProcess() -> i64 {
        let mut msg = Msg::HostEpollWaitProcess(HostEpollWaitProcess {});

        let ret = Self::HCall(&mut msg, false) as i64;
        return ret;
    }

    pub fn VcpuWait() -> i64 {
        let mut ret : i64 = 0;
        HyperCall64(HYPERCALL_VCPU_WAIT, 0, 0, &mut ret as * mut _ as u64);
        return ret as i64
    }

    pub fn NewTmpfsFile(typ: TmpfsFileType, addr: u64) -> i64 {
        let mut msg = Msg::NewTmpfsFile(NewTmpfsFile {
            typ,
            addr,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn IoUringEnter(idx: usize, toSubmit: u32, minComplete: u32, flags: u32) -> i64 {
        let mut msg = Msg::IoUringEnter(IoUringEnter {
            idx,
            toSubmit,
            minComplete,
            flags,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Chown(pathname: u64, owner: u32, group: u32) -> i64 {
        let mut msg = Msg::Chown(Chown {
            pathname,
            owner,
            group,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn FChown(fd: i32, owner: u32, group: u32) -> i64 {
        let mut msg = Msg::FChown(FChown {
            fd,
            owner,
            group,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Chmod(pathname: u64, mode: u32) -> i64 {
        let mut msg = Msg::Chmod(Chmod {
            pathname,
            mode,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn Fchmod(fd: i32, mode: u32) -> i64 {
        let mut msg = Msg::Fchmod(Fchmod {
            fd,
            mode,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn SymLinkAt(oldpath: u64, newdirfd: i32, newpath: u64) -> i64 {
        let mut msg = Msg::SymLinkAt(SymLinkAt {
            oldpath,
            newdirfd,
            newpath
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }

    pub fn ReadControlMsg(fd: i32, addr: u64, len: usize) -> i64 {
        let mut msg = Msg::ReadControlMsg(ReadControlMsg {
            fd,
            addr,
            len
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn WriteControlMsgResp(fd: i32, addr: u64, len: usize) -> i64 {
        let mut msg = Msg::WriteControlMsgResp(WriteControlMsgResp {
            fd: fd,
            addr: addr,
            len: len,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn UpdateWaitInfo(fd: i32, waitinfo: FdWaitInfo) -> i64 {
        let mut msg = Msg::UpdateWaitInfo(UpdateWaitInfo {
            fd: fd,
            waitinfo: waitinfo,
        });

        return HostSpace::HCall(&mut msg, false) as i64;
    }


    pub fn Futimens(fd: i32, times: u64) -> i64 {
        let mut msg = Msg::Futimens(Futimens {
            fd,
            times,
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn GetStdfds(addr: u64) -> i64 {
        let mut msg = Msg::GetStdfds(GetStdfds {
            addr
        });

        return HostSpace::Call(&mut msg, false) as i64;
    }

    pub fn MMapFile(len: u64, fd: i32, offset: u64, prot: i32) -> i64 {
        assert!(len % MemoryDef::PMD_SIZE == 0, "offset is {:x}, len is {:x}", offset, len);
        assert!(offset % MemoryDef::PMD_SIZE == 0, "offset is {:x}, len is {:x}", offset, len);
        let mut msg = Msg::MMapFile(MMapFile {
            len,
            fd,
            offset,
            prot,
        });

        let res = HostSpace::HCall(&mut msg, true) as i64;
        assert!(res as u64 % MemoryDef::PMD_SIZE == 0, "res {:x}", res);
        return res;
    }

    pub fn MUnmap(addr: u64, len: u64) {
        assert!(addr % MemoryDef::PMD_SIZE == 0, "addr is {:x}, len is {:x}", addr, len);
        assert!(len % MemoryDef::PMD_SIZE == 0, "addr is {:x}, len is {:x}", addr, len);
        let mut msg = Msg::MUnmap(qmsg::qcall::MUnmap {
            addr,
            len,
        });

        HostSpace::HCall(&mut msg, true);
    }

    pub fn WaitFDAsync(fd: i32, op: u32, mask: EventMask) {
        let msg = HostOutputMsg::WaitFDAsync(WaitFDAsync {
            fd,
            op,
            mask
        });

        super::SHARESPACE.AQCall(&msg);
    }

    pub fn EventfdWriteAsync(fd: i32) {
        let msg = HostOutputMsg::EventfdWriteAsync(EventfdWriteAsync {
            fd,
        });

        super::SHARESPACE.AQCall(&msg);
    }

    pub fn SyncPrint(level: DebugLevel, str: &str) {
        let msg = Print {
            level,
            str,
        };

        HyperCall64(HYPERCALL_PRINT, &msg as *const _ as u64, 0, 0);
    }

    pub fn Kprint(str: &str) {
        let bytes = str.as_bytes();
        let trigger = super::SHARESPACE.Log(bytes);
        if trigger {
            super::IOURING.LogFlush();
        }
    }

    #[inline(always)]
    pub fn KernelMsg(id: u64, val1: u64, val2: u64) {
        HyperCall64(HYPERCALL_MSG, id, val1, val2)
    }

    pub fn KernelOOM(size: u64, alignment: u64) {
        HyperCall64(HYPERCALL_OOM, size, alignment, 0)
    }

    pub fn KernelGetTime(clockId: i32) -> Result<i64> {
        let call = GetTimeCall {
            clockId,
            ..Default::default()
        };

        let addr = &call as *const _ as u64;
        HyperCall64(HYPERCALL_GETTIME, addr, 0, 0);

        use self::common::*;

        if call.res < 0 {
            return Err(Error::SysError(-call.res as i32))
        }

        return Ok(call.res);
    }

    pub fn KernelVcpuFreq() -> i64 {
        let call = VcpuFeq::default();

        let addr = &call as *const _ as u64;
        HyperCall64(HYPERCALL_VCPU_FREQ, addr, 0, 0);

        return call.res;
    }

    pub fn VcpuYield() {
        HyperCall64(HYPERCALL_VCPU_YIELD, 0, 0, 0);
    }

    pub fn VcpuDebug() {
        HyperCall64(HYPERCALL_VCPU_DEBUG, 0, 0, 0);
    }

    pub fn VcpuPrint() {
        HyperCall64(HYPERCALL_VCPU_PRINT, 0, 0, 0);
    }
}

pub fn GetSockOptI32(sockfd: i32, level: i32, optname: i32) -> Result<i32> {
    let mut val: i32 = 0;
    let len: i32 = 4;
    let res = HostSpace::GetSockOpt(sockfd,
                                    level,
                                    optname,
                                    &mut val as *mut i32 as u64,
                                    &len as *const i32 as u64) as i32;

    if res < 0 {
        return Err(Error::SysError(-res as i32));
    }

    return Ok(val);
}
