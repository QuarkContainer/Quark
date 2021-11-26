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

use super::super::task_mgr::*;
use super::super::linux_def::*;
use super::super::config::*;

#[repr(align(128))]
#[derive(Clone, Debug)]
pub enum Msg {
    //Qcall
    LoadProcessKernel(LoadProcessKernel),
    ControlMsgCall(ControlMsgCall),
    ControlMsgRet(ControlMsgRet),
    GetStdfds(GetStdfds),
    CreateMemfd(CreateMemfd),

    //Syscall
    Fallocate(Fallocate),
    RenameAt(RenameAt),
    Ftruncate(Ftruncate),
    Seek(Seek),
    ReadLinkAt(ReadLinkAt),
    GetTimeOfDay(GetTimeOfDay),
    IoCtl(IoCtl),
    Fcntl(Fcntl),
    Close(Close),

    Getxattr(Getxattr),
    Lgetxattr(Lgetxattr),
    Fgetxattr(Fgetxattr),
    Fstat(Fstat),
    BatchFstatat(BatchFstatat),
    Fstatat(Fstatat),
    Fstatfs(Fstatfs),
    GetDents64(GetDents64),

    TryOpenAt(TryOpenAt),
    CreateAt(CreateAt),
    Unlinkat(Unlinkat),
    Mkdirat(Mkdirat),
    SysSync(SysSync),
    SyncFs(SyncFs),
    SyncFileRange(SyncFileRange),
    FSync(FSync),
    MSync(MSync),
    MAdvise(MAdvise),
    FDataSync(FDataSync),
    FAccessAt(FAccessAt),

    Socket(Socket),
    SocketPair(SocketPair),
    GetPeerName(GetPeerName),
    GetSockName(GetSockName),
    GetSockOpt(GetSockOpt),
    SetSockOpt(SetSockOpt),
    Bind(Bind),
    Listen(Listen),
    Shutdown(Shutdown),

    SchedGetAffinity(SchedGetAffinity),
    GetRandom(GetRandom),
    Fchdir(Fchdir),
    Fadvise(Fadvise),
    Mlock2(Mlock2),
    MUnlock(MUnlock),
    Chown(Chown),
    FChown(FChown),
    Chmod(Chmod),
    Fchmod(Fchmod),
    SymLinkAt(SymLinkAt),
    Futimens(Futimens),

    IORead(IORead),
    IOTTYRead(IOTTYRead),
    IOWrite(IOWrite),
    IOReadAt(IOReadAt),
    IOWriteAt(IOWriteAt),
    IOAppend(IOAppend),
    IOAccept(IOAccept),
    IOConnect(IOConnect),
    IORecvMsg(IORecvMsg),
    IOSendMsg(IOSendMsg),
    MMapFile(MMapFile),
    MUnmap(MUnmap),
    NonBlockingPoll(NonBlockingPoll),
    NewTmpfsFile(NewTmpfsFile),
    IoUringSetup(IoUringSetup),
    IoUringEnter(IoUringEnter),
    Statm(Statm),
    NewFd(NewFd),
    HostEpollWaitProcess(HostEpollWaitProcess),
    WaitFD(WaitFD),
    VcpuWait(VcpuWait),
}

#[derive(Clone, Default, Debug)]
pub struct IoUringSetup {
    pub idx: usize,
    pub submission: u64,
    pub completion: u64
}

#[derive(Clone, Default, Debug)]
pub struct IoUringRegister {
    pub fd: i32,
    pub Opcode: u32,
    pub arg: u64,
    pub nrArgs: u32
}

#[derive(Clone, Default, Debug, Copy)]
pub struct IoUringEnter {
    pub idx: usize,
    pub toSubmit: u32,
    pub minComplete: u32,
    pub flags: u32,
}

#[derive(Clone, Default, Debug)]
pub struct InitPara {
    pub KernelPageTableRoot: u64,
    pub pagePoolBase: u64,
    pub pageCount: u32,
    pub next: u32,
}

#[derive(Clone, Default, Debug)]
pub struct MMapFile {
    pub len: u64,
    pub fd: i32,
    pub offset: u64,
    pub prot: i32,
}


#[derive(Clone, Default, Debug)]
pub struct MUnmap {
    pub addr: u64,
    pub len: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fallocate {
    pub fd: i32,
    pub mode: i32,
    pub offset: i64,
    pub len: i64,
}

// get vss/rss from /proc/self/statm
#[derive(Clone, Default, Debug)]
pub struct StatmInfo {
    pub vss: u64,
    pub rss: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Statm {
    pub buf: u64,
}

#[derive(Clone, Default, Debug)]
pub struct CreateMemfd {
    pub len: i64,
}

#[derive(Clone, Default, Debug)]
pub struct RenameAt {
    pub olddirfd: i32,
    pub oldpath: u64,
    pub newdirfd: i32,
    pub newpath: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Ftruncate {
    pub fd: i32,
    pub len: i64,
}

#[derive(Clone, Default, Debug)]
pub struct Seek {
    pub fd: i32,
    pub offset: i64,
    pub whence: i32,
}

#[derive(Clone, Default, Debug)]
pub struct IoCtl {
    pub fd: i32,
    pub cmd: u64,
    pub argp: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetTimeOfDay {
    pub tv: u64,
    pub tz: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Getxattr {
    pub path: u64,
    pub name: u64,
    pub value: u64,
    pub size: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Lgetxattr {
    pub path: u64,
    pub name: u64,
    pub value: u64,
    pub size: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fgetxattr {
    pub fd: i32,
    pub name: u64,
    pub value: u64,
    pub size: u64,
}

#[derive(Clone, Default, Debug)]
pub struct ReadLinkAt {
    pub dirfd: i32,
    pub path: u64,
    pub buf: u64,
    pub bufsize: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fcntl {
    pub fd: i32,
    pub cmd: i32,
    pub arg: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Close {
    pub fd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Fstatfs {
    pub fd: i32,
    pub buf: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fstat {
    pub fd: i32,
    pub buff: u64,
}

#[derive(Clone, Default, Debug)]
pub struct FileType {
    pub dirfd: i32,
    pub pathname: u64,
    pub mode: u32,
    pub device: u64,
    pub inode: u64,
    pub ret: i32,
}

#[derive(Clone, Default, Debug)]
pub struct BatchFstatat {
    pub addr: u64,
    pub count: usize
}

#[derive(Clone, Default, Debug)]
pub struct Fstatat {
    pub dirfd: i32,
    pub pathname: u64,
    pub buff: u64,
    pub flags: i32
}

#[derive(Clone, Default, Debug)]
pub struct GetDents64 {
    pub fd: i32,
    pub dirp: u64,
    pub count: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Unlinkat {
    pub dirfd: i32,
    pub pathname: u64,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Mkdirat {
    pub dirfd: i32,
    pub pathname: u64,
    pub mode_: u32,
    pub uid: u32,
    pub gid: u32,
}

#[derive(Clone, Default, Debug)]
pub struct SysSync {}

#[derive(Clone, Default, Debug)]
pub struct SyncFs {
    pub fd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct SyncFileRange {
    pub fd: i32,
    pub offset: i64,
    pub nbytes: i64,
    pub flags: u32,
}

#[derive(Clone, Default, Debug)]
pub struct FSync {
    pub fd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct MSync {
    pub addr: u64,
    pub len: usize,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct MAdvise {
    pub addr: u64,
    pub len: usize,
    pub advise: i32,
}

#[derive(Clone, Default, Debug)]
pub struct FDataSync {
    pub fd: i32,
}

pub struct TryOpenStruct<'a> {
    pub fstat: &'a LibcStat,
    pub writeable: bool,
}

#[derive(Clone, Default, Debug)]
pub struct TryOpenAt {
    pub dirfd: i32,
    pub name: u64,
    pub addr: u64,
}

#[derive(Clone, Default, Debug)]
pub struct CreateAt {
    pub dirfd: i32,
    pub pathName: u64,
    pub flags: i32,
    pub mode: i32,
    pub uid: u32,
    pub gid: u32,
    pub fstatAddr: u64,
}

#[derive(Clone, Default, Debug)]
pub struct FAccessAt {
    pub dirfd: i32,
    pub pathname: u64,
    pub mode: i32,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Socket {
    pub domain: i32,
    pub type_: i32,
    pub protocol: i32,
}

#[derive(Clone, Default, Debug)]
pub struct SocketPair {
    pub domain: i32,
    pub type_: i32,
    pub protocol: i32,
    pub socketVect: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetSockName {
    pub sockfd: i32,
    pub addr: u64,
    pub addrlen: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetPeerName {
    pub sockfd: i32,
    pub addr: u64,
    pub addrlen: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetSockOpt {
    pub sockfd: i32,
    pub level: i32,
    pub optname: i32,
    pub optval: u64,
    pub optlen: u64,
}

#[derive(Clone, Default, Debug)]
pub struct SetSockOpt {
    pub sockfd: i32,
    pub level: i32,
    pub optname: i32,
    pub optval: u64,
    pub optlen: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Connect {
    pub sockfd: i32,
    pub addr: u64,
    pub addrlen: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Bind {
    pub sockfd: i32,
    pub addr: u64,
    pub addrlen: u32,
    pub umask: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Listen {
    pub sockfd: i32,
    pub backlog: i32,
    pub block: bool,
}

#[derive(Clone, Default, Debug)]
pub struct Shutdown {
    pub sockfd: i32,
    pub how: i32,
}

#[derive(Clone, Default, Debug)]
pub struct SchedGetAffinity {
    pub pid: i32,
    pub cpuSetSize: u64,
    pub mask: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetRandom {
    pub buf: u64,
    pub len: u64,
    pub flags: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Fchdir {
    pub fd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Fadvise {
    pub fd: i32,
    pub offset: u64,
    pub len: u64,
    pub advice: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Mlock2 {
    pub addr: u64,
    pub len: u64,
    pub flags: u32,
}

#[derive(Clone, Default, Debug)]
pub struct MUnlock {
    pub addr: u64,
    pub len: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Chown {
    pub pathname: u64,
    pub owner: u32,
    pub group: u32,
}

#[derive(Clone, Default, Debug)]
pub struct FChown {
    pub fd: i32,
    pub owner: u32,
    pub group: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Chmod {
    pub pathname: u64,
    pub mode: u32,
}

#[derive(Clone, Default, Debug)]
pub struct Fchmod {
    pub fd: i32,
    pub mode: u32,
}

#[derive(Clone, Default, Debug)]
pub struct SymLinkAt {
    pub oldpath: u64,
    pub newdirfd: i32,
    pub newpath: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Futimens {
    pub fd: i32,
    pub times: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetStdfds {
    //pub fds : [i32; 3],
    pub addr: u64,
    //address of fds[0]
}

#[derive(Clone, Default, Debug)]
pub struct IORead {
    pub fd: i32,
    pub iovs: u64,
    pub iovcnt: i32,
}

#[derive(Clone, Default, Debug)]
pub struct IOTTYRead {
    pub fd: i32,
    pub iovs: u64,
    pub iovcnt: i32,
}

#[derive(Clone, Default, Debug)]
pub struct IOWrite {
    pub fd: i32,
    pub iovs: u64,
    pub iovcnt: i32,
}

#[derive(Clone, Default, Debug)]
pub struct IOReadAt {
    pub fd: i32,
    pub iovs: u64,
    pub iovcnt: i32,
    pub offset: u64,
}

#[derive(Clone, Default, Debug)]
pub struct IOWriteAt {
    pub fd: i32,
    pub iovs: u64,
    pub iovcnt: i32,
    pub offset: u64,
}

#[derive(Clone, Default, Debug)]
pub struct IOAppend {
    pub fd: i32,
    pub iovs: u64,
    pub iovcnt: i32,
    pub fileLenAddr: u64,
}

#[derive(Clone, Default, Debug)]
pub struct IOAccept {
    pub fd: i32,
    pub addr: u64,
    pub addrlen: u64,
    pub flags: i32,
    pub blocking: bool,
}

#[derive(Clone, Default, Debug)]
pub struct IOConnect {
    pub fd: i32,
    pub addr: u64,
    pub addrlen: u32,
    pub blocking: bool,
}

#[derive(Clone, Default, Debug)]
pub struct IORecvMsg {
    pub fd: i32,
    pub msghdr: u64,
    //address of MsgHdr
    pub flags: i32,
    pub blocking: bool,
}

#[derive(Clone, Default, Debug)]
pub struct IOSendMsg {
    pub fd: i32,
    pub msghdr: u64,
    //address of MsgHdr
    pub flags: i32,
    pub blocking: bool,
}

#[derive(Clone, Default, Debug)]
pub struct NewFd {
    pub fd: i32
}

#[derive(Clone, Default, Debug)]
pub struct HostEpollWaitProcess {
    pub addr: u64,
    pub count: usize,
}

#[derive(Clone, Default, Debug)]
pub struct VcpuWait {
    pub addr: u64,
    pub count: usize,
}

#[derive(Clone, Default, Debug)]
pub struct WaitFD {
    pub fd: i32,
    pub mask: EventMask,
}

#[derive(Clone, Default, Debug)]
pub struct NonBlockingPoll {
    pub fd: i32,
    pub mask: EventMask,
}

#[derive(Clone, Debug, Copy)]
pub enum TmpfsFileType {
    File,
    Fifo,
}

impl Default for TmpfsFileType {
    fn default() -> Self {
        return Self::Fifo
    }
}

#[derive(Clone, Default, Debug)]
pub struct NewTmpfsFile {
    pub typ: TmpfsFileType,
    pub addr: u64,
}

#[derive(Clone, Debug)]
pub struct LoadProcessKernel {
    pub processAddr: u64,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct ControlMsgCall {
    pub addr: u64,
    pub len: usize,
    pub taskId: TaskIdQ,
    pub ret : i64,
}

#[derive(Clone, Debug)]
pub struct ControlMsgRet {
    pub msgId: u64,
    pub addr: u64,
    pub len: usize,
}

pub struct Print<'a> {
    pub level: DebugLevel,
    pub str: &'a str,
}

#[derive(Debug)]
pub struct QMsg <'a> {
    pub taskId: TaskIdQ,
    pub globalLock: bool,
    pub ret: u64,
    pub msg: &'a Msg,
}

#[derive(Debug, Copy, Clone)]
pub enum HostOutputMsg {
    QCall(u64),
    WaitFDAsync(WaitFDAsync),
}

#[derive(Clone, Default, Debug, Copy)]
pub struct WaitFDAsync {
    pub fd: i32,
    pub mask: EventMask,
}

