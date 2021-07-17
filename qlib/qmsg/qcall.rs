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
    Print(u64, &'static str),
    LoadProcessKernel(LoadProcessKernel),
    LoadExecProcess(LoadExecProcess),
    ControlMsgCall(ControlMsgCall),
    ControlMsgRet(ControlMsgRet),
    MapAnon(MapAnon),
    MapFile(MapFile),
    PrintInt(PrintInt),
    SetBuff(SetBuff),
    WriteBuffTrigger(WriteBuffTrigger),
    ReadBuffTrigger(ReadBuffTrigger),
    GetStdfds(GetStdfds),
    CreateMemfd(CreateMemfd),

    //Syscall
    Fallocate(Fallocate),
    RenameAt(RenameAt),
    Truncate(Truncate),
    Ftruncate(Ftruncate),
    Eventfd(Eventfd),
    Seek(Seek),
    ReadLink(ReadLink),
    ReadLinkAt(ReadLinkAt),
    GetTimeOfDay(GetTimeOfDay),
    ClockGetRes(ClockGetRes),
    ClockGetTime(ClockGetTime),
    ClockSetTime(ClockSetTime),
    Times(Times),
    Interrupt(Interrupt),
    Wait,
    Exit(Exit),
    ExitThreadGroup(ExitThreadGroup),
    IoCtl(IoCtl),
    Fcntl(Fcntl),
    NanoSleep(NanoSleep),
    Time(Time),
    Close(Close),

    Getxattr(Getxattr),
    Lgetxattr(Lgetxattr),
    Fgetxattr(Fgetxattr),
    Stat(Stat),
    Fstat(Fstat),
    Fstatat(Fstatat),
    Statfs(Statfs),
    Fstatfs(Fstatfs),
    PRLimit(PRLimit),
    GetRLimit(GetRLimit),
    SetRLimit(SetRLimit),
    GetDents(GetDents),
    GetDents64(GetDents64),

    ForkFdTbl(ForkFdTbl),
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
    Uname(Uname),
    Umask(Umask),
    Access(Access),
    FAccessAt(FAccessAt),
    Pause(Pause),

    MinCore(MinCore),
    Dup(Dup),
    Dup2(Dup2),
    Dup3(Dup3),

    Socket(Socket),
    SocketPair(SocketPair),
    GetPeerName(GetPeerName),
    GetSockName(GetSockName),
    GetSockOpt(GetSockOpt),
    SetSockOpt(SetSockOpt),
    Bind(Bind),
    Listen(Listen),
    Shutdown(Shutdown),
    Select(Select),
    PSelect(PSelect),
    Poll(Poll),
    EpollCreate(EpollCreate),
    EpollCreate1(EpollCreate1),
    EpollPWait(EpollPWait),
    EpollCtl(EpollCtl),

    GetUid(GetUid),
    GetEUid(GetEUid),
    GetGid(GetGid),
    SetGid(SetGid),
    GetEGid(GetEGid),
    GetGroups(GetGroups),
    SetGroups(SetGroups),
    Sysinfo(Sysinfo),
    GetCwd(GetCwd),
    GetWd(GetWd),
    GetCurrentDirName(GetCurrentDirName),
    GetPGid(GetPGid),
    Pipe2(Pipe2),
    SchedGetAffinity(SchedGetAffinity),
    GetRandom(GetRandom),
    Chdir(Chdir),
    Fchdir(Fchdir),
    Fadvise(Fadvise),
    Prctl(Prctl),
    Mlock2(Mlock2),
    MUnlock(MUnlock),
    IOSetup(IOSetup),
    IOSubmit(IOSubmit),
    Rename(Rename),
    Chown(Chown),
    FChown(FChown),
    TimerFdCreate(TimerFdCreate),
    TimerFdSetTime(TimerFdSetTime),
    TimerFdGetTime(TimerFdGetTime),
    Chmod(Chmod),
    Fchmod(Fchmod),
    SetHostName(SetHostName),
    SymLinkAt(SymLinkAt),
    Futimens(Futimens),

    HostCPUInfo(HostCPUInfo),

    IORead(IORead),
    IOWrite(IOWrite),
    IOReadAt(IOReadAt),
    IOWriteAt(IOWriteAt),
    IOAppend(IOAppend),
    IOAccept(IOAccept),
    IOConnect(IOConnect),
    IORecvMsg(IORecvMsg),
    IOSendMsg(IOSendMsg),
    MMapAnon(MMapAnon),
    MMapFile(MMapFile),
    UnMapPma(UnMapPma),
    NonBlockingPoll(NonBlockingPoll),
    NewTmpfsFile(NewTmpfsFile),
    OpenFifo(OpenFifo),
    IoUringSetup(IoUringSetup),
    IoUringRegister(IoUringRegister),
    IoUringEnter(IoUringEnter),
    Statm(Statm),
}

#[derive(Clone, Default, Debug)]
pub struct IoUringSetup {
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
    pub fd: i32,
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
pub struct PRLimit {
    pub pid: i32,
    pub resource: i32,
    pub newLimit: u64,
    pub oldLimit: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetRLimit {
    pub resource: u32,
    pub rlimit: u64,
}

#[derive(Clone, Default, Debug)]
pub struct SetRLimit {
    pub resource: u32,
    pub rlimit: u64,
}

//GDT, IDT: descriptor table
#[derive(Clone, Default, Debug)]
pub struct DT {
    pub base: u64,
    pub limit: u16,
}

#[derive(Clone, Default, Debug)]
pub struct MMapAnon {
    pub len: u64,
    pub prot: i32,
}

#[derive(Clone, Default, Debug)]
pub struct MMapFile {
    pub len: u64,
    pub fd: i32,
    pub offset: u64,
    pub prot: i32,
}

/*
#[derive(Clone, Default, Debug)]
pub struct MUnmap {
    pub addr: u64,
    pub len: u64,
}*/

#[derive(Clone, Default, Debug)]
pub struct MapAnon {
    pub len: u64,
    pub hugePage: bool,
    pub share: bool,
}

#[derive(Clone, Default, Debug)]
pub struct MapFile {
    pub len: u64,
    pub hugePage: bool,
    pub fd: i32,
    pub offset: u64,
    pub share: bool,
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
pub struct Truncate {
    pub path: u64,
    pub len: i64,
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
pub struct Interrupt {
    pub target: TaskId,
}

#[derive(Clone, Default, Debug)]
pub struct Exit {
    pub target: TaskId,
}

#[derive(Clone, Default, Debug)]
pub struct ExitThreadGroup {
    pub tgid: i32,
}

#[derive(Clone, Default, Debug)]
pub struct IoCtl {
    pub fd: i32,
    pub cmd: u64,
    pub argp: u64,
}

#[derive(Clone, Default, Debug)]
pub struct UnMapPma {
    pub addr: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetTimeOfDay {
    pub tv: u64,
    pub tz: u64,
}

#[derive(Clone, Default, Debug)]
pub struct ClockGetRes {
    pub clkId: i32,
    pub ts: u64,
}

#[derive(Clone, Default, Debug)]
pub struct ClockGetTime {
    pub clkId: i32,
    pub ts: u64,
}

#[derive(Clone, Default, Debug)]
pub struct ClockSetTime {
    pub clkId: i32,
    pub ts: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Times {
    pub tms: u64,
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
pub struct ReadLink {
    pub path: u64,
    pub buf: u64,
    pub bufsize: u64,
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
pub struct NanoSleep {
    pub req: u64,
    pub rem: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Time {
    pub tloc: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Close {
    pub fd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Statfs {
    pub path: u64,
    pub buf: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fstatfs {
    pub fd: i32,
    pub buf: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Stat {
    pub pathName: u64,
    pub statBuff: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fstat {
    pub fd: i32,
    pub buff: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Fstatat {
    pub dirfd: i32,
    pub pathname: u64,
    pub buff: u64,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct GetDents {
    pub fd: i32,
    pub dirp: u64,
    pub count: u32,
}

#[derive(Clone, Default, Debug)]
pub struct GetDents64 {
    pub fd: i32,
    pub dirp: u64,
    pub count: u32,
}

#[derive(Clone, Default, Debug)]
pub struct CPUIDInfo {
    pub ax: u32,
    pub bx: u32,
    pub cx: u32,
    pub dx: u32,
}

#[derive(Clone, Default, Debug)]
pub struct HostCPUInfo {
    pub axArg: u32,
    pub cxArg: u32,
    pub addr: u64,
    //address of CPUIDInfo struct
}

#[derive(Clone, Default, Debug)]
pub struct GetUid {}

#[derive(Clone, Default, Debug)]
pub struct GetEUid {}

#[derive(Clone, Default, Debug)]
pub struct GetGid {}

#[derive(Clone, Default, Debug)]
pub struct SetGid {
    pub gid: u32,
}

#[derive(Clone, Default, Debug)]
pub struct GetEGid {}

#[derive(Clone, Default, Debug)]
pub struct GetGroups {
    pub size: i32,
    pub list: u64,
}

#[derive(Clone, Default, Debug)]
pub struct SetGroups {
    pub size: usize,
    pub list: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Sysinfo {
    pub info: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetCwd {
    pub buf: u64,
    pub size: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetWd {
    pub buf: u64,
}

#[derive(Clone, Default, Debug)]
pub struct GetCurrentDirName {}

#[derive(Clone, Default, Debug)]
pub struct GetPGid {}

//create a new fdTbl by fork from a tgid
#[derive(Clone, Default, Debug)]
pub struct ForkFdTbl {
    pub pTgid: i32,
    //parent tgid
    pub tgid: i32,
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
pub struct Uname {
    pub buff: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Pause {}

#[derive(Clone, Default, Debug)]
pub struct Access {
    pub pathName: u64,
    pub mode: i32,
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
}

#[derive(Clone, Default, Debug)]
pub struct Shutdown {
    pub sockfd: i32,
    pub how: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Select {
    pub nfds: i32,
    pub readfds: u64,
    pub writefds: u64,
    pub exceptfds: u64,
    pub timeout: u64,
}

#[derive(Clone, Default, Debug)]
pub struct PSelect {
    pub nfds: i32,
    pub readfds: u64,
    pub writefds: u64,
    pub exceptfds: u64,
    pub timeout: u64,
    pub sigmask: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Poll {
    pub fds: u64,
    pub nfds: u64,
    pub timeout: i32,
}

#[derive(Clone, Default, Debug)]
pub struct EpollCreate {
    pub size: i32,
}

#[derive(Clone, Default, Debug)]
pub struct EpollCreate1 {
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct EpollPWait {
    pub epfd: i32,
    pub events: u64,
    pub maxEvents: i32,
    pub timeout: i32,
    pub sigmask: u64,
}

#[derive(Clone, Default, Debug)]
pub struct EpollCtl {
    pub epfd: i32,
    pub op: i32,
    pub fd: i32,
    pub event: u64,
}

#[derive(Clone, Default, Debug)]
pub struct MinCore {
    pub addr: u64,
    pub len: u64,
    pub vec: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Dup {
    pub oldfd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Dup2 {
    pub oldfd: i32,
    pub newfd: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Dup3 {
    pub oldfd: i32,
    pub newfd: i32,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Pipe2 {
    pub fds: u64,
    pub flags: i32,
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
pub struct Chdir {
    pub path: u64,
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
pub struct Umask {
    pub mask: u32
}

#[derive(Clone, Default, Debug)]
pub struct Eventfd {
    pub initval: u32,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct Prctl {
    pub option: i32,
    pub arg2: u64,
    pub arg3: u64,
    pub arg4: u64,
    pub arg5: u64,
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
pub struct IOSetup {
    pub nr_events: u64,
    pub ctx_idp: u64,
}

#[derive(Clone, Default, Debug)]
pub struct IOSubmit {
    pub ctx_id: u64,
    pub nr: u64,
    pub iocbpp: u64,
}

#[derive(Clone, Default, Debug)]
pub struct Rename {
    pub oldpath: u64,
    pub newpath: u64,
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
pub struct TimerFdCreate {
    pub clockId: i32,
    pub flags: i32,
}

#[derive(Clone, Default, Debug)]
pub struct TimerFdSetTime {
    pub fd: i32,
    pub flags: i32,
    pub newValue: u64,
    pub oldValue: u64,
}

#[derive(Clone, Default, Debug)]
pub struct TimerFdGetTime {
    pub fd: i32,
    pub currVal: u64,
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
pub struct SetHostName {
    pub name: u64,
    pub len: usize,
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

//set read/write buff for buffered fd. Internal api
#[derive(Clone, Default, Debug)]
pub struct SetBuff {
    pub fd: i32,
    pub readBuf: u64,
    pub writeBuf: u64,
}

//write buff and find write buff is empty, notify host to write it to os, async call
#[derive(Clone, Default, Debug)]
pub struct WriteBuffTrigger {
    pub fd: i32,
}

//read buff and find read buff full, notify host to read more from os, async call
#[derive(Clone, Default, Debug)]
pub struct ReadBuffTrigger {
    pub fd: i32,
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

/*
#[derive(Clone, Default, Debug)]
pub struct WaitFD {
    pub fd: i32,
    pub mask: u32,
}*/

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

#[derive(Clone, Default, Debug)]
pub struct OpenFifo {
    pub UID: u64,
    pub flags: i32
}

#[derive(Clone, Default, Debug)]
pub struct PrintInt {
    pub val: i64,
}

#[derive(Clone, Debug)]
pub struct LoadProcessKernel {
    pub processAddr: u64,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct LoadExecProcess {
    pub processAddr: u64,
    pub len: usize,
}

#[derive(Clone, Debug)]
pub struct ControlMsgCall {
    pub addr: u64,
    pub len: usize,
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
pub struct Event<'a> {
    pub taskId: TaskIdQ,
    pub interrupted: bool,
    pub ret: u64,
    pub msg: &'a mut Msg,
}

