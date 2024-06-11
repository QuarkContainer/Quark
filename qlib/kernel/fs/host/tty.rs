// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor& Authors.
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
use alloc::sync::Arc;
use core::any::Any;
use core::ops::Deref;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::socket_buf::*;
use super::super::super::guestfdnotifier::*;
use super::super::super::kernel::waiter::*;
use super::super::super::quring::QUring;
use super::super::super::task::*;
use super::super::super::threadmgr::processgroup::*;
use super::super::super::threadmgr::session::*;
use super::super::super::SHARESPACE;

use super::super::attr::*;
use super::super::dentry::*;
use super::super::dirent::*;
use super::super::file::*;
use super::super::host::hostinodeop::*;
use super::super::inode::*;
use super::hostfileop::*;

use super::ioctl::*;

pub const NUM_CONTROL_CHARACTERS: usize = 19;
pub const DISABLED_CHAR: u8 = 0;

#[derive(Clone, Default, Copy, Debug)]
#[repr(C)]
pub struct Winsize {
    pub Row: u16,
    pub Col: u16,
    pub Xpixel: u16,
    pub Ypixel: u16,
}

#[derive(Clone, Default, Copy)]
#[repr(C)]
pub struct WindowSize {
    pub Rows: u16,
    pub Cols: u16,
    pub Pad: [u8; 4],
}

pub struct InputFlags {}

impl InputFlags {
    pub const IGNBRK: u32 = 0o0000001;
    pub const BRKINT: u32 = 0o0000002;
    pub const IGNPAR: u32 = 0o0000004;
    pub const PARMRK: u32 = 0o0000010;
    pub const INPCK: u32 = 0o0000020;
    pub const ISTRIP: u32 = 0o0000040;
    pub const INLCR: u32 = 0o0000100;
    pub const IGNCR: u32 = 0o0000200;
    pub const ICRNL: u32 = 0o0000400;
    pub const IUCLC: u32 = 0o0001000;
    pub const IXON: u32 = 0o0002000;
    pub const IXANY: u32 = 0o0004000;
    pub const IXOFF: u32 = 0o0010000;
    pub const IMAXBEL: u32 = 0o0020000;
    pub const IUTF8: u32 = 0o0040000;
}

pub struct OutputFlags {}

impl OutputFlags {
    pub const OPOST: u32 = 0o0000001;
    pub const OLCUC: u32 = 0o0000002;
    pub const ONLCR: u32 = 0o0000004;
    pub const OCRNL: u32 = 0o0000010;
    pub const ONOCR: u32 = 0o0000020;
    pub const ONLRET: u32 = 0o0000040;
    pub const OFILL: u32 = 0o0000100;
    pub const OFDEL: u32 = 0o0000200;
    pub const NLDLY: u32 = 0o0000400;
    pub const NL0: u32 = 0o0000000;
    pub const NL1: u32 = 0o0000400;
    pub const CRDLY: u32 = 0o0003000;
    pub const CR0: u32 = 0o0000000;
    pub const CR1: u32 = 0o0001000;
    pub const CR2: u32 = 0o0002000;
    pub const CR3: u32 = 0o0003000;
    pub const TABDLY: u32 = 0o0014000;
    pub const TAB0: u32 = 0o0000000;
    pub const TAB1: u32 = 0o0004000;
    pub const TAB2: u32 = 0o0010000;
    pub const TAB3: u32 = 0o0014000;
    pub const XTABS: u32 = 0o0014000;
    pub const BSDLY: u32 = 0o0020000;
    pub const BS0: u32 = 0o0000000;
    pub const BS1: u32 = 0o0020000;
    pub const VTDLY: u32 = 0o0040000;
    pub const VT0: u32 = 0o0000000;
    pub const VT1: u32 = 0o0040000;
    pub const FFDLY: u32 = 0o0100000;
    pub const FF0: u32 = 0o0000000;
    pub const FF1: u32 = 0o0100000;
}

pub struct ControlFlags {}

impl ControlFlags {
    pub const CBAUD: u32 = 0o0010017;
    pub const B0: u32 = 0o0000000;
    pub const B50: u32 = 0o0000001;
    pub const B75: u32 = 0o0000002;
    pub const B110: u32 = 0o0000003;
    pub const B134: u32 = 0o0000004;
    pub const B150: u32 = 0o0000005;
    pub const B200: u32 = 0o0000006;
    pub const B300: u32 = 0o0000007;
    pub const B600: u32 = 0o0000010;
    pub const B1200: u32 = 0o0000011;
    pub const B1800: u32 = 0o0000012;
    pub const B2400: u32 = 0o0000013;
    pub const B4800: u32 = 0o0000014;
    pub const B9600: u32 = 0o0000015;
    pub const B19200: u32 = 0o0000016;
    pub const B38400: u32 = 0o0000017;
    pub const EXTA: u32 = Self::B19200;
    pub const EXTB: u32 = Self::B38400;
    pub const CSIZE: u32 = 0o0000060;
    pub const CS5: u32 = 0o0000000;
    pub const CS6: u32 = 0o0000020;
    pub const CS7: u32 = 0o0000040;
    pub const CS8: u32 = 0o0000060;
    pub const CSTOPB: u32 = 0o0000100;
    pub const CREAD: u32 = 0o0000200;
    pub const PARENB: u32 = 0o0000400;
    pub const PARODD: u32 = 0o0001000;
    pub const HUPCL: u32 = 0o0002000;
    pub const CLOCAL: u32 = 0o0004000;
    pub const CBAUDEX: u32 = 0o0010000;
    pub const BOTHER: u32 = 0o0010000;
    pub const B57600: u32 = 0o0010001;
    pub const B115200: u32 = 0o0010002;
    pub const B230400: u32 = 0o0010003;
    pub const B460800: u32 = 0o0010004;
    pub const B500000: u32 = 0o0010005;
    pub const B576000: u32 = 0o0010006;
    pub const B921600: u32 = 0o0010007;
    pub const B1000000: u32 = 0o0010010;
    pub const B1152000: u32 = 0o0010011;
    pub const B1500000: u32 = 0o0010012;
    pub const B2000000: u32 = 0o0010013;
    pub const B2500000: u32 = 0o0010014;
    pub const B3000000: u32 = 0o0010015;
    pub const B3500000: u32 = 0o0010016;
    pub const B4000000: u32 = 0o0010017;
    pub const CIBAUD: u32 = 0o002003600000;
    pub const CMSPAR: u32 = 0o010000000000;
    pub const CRTSCTS: u32 = 0o020000000000;

    // IBSHIFT is the shift from CBAUD to CIBAUD.
    pub const IBSHIFT: u32 = 0o16;
}

pub struct LocalFlags {}

impl LocalFlags {
    pub const ISIG: u32 = 0o0000001;
    pub const ICANON: u32 = 0o0000002;
    pub const XCASE: u32 = 0o0000004;
    pub const ECHO: u32 = 0o0000010;
    pub const ECHOE: u32 = 0o0000020;
    pub const ECHOK: u32 = 0o0000040;
    pub const ECHONL: u32 = 0o0000100;
    pub const NOFLSH: u32 = 0o0000200;
    pub const TOSTOP: u32 = 0o0000400;
    pub const ECHOCTL: u32 = 0o0001000;
    pub const ECHOPRT: u32 = 0o0002000;
    pub const ECHOKE: u32 = 0o0004000;
    pub const FLUSHO: u32 = 0o0010000;
    pub const PENDIN: u32 = 0o0040000;
    pub const IEXTEN: u32 = 0o0100000;
    pub const EXTPROC: u32 = 0o0200000;
}

pub struct ControlFlagIndex {}

impl ControlFlagIndex {
    pub const VINTR: usize = 0;
    pub const VQUIT: usize = 1;
    pub const VERASE: usize = 2;
    pub const VKILL: usize = 3;
    pub const VEOF: usize = 4;
    pub const VTIME: usize = 5;
    pub const VMIN: usize = 6;
    pub const VSWTC: usize = 7;
    pub const VSTART: usize = 8;
    pub const VSTOP: usize = 9;
    pub const VSUSP: usize = 10;
    pub const VEOL: usize = 11;
    pub const VREPRINT: usize = 12;
    pub const VDISCARD: usize = 13;
    pub const VWERASE: usize = 14;
    pub const VLNEXT: usize = 15;
    pub const VEOL2: usize = 16;
}

pub const fn ControlCharacter(c: char) -> u8 {
    return c as u8 - 'A' as u8 + 1;
}

pub const DEFAULT_CONTROL_CHARACTERS: [u8; NUM_CONTROL_CHARACTERS] = [
    ControlCharacter('C'),  // VINTR = ^C
    ControlCharacter('\\'), // VQUIT = ^\
    '\x7f' as u8,           // VERASE = DEL
    ControlCharacter('U'),  // VKILL = ^U
    ControlCharacter('D'),  // VEOF = ^D
    0,                      // VTIME
    1,                      // VMIN
    0,                      // VSWTC
    ControlCharacter('Q'),  // VSTART = ^Q
    ControlCharacter('S'),  // VSTOP = ^S
    ControlCharacter('Z'),  // VSUSP = ^Z
    0,                      // VEOL
    ControlCharacter('R'),  // VREPRINT = ^R
    ControlCharacter('O'),  // VDISCARD = ^O
    ControlCharacter('W'),  // VWERASE = ^W
    ControlCharacter('V'),  // VLNEXT = ^V
    0,                      // VEOL2
    0,
    0,
];

pub const MASTER_TERMIOS: KernelTermios = KernelTermios {
    InputFlags: 0,
    OutputFlags: 0,
    ControlFlags: ControlFlags::B38400 | ControlFlags::CS8 | ControlFlags::CREAD,
    LocalFlags: 0,
    LineDiscipline: 0,
    ControlCharacters: DEFAULT_CONTROL_CHARACTERS,
    InputSpeed: 38400,
    OutputSpeed: 38400,
};

pub const DEFAULT_SLAVE_TERMIOS: KernelTermios = KernelTermios {
    InputFlags: InputFlags::ICRNL | InputFlags::IXON,
    OutputFlags: OutputFlags::OPOST | OutputFlags::ONLCR,
    ControlFlags: ControlFlags::B38400 | ControlFlags::CS8 | ControlFlags::CREAD,
    LocalFlags: LocalFlags::ISIG
        | LocalFlags::ICANON
        | LocalFlags::ECHO
        | LocalFlags::ECHOE
        | LocalFlags::ECHOK
        | LocalFlags::ECHOCTL
        | LocalFlags::ECHOKE
        | LocalFlags::IEXTEN,
    LineDiscipline: 0,
    ControlCharacters: DEFAULT_CONTROL_CHARACTERS,
    InputSpeed: 38400,
    OutputSpeed: 38400,
};

#[derive(Clone, Default, Copy, Debug)]
#[repr(C)]
pub struct Termios {
    pub InputFlags: u32,
    pub OutputFlags: u32,
    pub ControlFlags: u32,
    pub LocalFlags: u32,
    pub LineDiscipline: u8,
    pub ControlCharacters: [u8; NUM_CONTROL_CHARACTERS],
}

#[derive(Clone, Default)]
pub struct KernelTermios {
    pub InputFlags: u32,
    pub OutputFlags: u32,
    pub ControlFlags: u32,
    pub LocalFlags: u32,
    pub LineDiscipline: u8,
    pub ControlCharacters: [u8; NUM_CONTROL_CHARACTERS],
    pub InputSpeed: u32,
    pub OutputSpeed: u32,
}

impl KernelTermios {
    pub const VINTR: u8 = 0;
    pub const VQUIT: u8 = 1;
    pub const VERASE: u8 = 2;
    pub const VKILL: u8 = 3;
    pub const VEOF: u8 = 4;
    pub const VTIME: u8 = 5;
    pub const VMIN: u8 = 6;
    pub const VSWTC: u8 = 7;
    pub const VSTART: u8 = 8;
    pub const VSTOP: u8 = 9;
    pub const VSUSP: u8 = 10;
    pub const VEOL: u8 = 11;
    pub const VREPRINT: u8 = 12;
    pub const VDISCARD: u8 = 13;
    pub const VWERASE: u8 = 14;
    pub const VLNEXT: u8 = 15;
    pub const VEOL2: u8 = 16;

    pub fn IEnabled(&self, flag: u32) -> bool {
        return self.InputFlags & flag == flag;
    }

    pub fn OEnabled(&self, flag: u32) -> bool {
        return self.OutputFlags & flag == flag;
    }

    pub fn CEnabled(&self, flag: u32) -> bool {
        return self.ControlFlags & flag == flag;
    }

    pub fn LEnabled(&self, flag: u32) -> bool {
        return self.LocalFlags & flag == flag;
    }

    pub fn ToTermios(&self) -> Termios {
        return Termios {
            InputFlags: self.InputFlags,
            OutputFlags: self.OutputFlags,
            ControlFlags: self.ControlFlags,
            LocalFlags: self.LocalFlags,
            LineDiscipline: self.LineDiscipline,
            ControlCharacters: self.ControlCharacters,
        };
    }

    pub fn FromTermios(&mut self, term: &Termios) {
        self.InputFlags = term.InputFlags;
        self.OutputFlags = term.OutputFlags;
        self.ControlFlags = term.ControlFlags;
        self.LocalFlags = term.LocalFlags;
        self.LineDiscipline = term.LineDiscipline;
        self.ControlCharacters = term.ControlCharacters;
    }

    pub fn IsTerminating(&self, cBytes: &[u8]) -> bool {
        if cBytes.len() != 1 {
            return false;
        }

        let c = cBytes[0];

        if self.IsEOF(c) {
            return true;
        }

        if c == DISABLED_CHAR {
            return false;
        } else if c == '\n' as u8 || c == self.ControlCharacters[Self::VEOL as usize] {
            return true;
        } else if c == self.ControlCharacters[Self::VEOL2 as usize] {
            return self.LEnabled(LocalFlags::IEXTEN);
        }

        return false;
    }

    pub fn IsEOF(&self, c: u8) -> bool {
        return c == self.ControlCharacters[Self::VEOF as usize]
            && self.ControlCharacters[Self::VEOF as usize] == DISABLED_CHAR;
    }
}

pub struct TTYFileOpsInternal {
    pub fileOps: HostFileOp,
    pub termios: KernelTermios,
    pub session: Option<Session>,
    pub fgProcessgroup: Option<ProcessGroup>,
    pub fd: i32,
    pub buf: SocketBuff,
    pub queue: Queue,
}

impl TTYFileOpsInternal {
    fn checkChange(&self, _task: &Task, _sig: Signal) -> Result<()> {
        return Ok(());
        /*let thread = match &task.thread {
            // No task? Linux does not have an analog for this case, but
            // tty_check_change is more of a blacklist of cases than a
            // whitelist, and is surprisingly permissive. Allowing the
            // change seems most appropriate.
            None => return Ok(()),
            Some(ref t) => t.clone(),
        };

        let tg = thread.lock().tg.clone();
        let pg = tg.ProcessGroup();

        // If the session for the task is different than the session for the
        // controlling TTY, then the change is allowed. Seems like a bad idea,
        // but that's exactly what linux does.
        if tg.Session() != self.fgProcessgroup.Session() {
            return Ok(())
        }

        // If we are the foreground process group, then the change is allowed.
        if pg == self.fgProcessgroup {
            return Ok(())
        }

        // We are not the foreground process group.

        // Is the provided signal blocked or ignored?
        if (thread.SignalMask() & SignalSet::SignalSetOf(&sig).0 != 0 || tg.SignalHandlers())*/
    }
}

#[derive(Clone)]
pub struct TTYFileOps(Arc<QMutex<TTYFileOpsInternal>>);

impl Deref for TTYFileOps {
    type Target = Arc<QMutex<TTYFileOpsInternal>>;

    fn deref(&self) -> &Arc<QMutex<TTYFileOpsInternal>> {
        &self.0
    }
}

pub const ENABLE_RINGBUF: bool = true;

impl TTYFileOps {
    pub fn New(fops: HostFileOp) -> Self {
        let queue = fops.InodeOp.lock().queue.clone();
        let fd = fops.InodeOp.lock().HostFd;
        let internal = TTYFileOpsInternal {
            fileOps: fops,
            termios: DEFAULT_SLAVE_TERMIOS,
            session: None,
            fgProcessgroup: None,
            fd: fd,
            #[cfg(not(feature = "cc"))]
            buf: SocketBuff(Arc::new(SocketBuffIntern::Init(
                MemoryDef::DEFAULT_BUF_PAGE_COUNT,
            ))),
            #[cfg(feature = "cc")]
            buf: SocketBuff(Arc::new_in(
                SocketBuffIntern::Init(MemoryDef::DEFAULT_BUF_PAGE_COUNT),
                crate::GUEST_HOST_SHARED_ALLOCATOR,
            )),
            queue: queue,
        };

        if SHARESPACE.config.read().UringIO && ENABLE_RINGBUF {
            QUring::BufSockInit(
                internal.fd,
                internal.queue.clone(),
                internal.buf.clone(),
                false,
            )
            .unwrap();
        }

        return Self(Arc::new(QMutex::new(internal)));
    }

    pub fn InitForegroundProcessGroup(&self, pg: &ProcessGroup) {
        let mut t = self.lock();
        if t.fgProcessgroup.is_some() {
            panic!("foreground process group is already set");
        }

        t.fgProcessgroup = Some(pg.clone());
        t.session = Some(pg.Session());
    }

    pub fn ForegroundProcessGroup(&self) -> Option<ProcessGroup> {
        return self.lock().fgProcessgroup.clone();
    }
}

impl Waitable for TTYFileOps {
    fn Readiness(&self, task: &Task, mask: EventMask) -> EventMask {
        if SHARESPACE.config.read().UringIO && ENABLE_RINGBUF {
            return self.lock().buf.Events() & mask;
        }

        let fops = self.lock().fileOps.clone();
        return fops.Readiness(task, mask);
    }

    fn EventRegister(&self, task: &Task, e: &WaitEntry, mask: EventMask) {
        let queue = self.lock().queue.clone();
        queue.EventRegister(task, e, mask);
        let fd = self.lock().fd;
        if !SHARESPACE.config.read().UringIO && ENABLE_RINGBUF {
            UpdateFD(fd).unwrap();
        };
    }

    fn EventUnregister(&self, task: &Task, e: &WaitEntry) {
        let queue = self.lock().queue.clone();
        queue.EventUnregister(task, e);
        let fd = self.lock().fd;
        if !SHARESPACE.config.read().UringIO && ENABLE_RINGBUF {
            UpdateFD(fd).unwrap();
        };
    }
}

impl SpliceOperations for TTYFileOps {}

impl FileOperations for TTYFileOps {
    fn as_any(&self) -> &Any {
        return self;
    }

    fn FopsType(&self) -> FileOpsType {
        return FileOpsType::TTYFileOps;
    }

    fn Seekable(&self) -> bool {
        return false;
    }

    fn Seek(&self, task: &Task, f: &File, whence: i32, current: i64, offset: i64) -> Result<i64> {
        let fops = self.lock().fileOps.clone();
        let res = fops.Seek(task, f, whence, current, offset);
        return res;
    }

    fn ReadDir(
        &self,
        task: &Task,
        f: &File,
        offset: i64,
        serializer: &mut DentrySerializer,
    ) -> Result<i64> {
        let fops = self.lock().fileOps.clone();
        let res = fops.ReadDir(task, f, offset, serializer);
        return res;
    }

    fn ReadAt(
        &self,
        task: &Task,
        f: &File,
        dsts: &mut [IoVec],
        offset: i64,
        blocking: bool,
    ) -> Result<i64> {
        self.lock().checkChange(task, Signal(Signal::SIGTTIN))?;

        if SHARESPACE.config.read().UringIO && ENABLE_RINGBUF {
            let fd = self.lock().fd;
            let queue = self.lock().queue.clone();
            let ringBuf = self.lock().buf.clone();

            let ret = QUring::RingFileRead(task, fd, queue, ringBuf, dsts, false, false)?;
            return Ok(ret);
        }

        let fops = self.lock().fileOps.clone();
        let res = fops.ReadAt(task, f, dsts, offset, blocking);
        return res;
    }

    fn WriteAt(
        &self,
        task: &Task,
        f: &File,
        srcs: &[IoVec],
        offset: i64,
        blocking: bool,
    ) -> Result<i64> {
        {
            let t = self.lock();
            if t.termios.LEnabled(LocalFlags::TOSTOP) {
                t.checkChange(task, Signal(Signal::SIGTTOU))?;
            }
        }

        let size = IoVec::NumBytes(srcs);
        if size == 0 {
            return Ok(0);
        }

        if SHARESPACE.config.read().UringIO && ENABLE_RINGBUF {
            let fd = self.lock().fd;
            let queue = self.lock().queue.clone();
            let ringBuf = self.lock().buf.clone();

            return QUring::RingFileWrite(task, fd, queue, ringBuf, srcs, Arc::new(self.clone()));
        }

        let fops = self.lock().fileOps.clone();
        let res = fops.WriteAt(task, f, srcs, offset, blocking);
        return res;
    }

    fn Append(&self, task: &Task, f: &File, srcs: &[IoVec]) -> Result<(i64, i64)> {
        {
            let t = self.lock();
            if t.termios.LEnabled(LocalFlags::TOSTOP) {
                t.checkChange(task, Signal(Signal::SIGTTOU))?;
            }
        }

        let fops = self.lock().fileOps.clone();
        let res = fops.Append(task, f, srcs);
        return res;
    }

    fn Fsync(&self, task: &Task, f: &File, start: i64, end: i64, syncType: SyncType) -> Result<()> {
        let fops = self.lock().fileOps.clone();
        let res = fops.Fsync(task, f, start, end, syncType);
        return res;
    }

    fn Flush(&self, task: &Task, f: &File) -> Result<()> {
        let fops = self.lock().fileOps.clone();
        let res = fops.Flush(task, f);
        return res;
    }

    fn UnstableAttr(&self, task: &Task, f: &File) -> Result<UnstableAttr> {
        let fops = self.lock().fileOps.clone();
        let res = fops.UnstableAttr(task, f);
        return res;
    }

    fn Ioctl(&self, task: &Task, _f: &File, _fd: i32, request: u64, val: u64) -> Result<u64> {
        let fops = self.lock().fileOps.clone();
        let fd = fops
            .as_any()
            .downcast_ref::<HostFileOp>()
            .expect("Ioctl: not Hostfilop")
            .InodeOp
            .as_any()
            .downcast_ref::<HostInodeOp>()
            .expect("Ioctl: not HostInodeOp")
            .HostFd();
        let ioctl = request;

        match ioctl {
            IoCtlCmd::TCGETS => {
                //error!("TCGETS 1");
                let mut term = Termios::default();
                ioctlGetTermios(fd, &mut term)?;
                //error!("TCGETS 2 {:x?}", term);
                task.CopyOutObj(&term, val)?;
                return Ok(0);
            }

            IoCtlCmd::TCSETS | IoCtlCmd::TCSETSW | IoCtlCmd::TCSETSF => {
                self.lock().checkChange(task, Signal(Signal::SIGTTOU))?;

                let t: Termios = task.CopyInObj(val)?;
                ioctlSetTermios(fd, ioctl, &t)?;
                self.lock().termios.FromTermios(&t);
                return Ok(0);
            }
            IoCtlCmd::TIOCGPGRP => {
                let thread = task.Thread();
                let tg = thread.ThreadGroup();
                let pidns = tg.PIDNamespace();

                let internal = self.lock();

                let pgid = pidns.IDOfProcessGroup(internal.fgProcessgroup.as_ref().unwrap());
                info!("TIOCGPGRP pgid is {}, val is {:x}", pgid, val);

                task.CopyOutObj(&pgid, val)?;

                return Ok(0);
            }
            IoCtlCmd::TIOCSPGRP => {
                //error!("TIOCSPGRP 1");
                let thread = match &task.thread {
                    None => return Err(Error::SysError(SysErr::ENOTTY)),
                    Some(ref t) => t.clone(),
                };

                let mut t = self.lock();
                match t.checkChange(task, Signal(Signal::SIGTTOU)) {
                    // drivers/tty/tty_io.c:tiocspgrp() converts -EIO from
                    // tty_check_change() to -ENOTTY.
                    Err(Error::SysError(SysErr::EIO)) => {
                        return Err(Error::SysError(SysErr::ENOTTY))
                    }
                    Err(e) => return Err(e),
                    Ok(()) => (),
                }

                let tg = thread.ThreadGroup();
                let session = tg.Session();
                if session != t.session {
                    return Err(Error::SysError(SysErr::ENOTTY));
                }

                let pgid: i32 = task.CopyInObj(val)?;
                //error!("TIOCSPGRP 2 {}", pgid);
                if pgid < 0 {
                    return Err(Error::SysError(SysErr::EINVAL));
                }

                let pidns = tg.PIDNamespace();
                let pg = match pidns.ProcessGroupWithID(pgid) {
                    None => return Err(Error::SysError(SysErr::ESRCH)),
                    Some(pg) => pg,
                };

                // Check that new process group is in the TTY session.
                if pg.Session() != t.session.clone().unwrap() {
                    return Err(Error::SysError(SysErr::EPERM));
                }

                t.fgProcessgroup = Some(pg);
                return Ok(0);
            }
            IoCtlCmd::TIOCGWINSZ => {
                //error!("TIOCGWINSZ 1");
                let mut win = Winsize::default();
                ioctlGetWinsize(fd, &mut win)?;
                //error!("TIOCGWINSZ 2 {:x?}", win);
                task.CopyOutObj(&win, val)?;
                return Ok(0);
            }
            IoCtlCmd::TIOCSWINSZ => {
                let w: Winsize = task.CopyInObj(val)?;
                ioctlSetWinsize(fd, &w)?;
                return Ok(0);
            }
            IoCtlCmd::TIOCSETD
            | IoCtlCmd::TIOCSBRK
            | IoCtlCmd::TIOCCBRK
            | IoCtlCmd::TCSBRK
            | IoCtlCmd::TCSBRKP
            | IoCtlCmd::TIOCSTI
            | IoCtlCmd::TIOCCONS
            | IoCtlCmd::FIONBIO
            | IoCtlCmd::TIOCEXCL
            | IoCtlCmd::TIOCNXCL
            | IoCtlCmd::TIOCGEXCL
            | IoCtlCmd::TIOCNOTTY
            | IoCtlCmd::TIOCSCTTY
            | IoCtlCmd::TIOCGSID
            | IoCtlCmd::TIOCGETD
            | IoCtlCmd::TIOCVHANGUP
            | IoCtlCmd::TIOCGDEV
            | IoCtlCmd::TIOCMGET
            | IoCtlCmd::TIOCMSET
            | IoCtlCmd::TIOCMBIC
            | IoCtlCmd::TIOCMBIS
            | IoCtlCmd::TIOCGICOUNT
            | IoCtlCmd::TCFLSH
            | IoCtlCmd::TIOCSSERIAL
            | IoCtlCmd::TIOCGPTPEER => {
                //not implmentated
                return Err(Error::SysError(SysErr::ENOTTY));
            }
            _ => return Err(Error::SysError(SysErr::ENOTTY)),
        }
    }

    fn IterateDir(
        &self,
        task: &Task,
        d: &Dirent,
        dirCtx: &mut DirCtx,
        offset: i32,
    ) -> (i32, Result<i64>) {
        let fops = self.lock().fileOps.clone();

        return fops.IterateDir(task, d, dirCtx, offset);
    }

    fn Mappable(&self) -> Result<MMappable> {
        return Err(Error::SysError(SysErr::ENODEV));
    }
}

impl SockOperations for TTYFileOps {}
