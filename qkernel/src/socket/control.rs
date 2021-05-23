// Copyright (c) 2021 Quark Container Authors
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

use alloc::vec::Vec;
use alloc::slice;
use core::mem;
use core::ptr;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::auth::id::*;
use super::super::qlib::linux::socket::*;
use super::super::qlib::linux::time::*;
use super::super::kernel::fd_table::*;
use super::super::threadmgr::thread::*;
use super::super::fs::file::*;
use super::super::task::*;
use super::unix::transport::unix::*;
use super::epsocket::epsocket::*;

/*pub trait SCMRights {
    fn Files(&mut self, task: &Task, max: usize) -> (RightsFiles, bool);
}*/

// RightsFiles represents a SCM_RIGHTS socket control message. A reference is
// maintained for each fs.File and is release either when an FD is created or
// when the Release method is called.
#[derive(Clone)]
pub struct SCMRights(pub Vec<File>);

impl SCMRights {
    pub fn Files(&mut self, _task: &Task, max: usize) -> (SCMRights, bool) {
        let mut n = max;
        let mut trunc = false;

        let l = self.0.len();
        if n > l {
            n = l;
        } else if n < l {
            n = l;
            trunc = true;
        }

        let right = self.0.split_off(n);
        let left = self.0.split_off(0);
        self.0 = right;
        return (Self(left), trunc)
    }

    // NewSCMRights creates a new SCM_RIGHTS socket control message representation
    pub fn New(task: &Task, fds: &[i32]) -> Result<Self> {
        let mut files = Vec::new();

        for fd in fds {
            let file = task.GetFile(*fd)?;
            files.push(file);
        }

        return Ok(Self(files))
    }

    pub fn Clone(&self) -> Self {
        let mut ret = Vec::new();
        for f in &self.0 {
            ret.push(f.clone());
        }

        return Self(ret)
    }

    // rightsFDs gets up to the specified maximum number of FDs.
    pub fn RightsFDs(&mut self, task: &Task, cloexec: bool, max: usize) -> (Vec<i32>, bool) {
        info!("RightsFDs len is {}", self.0.len());
        let (files, trunc) = self.Files(task, max);
        let mut fds = Vec::with_capacity(files.0.len());
        for i in 0..core::cmp::min(max, files.0.len()) {
            let fd = match task.NewFDFrom(0, &files.0[i], &FDFlags{CloseOnExec: cloexec}) {
                Err(e) => {
                    info!("Error inserting FD: {:?}", e);
                    break;
                }
                Ok(fd) => fd,
            };

            info!("RightsFDs fd is {}", fd);
            fds.push(fd);
        }

        return (fds, trunc)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct ScmCredentials {
    pub thread: Thread,
    pub kuid: KUID,
    pub kgid: KGID,
}

impl ScmCredentials {
    pub fn New(task: &Task, creds: &ControlMessageCredentials) -> Result<Self> {
        let tcreds = task.Creds();
        let kuid = tcreds.UseUID(UID(creds.UID))?;
        let kgid = tcreds.UseGID(GID(creds.GID))?;
        let thread = task.Thread();
        let userns = thread.PIDNamespace().UserNamespace();
        if creds.PID != thread.ThreadGroup().ID() && !thread.HasCapabilityIn(Capability::CAP_SYS_ADMIN, &userns) {
            return Err(Error::SysError(SysErr::EPERM))
        }

        return Ok(Self {
            thread: thread,
            kuid: kuid,
            kgid: kgid,
        })
    }

    pub fn Clone(&self) -> Self {
        return Self{
            thread: self.thread.clone(),
            kuid: self.kuid,
            kgid: self.kgid,
        };
    }

    pub fn Credentials(&self) -> ControlMessageCredentials {
        return ControlMessageCredentials {
            PID: self.thread.ThreadID(),
            UID: self.kuid.0,
            GID: self.kgid.0,
        }
    }
}

/// Copy the in-memory representation of `src` into the byte slice `dst`.
///
/// Returns the remainder of `dst`.
///
/// Panics when `dst` is too small for `src` (more precisely, panics if
/// `mem::size_of_val(src) >= dst.len()`).

pub fn CopyBytes<'a, T: ?Sized>(src: &T, dst: &'a mut [u8]) -> &'a mut [u8] {
    let srclen = mem::size_of_val(src);
    info!("CopyBytes srclen is {}", srclen);
    assert!(dst.len() >= core::mem::size_of_val(src));
    unsafe {
        ptr::copy_nonoverlapping(src as * const T as * const u8, dst[..srclen].as_mut_ptr(), srclen);
    }

    return &mut dst[srclen..]
}

/// Fills `dst` with `len` zero bytes and returns the remainder of the slice.
///
/// Panics when `len >= dst.len()`.
pub fn PadBytes<'a> (len: usize, dst: &'a mut [u8]) -> &'a mut [u8] {
    for pad in &mut dst[..len] {
        *pad = 0
    }

    return &mut dst[len..]
}

pub const SCM_RIGHTS      : i32 = 0x1;
pub const SCM_CREDENTIALS : i32 = 0x2;
pub const SCM_TIMESTAMP   : i32 = SO_TIMESTAMP;

// A ControlMessageHeader is the header for a socket control message.
//
// ControlMessageHeader represents struct cmsghdr from linux/socket.h.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct ControlMessageHeader {
    pub Length : u64,
    pub Level: i32,
    pub Type: i32,
}

// SizeOfControlMessageHeader is the binary size of a ControlMessageHeader
// struct.
pub const SIZE_OF_CONTROL_MESSAGE_HEADER : usize = 16;

// A ControlMessageCredentials is an SCM_CREDENTIALS socket control message.
//
// ControlMessageCredentials represents struct ucred from linux/socket.h.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct ControlMessageCredentials {
    pub PID: i32,
    pub UID: u32,
    pub GID: u32,
}

impl ControlMessageCredentials {
    pub fn Empty() -> ControlMessageCredentials {
        ControlMessageCredentials {
            PID: 0,
            UID: NOBODY_KUID.0,
            GID: NOBODY_KGID.0,
        }
    }
}

impl ControlMessage for ControlMessageCredentials {
    fn CMsgLevel(&self) -> i32 {
        return SOL_SOCKET
    }

    fn Len(&self) -> usize {
        let headerLen = CMsgAlign(mem::size_of::<ControlMessageHeader>());
        let bodyLen = mem::size_of_val(&self);
        return headerLen + bodyLen;
    }

    fn CMsgType(&self) -> i32 {
        return SCM_CREDENTIALS;
    }

    fn EncodeInto<'a> (&self, buf: &'a mut [u8], flags: i32) -> (&'a mut [u8], i32) {
        let space = AlignDown(buf.len(), 4);
        let mut flags = flags;

        if space < mem::size_of::<ControlMessageHeader>(){
            flags |= MsgType::MSG_CTRUNC;
            return (buf, flags)
        }

        let mut length = 4 * 3 + mem::size_of::<ControlMessageHeader>();
        if length > space {
            flags |= MsgType::MSG_CTRUNC;
            length = space;
        }

        let cmsg = ControlMessageHeader {
            Length: length as u64, //self.Len() as _,
            Level: self.CMsgLevel(),
            Type: self.CMsgType(),
        };

        let buf = CopyBytes(&cmsg, buf);

        let buf = if buf.len() >= 4 {
            CopyBytes(&self.PID, buf)
        } else {
            return (buf, flags)
        };

        let buf = if buf.len() >= 4 {
            CopyBytes(&self.UID, buf)
        } else {
            return (buf, flags)
        };

        let buf = if buf.len() >= 4 {
            CopyBytes(&self.GID, buf)
        } else {
            return (buf, flags)
        };

        let aligned = AlignUp(length, ALIGNMENT) - length;
        if aligned > buf.len() {
            return (buf, flags)
        }

        return (&mut buf[aligned..], flags)
    }
}

// SizeOfControlMessageCredentials is the binary size of a
// ControlMessageCredentials struct.
pub const SIZE_OF_CONTROL_MESSAGE_CREDENTIALS : usize = 12;

pub trait ControlMessage {
    fn CMsgLevel(&self) -> i32;
    fn Len(&self) -> usize;

    fn Space(&self) -> usize {
        return CMsgAlign(self.Len());
    }

    fn CMsgType(&self) -> i32;
    fn EncodeInto<'a> (&self, buf: &'a mut [u8], flags: i32) -> (&'a mut [u8], i32);
}

pub const ALIGNMENT : usize = 8;

// A ControlMessageRights is an SCM_RIGHTS socket control message.
#[derive(Debug, Default, Clone)]
pub struct ControlMessageRights(pub Vec<i32>);

impl ControlMessage for ControlMessageRights {
    fn CMsgLevel(&self) -> i32 {
        return SOL_SOCKET
    }

    fn Len(&self) -> usize {
        let headerLen = CMsgAlign(mem::size_of::<ControlMessageHeader>());
        let bodyLen = mem::size_of_val(&self.0[..]);
        return headerLen + bodyLen;
    }

    fn CMsgType(&self) -> i32 {
        return SCM_RIGHTS;
    }

    fn EncodeInto<'a> (&self, buf: &'a mut [u8], flags: i32) -> (&'a mut [u8], i32) {
        let space = AlignDown(buf.len(), 4);
        let mut flags = flags;

        if space < mem::size_of::<ControlMessageHeader>() {
            flags |= MsgType::MSG_CTRUNC;
            return (buf, flags)
        }

        let mut length = 4 * self.0.len() + mem::size_of::<ControlMessageHeader>();
        if length > space {
            flags |= MsgType::MSG_CTRUNC;
            length = space;
        }

        let cmsg = ControlMessageHeader {
            Length: length as _, //self.Len() as _,
            Level: self.CMsgLevel(),
            Type: self.CMsgType(),
        };

        let buf = CopyBytes(&cmsg, buf);
        let cnt = core::cmp::min(length - mem::size_of::<ControlMessageHeader>(), buf.len());
        let buf = CopyBytes(&self.0[..cnt/4], buf);

        let aligned = AlignUp(length, ALIGNMENT) - length;
        if aligned > buf.len() {
            return (buf, flags)
        }

        return (&mut buf[aligned..], flags)
    }
}

pub fn AlignSlice<'a>(buf: &'a mut [u8], align: usize) -> &'a mut [u8] {
    let aligned = AlignUp(buf.len(), align);
    if aligned > buf.len() {
        // Linux allows unaligned data if there isn't room for alignment.
        // Since there isn't room for alignment, there isn't room for any
        // additional messages either.
        return buf
    }

    return &mut buf[aligned..]
}

#[derive(Debug, Default, Clone)]
pub struct ControlMessageTimeStamp(Timeval);

impl ControlMessage for ControlMessageTimeStamp {
    fn CMsgLevel(&self) -> i32 {
        return SOL_SOCKET
    }

    fn Len(&self) -> usize {
        let headerLen = CMsgAlign(mem::size_of::<ControlMessageHeader>());
        let bodyLen = mem::size_of_val(&self.0);
        return headerLen + bodyLen;
    }

    fn CMsgType(&self) -> i32 {
        return SCM_TIMESTAMP;
    }

    fn EncodeInto<'a> (&self, buf: &'a mut [u8], flags: i32) -> (&'a mut [u8], i32) {
        let space = AlignDown(buf.len(), 4);
        let mut flags = flags;

        if space < mem::size_of::<ControlMessageHeader>() {
            flags |= MsgType::MSG_CTRUNC;
            return (buf, flags)
        }

        let length = 2 * 8 + mem::size_of::<ControlMessageHeader>();
        if length > space {
            flags |= MsgType::MSG_CTRUNC;
            return (buf, flags)
        }

        let cmsg = ControlMessageHeader {
            Length: self.Len() as _,
            Level: self.CMsgLevel(),
            Type: self.CMsgType(),
        };

        let buf = CopyBytes(&cmsg, buf);
        let buf = CopyBytes(&self.0, buf);

        let aligned = AlignUp(length, ALIGNMENT) - length;
        if aligned > buf.len() {
            return (buf, flags)
        }

        return (&mut buf[aligned..], flags)
    }
}

pub type AlignedOfCmsgData = usize;

// Round `len` up to meet the platform's required alignment for
// `cmsghdr`s and trailing `cmsghdr` data.  This should match the
// behaviour of CMSG_ALIGN from the Linux headers and do the correct
// thing on other platforms that don't usually provide CMSG_ALIGN.
#[inline]
pub fn CMsgAlign(len: usize) -> usize {
    let alignBytes = mem::size_of::<AlignedOfCmsgData>() - 1;
    return (len + alignBytes) & !alignBytes;
}

// CmsgSpace returns the number of bytes an ancillary element with
// payload of the passed data length occupies.
#[inline]
pub fn CMsgSpace(datalen: usize) -> usize {
    return CMsgAlign(SIZEOF_CMSGHDR) + CMsgAlign(datalen);
}

// SizeOfControlMessageRight is the size of a single element in
// ControlMessageRights.
pub const SIZE_OF_CONTROL_MESSAGE_RIGHT : usize = 4;

// SCM_MAX_FD is the maximum number of FDs accepted in a single sendmsg call.
// From net/scm.h.
pub const SCM_MAX_FD : usize = 253;

// SO_ACCEPTCON is defined as __SO_ACCEPTCON in
// include/uapi/linux/net.h, which represents a listening socket
// state. Note that this is distinct from SO_ACCEPTCONN, which is a
// socket option for querying whether a socket is in a listening
// state.
pub const SO_ACCEPTCON : i32 = 1 << 16;

// A ControlMessages represents a collection of socket control messages.
#[derive(Debug, Default, Clone)]
pub struct ControlMessages {
    pub Rights: Option<ControlMessageRights>,
    pub Credentials: Option<ControlMessageCredentials>,
    pub Timestamps: Option<ControlMessageTimeStamp>,
}

impl ControlMessages {
    pub fn Empty(&self) -> bool {
        return self.Rights.is_none() && self.Credentials.is_none() && self.Timestamps.is_none()
    }

    pub fn ToSCMUnix(&self, task: &Task, ep: &BoundEndpoint, toEp: &Option<BoundEndpoint>) -> Result<SCMControlMessages> {
        let rights = match self.Rights {
            None => None,
            Some(ref rights) => {
                Some(SCMRights::New(task, &rights.0[..])?)
            }
        };

        let creds = match self.Credentials {
            None => {
                if ep.Passcred() || ep.ConnectedPasscred() {
                    MakeCreds(task, None)
                } else if toEp.is_some() && toEp.as_ref().unwrap().Passcred(){
                    MakeCreds(task, None)
                } else {
                    None
                }
            },
            Some(ref creds) => {
                Some(ScmCredentials::New(task, &creds)?)
            }
        };

        return Ok(SCMControlMessages {
            Rights: rights,
            Credentials: creds,
        })
    }
}

// AlignUp rounds a length up to an alignment. align must be a power of 2.
pub fn AlignUp(length: usize, align: usize) -> usize {
    return (length + align - 1) & !(align - 1)
}

// AlignDown rounds a down to an alignment. align must be a power of 2.
pub fn AlignDown(length: usize, align: usize) -> usize {
    return length & !(align - 1)
}

pub fn Parse(buf : &[u8]) -> Result<ControlMessages> {
    let mut fds = ControlMessageRights::default();
    let mut creds = ControlMessageCredentials::default();
    let mut hasCreds = false;

    let mut i = 0;
    while i < buf.len() {
        if i + SIZE_OF_CONTROL_MESSAGE_HEADER > buf.len() {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        let h = unsafe {
            &*(buf[i..i + SIZE_OF_CONTROL_MESSAGE_HEADER].as_ptr() as * const ControlMessageHeader)
        };

        if (h.Length as usize) < SIZE_OF_CONTROL_MESSAGE_HEADER || h.Length as usize > buf.len() - i {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        if h.Level != LibcConst::SOL_SOCKET as i32 {
            return Err(Error::SysError(SysErr::EINVAL))
        }

        i += SIZE_OF_CONTROL_MESSAGE_HEADER;
        let length = h.Length as usize - SIZE_OF_CONTROL_MESSAGE_HEADER;

        let width = 8;

        match h.Type {
            SCM_RIGHTS => {
                let rightsSize = AlignDown(length, SIZE_OF_CONTROL_MESSAGE_RIGHT);
                let numRights = rightsSize / SIZE_OF_CONTROL_MESSAGE_RIGHT;

                let cnt = fds.0.len();
                if cnt + numRights > SCM_MAX_FD {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                let ptr = &buf[i] as *const _ as * const i32;
                assert!(buf[i..].len() >= 4 * numRights);
                let rights = unsafe { slice::from_raw_parts(ptr, numRights) };

                for r in rights {
                    fds.0.push(*r);
                }

                i += AlignUp(length, width);
            }
            SCM_CREDENTIALS => {
                if length < SIZE_OF_CONTROL_MESSAGE_CREDENTIALS {
                    return Err(Error::SysError(SysErr::EINVAL))
                }

                assert!(buf[i..].len() >= core::mem::size_of::<ControlMessageCredentials>());
                let c = unsafe {
                    &*(&buf[i] as * const _ as * const ControlMessageCredentials)
                };

                creds = *c;
                hasCreds = true;
                i += AlignUp(length, width) ;
            }
            _ => {
                return Err(Error::SysError(SysErr::EINVAL))
            }
        }
    }

    let mut ret = ControlMessages::default();
    if hasCreds {
        ret.Credentials = Some(creds);
    }

    if fds.0.len() > 0 {
        let mut rights = ControlMessageRights::default();
        rights.0.append(&mut fds.0);
        ret.Rights = Some(rights)
    }

    return Ok(ret)
}

pub fn MakeCreds(task: &Task, _cred: Option<BoundEndpoint>) -> Option<ScmCredentials> {
    //TODO: this is duplicating the function of scmCredentials::new, refactoring this
    /*let cr = match cred {
        None => return None,
        Some(cr) => cr,
    };*/

    //if cr.Passcred() || cr.ConnectedPasscred() || true {
    if true {
        //info!("MakeCreds cr.Passcred() in...");
        let tcred = task.Creds();
        let kuid = tcred.lock().EffectiveKUID;
        let kgid = tcred.lock().EffectiveKGID;
        return Some(ScmCredentials {
            thread: task.Thread(),
            kuid: kuid,
            kgid: kgid,
        });
    }

    return None
}

pub fn NewControlMessage(task: &Task, cred: Option<BoundEndpoint>, rights: Option<SCMRights>) -> SCMControlMessages {
    return SCMControlMessages {
        Credentials: MakeCreds(task, cred),
        Rights: rights,
    }
}