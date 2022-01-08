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

use super::super::super::task::*;
use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::bytestream::*;
use super::super::host::tty::*;
use super::line_discipline::*;

const WAIT_BUF_MAX_BYTES: usize = 131072;
const WAIT_BUF_DEFAULT_PAGE_COUNT: u64 = 128 / 4;

pub struct Queue {
    pub buf: ByteStream,
    pub transform: fn(l: &mut LineDiscipline, q: &mut Queue, buf: &mut [u8]) -> usize,
    pub readable: bool,
}

impl Queue {
    pub fn NewInputQueue() -> Self {
        return Self {
            buf: ByteStream::Init(WAIT_BUF_DEFAULT_PAGE_COUNT),
            transform: inputQTransform,
            readable: false,
        }
    }

    pub fn NewOutputQueue() -> Self {
        return Self {
            buf: ByteStream::Init(WAIT_BUF_DEFAULT_PAGE_COUNT),
            transform: outputQTransform,
            readable: false,
        }
    }

    pub fn ReableSize(&self, task: &Task, dstAddr: u64) -> Result<()> {
        let size: i32 = if self.readable {
            self.buf.AvailableDataSize() as i32
        } else {
            0
        };

        task.CopyOutObj(&size, dstAddr)?;
        return Ok(())
    }

    pub fn Read(&mut self, dst: &mut [u8]) -> Result<i64> {
        if !self.readable {
            return Err(Error::SysError(SysErr::EAGAIN))
        }

        let mut dst = dst;
        if dst.len() > CANON_MAX_BYTES {
            dst = &mut dst[0..CANON_MAX_BYTES]
        }

        let (_, n) = self.buf.read(dst)?;

        if self.buf.AvailableDataSize() == 0 {
            self.readable = false;
        }

        return Ok(n as i64)
    }

    pub fn Write(&mut self, src: &mut [u8], l: &mut LineDiscipline) -> Result<i64> {
        if self.buf.AvailableSpace() == 0 {
            return Err(Error::SysError(SysErr::EAGAIN))
        }

        let n = (self.transform)(l, self, src);
        return Ok(n as i64)
    }
}

pub fn outputQTransform(l: &mut LineDiscipline, q: &mut Queue, buf: &mut [u8]) -> usize {
    if !l.termios.OEnabled(OutputFlags::OPOST) {
        let (_, len) = q.buf.write(buf).unwrap();
        if q.buf.AvailableDataSize() > 0 {
            q.readable = true;
        }
        return len
    }

    let mut ret = 0;
    let mut buf = buf;
    while buf.len() > 0 {
        let size = l.Peek(buf);
        let mut cBytes = buf[..size].to_vec();
        if q.buf.AvailableSpace() < size || q.buf.AvailableSpace() < 2 {
            return ret;
        }

        buf = &mut buf[size..];

        ret += size;

        match cBytes[0] as char {
            '\n' => {
                if l.termios.OEnabled(OutputFlags::ONLRET) {
                    l.column = 0;
                }

                if l.termios.OEnabled(OutputFlags::ONLCR) {
                    q.buf.write(&['\r' as u8, '\n' as u8]).unwrap();
                    continue;
                }
            }
            '\r' => {
                if l.termios.OEnabled(OutputFlags::ONOCR) {
                    continue
                }

                if l.termios.OEnabled(OutputFlags::OCRNL) {
                    cBytes[0] = '\n' as u8;
                    if l.termios.OEnabled(OutputFlags::ONLRET) {
                        l.column = 0;
                    }
                    break;
                }
                l.column = 0;
            }
            '\t' => {
                let spaces = SPACES_PER_TAB - l.column as usize % SPACES_PER_TAB;
                if l.termios.OutputFlags & OutputFlags::TABDLY == OutputFlags::XTABS {
                    l.column += spaces as i32;
                    for _i in 0..spaces {
                        q.buf.write(&[' ' as u8]).unwrap();
                    }

                    continue;
                }

                l.column += spaces as i32;
            }
            //'\b'
            '\x08' => {
                if l.column > 0 {
                    l.column -= 1;
                }
            }
            _ => {
                l.column += 1;
            }
        }

        q.buf.write(&cBytes).unwrap();
    }

    if q.buf.AvailableDataSize() > 0 {
        q.readable = true;
    }

    return ret;
}

pub fn inputQTransform(l: &mut LineDiscipline, q: &mut Queue, buf: &mut [u8]) -> usize {
    //todo: don't unerstand
    // If there's a line waiting to be read in canonical mode, don't write
    // anything else to the read buffer.
    if l.termios.LEnabled(LocalFlags::ICANON) && q.readable {
        return 0;
    }

    let mut maxBytes = NON_CANON_MAX_BYTES;
    if l.termios.LEnabled(LocalFlags::ICANON) {
        maxBytes = CANON_MAX_BYTES;
    }

    let mut ret = 0;

    let mut buf = buf;
    while buf.len() > 0 && q.buf.AvailableDataSize() < CANON_MAX_BYTES {
        let size = l.Peek(buf);
        let mut cBytes = buf[..size].to_vec();

        match cBytes[0] as char {
            '\r' => {
                if l.termios.IEnabled(InputFlags::IGNCR) {
                    buf = &mut buf[size..];
                    ret += size;
                    continue;
                }

                if l.termios.IEnabled(InputFlags::ICRNL) {
                    cBytes[0] = '\n' as u8;
                }
            }
            '\n' => {
                if l.termios.IEnabled(InputFlags::INLCR) {
                    cBytes[0] = '\r' as u8;
                }
            }
            _ => (),
        }

        if l.ShouldDiscard(q, &cBytes) {
            buf = &mut buf[size..];
            ret += size;
            continue;
        }

        if q.buf.AvailableDataSize() + size > maxBytes {
            break;
        }

        buf = &mut buf[size..];
        ret += size;

        if l.termios.LEnabled(LocalFlags::ICANON) && l.termios.IsEOF(cBytes[0]) {
            q.readable = true;
            break;
        }

        q.buf.write(&cBytes).unwrap();

        if l.termios.LEnabled(LocalFlags::ECHO) {
            let outQueue = l.outQueue.clone();
            outQueue.lock().Write(&mut cBytes, l).unwrap();
        }

        if l.termios.LEnabled(LocalFlags::ICANON) && l.termios.IsTerminating(&cBytes) {
            q.readable = true;
            break;
        }
    }

    if !l.termios.LEnabled(LocalFlags::ICANON) && q.buf.AvailableDataSize() > 0 {
        q.readable = true;
    }

    return ret;
}