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

use alloc::vec::Vec;
use core::cmp;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::mem::stackvec::*;

// copyToMulti copies as many bytes from src to dst as possible.
pub fn CopytoMulti(dsts: &mut [IoVec], src: &[u8]) {
    let mut src = src;

    for dst in dsts {
        let dst = dst.ToSliceMut();
        let count = cmp::min(dst.len(), src.len());
        if count == 0 {
            break;
        }

        dst[..count].clone_from_slice(&src[..count]);
        src = &src[count..];
    }
}

// copyFromMulti copies as many bytes from src to dst as possible.
pub fn CopyFromMulti(dst: &mut [u8], srcs: &[IoVec]) {
    let mut idx = 0;
    for src in srcs {
        let src = src.ToSlice();
        for b in src {
            dst[idx] = *b;
            idx += 1;
        }
    }
}

// buildIovec builds an iovec slice from the given []byte slice.
//
// If truncate, truncate bufs > maxlen. Otherwise, immediately return an error.
//
// If length < the total length of bufs, err indicates why, even when returning
// a truncated iovec.
//
// If intermediate != nil, iovecs references intermediate rather than bufs and
// the caller must copy to/from bufs as necessary.

//ret: Result<totalLength, Option<intermedatebuf>, PartialWrite>
pub fn BuildIovec(
    bufs: &[IoVec],
    vecs: &mut StackVec<IoVec>,
    maxlen: usize,
    truncate: bool,
) -> Result<(usize, Option<Vec<u8>>, bool)> {
    let mut iovsRequired = 0;
    let mut length = 0;
    for b in bufs {
        length += b.Len();
        if b.Len() > 0 {
            iovsRequired += 1;
        }
    }

    let mut partial = false;

    let mut stopLen = length;
    if length > maxlen {
        if truncate {
            stopLen = maxlen;
            partial = true;
        } else {
            return Err(Error::SysError(SysErr::EMSGSIZE));
        }
    }

    if iovsRequired > UIO_MAXIOV {
        // The kernel will reject our call if we pass this many iovs.
        // Use a single intermediate buffer instead.
        let mut b = Vec::with_capacity(stopLen);
        unsafe {
            b.set_len(stopLen);
        }

        let iovec = IoVec {
            start: &b[0] as *const _ as u64,
            len: b.len(),
        };
        vecs.Push(iovec);

        return Ok((stopLen, Some(b), partial));
    }

    let mut total = 0;

    for b in bufs {
        let l = b.Len();
        if l == 0 {
            continue;
        }

        let mut stop = l;

        if total + stop > stopLen {
            stop = stopLen - total;
        }

        let iovec = IoVec {
            start: b.start,
            len: stop,
        };

        vecs.Push(iovec);
        total += stop;

        if total >= stopLen {
            break;
        }
    }

    return Ok((total, None, partial));
}
