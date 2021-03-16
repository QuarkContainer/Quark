// Copyright (c) 2021 QuarkSoft LLC
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
use core::ops::Deref;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

#[derive(Default, Debug)]
pub struct IOInternal {
    // CharsRead is the number of bytes read by read syscalls.
    pub CharsRead: AtomicU64,

    // CharsWritten is the number of bytes written by write syscalls.
    pub CharsWritten: AtomicU64,

    // ReadSyscalls is the number of read syscalls.
    pub ReadSyscalls: AtomicU64,

    // WriteSyscalls is the number of write syscalls.
    pub WriteSyscalls: AtomicU64,

    // BytesRead is the number of bytes actually read into pagecache.
    pub BytesRead: AtomicU64,

    // BytesWritten is the number of bytes actually written from pagecache.
    pub BytesWritten: AtomicU64,

    // BytesWriteCancelled is the number of bytes not written out due to
    // truncation.
    pub BytesWriteCancelled: AtomicU64,
}

#[derive(Clone, Default, Debug)]
pub struct IO(Arc<IOInternal>);

impl Deref for IO {
    type Target = Arc<IOInternal>;

    fn deref(&self) -> &Arc<IOInternal> {
        &self.0
    }
}

impl IO {
    pub fn AccountReadSyscall(&self, bytes: i64) {
        self.ReadSyscalls.fetch_add(1, Ordering::SeqCst);
        if bytes > 0 {
            self.CharsRead.fetch_add(bytes as u64, Ordering::SeqCst);
        }
    }

    pub fn WriteSyscallAddr(&self) -> u64 {
        return &self.WriteSyscalls as * const _ as u64;
    }

    pub fn AccountWriteSyscall(&self, bytes: i64) {
        self.WriteSyscalls.fetch_add(1, Ordering::SeqCst);
        if bytes > 0 {
            self.CharsWritten.fetch_add(bytes as u64, Ordering::SeqCst);
        }
    }

    pub fn AccountReadIO(&self, bytes: i64) {
        if bytes > 0 {
            self.BytesRead.fetch_add(bytes as u64, Ordering::SeqCst);
        }
    }

    pub fn AccountWriteIO(&self, bytes: i64) {
        if bytes > 0 {
            self.BytesWritten.fetch_add(bytes as u64, Ordering::SeqCst);
        }
    }

    pub fn Accumulate(&self, io: &IO) {
        self.CharsRead.fetch_add(io.CharsRead.load(Ordering::SeqCst), Ordering::SeqCst);
        self.CharsWritten.fetch_add(io.CharsWritten.load(Ordering::SeqCst), Ordering::SeqCst);
        self.ReadSyscalls.fetch_add(io.ReadSyscalls.load(Ordering::SeqCst), Ordering::SeqCst);
        self.WriteSyscalls.fetch_add(io.WriteSyscalls.load(Ordering::SeqCst), Ordering::SeqCst);
        self.BytesRead.fetch_add(io.BytesRead.load(Ordering::SeqCst), Ordering::SeqCst);
        self.BytesWritten.fetch_add(io.BytesWritten.load(Ordering::SeqCst), Ordering::SeqCst);
        self.BytesWriteCancelled.fetch_add(io.BytesWriteCancelled.load(Ordering::SeqCst), Ordering::SeqCst);
    }
}