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

use core::ops::Deref;
use super::super::mutex::*;
//use core::sync::atomic::AtomicU64;
//use core::sync::atomic::Ordering;

pub enum MemoryKind {
    // System represents miscellaneous system memory. This may include
    // memory that is in the process of being reclaimed, system caches,
    // page tables, swap, etc.
    //
    // This memory kind is backed by platform memory.
    System,

    // Anonymous represents anonymous application memory.
    //
    // This memory kind is backed by platform memory.
    Anonymous,

    // PageCache represents memory allocated to back sandbox-visible files that
    // do not have a local fd. The contents of these files are buffered in
    // memory to support application mmaps.
    //
    // This memory kind is backed by platform memory.
    PageCache,

    // Tmpfs represents memory used by the sandbox-visible tmpfs.
    //
    // This memory kind is backed by platform memory.
    Tmpfs,

    // Ramdiskfs represents memory used by the ramdiskfs.
    //
    // This memory kind is backed by platform memory.
    Ramdiskfs,

    // Mapped represents memory related to files which have a local fd on the
    // host, and thus can be directly mapped. Typically these are files backed
    // by gofers with donated-fd support. Note that this value may not track the
    // exact amount of memory used by mapping on the host, because we don't have
    // any visibility into the host kernel memory management. In particular,
    // once we map some part of a host file, the host kernel is free to
    // abitrarily populate/decommit the pages, which it may do for various
    // reasons (ex. host memory reclaim, NUMA balancing).
    //
    // This memory kind is backed by the host pagecache, via host mmaps.
    Mapped,
}

// MemoryStats tracks application memory usage in bytes. All fields correspond to the
// memory category with the same name. This object is thread-safe if accessed
// through the provided methods. The public fields may be safely accessed
// directly on a copy of the object obtained from Memory.Copy().
#[derive(Copy, Clone)]
pub struct MemoryStats {
    pub System      : u64,
    pub Anonymous   : u64,
    pub PageCache   : u64,
    pub Tmpfs       : u64,
    // Lazily updated based on the value in RTMapped.
    pub Mapped      : u64,
    pub Ramdiskfs   : u64,
    pub RTMapped    : u64,
}

pub struct MemoryAccounting(QMutex<MemoryStats>);

impl Deref for MemoryAccounting {
    type Target = QMutex<MemoryStats>;

    fn deref(&self) -> &QMutex<MemoryStats> {
        &self.0
    }
}

impl MemoryStats {
    pub fn IncLocked(&mut self, val: u64, kind: MemoryKind) {
        match kind {
            MemoryKind::System => self.System += val,
            MemoryKind::Anonymous => self.Anonymous += val,
            MemoryKind::PageCache => self.PageCache += val,
            MemoryKind::Tmpfs => self.Tmpfs += val,
            MemoryKind::Mapped => self.RTMapped += val,
            MemoryKind::Ramdiskfs => self.Ramdiskfs += val,
        };
    }

    pub fn DecLocked(&mut self, val: u64, kind: MemoryKind) {
        match kind {
            MemoryKind::System => self.System -= val,
            MemoryKind::Anonymous => self.Anonymous -= val,
            MemoryKind::PageCache => self.PageCache -= val,
            MemoryKind::Tmpfs => self.Tmpfs -= val,
            MemoryKind::Mapped => self.RTMapped -= val,
            MemoryKind::Ramdiskfs => self.Ramdiskfs -= val,
        };
    }

    pub fn MoveLocked(&mut self, val: u64, to: MemoryKind, from: MemoryKind) {
        self.DecLocked(val, from);
        self.IncLocked(val, to);
    }

    pub fn TotalLocked(&self) -> u64 {
        let mut total = 0;
        total += self.System;
        total += self.Anonymous;
        total += self.PageCache;
        total += self.RTMapped;
        total += self.Tmpfs;
        total += self.Ramdiskfs;
        return total
    }
}

impl MemoryAccounting {
    pub fn Inc(&self, val: u64, kind: MemoryKind) {
        let mut m = self.lock();
        m.IncLocked(val, kind)
    }

    pub fn Dec(&self, val: u64, kind: MemoryKind) {
        let mut m = self.lock();
        m.DecLocked(val, kind)
    }

    pub fn Move(&self, val: u64, to: MemoryKind, from: MemoryKind) {
        let mut m = self.lock();
        m.MoveLocked(val, to, from)
    }

    pub fn Total(&self) -> u64 {
        let m = self.lock();
        return m.TotalLocked()
    }

    pub fn Copy(&self) -> MemoryStats {
        return *self.lock();
    }
}

// MinimumTotalMemoryBytes is the minimum reported total system memory.
pub const MINIMUM_TOTAL_MEMORY_BYTES: u64 = 1 << 30; // 2GB

// TotalMemory returns the "total usable memory" available.
//
// This number doesn't really have a true value so it's based on the following
// inputs and further bounded to be above some minimum guaranteed value (2GB),
// additionally ensuring that total memory reported is always less than used.
//
// memSize should be the platform.Memory size reported by platform.Memory.TotalSize()
// used is the total memory reported by MemoryLocked.Total()
pub fn TotalMemory(memSize: u64, used: u64) -> u64 {
    let mut memSize = memSize;
    if memSize < MINIMUM_TOTAL_MEMORY_BYTES {
        memSize = MINIMUM_TOTAL_MEMORY_BYTES;
    }

    if memSize < used {
        memSize = used;
        // Bump totalSize to the next largest power of 2, if one exists, so
        // that MemFree isn't 0.
        for i in 0..64 {
            let size = 1 << i;
            if size as u64 >= memSize {
                return size;
            }
        }
    }

    return memSize;
}