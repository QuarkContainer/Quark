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

use alloc::string::String;
use alloc::string::ToString;
use core::u64;

use super::pagetable;
use super::pagetable::PageTableFlags;
use super::common::*;
use super::linux_def::*;
use super::range::*;

pub const PAGE_SHIFT: u64 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

pub const HUGE_PAGE_SHIFT: u64 = 21;
pub const HUGE_PAGE_SIZE: u64 = 1 << HUGE_PAGE_SHIFT;
pub const HUGE_PAGE_MASK: u64 = HUGE_PAGE_SIZE - 1;

pub const ONE_TB: u64 = 0x1_000_000_000; //0x10_000_000_000;
pub const KERNEL_BASE_ADDR: u64 = 7 * ONE_TB;
pub const KERNEL_ADDR_SIZE: u64 = 128 * ONE_TB;
pub const PHY_MEM_SPACE: u64 = 8 * ONE_TB;

pub const CHUNK_SHIFT: u64 = HUGE_PAGE_SHIFT;
pub const CHUNK_SIZE: u64 = 1 << CHUNK_SHIFT;
pub const CHUNK_MASK: u64 = CHUNK_SIZE - 1;
pub const CHUNK_PAGE_COUNT: u64 = CHUNK_SIZE / MemoryDef::PAGE_SIZE;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AccessType(pub u64);

impl AccessType {
    pub fn Default() -> Self {
        return AccessType(0);
    }

    #[cfg(target_arch = "x86_64")]
    pub fn NewFromPageFlags(flags: PageTableFlags) -> Self {
        let present: bool = flags & PageTableFlags::PRESENT == PageTableFlags::PRESENT;
        let useraccess = flags & PageTableFlags::USER_ACCESSIBLE == PageTableFlags::USER_ACCESSIBLE;
        if !present || !useraccess {
            return Self::New(false, false, false);
        }

        let write = flags & PageTableFlags::WRITABLE == PageTableFlags::WRITABLE;
        let exec = flags & PageTableFlags::NO_EXECUTE != PageTableFlags::NO_EXECUTE;
        return Self::New(present, write, exec);
    }

    #[cfg(target_arch = "aarch64")]
    pub fn NewFromPageFlags(flags: PageTableFlags) -> Self {
        let present = flags.contains(PageTableFlags::VALID);
        let useraccess = flags.contains(PageTableFlags::USER_ACCESSIBLE);
        if !present || !useraccess {
            return Self::New(false, false, false);
        }
        let write = !flags.contains(PageTableFlags::READ_ONLY);
        let exec = !flags.contains(PageTableFlags::UXN);

        return Self::New(present, write, exec);
    }

    #[cfg(target_arch = "x86_64")]
    pub fn ToUserPageFlags(&self) -> PageTableFlags {
        let mut flags = PageTableFlags::NO_EXECUTE;
        if self.Read() {
            flags |= PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
        }

        if self.Write() {
            flags |= PageTableFlags::WRITABLE;
        }

        if self.Exec() {
            flags &= !PageTableFlags::NO_EXECUTE;
        }

        return flags;
    }

    #[cfg(target_arch = "aarch64")]
    pub fn ToUserPageFlags(&self) -> PageTableFlags {
        let mut flags = PageTableFlags::UXN;
        if self.Read() {
            flags |= PageTableFlags::VALID | PageTableFlags::USER_ACCESSIBLE;
        }

        if !self.Write() {
            flags |= PageTableFlags::READ_ONLY;
        }

        if self.Exec() {
            flags &= !PageTableFlags::UXN;
        }

        return flags;
    }

    pub fn New(read: bool, write: bool, exec: bool) -> Self {
        let mut prot = 0;
        if read {
            prot |= MmapProt::PROT_READ;
        }

        if write {
            prot |= MmapProt::PROT_WRITE;
        }

        if exec {
            prot |= MmapProt::PROT_EXEC;
        }

        return Self(prot);
    }

    pub fn String(&self) -> String {
        let mut ret = "".to_string();
        if self.Read() {
            ret += "r";
        } else {
            ret += "-";
        }

        if self.Write() {
            ret += "w";
        } else {
            ret += "-";
        }

        if self.Exec() {
            ret += "x";
        } else {
            ret += "-";
        }

        return ret;
    }

    pub fn ReadOnly() -> Self {
        return AccessType(MmapProt::PROT_READ);
    }

    pub fn ReadWrite() -> Self {
        return AccessType(MmapProt::PROT_READ | MmapProt::PROT_WRITE);
    }

    pub fn Executable() -> Self {
        return AccessType(MmapProt::PROT_READ | MmapProt::PROT_EXEC);
    }

    pub fn AnyAccess() -> Self {
        return AccessType(MmapProt::PROT_READ | MmapProt::PROT_WRITE | MmapProt::PROT_EXEC);
    }

    pub fn Any(&self) -> bool {
        return self.Read() || self.Write() || self.Exec();
    }

    pub fn Read(&self) -> bool {
        self.0 & MmapProt::PROT_READ != 0
    }
    pub fn Write(&self) -> bool {
        self.0 & MmapProt::PROT_WRITE != 0
    }
    pub fn Exec(&self) -> bool {
        self.0 & MmapProt::PROT_EXEC != 0
    }

    pub fn SetProt(&mut self, prot: u64) {
        self.0 = prot;
    }

    pub fn SetRead(&mut self) -> &mut Self {
        self.0 |= MmapProt::PROT_READ;
        return self;
    }

    pub fn SetWrite(&mut self) -> &mut Self {
        self.0 |= MmapProt::PROT_WRITE;
        return self;
    }

    pub fn ClearWrite(&mut self) -> &mut Self {
        self.0 &= !MmapProt::PROT_WRITE;
        return self;
    }

    pub fn SetExec(&mut self) -> &mut Self {
        self.0 |= MmapProt::PROT_EXEC;
        return self;
    }

    pub fn Val(&self) -> u64 {
        return self.0;
    }

    // Effective returns the set of effective access types allowed by a, even if
    // some types are not explicitly allowed.
    pub fn Effective(&self) -> Self {
        // In Linux, Write and Execute access generally imply Read access. See
        // mm/mmap.c:protection_map.
        //
        // The notable exception is get_user_pages, which only checks against
        // the original vma flags. That said, most user memory accesses do not
        // use GUP.
        let mut res = *self;
        if res.Write() || res.Exec() {
            res.SetRead();
        }

        return res;
    }

    pub fn SupersetOf(&self, other: &Self) -> bool {
        if !self.Read() && other.Read() {
            return false;
        }

        if !self.Write() && other.Write() {
            return false;
        }

        if !self.Exec() && other.Exec() {
            return false;
        }

        return true;
    }
}

pub struct PageOpts(PageTableFlags);

#[cfg(target_arch = "aarch64")]
impl PageOpts {
    //const Empty : PageTableFlags = PageTableFlags::PRESENT & PageTableFlags::WRITABLE; //set 0

    pub fn New(user: bool, write: bool, exec: bool) -> Self {
        let mut flags = PageTableFlags::VALID | PageTableFlags::MT_NORMAL
                        | PageTableFlags::ACCESSED;
        if !write {
            flags |= PageTableFlags::READ_ONLY;
        }

        if user {
            flags |= PageTableFlags::USER_ACCESSIBLE;
        }

        if !exec {
            flags |= PageTableFlags::UXN;
        }

        return Self(flags);
    }

    pub fn All() -> Self {
        return PageOpts(
            PageTableFlags::VALID | PageTableFlags::USER_ACCESSIBLE,
        );
    }

    pub fn Zero() -> Self {
        return PageOpts(PageTableFlags::ZERO); //set 0
    }

    pub fn Kernel() -> Self {
        return PageOpts(
            PageTableFlags::VALID |
            PageTableFlags::MT_NORMAL |
            PageTableFlags::DIRTY |
            PageTableFlags::ACCESSED
        );
    }

    pub fn UserReadOnly() -> Self {
        return PageOpts(
            PageTableFlags::VALID |
            PageTableFlags::USER_ACCESSIBLE |
            PageTableFlags::READ_ONLY |
            PageTableFlags::MT_NORMAL |
            PageTableFlags::ACCESSED
        );
    }

    pub fn UserNonAccessable() -> Self {
        return PageOpts(PageTableFlags::VALID | PageTableFlags::ACCESSED | PageTableFlags::MT_NORMAL);
    }

    pub fn UserReadWrite() -> Self {
        return PageOpts(
            PageTableFlags::VALID |
            PageTableFlags::USER_ACCESSIBLE |
            PageTableFlags::MT_NORMAL |
            PageTableFlags::ACCESSED
        );
    }

    pub fn KernelReadOnly() -> Self {
        return PageOpts(PageTableFlags::VALID | PageTableFlags::READ_ONLY | PageTableFlags::MT_NORMAL | PageTableFlags::ACCESSED);
    }
    
    pub fn KernelReadWrite() -> Self {
        return PageOpts(PageTableFlags::VALID | PageTableFlags::MT_NORMAL | PageTableFlags::ACCESSED);
    }

    pub fn Present(&self) -> bool {
        return (self.0 & PageTableFlags::VALID) != Self::Zero().0;
    }

    pub fn Write(&self) -> bool {
        return (self.0 & PageTableFlags::READ_ONLY) == Self::Zero().0;
    }

    pub fn Global(&self) -> bool {
        return (self.0 & PageTableFlags::NON_GLOBAL) == Self::Zero().0;
    }

    pub fn UserAccess(&self) -> bool {
        return (self.0 & PageTableFlags::USER_ACCESSIBLE) != Self::Zero().0;
    }

    pub fn SetPresent(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::VALID;
        return self;
    }

    pub fn SetWrite(&mut self) -> &mut Self {
        self.0 &= !PageTableFlags::READ_ONLY;
        return self;
    }

    pub fn SetUserAccess(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::USER_ACCESSIBLE;
        return self;
    }

    pub fn SetGlobal(&mut self) -> &mut Self {
        self.0 &= !PageTableFlags::NON_GLOBAL;
        return self;
    }

	// Set a collection of PT flags to configure
	// device memory for MMIO.
    pub fn SetMMIOPage(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::MT_NORMAL_NC |
                  PageTableFlags::VALID |
                  PageTableFlags::PAGE |
                  PageTableFlags::INNER_SHAREABLE |
                  PageTableFlags::UXN |
                  PageTableFlags::PXN;
        return self;
    }

    pub fn SetBlock(&mut self) -> &mut Self {
        self.0 &= !PageTableFlags::TABLE;
        return self;
    }

    pub fn SetMtNormal(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::MT_NORMAL;
        return self;
    }

    pub fn SetDirty(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::DIRTY;
        return self;
    }

    pub fn SetAccessed(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::ACCESSED;
        return self;
    }

    pub fn Val(&self) -> PageTableFlags {
        return self.0;
    }
}

#[cfg(target_arch = "x86_64")]
impl PageOpts {
    //const Empty : PageTableFlags = PageTableFlags::PRESENT & PageTableFlags::WRITABLE; //set 0
    pub fn New(user: bool, write: bool, exec: bool) -> Self {
        let mut flags = PageTableFlags::PRESENT;
        if write {
            flags |= PageTableFlags::WRITABLE;
        }

        if user {
            flags |= PageTableFlags::USER_ACCESSIBLE;
        }

        if !exec {
            flags |= PageTableFlags::NO_EXECUTE;
        }

        return Self(flags);
    }

    pub fn All() -> Self {
        return PageOpts(
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::USER_ACCESSIBLE,
        );
    }

    pub fn Zero() -> Self {
        return PageOpts(PageTableFlags::PRESENT & PageTableFlags::WRITABLE); //set 0
    }

    pub fn Kernel() -> Self {
        return PageOpts(
            PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::GLOBAL,
        );
    }

    pub fn UserReadOnly() -> Self {
        return PageOpts(PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE);
    }

    pub fn UserNonAccessable() -> Self {
        return PageOpts(PageTableFlags::PRESENT | PageTableFlags::ACCESSED);
    }

    pub fn UserReadWrite() -> Self {
        return PageOpts(
            PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE | PageTableFlags::WRITABLE,
        );
    }

    pub fn KernelReadOnly() -> Self {
        return PageOpts(PageTableFlags::PRESENT);
    }

    pub fn KernelReadWrite() -> Self {
        return PageOpts(PageTableFlags::PRESENT | PageTableFlags::WRITABLE);
    }

    pub fn Present(&self) -> bool {
        return (self.0 & PageTableFlags::PRESENT) != Self::Zero().0;
    }

    pub fn Write(&self) -> bool {
        return (self.0 & PageTableFlags::WRITABLE) != Self::Zero().0;
    }

    pub fn Global(&self) -> bool {
        return (self.0 & PageTableFlags::GLOBAL) != Self::Zero().0;
    }

    pub fn UserAccess(&self) -> bool {
        return (self.0 & PageTableFlags::USER_ACCESSIBLE) != Self::Zero().0;
    }

    pub fn SetPresent(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::PRESENT;
        return self;
    }

    pub fn SetWrite(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::WRITABLE;
        return self;
    }

    pub fn SetUserAccess(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::USER_ACCESSIBLE;
        return self;
    }

    pub fn SetGlobal(&mut self) -> &mut Self {
        self.0 |= PageTableFlags::GLOBAL;
        return self;
    }

    pub fn Val(&self) -> PageTableFlags {
        return self.0;
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Addr(pub u64);

impl Addr {
    pub fn AddLen(&self, len: u64) -> Result<Addr> {
        if core::u64::MAX - self.0 < len {
            return Err(Error::SysError(SysErr::EFAULT));
        }
        let end = self.0 + len;
        return Ok(Addr(end));
    }

    pub const fn RoundDown(&self) -> Result<Addr> {
        return Ok(Addr(self.0 & !(PAGE_SIZE - 1)));
    }

    pub const fn RoundUp(&self) -> Result<Addr> {
        if u64::MAX - PAGE_SIZE + 1 < self.0 {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        if self.0 == 0 {
            return Ok(Addr(0));
        }
        let addr = self.0 - 1 + PAGE_SIZE;
        return Addr(addr).RoundDown();
    }

    pub fn MustRoundUp(&self) -> Addr {
        return self
            .RoundUp()
            .expect(&format!("usermem.Addr({}).RoundUp() wraps", self.0));
    }

    pub fn PageOffset(&self) -> u64 {
        self.0 & (PAGE_SIZE - 1)
    }

    pub fn IsPageAligned(&self) -> bool {
        self.PageOffset() == 0
    }

    pub fn is_huge_page_aligned(&self, _type: pagetable::HugePageType) -> bool {
        let align_size = _type.size();
        return (self.0 & (align_size -1)) == 0
    }

    pub fn PageAligned(&self) -> Result<()> {
        if !self.IsPageAligned() {
            return Err(Error::UnallignedAddress(format!("PageAligned {:?}", self)));
        }

        Ok(())
    }

    pub fn AddPages(&self, pageCount: u32) -> Addr {
        return Addr(self.0 + pageCount as u64 * PAGE_SIZE);
    }

    pub fn PageOffsetIdx(&self, addr: Addr) -> Result<u32> {
        let addr = addr.RoundDown()?;

        if addr.0 < self.0 {
            return Err(Error::AddressNotInRange);
        }

        return Ok(((addr.0 - self.0) / PAGE_SIZE as u64) as u32);
    }

    pub fn Offset(&self, startAddr: Addr) -> Result<Addr> {
        if self.0 < startAddr.0 {
            return Err(Error::AddressNotInRange);
        }

        return Ok(Addr(self.0 - startAddr.0));
    }

    pub fn ToRange(&self, length: u64) -> Result<Range> {
        let _end = self.AddLen(length)?;
        return Ok(Range::New(self.0, length));
    }
}
