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

use super::super::qlib::linux_def::*;

#[derive(Debug, Copy, Clone, Default)]
pub struct FileFlags {
    pub Direct: bool,
    pub NonBlocking: bool,
    pub DSync: bool,
    pub Sync: bool,
    pub Append: bool,
    pub Read: bool,
    pub Write: bool,
    pub Pread: bool,
    pub PWrite: bool,
    pub Directory: bool,
    pub Async: bool,
    pub LargeFile: bool,
    pub NonSeekable: bool,
    pub Truncate: bool,
}

impl FileFlags {
    pub fn FromFlags(mask: u32) -> Self {
        return Self {
            Direct: mask & Flags::O_DIRECT as u32 != 0,
            DSync: mask & (Flags::O_DSYNC | Flags::O_SYNC) as u32 != 0,
            Sync: mask & Flags::O_SYNC as u32 != 0,
            NonBlocking: mask & Flags::O_NONBLOCK as u32 != 0,
            Read: mask & Flags::O_ACCMODE as u32 != Flags::O_WRONLY as u32,
            Write: mask & Flags::O_ACCMODE as u32 != Flags::O_RDONLY as u32,
            Append: mask & Flags::O_APPEND as u32 != 0,
            Directory: mask & Flags::O_DIRECTORY as u32 != 0,
            Async: mask & Flags::O_ASYNC as u32 != 0,
            LargeFile: mask & Flags::O_LARGEFILE as u32 != 0,
            Truncate: mask & Flags::O_TRUNC as u32 != 0,
            ..Default::default()
        }
    }

    //from the Fcntl GETTFL result
    pub fn FromFcntl(mask: u32) -> Self {
        let accmode = mask & Flags::O_ACCMODE as u32;

        return Self {
            Direct: mask & Flags::O_DIRECT as u32 != 0,
            NonBlocking: mask & Flags::O_NONBLOCK as u32 != 0,
            Sync: mask & Flags::O_SYNC as u32 != 0,
            Append: mask & Flags::O_APPEND as u32 != 0,
            Read: accmode == Flags::O_RDONLY as u32 || accmode == Flags::O_RDWR as u32,
            Write: accmode == Flags::O_WRONLY as u32 || accmode == Flags::O_RDWR as u32,
            ..Default::default()
        }
    }

    pub fn SettableFileFlags(&self) -> SettableFileFlags {
        return SettableFileFlags {
            Direct: self.Direct,
            NonBlocking: self.NonBlocking,
            Append: self.Append,
            Async: self.Async,
        }
    }

    pub fn ToLinux(&self) -> i32 {
        let mut mask: i32 = 0;

        if self.Direct {
            mask |= Flags::O_DIRECT;
        }

        if self.NonBlocking {
            mask |= Flags::O_NONBLOCK;
        }

        if self.DSync {
            mask |= Flags::O_DSYNC;
        }

        if self.Sync {
            mask |= Flags::O_SYNC;
        }

        if self.Append {
            mask |= Flags::O_APPEND;
        }

        if self.Directory {
            mask |= Flags::O_DIRECTORY;
        }

        if self.Async {
            mask |= Flags::O_ASYNC;
        }

        if self.LargeFile {
            mask |= Flags::O_LARGEFILE;
        }

        if self.Truncate {
            mask |= Flags::O_TRUNC;
        }

        if self.Read && self.Write {
            mask |= Flags::O_RDWR;
        } else if self.Write {
            mask |= Flags::O_WRONLY;
        } else if self.Read {
            mask |= Flags::O_RDONLY;
        }

        return mask;
    }
}

#[derive(Debug, Copy, Clone, Default)]
pub struct SettableFileFlags {
    pub Direct: bool,
    pub NonBlocking: bool,
    pub Append: bool,
    pub Async: bool,
}

