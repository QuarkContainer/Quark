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

pub mod inode;
pub mod dirent;
pub mod file;
//pub mod inodeOperations;
pub mod flags;
pub mod attr;
pub mod filesystems;
pub mod mount;
pub mod host;
pub mod dentry;
pub mod lock;
pub mod ramfs;
pub mod fsutil;
pub mod mount_overlay;
pub mod overlay;
pub mod inode_overlay;
pub mod copy_up;
pub mod file_overlay;
pub mod dev;
pub mod tty;
pub mod procfs;
pub mod sys;
pub mod anon;
pub mod timerfd;
pub mod tmpfs;

pub fn Init() {
    self::tty::Init();
    self::dev::Init();
    self::procfs::Init();
    self::sys::Init();
    self::tmpfs::Init();
}