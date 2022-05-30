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

use core::sync::atomic;
use core::sync::atomic::AtomicU64;

use super::qlib::singleton::*;

pub type UniqueID = u64;
pub static UID: Singleton<AtomicU64> = Singleton::<AtomicU64>::New();
pub static INOTIFY_COOKIE: Singleton<AtomicU32> = Singleton::<AtomicU32>::New();

pub fn NewUID() -> u64 {
    return UID.fetch_add(1, atomic::Ordering::SeqCst);
}

pub fn NewInotifyCookie() -> u64 {
    return INOTIFY_COOKIE.fetch_add(1, atomic::Ordering::SeqCst);
}
