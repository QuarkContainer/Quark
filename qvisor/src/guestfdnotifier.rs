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

use super::qlib::kernel::waiter::*;
use super::qlib::Common::*;

pub fn Notify(_fd: i32, _mask: u32) {}

pub fn AddFD(_fd: i32, _queue: &Queue) {}

pub fn RemoveFD(_fd: i32) {}

pub fn UpdateFD(_fd: i32) -> Result<()> {
    return Err(Error::None);
}

pub fn NonBlockingPoll(_fd: i32, _mask: EventMask) -> EventMask {
    0
}
