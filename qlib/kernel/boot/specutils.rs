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
use alloc::string::ToString;

use super::super::super::path::*;
use super::*;

pub fn ContainsStr(strs: &Vec<&str>, str: &str) -> bool {
    for s in strs {
        if *s == str {
            return true;
        }
    }

    return false;
}

pub fn IsSupportedDevMount(m: &oci::Mount) -> bool {
    let existingDevices = vec!["/dev/fd", "/dev/stdin", "/dev/stdout", "/dev/stderr",
                               "/dev/null", "/dev/zero", "/dev/full", "/dev/random",
                               "/dev/urandom", "/dev/shm", "/dev/pts", "/dev/ptmx"];

    let dst = Clean(&m.destination);
    if dst.as_str() == "dev" {
        return false;
    }

    for dev in existingDevices {
        if dst.as_str() == dev || HasPrefix(&dst, &(dev.to_string() + "/")) {
            return false
        }
    }

    return true
}
