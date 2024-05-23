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

use std::fs::File;
use std::os::unix::io::AsRawFd;

pub const RAWLOG_FILE_DEFAULT: &str = "/var/log/quark/raw.log";

fn main() {
    parse();
}

pub fn parse() {
    let rawfile = File::open(RAWLOG_FILE_DEFAULT).expect("Log Open fail");

    let data: [u64; 6] = [0; 6];

    let fd = rawfile.as_raw_fd();

    loop {
        let count = unsafe { libc::read(fd, data.as_ptr() as u64 as *mut _, data.len() * 8) };
        if count > 0 {
            println!("{:x?}", &data);
        } else {
            break;
        }
    }

    return;
}
