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
use alloc::vec::Vec;

use super::elf::*;
use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::fs::file::*;
use super::super::task::*;

// interpMaxLineLength is the maximum length for the first line of an
// interpreter script.
//
// From execve(2): "A maximum line length of 127 characters is allowed
// for the first line in a #! executable shell script."
pub const INTERP_MAX_LINE_LENGTH : usize = 127;

pub fn ParseInterpreterScript(task: &mut Task, fileName: &str, f: &File, mut argv: Vec<String>) -> Result<(String, Vec<String>)> {
    let mut buf : [u8; INTERP_MAX_LINE_LENGTH] = [0; INTERP_MAX_LINE_LENGTH];

    let n = ReadAll(task, f, &mut buf, 0)?;
    // Allow unexpected EOF, as a valid executable could be only three
    // bytes (e.g., #!a).
    let line = &buf[0..n];
    // Ignore #!.
    let line = &line[2..];

    // Ignore everything after newline.
    // Linux silently truncates the remainder of the line if it exceeds
    // interpMaxLineLength.
    let line = match line.iter().position(|&r| r == '\n' as u8) {
        None => line,
        Some(n) => &line[..n],
    };

    // Skip any whitespace before the interpeter.
    let mut idx = 0;
    while idx < line.len() && (line[idx] == ' ' as u8 || line[idx] == '\t' as u8) {
        idx += 1;
    }
    let line = &line[idx..];

    let (line, right) = match line.iter().position(|&r| r == '\t' as u8 || r == ' ' as u8 ) {
        None => (line, &line[..0]),
        Some(n) => (&line[..n], &line[n..]),
    };

    let interp = String::from_utf8(line.to_vec()).unwrap();
    let arg = String::from_utf8(right.to_vec()).unwrap().trim().to_string();

    if interp.len() == 0 {
        return Err(Error::SysError(SysErr::ENOEXEC));
    }

    let mut newargv = Vec::new();
    // Build the new argument list:
    //
    // 1. The interpreter.
    newargv.push(interp.to_string());

    // 2. The optional interpreter argument.
    if arg.len() > 0 {
        newargv.push(arg.to_string())
    }

    // 3. The original arguments. The original argv[0] is replaced with the
    // full script filename.
    if argv.len() > 0 {
        argv[0] = fileName.to_string();
    } else {
        argv.push(fileName.to_string());
    }

    newargv.append(&mut argv);

    return Ok((interp, newargv))
}