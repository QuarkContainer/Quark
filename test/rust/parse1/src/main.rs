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

#![allow(non_snake_case)]

extern crate libc;

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{self, prelude::*, BufReader};
use std::os::unix::io::IntoRawFd;

pub const RAWLOG_FILE_DEFAULT: &str = "/var/log/quark/raw.log";
pub const LOG_FILE_DEFAULT: &str = "/var/log/quark/quark.log";

#[derive(Default, Debug)]
pub struct Line {
    pub processid: u64,
    pub line_num: u64,
    pub op: u64,
    pub class: u64,
    pub addr: u64,
    pub cpuid: u64,
}

pub fn main() {
    let logMap = LogParse().unwrap();

    // println!("{:#?}", &logMap);

    let mut map = BTreeMap::new();
    for (_pid, (line, o)) in &logMap {
        map.insert(line, o);
    }

    for (line, o) in map {
        println!("{} {}", line, o);
    }

    // let rawlogMap = RawFileParse(pid);
    // for (line_num, (addr, cpuid)) in rawlogMap.iter() {
    //     let count = *line_num;
    //     for i in (0..count as usize).rev() {
    //         let i = i as u64;
    //         match logMap.get(&i) {
    //             None => continue,
    //             Some((cpu, line)) => {
    //                 if *cpu as u64 == *cpuid {
    //                     ////////////////////////////////////////////
    //                     println!("{:x} {} {} {} *** {}", *addr, *cpuid, *line_num, i, line);
    //                     break;
    //                 }
    //             }
    //         }
    //     }
    //     //println!("{:x} {} {}", *addr, *line_num - 1, *cpuid);
    // }

    //println!("total: {:?}", rawlogMap);
}

pub fn LogParse() -> io::Result<BTreeMap<String, (i32, String)>> {
    let file = File::open(LOG_FILE_DEFAULT)?;
    let reader = BufReader::new(file);
    let mut map = BTreeMap::new();

    let mut linenum = 0;
    for line in reader.lines() {
        linenum += 1;
        //println!("{}", line?);
        let line = line?;
        // let (pid, lineNum, cpuid) = ParseLine(&line);

        // if pid == 0 || lineNum == 0 || cpuid == -1 {
        //     continue;
        // }
        // if pid == requestPid {
        //     map.insert(lineNum, (cpuid, line));
        //     //println!("{} {} {}", pid, lineNum, cpuid);
        // }
        let pid = ParseLineFromPid(&line);
        if pid.len() == 0 {
            continue;
        }

        map.insert(pid, (linenum, line.to_string()));
    }

    return Ok(map);
}

pub fn ParseLineFromPid(line: &str) -> String {
    let split: Vec<&str> = line.split(" ").collect();
    //println!("ParseLine split is {:?}", &split);
    if split.len() <= 3 {
        return String::new();
    }

    let s: Vec<&str> = split[2].split(")").collect();
    if s.len() < 2 {
        return String::new();
    }
    return s[0].to_string();
}

pub fn ParseLine(line: &str) -> (u64, u64, i32) {
    let split: Vec<&str> = line.split(" ").collect();
    //println!("ParseLine split is {:?}", &split);
    if split.len() <= 3 {
        return (0, 0, -1);
    }

    let pid = match split[0].parse::<u64>() {
        Ok(v) => v,
        _ => 0,
    };
    let lineNum = match split[1].parse::<u64>() {
        Ok(v) => v,
        _ => 0,
    };

    println!("pid {} linenum is {}", pid, lineNum);
    let taskSeg = split[3];
    let cpuid: i32 = {
        if pid == 0 || lineNum == 0 || taskSeg.as_bytes()[0] != '[' as u8 {
            0
        } else {
            let segs: Vec<&str> = taskSeg[1..].split("/").collect();
            match segs[0].parse::<i32>() {
                Ok(v) => v,
                _ => -1,
            }
        }
    };

    return (pid, lineNum, cpuid);
}

pub fn RawFileParse(pid: u64) -> BTreeMap<u64, (u64, u64)> {
    let file = OpenOptions::new()
        .read(true)
        .open(RAWLOG_FILE_DEFAULT)
        .expect("Log Open fail");

    let mut map = BTreeMap::new();
    let mut processes = BTreeMap::new();

    let fd = file.into_raw_fd();
    let mut data = Line::default();
    let mut total = 0;
    loop {
        let ret = unsafe { libc::read(fd, &mut data as *mut Line as u64 as _, 6 * 8) };

        if ret <= 0 {
            println!("ret is {}", ret);
            break;
        }

        assert!(ret == 6 * 8);
        if data.op == 1 {
            total += 1;
        }

        let cpuid = data.cpuid & 0xff;
        //let idx = data.cpuid >> 8;
        //println!("{} {} {}/{}/{:x}/{}/{}", data.processid, data.line_num, data.op, data.class, data.addr, cpuid, idx);
        processes.insert(data.processid, 0);
        if data.processid != pid {
            continue;
        }

        if data.op == 1 {
            // allocate
            assert!(!map.contains_key(&data.addr));
            map.insert(data.addr, (data.line_num, cpuid));
        } else {
            assert!(data.op == 2); // deallocate
            assert!(map.contains_key(&data.addr));
            map.remove(&data.addr);
        }
    }

    let mut line2addr = BTreeMap::new();

    for (addr, (line_num, cpuid)) in map.iter() {
        line2addr.insert(*line_num, (*addr, *cpuid));
    }

    println!("*******total is {}/{:#?}", total, processes);

    return line2addr;

    /*for (line_num, (addr, cpuid)) in line2addr.iter() {
        println!("{:x} {} {}", *addr, *line_num - 1, *cpuid);
    }

    println!("total {}", line2addr.len());*/
}
