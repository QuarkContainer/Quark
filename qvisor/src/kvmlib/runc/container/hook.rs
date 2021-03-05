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

use alloc::string::ToString;
use std::process::Command;
use std::io::Write;
use std::process::Stdio;
use std::{thread, time};
use std::collections::HashMap;

use super::super::super::qlib::common::*;
use super::super::super::qlib::path::*;
use super::super::oci::*;
use super::exec_hook::execute_hook;

// This file implements hooks as defined in OCI spec:
// https://github.com/opencontainers/runtime-spec/blob/master/config.md#toc22
//
// "hooks":{
// 		"prestart":[{
// 			"path":"/usr/bin/dockerd",
// 			"args":[
// 				"libnetwork-setkey", "arg2",
// 			]
// 		}]
// },

// executeHooksBestEffort executes hooks and logs warning in case they fail.
// Runs all hooks, always.
pub fn executeHooksBestEffort(hooks: &[Hook], s: &State) {
    for h in hooks {
        match execute_hook(h, s) {
            Ok(_) => (),
            Err(e) => info!("Failure to execute hook {:?}, err: {:?}", h, e)
        }
    }
}

// executeHooks executes hooks until the first one fails or they all execute.
pub fn executeHooks(hooks: &[Hook], s: &State) -> Result<()> {
    for h in hooks {
        match execute_hook(h, s) {
            Ok(_) => (),
            Err(e) => {
                info!("Failure to execute hook {:?}, err: {:?}", h, e);
                return Err(e);
            }
        }
    }

    return Ok(())
}

pub fn executeHook(h: &Hook, s: &State) -> Result<()> {
    info!("Executing hook {:#?}, state: {:#?}", h, s);

    if h.path.trim().len() == 0 {
        return Err(Error::Common("empty path for hook".to_string()));
    }

    if !IsAbs(&h.path) {
        return Err(Error::Common(format!("path for hook is not absolute: {:?}", &h.path)));
    }

    let b = s.to_string().map_err(|e| Error::Common(format!("executeHook error is {:?}", e)))?;

    info!("Executing state: {}", &b);

    let mut envs = HashMap::new();
    for s in &h.env {
        let ss : Vec<&str> = s.split('=').collect();
        assert!(ss.len() == 0);
        envs.insert(ss[0].to_string(), ss[1].to_string());
    }

    let mut cmd = Command::new(&h.path)
        .args(&h.args)
        .envs(&envs)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Common(format!("executeHook spawn error is {:?}", e)))?;

    //error!("Executing args is {:?}", cmd.get_args().collect());
    //error!("Executing envs is {:?}", cmd.get_envs().collect());

    let stdin = cmd.stdin.as_mut().expect("Failed to open stdin"); //.map_err(|e| Error::Common(format!("executeHook get stdin error is {:?}", e)))?;
    stdin.write_all(b.as_bytes()).map_err(|e| Error::Common(format!("executeHook stdin write error is {:?}", e)))?;

    match h.timeout {
        None => {
            let output = cmd.wait_with_output().expect("Failed to read stdout");
            //assert_eq!(String::from_utf8_lossy(&output.stdout), "!dlrow ,olleH");

            //let exitStatus = cmd.wait().map_err(|e| Error::Common(format!("executeHook wait error is {:?}", e)))?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("status: {}", output.status);
            //info!("Execute hook {} success!, status is {:?}", &h.path, &exitStatus);
            info!("stdiout is {}, stderr is {}", stdout, stderr);
            return Ok(())
        },
        Some(timeout) => {
            let ms = timeout * 1000;
            for _i in 0 .. (ms/10) as usize {
                match cmd.try_wait() {
                    Ok(Some(status)) => {
                        //let stdout = String::from_utf8_lossy(&cmd.stdout);
                        //let stderr = String::from_utf8_lossy(&cmd.stderr);

                        info!("Execute hook timeout {} success!, status is {:?}", &h.path, &status);
                        //info!("stdiout is {}, stderr is {}", stdout, stderr);
                        return Ok(())
                    }
                    Ok(None) => (),
                    Err(e) => {
                        return Err(Error::Common(format!("executeHook wait error is {:?}", e)));
                    }
                }

                let ten_millis = time::Duration::from_millis(10);
                thread::sleep(ten_millis);
            }
        }
    }

    //let stdout = String::from_utf8_lossy(&cmd.stdout);
    //let stderr = String::from_utf8_lossy(&cmd.stderr);

    cmd.kill().map_err(|e| Error::Common(format!("executeHook kill error is {:?}", e)))?;
    //info!("timeout executing hook {}\nstdout: {}\nstderr: {}", h.Path, stdout.String(), stderr.String());
    info!("timeout executing hook {}", &h.path);
    return Ok(())
}