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

use alloc::collections::btree_map::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::fs::File;
use std::io::prelude::*;
use std::os::unix::io::AsRawFd;
use std::os::unix::io::FromRawFd;
use std::path::Path;
use std::process::Stdio;
use std::{thread, time};
use tempfile::Builder;

use super::super::super::console::pty::*;
use super::super::super::console::unix_socket::*;
use super::super::super::qlib::auth::cap_set::*;
use super::super::super::qlib::auth::id::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux::time::*;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::super::oci::serialize::*;
use super::super::oci::*;
use super::super::specutils::specutils::*;
use super::command::*;

#[derive(Default, Debug)]
pub struct User {
    kuid: KUID,
    kgid: KGID,
}

#[derive(Default, Debug)]
pub struct ExecCmd {
    pub id: String,
    pub cwd: String,
    pub user: String,
    pub envv: Vec<String>,
    pub extraKGIDs: Vec<String>,
    pub caps: Vec<String>,
    pub detach: bool,
    pub processPath: String,
    pub pid: String,
    pub internalPidFile: String,
    pub consoleSocket: String,
    pub argv: Vec<String>,
    pub clearStatus: bool,
    pub terminal: bool,
}

impl ExecCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        let ret = Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
            cwd: cmd_matches.value_of("cwd").unwrap().to_string(),
            user: cmd_matches.value_of("user").unwrap().to_string(),
            envv: match cmd_matches.values_of("env") {
                None => Vec::new(),
                Some(iter) => iter.map(|s| s.to_string()).collect(),
            },
            extraKGIDs: match cmd_matches.values_of("additional-gids") {
                None => Vec::new(),
                Some(iter) => iter.map(|s| s.to_string()).collect(),
            },
            caps: match cmd_matches.values_of("cap") {
                None => Vec::new(),
                Some(iter) => iter.map(|s| s.to_string()).collect(),
            },
            detach: cmd_matches.is_present("detach"),
            processPath: cmd_matches.value_of("process").unwrap().to_string(),
            pid: cmd_matches.value_of("p").unwrap().to_string(),
            internalPidFile: cmd_matches
                .value_of("internal-pid-file")
                .unwrap()
                .to_string(),
            consoleSocket: cmd_matches.value_of("console-socket").unwrap().to_string(),
            argv: match cmd_matches.values_of("command") {
                None => Vec::new(),
                Some(iter) => iter.map(|s| s.to_string()).collect(),
            },
            clearStatus: cmd_matches.value_of("clear-status").unwrap() == "true",
            terminal: cmd_matches.is_present("terminal"),
        };

        if ret.processPath.len() == 0 && ret.argv.len() == 0 {
            println!("{}", cmd_matches.usage());
            return Err(Error::Common(format!(
                "either process or command is required"
            )));
        }

        return Ok(ret);
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("exec")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .arg(
                Arg::with_name("cwd")
                    .help("current working directory")
                    .default_value("")
                    .takes_value(true)
                    .long("cwd"),
            )
            .arg(
                Arg::with_name("env")
                    .help("set environment variables (e.g. '--env PATH=/bin --env TERM=xterm')")
                    .takes_value(true)
                    .multiple(true)
                    .long("env")
                    .short("e"),
            )
            .arg(
                Arg::with_name("terminal")
                    .help("allocate a pseudo-TTY")
                    .long("tty")
                    .short("t"),
            )
            .arg(
                Arg::with_name("user")
                    .help("UID (format: <uid>[:<gid>])")
                    .takes_value(true)
                    .default_value("")
                    .long("user")
                    .short("u"),
            )
            .arg(
                Arg::with_name("additional-gids")
                    .help("additional gids")
                    .takes_value(true)
                    .multiple(true)
                    .long("additional-gids")
                    .short("g"),
            )
            .arg(
                Arg::with_name("cap")
                    .help("add a capability to the bounding set for the process")
                    .takes_value(true)
                    .multiple(true)
                    .long("cap"),
            )
            .arg(&common.detach_arg)
            .arg(
                Arg::with_name("process")
                    .help("path to the process.json")
                    .default_value("")
                    .takes_value(true)
                    .long("process")
                    .short("p"),
            )
            .arg(
                Arg::with_name("p")
                    .takes_value(true)
                    .default_value("")
                    .long("pid-file")
                    .help("specify the file to write the container pid to"),
            )
            .arg(
                Arg::with_name("internal-pid-file")
                    .takes_value(true)
                    .default_value("")
                    .long("internal-pid-file")
                    .help("filename that the container-internal pid will be written to"),
            )
            .arg(&common.consoleSocket_arg)
            .arg(
                Arg::with_name("clear-status")
                    .takes_value(true)
                    .default_value("true")
                    .long("clear-status")
                    .help("clear the status of the exec'd process upon completion"),
            )
            .setting(AppSettings::TrailingVarArg)
            .arg(Arg::with_name("command").multiple(true))
            .about("Run a container");
    }

    pub fn ArgsFromCLI(&mut self) -> Result<ExecArgs> {
        let mut extraKGIDs = Vec::new();
        for g in &self.extraKGIDs {
            let kgid = match g.parse::<u32>() {
                Err(e) => panic!("parsing gid: {} fail, err is {:?}", g, e),
                Ok(id) => id,
            };

            extraKGIDs.push(KGID(kgid));
        }

        //todo: handle capacities
        let caps = TaskCaps::default();

        let mut argv = Vec::new();
        argv.append(&mut self.argv);

        let mut envv = Vec::new();
        envv.append(&mut self.envv);

        let ids: Vec<&str> = self.user.split(':').collect();
        let uid = match ids[0].parse::<u32>() {
            Err(e) => panic!("parsing uid: {} fail, err is {:?}", ids[1], e),
            Ok(id) => id,
        };

        let gid = if ids.len() > 2 {
            panic!("user's format should be <uid>[:<gid>]");
        } else if ids.len() == 2 {
            match ids[1].parse::<u32>() {
                Err(e) => panic!("parsing gid: {} fail, err is {:?}", ids[1], e),
                Ok(id) => id,
            }
        } else {
            0
        };

        if self.detach && self.terminal && self.consoleSocket.len() == 0 {
            return Err(Error::Common(
                "cannot allocate tty if runc will detach without setting console socket"
                    .to_string(),
            ));
        }

        if (!self.detach || !self.terminal) && self.consoleSocket.len() > 0 {
            return Err(Error::Common(
                "annot use console socket if runc will not detach or allocate tty".to_string(),
            ));
        }

        return Ok(ExecArgs {
            Argv: argv,
            Envv: envv,
            Root: "".to_string(),
            WorkDir: self.cwd.to_string(),
            KUID: KUID(uid),
            KGID: KGID(gid),
            ExtraKGIDs: extraKGIDs,
            Capabilities: caps,
            Terminal: self.terminal,
            Detach: self.detach,
            ContainerID: self.id.to_string(),
            ConsoleSocket: self.consoleSocket.to_string(),
            ExecId: "".to_string(),
            Fds: Vec::new(),
        });
    }

    pub fn ArgsFromProcess(&self) -> Result<ExecArgs> {
        let mut process: Process = deserialize(&self.processPath)
            .map_err(|e| Error::Common(format!("deserialize process with error {:?}", e)))?;

        let caps = Capabilities(false, &process.capabilities);

        let mut extraKGIDs: Vec<KGID> = Vec::with_capacity(process.user.additional_gids.len());
        extraKGIDs.append(
            &mut process
                .user
                .additional_gids
                .iter()
                .map(|id| KGID(*id))
                .collect(),
        );

        let mut argv = Vec::new();
        argv.append(&mut process.args);

        let mut envv = Vec::new();
        envv.append(&mut process.env);

        return Ok(ExecArgs {
            Argv: argv,
            Envv: envv,
            Root: "".to_string(),
            WorkDir: process.cwd.to_string(),
            KUID: KUID(process.user.uid),
            KGID: KGID(process.user.gid),
            ExtraKGIDs: extraKGIDs,
            Capabilities: caps,
            Terminal: process.terminal,
            ContainerID: self.id.to_string(),
            Detach: self.detach,
            ConsoleSocket: self.consoleSocket.to_string(),
            ExecId: "".to_string(),
            Fds: Vec::new(),
        });
    }

    pub fn ParseArgs(&mut self) -> Result<ExecArgs> {
        if self.processPath.len() == 0 {
            return self.ArgsFromCLI();
        } else {
            return self.ArgsFromProcess();
        }
    }

    pub fn Run(&mut self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: Exec ....");
        if self.detach {
            let ret = self.ExecAndWait(gCfg);
            error!("exec return .....");
            return ret;
        }

        if !self.clearStatus {
            let sid = unsafe {
                //signal (SIGHUP, SIG_IGN);
                libc::setsid()
            };

            if sid < 0 {
                panic!("Exec process setsid fail");
            }
        }

        let mut execArgs = self.ParseArgs()?;

        let mut container = Container::Load(&gCfg.RootDir, &self.id)?;

        if execArgs.WorkDir.len() == 0 {
            execArgs.WorkDir = container.Spec.process.cwd.to_string();
        }

        if execArgs.Envv.len() == 0 {
            execArgs.Envv = ResolveEnvs(&[&container.Spec.process.env, &self.envv])?;
        }

        //todo: handle caps

        let _pid = container.Execute(execArgs, self)?;

        return Ok(());
    }

    pub fn ExecAndWait(&self, gCfg: &GlobalConfig) -> Result<()> {
        let mut cmd = std::process::Command::new(&ReadLink(EXE_PATH)?);

        cmd.arg("--root");
        cmd.arg(&gCfg.RootDir);

        cmd.arg("--log");
        cmd.arg(&gCfg.DebugLog);

        cmd.arg("exec");
        cmd.arg(&self.id);

        if self.cwd.len() > 0 {
            cmd.arg("--cwd");
            cmd.arg(&self.cwd);
        }

        if self.user.len() > 0 {
            cmd.arg("--user");
            cmd.arg(&self.user);
        }

        if self.envv.len() > 0 {
            cmd.arg("--env");
            for e in &self.envv {
                cmd.arg(e);
            }
        }

        if self.terminal {
            cmd.arg("--tty");
        }

        if self.extraKGIDs.len() > 0 {
            cmd.arg("--additional-gids");
            for g in &self.extraKGIDs {
                cmd.arg(g);
            }
        }

        if self.caps.len() > 0 {
            cmd.arg("--cap");
            for cap in &self.caps {
                cmd.arg(cap);
            }
        }

        if self.detach {
            cmd.arg("--clear-status");
            cmd.arg("false");
        }

        if self.processPath.len() > 0 {
            cmd.arg("--process");
            cmd.arg(&self.processPath);
        }

        // The command needs to write a pid file so that execAndWait can tell
        // when it has started. If no pid-file was provided, we should use a
        // filename in a temp directory.
        let mut pidFile = self.pid.to_string();
        cmd.arg("--pid-file");
        if pidFile.len() == 0 {
            let tmpDir = Builder::new()
                .prefix("exec-pid-")
                .tempdir()
                .expect("create temp folder exec-pid- fail ");

            pidFile = tmpDir.path().join("pid").to_str().unwrap().to_string();
            cmd.arg(&pidFile);
        } else {
            cmd.arg(&pidFile);
        }

        if self.internalPidFile.len() > 0 {
            cmd.arg("--internal-pid-file");
            cmd.arg(&self.internalPidFile);
        }

        cmd.stdin(Stdio::inherit());
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::inherit());

        if self.consoleSocket.len() > 0 {
            cmd.arg("--console-socket");
            cmd.arg(&self.consoleSocket);

            let (master, slave) = NewPty()?;
            unsafe {
                let tty = slave.dup()?;
                cmd.stdin(Stdio::from_raw_fd(tty));
                cmd.stdout(Stdio::from_raw_fd(tty));
                cmd.stderr(Stdio::from_raw_fd(tty));
            }

            let client = UnixSocket::NewClient(&self.consoleSocket)?;
            client.SendFd(master.as_raw_fd())?;
        }

        for a in &self.argv {
            cmd.arg(a);
        }

        let child = cmd.spawn().unwrap();

        info!("quark exec: before wait for ready");
        WaitForReady(&pidFile, child.id() as i32, 10 * SECOND)?;
        info!("quark exec: after wait for ready");
        return Ok(());
    }
}

pub fn WaitForReady(pidfile: &str, pid: i32, timeout: i64) -> Result<()> {
    let count = timeout / 1 * 100 * MILLISECOND;

    for _i in 0..count as usize {
        let period = time::Duration::from_millis(100);
        thread::sleep(period);

        if !Path::new(pidfile).exists() {
            continue;
        }

        error!("{} exist", pidfile);
        let mut f = match File::open(pidfile) {
            Err(e) => {
                return Err(Error::Common(format!(
                    "WaitForReady fail to open {} with error {:?}",
                    pidfile, e
                )));
            }
            Ok(f) => f,
        };
        let mut pidstr = String::new();
        f.read_to_string(&mut pidstr).map_err(|e| {
            Error::Common(format!(
                "WaitForReady fail to read {} with error {:?}",
                pidfile, e
            ))
        })?;
        let pidInt = pidstr
            .parse::<i32>()
            .map_err(|_e| Error::Common(format!("WaitForReady cant covert {} to i32", pidstr)))?;
        if pidInt == pid {
            return Ok(());
        }

        let mut ws: i32 = 0;
        let child = unsafe { libc::wait4(pid, &mut ws, libc::WNOHANG, 0 as *mut libc::rusage) };

        if child < 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        if child == pid {
            return Err(Error::Common(format!("process {} has terminated", pid)));
        }
    }

    return Err(Error::Common(format!("wait process {} timeout", pid)));
}

// resolveEnvs transforms lists of environment variables into a single list of
// environment variables. If a variable is defined multiple times, the last
// value is used.
pub fn ResolveEnvs(envs: &[&[String]]) -> Result<Vec<String>> {
    let mut envMap = BTreeMap::new();

    for env in envs {
        for str in *env {
            let parts: Vec<&str> = str.split('=').collect();
            if parts.len() != 2 {
                return Err(Error::Common(format!("invlid env {}", str)));
            }

            envMap.insert(parts[0].to_string(), parts[1].to_string());
        }
    }

    let mut ret = Vec::with_capacity(envMap.len());
    for (key, val) in envMap {
        ret.push(format!("{}={}", key, val))
    }

    return Ok(ret);
}
