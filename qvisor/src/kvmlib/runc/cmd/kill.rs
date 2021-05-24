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

use lazy_static::lazy_static;
use alloc::collections::btree_map::BTreeMap;
use clap::{App, AppSettings, SubCommand, ArgMatches, Arg};
use alloc::string::String;

use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::common::*;
use super::super::cmd::config::*;
use super::super::container::container::*;
use super::command::*;

#[derive(Default, Debug)]
pub struct KillCmd  {
    pub id: String,
    pub all: bool,
    pub pid: i32,
    pub sig: Vec<String>
}

impl KillCmd {
    pub fn Init(cmd_matches: &ArgMatches) -> Result<Self> {
        let pidStr = cmd_matches.value_of("pid").unwrap().to_string();
        let pid = match pidStr.parse::<i32>() {
            Err(_e) => return Err(Error::Common(format!("pid {} cant not be parsed as int type", pidStr))),
            Ok(v) => v,
        };

        let sig = match cmd_matches.values_of("sig") {
            None => Vec::new(),
            Some(iter) => iter.map(|s| s.to_string()).collect(),
        };

        if sig.len() >= 2 {
            return Err(Error::Common(format!("too many signals {:?}", &sig)));
        }

        return Ok(Self {
            id: cmd_matches.value_of("id").unwrap().to_string(),
            all: cmd_matches.is_present("all"),
            pid: pid,
            sig: sig,
        })
    }

    pub fn SubCommand<'a, 'b>(common: &CommonArgs<'a, 'b>) -> App<'a, 'b> {
        return SubCommand::with_name("kill")
            .setting(AppSettings::ColoredHelp)
            .arg(&common.id_arg)
            .arg(
                Arg::with_name("all")
                    .long("all")
                    .short("a")
                    .help("send the specified signal to all processes inside the container"),
            )
            .arg(
                Arg::with_name("pid")
                    .default_value("0")
                    .long("pid")
                    .takes_value(true)
                    .help("send the specified signal to a specific process"),
            )
            .setting(AppSettings::TrailingVarArg)
            .arg(
                Arg::with_name("sig")
                    .multiple(false),
            )
            .about("sends a signal to the container");
    }

    pub fn Run(&self, gCfg: &GlobalConfig) -> Result<()> {
        info!("Container:: Kill ....");
        let mut signal = if self.sig.len() == 0 {
            "".to_string()
        } else {
            self.sig[0].to_string()
        };

        if signal.len() == 0 {
            signal = "KILL".to_string();
        }

        let sig = ParseSignal(&signal)?;

        let container = Container::Load(&gCfg.RootDir, &self.id)?;

        if self.pid != 0 {
            return container.SignalProcess(sig, self.pid)
        } else {
            return container.SignalContainer(sig, self.all)
        }
    }
}

pub fn ParseSignal(s: &str) -> Result<i32> {
    match s.parse::<i32>() {
        Ok(n) => {
            for (_, id) in SIGNAL_MAP.iter() {
                if n == *id {
                    return Ok(n)
                }
            }

            return Err(Error::Common(format!("unknown signal {}", n)));
        }
        Err(_) => (),
    }

    let str = s.to_uppercase();
    let str = if str.starts_with("SIG") {
        &s["SIG".len()..]
    } else {
        &str
    };

    match SIGNAL_MAP.get(&str) {
        None => return Err(Error::Common(format!("unknown signal {}", s))),
        Some(sig) => Ok(*sig),
    }
}

lazy_static! {
    static ref SIGNAL_MAP: BTreeMap<&'static str, i32> = {
        let mut map = BTreeMap::new();

        map.insert("ABRT",   Signal::SIGABRT);
        map.insert("ALRM",   Signal::SIGALRM);
        map.insert("BUS",    Signal::SIGBUS);
        map.insert("CHLD",   Signal::SIGCHLD);
        map.insert("CLD",    Signal::SIGCLD);
        map.insert("CONT",   Signal::SIGCONT);
        map.insert("FPE",    Signal::SIGFPE);
        map.insert("HUP",    Signal::SIGHUP);
        map.insert("ILL",    Signal::SIGILL);
        map.insert("INT",    Signal::SIGINT);
        map.insert("IO",     Signal::SIGIO);
        map.insert("IOT",    Signal::SIGIOT);
        map.insert("KILL",   Signal::SIGKILL);
        map.insert("PIPE",   Signal::SIGPIPE);
        map.insert("POLL",   Signal::SIGPOLL);
        map.insert("PROF",   Signal::SIGPROF);
        map.insert("PWR",    Signal::SIGPWR);
        map.insert("QUIT",   Signal::SIGQUIT);
        map.insert("SEGV",   Signal::SIGSEGV);
        map.insert("STKFLT", Signal::SIGSTKFLT);
        map.insert("STOP",   Signal::SIGSTOP);
        map.insert("SYS",    Signal::SIGSYS);
        map.insert("TERM",   Signal::SIGTERM);
        map.insert("TRAP",   Signal::SIGTRAP);
        map.insert("TSTP",   Signal::SIGTSTP);
        map.insert("TTIN",   Signal::SIGTTIN);
        map.insert("TTOU",   Signal::SIGTTOU);
        map.insert("URG",    Signal::SIGURG);
        map.insert("USR1",   Signal::SIGUSR1);
        map.insert("USR2",   Signal::SIGUSR2);
        map.insert("VTALRM", Signal::SIGVTALRM);
        map.insert("WINCH",  Signal::SIGWINCH);
        map.insert("XCPU",   Signal::SIGXCPU);
        map.insert("XFSZ",   Signal::SIGXFSZ);

        map
    };
}