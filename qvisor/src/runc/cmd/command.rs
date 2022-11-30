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

use std::env;

use clap::{App, AppSettings, Arg};

use super::boot::*;
use super::cmd::*;
use super::config;
use super::config::*;
use super::create::*;
use super::delete::*;
use super::exec::*;
use super::kill::*;
use super::list::*;
use super::pause::*;
use super::ps::*;
use super::resume::*;
use super::run::*;
use super::sandbox::*;
use super::start::*;
use super::state::*;
use super::super::super::qlib::common::*;
use super::wait::*;

fn id_validator(val: String) -> core::result::Result<(), String> {
    if val.contains("..") || val.contains('/') {
        return Err(format!("id {} may cannot contain '..' or '/'", val));
    }
    Ok(())
}

fn get_args() -> Vec<String> {
    return env::args().collect();
}

pub struct CommonArgs<'a, 'b> {
    pub id_arg: Arg<'a, 'b>,
    pub bundle_arg: Arg<'a, 'b>,
    pub consoleSocket_arg: Arg<'a, 'b>,
    pub detach_arg: Arg<'a, 'b>,
    pub no_pivot_arg: Arg<'a, 'b>,
    pub pid_arg: Arg<'a, 'b>,
    pub init_arg: Arg<'a, 'b>,
    pub format_arg: Arg<'a, 'b>,
    // adhoc: for working with gvisor
    pub user_log_arg: Arg<'a, 'b>,
}

impl<'a, 'b> CommonArgs<'a, 'b> {
    pub fn New() -> Self {
        let id_arg = Arg::with_name("id")
            .required(true)
            .takes_value(true)
            .validator(id_validator)
            .help("Unique identifier");
        let bundle_arg = Arg::with_name("bundle")
            .required(true)
            .default_value(".")
            .long("bundle")
            .short("b")
            .help("path to the root of the bundle directory, defaults to the current directory");
        let consoleSocket_arg = Arg::with_name("console-socket")
            .required(true)
            .long("console-socket")
            .default_value("")
            .takes_value(true)
            .help("path to an AF_UNIX socket which will receive a file descriptor referencing the master end of the console's pseudoterminal");
        let detach_arg = Arg::with_name("detach")
            .help("detach from the container's process")
            .long("detach")
            .short("d");
        let no_pivot_arg = Arg::with_name("no-pivot")
            .long("no_pivot")
            .help("do not use pivot root to jail process inside rootfs.  This should be used whenever the rootfs is on top of a ramdisk");
        let pid_arg = Arg::with_name("p")
            .takes_value(true)
            .default_value("")
            .long("pid-file")
            .short("p")
            .help("Additional location to write pid");
        let init_arg = Arg::with_name("n")
            .help("Do not create an init process")
            .long("no-init")
            .short("n");
        let format_arg = Arg::with_name("format")
            .help("select one of: table or json (default: 'table')")
            .default_value("table")
            .long("format")
            .short("f")
            .takes_value(true);
        let user_log_arg = Arg::with_name("user-log")
            .default_value("/var/log/")
            .long("user-log")
            .short("u")
            .takes_value(true);

        return Self {
            id_arg: id_arg,
            bundle_arg: bundle_arg,
            consoleSocket_arg: consoleSocket_arg,
            detach_arg: detach_arg,
            no_pivot_arg: no_pivot_arg,
            pid_arg: pid_arg,
            init_arg: init_arg,
            format_arg: format_arg,
            user_log_arg: user_log_arg,
        };
    }
}

pub fn Parse() -> Result<Arguments> {
    let common = CommonArgs::New();

    let matches = App::new("qvisor")
        .about("qvisor - run a secure container from an oci-runtime spec file")
        .setting(AppSettings::ColoredHelp)
        .author(crate_authors!("\n"))
        .setting(AppSettings::SubcommandRequired)
        .version(crate_version!())
        .arg(
            Arg::with_name("v")
                .multiple(true)
                .help("Sets the level of verbosity")
                .short("v"),
        )
        .arg(
            Arg::with_name("d")
                .help("Daemonize the process")
                .long("daemonize")
                .short("d"),
        )
        .arg(
            Arg::with_name("log")
                .help("Compatibility (ignored)")
                .default_value("/home/brad/rust/quark/qvisor.log")
                .long("log")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("systemd-cgroup")
                .help("systemd cgroup driver")
                .long("systemd-cgroup")
                .takes_value(false),
        )
        .arg(
            Arg::with_name("log-format")
                .help("Compatibility (ignored)")
                .long("log-format")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("r")
                .default_value("/run/qvisor")
                .help("Dir for state")
                .long("root")
                .short("r")
                .takes_value(true),
        )
        .subcommand(RunCmd::SubCommand(&common))
        .subcommand(CreateCmd::SubCommand(&common))
        .subcommand(StartCmd::SubCommand(&common))
        .subcommand(WaitCmd::SubCommand(&common))
        .subcommand(ListCmd::SubCommand(&common))
        .subcommand(CmdCmd::SubCommand(&common))
        .subcommand(BootCmd::SubCommand(&common))
        .subcommand(ExecCmd::SubCommand(&common))
        .subcommand(PauseCmd::SubCommand(&common))
        .subcommand(ResumeCmd::SubCommand(&common))
        .subcommand(PsCmd::SubCommand(&common))
        .subcommand(KillCmd::SubCommand(&common))
        .subcommand(DeleteCmd::SubCommand(&common))
        .subcommand(StateCmd::SubCommand(&common))
        .subcommand(SandboxCmd::SubCommand(&common))
        .get_matches_from(get_args());

    let level = match matches.occurrences_of("v") {
        0 => DebugLevel::Info, //default
        1 => DebugLevel::Debug,
        _ => DebugLevel::Trace,
    };

    let systemdCgroup = matches.occurrences_of("systemd-cgroup") > 0;

    //let _ = log::set_logger(&logger::SIMPLE_LOGGER)
    //    .map(|()| log::set_max_level(level));

    // create empty log file to avoid warning
    let logFile = matches.value_of("log").unwrap_or_default();

    let rootDir = matches.value_of("r").unwrap().to_string();
    /*debug!("ensuring railcar state dir {}", &state_dir);
    let chain = || format!("ensuring railcar state dir {} failed", &state_dir);
    create_dir_all(&state_dir).chain_err(chain)?;*/

    let gConfig = config::GlobalConfig {
        RootDir: rootDir.to_string(),
        DebugLevel: level,
        DebugLog: logFile.to_string(),
        FileAccess: config::FileAccessType::default(),
        Network: config::NetworkType::default(),
        SystemdCgroup: systemdCgroup,
    };

    let args = match matches.subcommand() {
        ("run", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::RunCmd(RunCmd::Init(&cmd_matches)?),
        },
        ("create", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::CreateCmd(CreateCmd::Init(&cmd_matches)?),
        },
        ("start", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::StartCmd(StartCmd::Init(&cmd_matches)?),
        },
        ("list", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::ListCmd(ListCmd::Init(&cmd_matches)?),
        },
        ("cmd", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::CmdCmd(CmdCmd::Init(&cmd_matches)?),
        },
        ("boot", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::BootCmd(BootCmd::Init(&cmd_matches)?),
        },
        ("exec", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::ExecCmd(ExecCmd::Init(&cmd_matches)?),
        },
        ("pause", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::PauseCmd(PauseCmd::Init(&cmd_matches)?),
        },
        ("resume", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::ResumeCmd(ResumeCmd::Init(&cmd_matches)?),
        },
        ("ps", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::PsCmd(PsCmd::Init(&cmd_matches)?),
        },
        ("wait", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::WaitCmd(WaitCmd::Init(&cmd_matches)?),
        },
        ("kill", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::KillCmd(KillCmd::Init(&cmd_matches)?),
        },
        ("delete", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::DeleteCmd(DeleteCmd::Init(&cmd_matches)?),
        },
        ("state", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::StateCmd(StateCmd::Init(&cmd_matches)?),
        },
        ("sandbox", Some(cmd_matches)) => Arguments {
            config: gConfig,
            cmd: Command::SandboxCmd(SandboxCmd::Init(&cmd_matches)?),
        },
        // We should never reach here because clap already enforces this
        _ => panic!("command not recognized"),
    };

    return Ok(args);
}

#[derive(Debug)]
pub struct Arguments {
    pub config: GlobalConfig,
    pub cmd: Command,
}

#[derive(Debug)]
pub enum Command {
    RunCmd(RunCmd),
    CreateCmd(CreateCmd),
    StartCmd(StartCmd),
    ListCmd(ListCmd),
    CmdCmd(CmdCmd),
    WaitCmd(WaitCmd),
    BootCmd(BootCmd),
    ExecCmd(ExecCmd),
    PauseCmd(PauseCmd),
    ResumeCmd(ResumeCmd),
    PsCmd(PsCmd),
    KillCmd(KillCmd),
    DeleteCmd(DeleteCmd),
    StateCmd(StateCmd),
    SandboxCmd(SandboxCmd),
}

pub fn Run(args: &mut Arguments) -> Result<()> {
    match &mut args.cmd {
        Command::RunCmd(cmd) => return cmd.Run(&mut args.config),
        Command::CreateCmd(cmd) => return cmd.Run(&mut args.config),
        Command::StartCmd(cmd) => return cmd.Run(&mut args.config),
        Command::ListCmd(cmd) => return cmd.Run(&mut args.config),
        Command::CmdCmd(cmd) => return cmd.Run(&mut args.config),
        Command::WaitCmd(cmd) => return cmd.Run(&mut args.config),
        Command::BootCmd(cmd) => return cmd.Run(&mut args.config),
        Command::ExecCmd(cmd) => return cmd.Run(&mut args.config),
        Command::PauseCmd(cmd) => return cmd.Run(&mut args.config),
        Command::ResumeCmd(cmd) => return cmd.Run(&mut args.config),
        Command::PsCmd(cmd) => return cmd.Run(&mut args.config),
        Command::KillCmd(cmd) => return cmd.Run(&mut args.config),
        Command::DeleteCmd(cmd) => return cmd.Run(&mut args.config),
        Command::StateCmd(cmd) => return cmd.Run(&mut args.config),
        Command::SandboxCmd(cmd) => return cmd.Run(&mut args.config),
    }
}
