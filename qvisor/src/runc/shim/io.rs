/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
use std::fmt::Debug;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam::sync::WaitGroup;

use containerd_shim::util::IntoOption;
use containerd_shim::{
    error::{Error, Result},
    io_error,
};

use std::fs::File;
use std::io::Result as IOResult;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::process::Stdio as ProcessStdio;
use std::sync::Mutex;
use std::process::Command;

use nix::unistd::{Gid, Uid};
use os_pipe::{PipeReader, PipeWriter};

pub trait Io: Debug + Send + Sync {
    /// Return write side of stdin
    fn stdin(&self) -> Option<Box<dyn Write + Send + Sync>> {
        None
    }

    /// Return read side of stdout
    fn stdout(&self) -> Option<Box<dyn Read + Send>> {
        None
    }

    /// Return read side of stderr
    fn stderr(&self) -> Option<Box<dyn Read + Send>> {
        None
    }

    /// Set IO for passed command.
    /// Read side of stdin, write side of stdout and write side of stderr should be provided to command.
    fn set(&self, cmd: &mut Command) -> Result<()>;

    /// Only close write side (should be stdout/err "from" runc process)
    fn close_after_start(&self);
}

#[derive(Debug, Clone)]
pub struct IOOption {
    pub open_stdin: bool,
    pub open_stdout: bool,
    pub open_stderr: bool,
}

impl Default for IOOption {
    fn default() -> Self {
        Self {
            open_stdin: true,
            open_stdout: true,
            open_stderr: true,
        }
    }
}

/// Struct to represent a pipe that can be used to transfer stdio inputs and outputs.
///
/// With this Io driver, methods of [crate::Runc] may capture the output/error messages.
/// When one side of the pipe is closed, the state will be represented with [`None`].
#[derive(Debug)]
pub struct Pipe {
    rd: PipeReader,
    wr: PipeWriter,
}

#[derive(Debug)]
pub struct PipedIo {
    stdin: Option<Pipe>,
    stdout: Option<Pipe>,
    stderr: Option<Pipe>,
}

impl Pipe {
    fn new() -> std::io::Result<Self> {
        let (rd, wr) = os_pipe::pipe()?;
        Ok(Self { rd, wr })
    }
}

impl PipedIo {
    pub fn new(uid: u32, gid: u32, opts: &IOOption) -> std::io::Result<Self> {
        Ok(Self {
            stdin: Self::create_pipe(uid, gid, opts.open_stdin, true)?,
            stdout: Self::create_pipe(uid, gid, opts.open_stdout, false)?,
            stderr: Self::create_pipe(uid, gid, opts.open_stderr, false)?,
        })
    }

    fn create_pipe(
        uid: u32,
        gid: u32,
        enabled: bool,
        stdin: bool,
    ) -> std::io::Result<Option<Pipe>> {
        if !enabled {
            return Ok(None);
        }

        let pipe = Pipe::new()?;
        let uid = Some(Uid::from_raw(uid));
        let gid = Some(Gid::from_raw(gid));
        if stdin {
            let rd = pipe.rd.try_clone()?;
            nix::unistd::fchown(rd.as_raw_fd(), uid, gid)?;
        } else {
            let wr = pipe.wr.try_clone()?;
            nix::unistd::fchown(wr.as_raw_fd(), uid, gid)?;
        }
        Ok(Some(pipe))
    }
}

impl Io for PipedIo {
    fn stdin(&self) -> Option<Box<dyn Write + Send + Sync>> {
        self.stdin.as_ref().and_then(|pipe| {
            pipe.wr
                .try_clone()
                .map(|x| Box::new(x) as Box<dyn Write + Send + Sync>)
                .ok()
        })
    }

    fn stdout(&self) -> Option<Box<dyn Read + Send>> {
        self.stdout.as_ref().and_then(|pipe| {
            pipe.rd
                .try_clone()
                .map(|x| Box::new(x) as Box<dyn Read + Send>)
                .ok()
        })
    }

    fn stderr(&self) -> Option<Box<dyn Read + Send>> {
        self.stderr.as_ref().and_then(|pipe| {
            pipe.rd
                .try_clone()
                .map(|x| Box::new(x) as Box<dyn Read + Send>)
                .ok()
        })
    }

    // Note that this internally use [`std::fs::File`]'s `try_clone()`.
    // Thus, the files passed to commands will be not closed after command exit.
    fn set(&self, cmd: &mut Command) -> std::io::Result<()> {
        if let Some(p) = self.stdin.as_ref() {
            let pr = p.rd.try_clone()?;
            cmd.stdin(pr);
        }

        if let Some(p) = self.stdout.as_ref() {
            let pw = p.wr.try_clone()?;
            cmd.stdout(pw);
        }

        if let Some(p) = self.stderr.as_ref() {
            let pw = p.wr.try_clone()?;
            cmd.stdout(pw);
        }

        Ok(())
    }

    fn close_after_start(&self) {
        if let Some(p) = self.stdout.as_ref() {
            nix::unistd::close(p.wr.as_raw_fd()).unwrap_or_else(|e| debug!("close stdout: {}", e));
        }

        if let Some(p) = self.stderr.as_ref() {
            nix::unistd::close(p.wr.as_raw_fd()).unwrap_or_else(|e| debug!("close stderr: {}", e));
        }
    }
}

/// IO driver to direct output/error messages to /dev/null.
///
/// With this Io driver, all methods of [crate::Runc] can't capture the output/error messages.
#[derive(Debug)]
pub struct NullIo {
    dev_null: Mutex<Option<File>>,
}

impl NullIo {
    pub fn new() -> std::io::Result<Self> {
        let fd = nix::fcntl::open(
            "/dev/null",
            nix::fcntl::OFlag::O_RDONLY,
            nix::sys::stat::Mode::empty(),
        )?;
        let dev_null = unsafe { Mutex::new(Some(std::fs::File::from_raw_fd(fd))) };
        Ok(Self { dev_null })
    }
}

impl Io for NullIo {
    fn set(&self, cmd: &mut Command) -> std::io::Result<()> {
        if let Some(null) = self.dev_null.lock().unwrap().as_ref() {
            cmd.stdout(null.try_clone()?);
            cmd.stderr(null.try_clone()?);
        }
        Ok(())
    }

    fn close_after_start(&self) {
        let mut m = self.dev_null.lock().unwrap();
        let _ = m.take();
    }
}

/// Io driver based on Stdio::inherited(), to direct outputs/errors to stdio.
///
/// With this Io driver, all methods of [crate::Runc] can't capture the output/error messages.
#[derive(Debug)]
pub struct InheritedStdIo {}

impl InheritedStdIo {
    pub fn new() -> std::io::Result<Self> {
        Ok(InheritedStdIo {})
    }
}

impl Io for InheritedStdIo {
    fn set(&self, cmd: &mut Command) -> std::io::Result<()> {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        Ok(())
    }

    fn close_after_start(&self) {}
}

/// Io driver based on Stdio::piped(), to capture outputs/errors from runC.
///
/// With this Io driver, methods of [crate::Runc] may capture the output/error messages.
#[derive(Debug)]
pub struct PipedStdIo {}

impl PipedStdIo {
    pub fn new() -> std::io::Result<Self> {
        Ok(PipedStdIo {})
    }
}

impl Io for PipedStdIo {
    fn set(&self, cmd: &mut Command) -> std::io::Result<()> {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        Ok(())
    }

    fn close_after_start(&self) {}
}

/// FIFO for the scenario that set FIFO for command Io.
#[derive(Debug)]
pub struct FIFO {
    pub stdin: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
}

impl Io for FIFO {
    fn set(&self, cmd: &mut Command) -> Result<()> {
        if let Some(path) = self.stdin.as_ref() {
            let stdin = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(path)?;
            cmd.stdin(stdin);
        }

        if let Some(path) = self.stdout.as_ref() {
            let stdout = OpenOptions::new().write(true).open(path)?;
            cmd.stdout(stdout);
        }

        if let Some(path) = self.stderr.as_ref() {
            let stderr = OpenOptions::new().write(true).open(path)?;
            cmd.stderr(stderr);
        }

        Ok(())
    }

    fn close_after_start(&self) {}
}


pub fn spawn_copy<R: Read + Send + 'static, W: Write + Send + 'static>(
    mut from: R,
    mut to: W,
    wg_opt: Option<&WaitGroup>,
    on_close_opt: Option<Box<dyn FnOnce() + Send + Sync>>,
) -> JoinHandle<()> {
    let wg_opt_clone = wg_opt.cloned();
    std::thread::spawn(move || {
        if let Err(e) = std::io::copy(&mut from, &mut to) {
            debug!("copy io error: {}", e);
        }
        if let Some(x) = on_close_opt {
            x()
        };
        if let Some(x) = wg_opt_clone {
            std::mem::drop(x)
        };
    })
}

#[derive(Clone, Debug)]
pub struct Stdio {
    pub stdin: String,
    pub stdout: String,
    pub stderr: String,
    pub terminal: bool,
}

impl Stdio {
    pub fn is_null(&self) -> bool {
        self.stdin.is_empty() && self.stdout.is_empty() && self.stderr.is_empty()
    }
}

pub struct ProcessIO {
    pub uri: Option<String>,
    pub io: Option<Arc<dyn Io>>,
    pub copy: bool,
}

impl ProcessIO {
    pub fn copy(&self, stdio: &Stdio) -> Result<WaitGroup> {
        let wg = WaitGroup::new();
        if !self.copy {
            return Ok(wg);
        };
        if let Some(pio) = &self.io {
            if let Some(w) = pio.stdin() {
                debug!("copy_io: pipe stdin from {}", stdio.stdin.as_str());
                if !stdio.stdin.is_empty() {
                    let stdin = OpenOptions::new()
                        .read(true)
                        .open(stdio.stdin.as_str())
                        .map_err(io_error!(e, "open stdin"))?;
                    spawn_copy(stdin, w, None, None);
                }
            }

            if let Some(r) = pio.stdout() {
                debug!("copy_io: pipe stdout from to {}", stdio.stdout.as_str());
                if !stdio.stdout.is_empty() {
                    let stdout = OpenOptions::new()
                        .write(true)
                        .open(stdio.stdout.as_str())
                        .map_err(io_error!(e, "open stdout"))?;
                    // open a read to make sure even if the read end of containerd shutdown,
                    // copy still continue until the restart of containerd succeed
                    let stdout_r = OpenOptions::new()
                        .read(true)
                        .open(stdio.stdout.as_str())
                        .map_err(io_error!(e, "open stdout for read"))?;
                    spawn_copy(
                        r,
                        stdout,
                        Some(&wg),
                        Some(Box::new(move || {
                            drop(stdout_r);
                        })),
                    );
                }
            }

            if let Some(r) = pio.stderr() {
                if !stdio.stderr.is_empty() {
                    debug!("copy_io: pipe stderr from to {}", stdio.stderr.as_str());
                    let stderr = OpenOptions::new()
                        .write(true)
                        .open(stdio.stderr.as_str())
                        .map_err(io_error!(e, "open stderr"))?;
                    // open a read to make sure even if the read end of containerd shutdown,
                    // copy still continue until the restart of containerd succeed
                    let stderr_r = OpenOptions::new()
                        .read(true)
                        .open(stdio.stderr.as_str())
                        .map_err(io_error!(e, "open stderr for read"))?;
                    spawn_copy(
                        r,
                        stderr,
                        Some(&wg),
                        Some(Box::new(move || {
                            drop(stderr_r);
                        })),
                    );
                }
            }
        }

        Ok(wg)
    }
}

pub fn create_io(id: &str, _io_uid: u32, _io_gid: u32, stdio: &Stdio) -> Result<ProcessIO> {
    if stdio.is_null() {
        let nio = NullIo::new().map_err(io_error!(e, "new Null Io"))?;
        let pio = ProcessIO {
            uri: None,
            io: Some(Arc::new(nio)),
            copy: false,
        };
        return Ok(pio);
    }
    let stdout = stdio.stdout.as_str();
    let scheme_path = stdout.trim().split("://").collect::<Vec<&str>>();
    let scheme: &str;
    let uri: String;
    if scheme_path.len() <= 1 {
        // no scheme specified
        // default schema to fifo
        uri = format!("fifo://{}", stdout);
        scheme = "fifo"
    } else {
        uri = stdout.to_string();
        scheme = scheme_path[0];
    }

    let mut pio = ProcessIO {
        uri: Some(uri),
        io: None,
        copy: false,
    };

    if scheme == "fifo" {
        debug!(
        "create named pipe io for container {}, stdin: {}, stdout: {}, stderr: {}",
        id,
        stdio.stdin.as_str(),
        stdio.stdout.as_str(),
        stdio.stderr.as_str()
        );
        let io = FIFO {
            stdin: stdio.stdin.to_string().none_if(|x| x.is_empty()),
            stdout: stdio.stdout.to_string().none_if(|x| x.is_empty()),
            stderr: stdio.stderr.to_string().none_if(|x| x.is_empty()),
        };
        pio.io = Some(Arc::new(io));
        pio.copy = false;
    }
    Ok(pio)
}
