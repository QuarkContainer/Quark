// Copyright (c) 2017, Oracle and/or its affiliates.  All rights reserved.
//
// copy and modified from https://github.com/oracle/railcar/
// license https://github.com/oracle/railcar/blob/master/LICENSE.txt

use libc;
use nix::errno::Errno;
use std::ffi::CString;
use nix::unistd::*;
use nix::sys::wait::{waitpid, WaitStatus as NixWaitStatus};
use nix::sys::wait::WaitPidFlag;
use nix::poll::{poll, EventFlags, PollFd as NixPollFd};
use nix::unistd::{close as NixClose, dup2, fork, pipe2, read as NixRead, write, ForkResult};
use nix::fcntl::{OFlag as NixOFlag};
use std::fs::{File};
use std::os::unix::io::{FromRawFd, RawFd};

use super::super::oci;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;

#[inline]
pub fn clearenv() -> Result<()> {
    let res = unsafe { libc::clearenv() };
    if res < 0 {
        return Err(Error::SysError(-res))
    }

    return Ok(())
}

pub fn putenv(string: &CString) -> Result<()> {
    let ptr = string.clone().into_raw();
    let res = unsafe { libc::putenv(ptr as *mut libc::c_char) };
    if res < 0 {
        return Err(Error::SysError(-res))
    }

    return Ok(())
}

fn do_exec(path: &str, args: &[String], env: &[String]) -> Result<()> {
    let p = CString::new(path.to_string()).unwrap();
    let a: Vec<CString> = args
        .iter()
        .map(|s| CString::new(s.to_string()).unwrap_or_default())
        .collect();
    let env: Vec<CString> = env
        .iter()
        .map(|s| CString::new(s.to_string()).unwrap_or_default())
        .collect();
    // execvp doesn't use env for the search path, so we set env manually
    clearenv()?;
    for e in &env {
        debug!("adding {:?} to env", e);
        putenv(e)?;
    }
    execvp(&p, &a).expect("failed to exec");
    // should never reach here
    Ok(())
}

fn reap_children() -> Result<NixWaitStatus> {
    let mut result = NixWaitStatus::Exited(Pid::from_raw(0), 0);
    loop {
        match waitpid(Pid::from_raw(-1), Some(WaitPidFlag::WNOHANG)) {
            Err(e) => {
                if e != ::nix::Error::Sys(Errno::ECHILD) {
                    return Err(Error::Common(format!("reap_children fail {:?}", e)));
                }
                // ECHILD means no processes are left
                break;
            }
            Ok(s) => {
                result = s;
                if result == NixWaitStatus::StillAlive {
                    break;
                }
            }
        }
    }
    Ok(result)
}

fn wait_for_child(child: Pid) -> Result<(i32, Option<Signal>)> {
    loop {
        // wait on all children, but only return if we match child.
        let result = match waitpid(Pid::from_raw(-1), None) {
            Err(::nix::Error::Sys(errno)) => {
                // ignore EINTR as it gets sent when we get a SIGCHLD
                if errno == Errno::EINTR {
                    continue;
                }
                let msg = format!("could not waitpid on {}", child);
                return Err(Error::Common(msg))
            }
            Err(e) => {
                return Err(Error::Common(format!("reap_children fail {:?}", e)));
            }
            Ok(s) => s,
        };
        match result {
            NixWaitStatus::Exited(pid, code) => {
                if child != Pid::from_raw(-1) && pid != child {
                    continue;
                }
                reap_children()?;
                return Ok((code as i32, None));
            }
            NixWaitStatus::Signaled(pid, signal, _) => {
                if child != Pid::from_raw(-1) && pid != child {
                    continue;
                }
                reap_children()?;
                return Ok((0, Some(Signal(signal as i32))));
            }
            _ => {}
        };
    }
}

fn wait_for_pipe_vec(
    rfd: RawFd,
    timeout: i32,
    num: usize,
) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    while result.len() < num {
        let pfds =
            &mut [NixPollFd::new(rfd, EventFlags::POLLIN | EventFlags::POLLHUP)];
        match poll(pfds, timeout) {
            Err(e) => {
                if e != ::nix::Error::Sys(Errno::EINTR) {
                    return Err(Error::Common(format!("unable to poll rfd {:?}", e)))
                }
                continue;
            }
            Ok(n) => {
                if n == 0 {
                    return Err(Error::SysError(SysErr::ETIMEDOUT));
                }
            }
        }
        let events = pfds[0].revents();
        if events.is_none() {
            // continue on no events
            continue;
        }
        if events.unwrap() == EventFlags::POLLNVAL {
            return Err(Error::SysError(SysErr::EPIPE));
        }
        if !events
            .unwrap()
            .intersects(EventFlags::POLLIN | EventFlags::POLLHUP)
            {
                // continue on other events (should not happen)
                debug!("got a continue on other events {:?}", events);
                continue;
            }
        let data: &mut [u8] = &mut [0];
        let n = NixRead(rfd, data).map_err(|e| Error::Common(format!("could not read from rfd {:?}", e)))?;
        if n == 0 {
            // the wfd was closed so close our end
            NixClose(rfd).map_err(|e| Error::Common(format!("could not close rfd {:?}", e)))?;
            break;
        }
        result.extend(data.iter().cloned());
    }
    Ok(result)
}

fn wait_for_pipe_sig(rfd: RawFd, timeout: i32) -> Result<Option<Signal>> {
    let result = wait_for_pipe_vec(rfd, timeout, 1)?;
    if result.len() < 1 {
        return Ok(None);
    }

    let s = Signal(result[0] as i32);
    Ok(Some(s))
}

pub fn execute_hook(hook: &oci::Hook, state: &oci::State) -> Result<()> {
    debug!("executing hook {:?}", hook);
    let (rfd, wfd) =
        pipe2(NixOFlag::O_CLOEXEC).map_err(|_| Error::Common("failed to create pipe".to_string()))?;
    match fork().map_err(|_| Error::Common("for fail".to_string()))? {
        ForkResult::Child => {
            close(rfd).map_err(|_| Error::Common("could not close rfd".to_string()))?;
            let (rstdin, wstdin) =
                pipe2(NixOFlag::empty()).map_err(|_| Error::Common("failed to create pipe".to_string()))?;
            // fork second child to execute hook
            match fork().map_err(|_| Error::Common("for fail".to_string()))? {
                ForkResult::Child => {
                    close(0).map_err(|_| Error::Common("could not close stdin".to_string()))?;
                    dup2(rstdin, 0).map_err(|_| Error::Common("could not dup to stdin".to_string()))?;
                    close(rstdin).map_err(|_| Error::Common("could not close rstdin".to_string()))?;
                    close(wstdin).map_err(|_| Error::Common("could not close wstdin".to_string()))?;
                    do_exec(&hook.path, &hook.args, &hook.env)?;
                }
                ForkResult::Parent { child } => {
                    close(rstdin).map_err(|_| Error::Common("could not close rstdin".to_string()))?;
                    unsafe {
                        // closes the file descriptor autmotaically
                        state
                            .to_writer(File::from_raw_fd(wstdin))
                            .map_err(|_| Error::Common("could not write state".to_string()))?;
                    }
                    let (exit_code, sig) = wait_for_child(child)?;
                    if let Some(signal) = sig {
                        // write signal to pipe.
                        let data: &[u8] = &[signal.0 as u8];
                        write(wfd, data)
                            .map_err(|_| Error::Common("failed to write signal hook".to_string()))?;
                    }
                    close(wfd).map_err(|_| Error::Common("could not close wfd".to_string()))?;
                    std::process::exit(exit_code as i32);
                }
            }
        }
        ForkResult::Parent { child } => {
            // the wfd is only used by the child so close it
            close(wfd).map_err(|_| Error::Common("could not close wfd".to_string()))?;
            let mut timeout = -1 as i32;
            if let Some(t) = hook.timeout {
                timeout = t as i32 * 1000;
            }
            // a timeout will cause a failure and child will be killed on exit
            if let Some(sig) = wait_for_pipe_sig(rfd, timeout)? {
                let msg = format!{"hook exited with signal: {:?}", sig};
                return Err(Error::Common(msg));
            }
            let (exit_code, _) = wait_for_child(child)?;
            if exit_code != 0 {
                let msg = format!{"hook exited with exit code: {}", exit_code};
                return Err(Error::Common(msg));
            }
        }
    };
    Ok(())
}