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
use libc::*;
use std::{thread, time};
use alloc::str;
use lazy_static::lazy_static;
use spin::Mutex;
use nix::sys::signal;

use super::super::super::qlib::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::control_msg::*;
use super::super::super::ucall::ucall::*;
use super::super::super::ucall::ucall_client::*;
use super::super::super::vmspace::syscall::*;
use super::super::runtime::console::*;
use super::super::cgroup::*;
use super::super::oci::*;
use super::super::container::container::*;
use super::super::cmd::config::*;
use super::super::runtime::sandbox_process::*;

lazy_static! {
    static ref SIGNAL_STRUCT : Mutex<Option<SignalStruct>> = Mutex::new(None);
}

extern fn handle_sigint(signal :i32) {
    if SIGNAL_STRUCT.lock().is_none() {
        return
    }

    error!("exec signal {}", signal);

    SIGNAL_STRUCT.lock().as_ref().unwrap().SignalProcess(signal).unwrap();
}

// numSignals is the number of normal (non-realtime) signals on Linux.
pub const NUM_SIGNALS : usize = 32;

pub struct SignalStruct {
    pub cid: String,
    pub pid: i32,
}

impl SignalStruct {
    pub fn New(cid: &str, pid: i32) {
        let data = Self {
            cid: cid.to_string(),
            pid: pid,
        };

        error!("enable forward signal in exec");
        unsafe {
            libc::ioctl( 0, libc::TIOCSCTTY, 0);
        }

        *SIGNAL_STRUCT.lock() = Some(data);

        let sig_action = signal::SigAction::new(signal::SigHandler::Handler(handle_sigint),
                                                signal::SaFlags::empty(),
                                                signal::SigSet::empty());

        for i in 1..NUM_SIGNALS {
            if i == 9           //SIGKILL
                || i == 19 {    //SIGSTOP
                continue
            }

            unsafe {
                signal::sigaction(signal::Signal::from_c_int(i as i32).unwrap(), &sig_action)
                    .map_err(|e| Error::Common(format!("sigaction fail with err {:?} for signal {}", e, i)))
                    .unwrap();
            }
        }
    }

    pub fn StopSignal() {
        *SIGNAL_STRUCT.lock() = None;
    }

    pub fn SignalProcess(&self, signo: i32) -> Result<()> {
        return SignalProcess(&self.cid, self.pid, signo, true);
    }
}

pub fn SignalProcess(cid: &str, pid: i32, signo: i32, fgProcess: bool) -> Result<()> {
    info!("Signal sandbox {}", cid);

    let addr = ControlSocketAddr(cid);
    info!("SandboxConnect connect address is {}", &addr);
    let client = UCallClient::Init(&addr)?;

    let mut mode = SignalDeliveryMode::DeliverToProcess;
    if fgProcess {
        mode = SignalDeliveryMode::DeliverToForegroundProcessGroup;
    }

    let req = UCallReq::Signal(SignalArgs{
        Signo: signo,
        PID: pid,
        Mode: mode,
    });

    let resp = client.Call(&req)?;
    match resp {
        UCallResp::SignalResp => {
            return Ok(())
        },
        resp => {
            panic!("SignalProcess get unknow resp {:?}", resp);
        }
    }
}

// Sandbox wraps a sandbox process.
//
// Note: Sandbox must be immutable because a copy of it is saved for each
// container and changes would not be synchronized to all of them.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Sandbox {
    // ID is the id of the sandbox (immutable). By convention, this is the same
    // ID as the first container run in the sandbox.
    pub ID: String,

    // Pid is the pid of the running sandbox (immutable). May be 0 is the sandbox
    // is not running.
    pub Pid: i32,

    // Cgroup has the cgroup configuration for the sandbox.
    pub Cgroup: Option<Cgroup>,

    // child is set if a sandbox process is a child of the current process.
    //
    // This field isn't saved to json, because only a creator of sandbox
    // will have it as a child process.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub child: bool,

    // status is an exit status of a sandbox process.
    #[serde(default, skip_serializing)]
    pub status: u32,

    #[serde(default, skip_serializing, skip_deserializing)]
    pub autoStart: bool,

    #[serde(default, skip_serializing, skip_deserializing)]
    pub pivot: bool,

    #[serde(skip_serializing, skip_deserializing)]
    pub console: Console,
}

impl Sandbox {
    pub fn New(id: &str,
               action: RunAction,
               spec: &Spec,
               conf: &GlobalConfig,
               bundleDir: &str,
               consoleSocket: &str,
               _userlog: &str,
               cg: Option<Cgroup>,
               detach: bool,
               pivot: bool) -> Result<Self> {
        let mut s = Self {
            ID: id.to_string(),
            Cgroup: cg,
            ..Default::default()
        };

        //let pid = CreateSandboxProcess(conf, id, bundleDir, ptyfd, autoStart)?;

        let process = &spec.process;
        let terminal = process.terminal;

        let process = SandboxProcess::New(conf, action, id, bundleDir, pivot)?;
        //let pid = process.Fork()?;
        let (pid, console) = process.Execv(terminal, consoleSocket, detach)?;

        s.console = console;
        s.child = true;
        s.Pid = pid;

        return Ok(s)
    }

    pub fn ForwardSignals(&self, pid: i32) {
        SignalStruct::New(&self.ID, pid);
    }

    pub fn StopSignal(&self) {
        SignalStruct::StopSignal();
    }

    pub fn GetCwd(&self, buf: u64, size: u64) -> i64 {
        let nr = SysCallID::sys_getcwd as usize;

        unsafe {
            let res = syscall2(nr, buf as usize, size as usize) as i64;
            return res
        }
    }

    pub fn Pause(&self, cid: &str) -> Result<()> {
        info!("Pause sandbox {}", cid);

        let client = self.SandboxConnect()?;

        let req = UCallReq::Pause;

        let _resp = client.Call(&req)?;

        return Ok(());
    }

    pub fn Unpause(&self, cid: &str) -> Result<()> {
        info!("Unpause sandbox {}", cid);

        let client = self.SandboxConnect()?;

        let req = UCallReq::Unpause;

        let _resp = client.Call(&req)?;

        return Ok(());
    }

    pub fn Processes(&self, cid: &str) -> Result<Vec<ProcessInfo>> {
        info!("Getting processes for container {} in sandbox {}", cid, self.ID);
        let client = self.SandboxConnect()?;

        let req = UCallReq::Ps(cid.to_string());

        let resp = client.Call(&req)?;
        match resp {
            UCallResp::PsResp(ps) => Ok(ps),
            resp => {
                panic!("Processes get unknow resp {:?}", resp);
            }
        }
    }

    pub fn StartRootContainer(&self) -> Result<()> {
        let client = self.SandboxConnect()?;

        let req = UCallReq::RootContainerStart(RootContainerStart{
            cid: self.ID.to_string(),
        });

        let _resp = client.Call(&req)?;

        return Ok(());
    }

    pub fn Execute(&self, mut args: ExecArgs) -> Result<i32> {
        info!("Executing new process in container {} in sandbox {}", &args.ContainerID, &self.ID);

        args.Fds.push(0);
        args.Fds.push(1);
        args.Fds.push(2);

        let client = self.SandboxConnect()?;
        let req = UCallReq::ExecProcess(args);
        let pid = match client.Call(&req)? {
            UCallResp::ExecProcessResp(pid) => pid,
            resp => panic!("sandbox::Execute get error {:?}", resp),
        };

        return Ok(pid)
    }

    pub fn Destroy(&mut self) -> Result<()> {
        info!("Destroy sandbox {}", &self.ID);

        if self.Pid != 0 {
            info!("Killing sandbox {}", &self.ID);
            let ret = unsafe {
                kill(self.Pid, SIGKILL)
            };

            if ret < 0 && errno::errno().0 != ESRCH {
                return Err(Error::Common(format!("killing sandbox {} PID {}: {}",
                                                 &self.ID, &self.Pid, errno::errno().0)));
            }

            return self.WaitForStopped();
        }

        return Ok(())
    }

    pub fn WaitPID(&mut self, _cid: &str, pid: i32, clearStatus: bool) -> Result<u32> {
        let client = self.SandboxConnect()?;

        let req = UCallReq::WaitPid(WaitPid{
            pid: pid,
            clearStatus: clearStatus,
        });

        let resp = client.Call(&req)?;
        match resp {
            UCallResp::WaitPidResp(status) => {
                println!("WaitPID status is {}", WaitStatus(status).ExitStatus());
                return Ok(status);
            },
            resp => {
                panic!("WaitPID get unknow resp {:?}", resp);
            }
        }
    }

    pub fn Wait(&mut self, _cid: &str) -> Result<u32> {
        match self.SandboxConnect() {
            Ok(client) => {
                let req = UCallReq::WaitContainer;

                match client.Call(&req) {
                    Ok(UCallResp::WaitContainerResp(status)) => {
                        println!("Wait status is {}", WaitStatus(status).ExitStatus());
                    }
                    Ok(resp) => {
                        println!("wait resp is {:?}", resp);
                    }
                    Err(e) => {
                        println!("wait resp error is {:?}", e);
                    }
                };
            }
            //the container has exited
            Err(Error::SysError(SysErr::ECONNREFUSED)) =>{
                info!("Wait: connect fail....");
            },
            Err(e) => return Err(e),
        }

        // The sandbox may have already exited, or exited while handling the
        // Wait RPC. The best we can do is ask Linux what the sandbox exit
        // status was, since in most cases that will be the same as the
        // container exit status.
        self.WaitForStopped()?;
        return Ok(self.status)
    }

    pub fn WaitForStopped(&mut self) -> Result<()> {
        info!("self child is {}, pid is {}", self.child, self.Pid);
        let ms = 5 * 1000; //5 sec
        for _i in 0 .. (ms/10) as usize {
            if self.child {
                if self.Pid == 0 {
                    return Ok(())
                }

                // The sandbox process is a child of the current process,
                // so we can wait it and collect its zombie.
                //info!("start to wait pid {}", self.Pid);
                let ret = unsafe {
                    wait4(self.Pid, &mut self.status as * mut _ as * mut i32, WNOHANG, 0 as *mut rusage)
                };

                if ret > 0 {
                    self.Pid = 0;
                    return Ok(())
                }

                if ret < 0 {
                    info!("wait sandbox fail use error {}", errno::errno().0);
                }
            } else if self.IsRunning() {
                continue;
            } else {
                return Ok(())
            }

            let ten_millis = time::Duration::from_millis(10);
            thread::sleep(ten_millis);
        }

        return Err(Error::Common(format!("wait sandbox {} timeout", self.ID)));
    }

    pub fn SandboxConnect(&self) -> Result<UCallClient> {
        let addr = ControlSocketAddr(&self.ID);
        info!("SandboxConnect connect address is {}", &addr);
        let client = UCallClient::Init(&addr)?;
        return Ok(client)
    }

    // IsRunning returns true if the sandbox is running.
    pub fn IsRunning(&self) -> bool {
        /*if self.Pid != 0 {
            // Send a signal 0 to the sandbox process.
            let ret = unsafe {
                kill(self.Pid, 0)
            };

            if ret == 0 {
                // Succeeded, process is running.
                return true;
            }
        }

        return false;*/
        return true;
    }

    pub fn SignalContainer(&self, _cid: &str, signo: i32, all: bool) -> Result<()> {
        info!("Signal container sandbox {}", &self.ID);

        let client = self.SandboxConnect()?;

        let mut mode = SignalDeliveryMode::DeliverToProcess;
        if all {
            mode = SignalDeliveryMode::DeliverToAllProcesses;
        }

        let req = UCallReq::Signal(SignalArgs{
            Signo: signo,
            PID: 0,
            Mode: mode,
        });

        let resp = client.Call(&req)?;
        match resp {
            UCallResp::SignalResp => {
                return Ok(())
            },
            resp => {
                panic!("SignalContainer get unknow resp {:?}", resp);
            }
        }
    }

    pub fn SignalProcess(&self, _cid: &str, pid: i32, signo: i32, fgProcess: bool) -> Result<()> {
        return SignalProcess(&self.ID, pid, signo, fgProcess)
    }

    pub fn DestroyContainer(&self, _cid: &str) -> Result<()> {
        let client = self.SandboxConnect()?;
        let req = UCallReq::ContainerDestroy;
        let resp = client.Call(&req)?;
        match resp {
            UCallResp::ContainerDestroyResp => {
                return Ok(())
            },
            resp => {
                panic!("DestroyContainer get unknow resp {:?}", resp);
            }
        }
    }

    pub fn IsRootContainer(&self, cid: &str) -> bool {
        return self.ID.as_str() == cid
    }

}