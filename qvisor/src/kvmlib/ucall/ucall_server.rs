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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::control_msg::*;
use super::super::qlib::loader;
use super::super::{IO_MGR};
use super::ucall::*;
use super::usocket::*;
use super::super::runc::container::container::*;
use super::super::vmspace::*;

pub fn ReadControlMsg(fd: i32) -> Result<ControlMsg> {
    let usock = USocket {
        socket: fd,
    };

    let (mut req, fds) = match usock.GetReq() {
        Ok((req, fds)) => ((req, fds)),
        Err(e) => {
            let err = UCallResp::UCallRespErr(format!("{:?}", e));
            usock.SendResp(&err)?;
            usock.Drop();
            return Err(e)
        }
    };

    let msg = ProcessReqHandler(&mut req, &fds);
    return msg
}

pub fn RootContainerStartHandler(start: &RootContainerStart) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::RootContainerStart(RootProcessStart{
        cid: start.cid.to_string(),
    }));

    return Ok(msg);
}

pub fn ExecProcessHandler(execArgs: &mut ExecArgs, fds: &[i32]) -> Result<ControlMsg> {
    execArgs.SetFds(fds);

    let mut process = loader::Process::default();
    process.ID = execArgs.ContainerID.to_string();
    process.Cwd = execArgs.WorkDir.to_string();
    process.Args.append(&mut execArgs.Argv);
    process.Envs.append(&mut execArgs.Envv);
    process.UID = execArgs.KUID.0;
    process.GID = execArgs.KGID.0;
    process.AdditionalGids.append(&mut execArgs.ExtraKGIDs.iter().map(| gid | gid.0).collect());
    process.Terminal = execArgs.Terminal;

    for i in 0..execArgs.Fds.len() {
        let osfd = execArgs.Fds[i];
        VMSpace::UnblockFd(osfd);

        let hostfd = IO_MGR.AddFile(osfd);

        process.Stdiofds[i] = hostfd;
    }

    let msg = ControlMsg::New(Payload::ExecProcess(process));

    return Ok(msg);
}

pub fn PauseHandler() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::Pause);
    return Ok(msg)
}

pub fn UnpauseHandler() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::Unpause);
    return Ok(msg)
}

pub fn PsHandler(cid: &str) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::Ps(cid.to_string()));
    return Ok(msg)
}

pub fn WaitHandler() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::WaitContainer);
    return Ok(msg)
}

pub fn WaitPidHandler(waitpid: &WaitPid) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::WaitPid(*waitpid));
    return Ok(msg)
}

pub fn SignalHandler(signalArgs: &SignalArgs) -> Result<ControlMsg> {
    if signalArgs.Signo == Signal::SIGKILL && signalArgs.Mode == SignalDeliveryMode::DeliverToAllProcesses {
        unsafe {
            // ucallServer::HandleSignal SIGKILL all processes
            libc::kill(0, 9);
            panic!("SignalHandler kill whole process")
        }
    }

    let msg = ControlMsg::New(Payload::Signal(*signalArgs));
    return Ok(msg)
}

pub fn ContainerDestroyHandler() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::ContainerDestroy);
    return Ok(msg)
}

pub fn ProcessReqHandler(req: &mut UCallReq, fds: &[i32]) -> Result<ControlMsg> {
    let msg = match req {
        UCallReq::RootContainerStart(start) => RootContainerStartHandler(start)?,
        UCallReq::ExecProcess(ref mut execArgs) => ExecProcessHandler(execArgs, fds)?,
        UCallReq::Pause => PauseHandler()?,
        UCallReq::Unpause => UnpauseHandler()?,
        UCallReq::Ps(cid) => PsHandler(cid)?,
        UCallReq::WaitContainer => WaitHandler()?,
        UCallReq::WaitPid(waitpid) => WaitPidHandler(waitpid)?,
        UCallReq::Signal(signalArgs) => SignalHandler(signalArgs)?,
        UCallReq::ContainerDestroy => ContainerDestroyHandler()?,
    };

    return Ok(msg)
}
