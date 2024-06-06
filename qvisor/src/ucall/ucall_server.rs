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

use crate::qlib::kernel::GlobalIOMgr;

use super::super::qlib::common::*;
use super::super::qlib::control_msg::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::loader;
use super::super::runc::container::container::*;
use super::super::vmspace::*;
use super::ucall::*;
use super::usocket::*;

pub fn ReadControlMsg(fd: i32) -> Result<ControlMsg> {
    let usock = USocket { socket: fd };

    let (mut req, fds) = match usock.GetReq() {
        Ok((req, fds)) => (req, fds),
        Err(e) => {
            let err = UCallResp::UCallRespErr(format!("{:?}", e));
            usock.SendResp(&err)?;
            usock.Drop();
            return Err(e);
        }
    };

    let msg = ProcessReqHandler(&mut req, &fds);
    return msg;
}

pub fn RootContainerStartHandler(start: &RootContainerStart) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::RootContainerStart(RootProcessStart {
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
    process.Caps = execArgs.Capabilities;
    process
        .AdditionalGids
        .append(&mut execArgs.ExtraKGIDs.iter().map(|gid| gid.0).collect());
    process.Terminal = execArgs.Terminal;
    process.ExecId = Some(execArgs.ExecId.clone());

    for i in 0..execArgs.Fds.len() {
        let osfd = execArgs.Fds[i];
        //VMSpace::UnblockFd(osfd);

        let hostfd = GlobalIOMgr().AddFile(osfd);
        process.Stdiofds[i] = hostfd;
    }

    let msg = ControlMsg::New(Payload::ExecProcess(process));

    return Ok(msg);
}

pub fn PauseHandler() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::Pause);
    return Ok(msg);
}

pub fn UnpauseHandler() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::Unpause);
    return Ok(msg);
}

pub fn PsHandler(cid: &str) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::Ps(cid.to_string()));
    return Ok(msg);
}

pub fn WaitHandler(cid: &str) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::WaitContainer(cid.to_string()));
    return Ok(msg);
}

pub fn WaitAll() -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::WaitAll);
    return Ok(msg);
}

pub fn WaitPidHandler(waitpid: &WaitPid) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::WaitPid(waitpid.clone()));
    return Ok(msg);
}

pub fn SignalHandler(signalArgs: &SignalArgs) -> Result<ControlMsg> {
    if signalArgs.Signo == Signal::SIGKILL
        && signalArgs.Mode == SignalDeliveryMode::DeliverToAllProcesses
    {
        unsafe {
            // ucallServer::HandleSignal SIGKILL all processes
            libc::kill(0, 9);
            panic!("SignalHandler kill whole process")
        }
    }

    let msg = ControlMsg::New(Payload::Signal(signalArgs.clone()));
    return Ok(msg);
}

pub fn ContainerDestroyHandler(cid: &String) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::ContainerDestroy(cid.clone()));
    return Ok(msg);
}

pub fn CreateSubContainerHandler(args: &mut CreateArgs, fds: &[i32]) -> Result<ControlMsg> {
    //set fds back to args,
    if fds.len() == 1 {
        args.fds[0] = fds[0];
    } else if fds.len() == 3 {
        args.fds[0] = fds[0];
        args.fds[1] = fds[1];
        args.fds[2] = fds[2];
    }

    for i in 0..fds.len() {
        let osfd = args.fds[i];
        VMSpace::UnblockFd(osfd);

        let hostfd = GlobalIOMgr().AddFile(osfd);
        args.fds[i] = hostfd;
    }

    let msg = ControlMsg::New(Payload::CreateSubContainer(args.clone()));
    return Ok(msg);
}

pub fn StartSubContainerHandler(args: &mut StartArgs) -> Result<ControlMsg> {
    let msg = ControlMsg::New(Payload::StartSubContainer(args.clone()));
    return Ok(msg);
}

pub fn ProcessReqHandler(req: &mut UCallReq, fds: &[i32]) -> Result<ControlMsg> {
    let msg = match req {
        UCallReq::RootContainerStart(start) => RootContainerStartHandler(start)?,
        UCallReq::ExecProcess(ref mut execArgs) => ExecProcessHandler(execArgs, fds)?,
        UCallReq::Pause => PauseHandler()?,
        UCallReq::Unpause => UnpauseHandler()?,
        UCallReq::Ps(cid) => PsHandler(cid)?,
        UCallReq::WaitContainer(cid) => WaitHandler(cid)?,
        UCallReq::WaitPid(waitpid) => WaitPidHandler(waitpid)?,
        UCallReq::Signal(signalArgs) => SignalHandler(signalArgs)?,
        UCallReq::ContainerDestroy(cid) => ContainerDestroyHandler(cid)?,
        UCallReq::CreateSubContainer(args) => CreateSubContainerHandler(args, fds)?,
        UCallReq::StartSubContainer(args) => StartSubContainerHandler(args)?,
        UCallReq::WaitAll => WaitAll()?,
    };

    return Ok(msg);
}
