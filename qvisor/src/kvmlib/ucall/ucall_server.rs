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

use libc::*;
use spin::Mutex;
use core::sync::atomic::AtomicBool;
use core::sync::atomic::Ordering;
use lazy_static::lazy_static;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::control_msg::*;
use super::super::qlib::loader;
use super::super::IO_MGR;
use super::ucall::*;
use super::usocket::*;
use super::super::runc::container::container::*;
use super::super::vmspace::*;

lazy_static! {
    pub static ref UCALL_SRV : Mutex<UCallController> = Mutex::new(UCallController::New());
    pub static ref STOP : AtomicBool = AtomicBool::new(false);
}

pub fn InitUCallController(sock: i32) -> Result<()> {
    error!("InitUCallController sock is {}", sock);
    UCALL_SRV.lock().Init(sock)?;

    return Ok(())
}

pub fn HandleSrv(srvSock: i32) -> Result<()> {
    loop {
        let sock = unsafe {
            accept(srvSock, 0 as *mut sockaddr, 0 as *mut socklen_t)
        };

        if sock < 0 {
            let errno = errno::errno().0 as i32;

            //no connection
            if errno == SysErr::EAGAIN {
                return Ok(())
            }

            return Err(Error::SysError(errno))
        }

        let usock = USocket {
            socket: sock,
        };

        let (mut req, fds) = match usock.GetReq() {
            Ok((req, fds)) => ((req, fds)),
            Err(e) => {
                let err = UCallResp::UCallRespErr(format!("{:?}", e));
                usock.SendResp(&err)?;
                return Ok(())
            }
        };

        return ProcessReq(usock, &mut req, &fds);
    }
}

pub fn HandleRootContainerStart(usock: USocket, start: &RootContainerStart) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::RootContainerStart(RootProcessStart{
        cid: start.cid.to_string(),
    })))?;

    return Ok(());
}

pub fn HandleExecProcess(usock: USocket, execArgs: &mut ExecArgs, fds: &[i32]) -> Result<()> {
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
        let stat = VMSpace::LibcFstat(osfd)?;

        VMSpace::UnblockFd(osfd);

        let st_mode = stat.st_mode & ModeType::S_IFMT as u32;
        let epollable = st_mode == S_IFIFO || st_mode == S_IFSOCK || st_mode == S_IFCHR;

        let hostfd = IO_MGR.lock().AddFd(osfd, epollable);

        // can block wait
        /*if epollable {
            FD_NOTIFIER.AddFd(osfd, Box::new(GuestFd{hostfd: hostfd}));
        }*/

        process.Stdiofds[i] = hostfd;
    }

    SendControlMsg(usock, ControlMsg::New(Payload::ExecProcess(process)))?;

    return Ok(());
}

pub fn HandlePause(usock: USocket) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::Pause))?;
    return Ok(())
}

pub fn HandleUnpause(usock: USocket) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::Unpause))?;
    return Ok(())
}

pub fn HandlePs(usock: USocket, cid: &str) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::Ps(cid.to_string())))?;
    return Ok(())
}

pub fn HandleWait(usock: USocket) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::WaitContainer))?;
    return Ok(())
}

pub fn HandleWaitPid(usock: USocket, waitpid: &WaitPid) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::WaitPid(*waitpid)))?;
    return Ok(())
}

pub fn HandleSignal(usock: USocket, signalArgs: &SignalArgs) -> Result<()> {
    //error!("ucallServer::HandleSignal signalArgs is {:?}", signalArgs);

    // workaournd: in case the guest threads crash.
    // todo: fix this
    if signalArgs.Signo == Signal::SIGKILL && signalArgs.Mode == SignalDeliveryMode::DeliverToAllProcesses {
        unsafe {
            // ucallServer::HandleSignal SIGKILL all processes
            libc::kill(0, 9);
            return Ok(())
        }
    }

    SendControlMsg(usock, ControlMsg::New(Payload::Signal(*signalArgs)))?;
    return Ok(())
}

pub fn HandleContainerDestroy(usock: USocket) -> Result<()> {
    SendControlMsg(usock, ControlMsg::New(Payload::ContainerDestroy))?;
    return Ok(())
}

pub fn ProcessReq(usock: USocket, req: &mut UCallReq, fds: &[i32]) -> Result<()> {
    match req {
        UCallReq::RootContainerStart(start) => HandleRootContainerStart(usock, start)?,
        UCallReq::ExecProcess(ref mut execArgs) => HandleExecProcess(usock, execArgs, fds)?,
        UCallReq::Pause => HandlePause(usock)?,
        UCallReq::Unpause => HandleUnpause(usock)?,
        UCallReq::Ps(cid) => HandlePs(usock, cid)?,
        UCallReq::WaitContainer => HandleWait(usock)?,
        UCallReq::WaitPid(waitpid) => HandleWaitPid(usock, waitpid)?,
        UCallReq::Signal(signalArgs) => HandleSignal(usock, signalArgs)?,
        UCallReq::ContainerDestroy => HandleContainerDestroy(usock)?,
    };

    return Ok(())
}

pub fn HandleEvent() -> Result<()> {
    return Ok(())
}

pub fn Stop() -> Result<()> {
    STOP.store(true, Ordering::SeqCst);
    return UCALL_SRV.lock().Notify();
}

pub fn UcallSrvProcess() -> Result<()> {
    let epollSock = UCALL_SRV.lock().epollSock;
    let srvSock = UCALL_SRV.lock().srvSock;
    let _eventfd = UCALL_SRV.lock().eventfd;

    let mut events = [epoll_event { events: 0, u64: 0 }; 2];

    while !STOP.load(Ordering::SeqCst) {
        let nfds = unsafe {
            epoll_wait(epollSock, &mut events[0], 2, -1)
        };

        if nfds == -1 {
            return Err(Error::Common(format!("UCallController wait fail with err {}", errno::errno().0)));
        }

        for i in 0..nfds as usize {
            let fd = events[i].u64 as i32;
            if fd == srvSock {
                HandleSrv(srvSock)?;
            } else { //eventfd
                HandleEvent()?;
            }
        }
    }

    UCALL_SRV.lock().Close();
    return Ok(())
}

pub struct UCallController {
    pub epollSock: i32,
    pub srvSock: i32,
    pub eventfd: i32,
}

impl UCallController {
    pub fn New() -> Self {
        return Self {
            epollSock: 0,
            srvSock: 0,
            eventfd: 0,
        }
    }

    pub fn Close(&mut self) {
        unsafe {
            close(self.epollSock);
            close(self.srvSock);
            close(self.eventfd);
        }
    }

    pub fn Init(&mut self, sock: i32) -> Result<()> {
        let eventfd = unsafe {
            eventfd(0, 0)
        };

        if eventfd < 0 {
            info!("USrvSocket eventfd fail");
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        let epollfd = unsafe {
            epoll_create1(0)
        };

        if epollfd == -1 {
            info!("USrvSocket epoll_create fail");
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        let mut event = epoll_event {
            events: (EPOLLIN | EPOLLHUP | EPOLLERR) as u32,
            u64: sock as u64,
        };

        let ret = unsafe {
            epoll_ctl(epollfd, EPOLL_CTL_ADD, sock, &mut event)
        };

        if ret < 0 {
            error!("USrvSocket epoll_ctl add fd fail with err {}", errno::errno().0 as i32);
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        let mut event = epoll_event {
            events: (EPOLLIN | EPOLLHUP | EPOLLERR) as u32,
            u64: eventfd as u64,
        };

        let ret = unsafe {
            epoll_ctl(epollfd, EPOLL_CTL_ADD, eventfd, &mut event)
        };

        if ret < 0 {
            error!("USrvSocket epoll_ctl add fd fail with err {}", errno::errno().0 as i32);
            return Err(Error::SysError(errno::errno().0 as i32))
        }

        self.srvSock = sock;
        self.eventfd = eventfd;
        self.epollSock = epollfd;

        return Ok(())
    }

    pub fn Notify(&self) -> Result<()> {
        let data: u64 = 1;
        unsafe {
            write(self.eventfd, &data as *const _ as *const c_void, 8);
        }

        return Ok(())
    }

}

pub trait UcallCallback {
    fn Callback(&self, payload: &Payload) -> Result<()>;
}