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

use alloc::vec::Vec;
use core::{ptr};
use alloc::string::String;

use super::super::super::common::*;
use super::super::super::control_msg::*;
use super::super::super::vcpu_mgr::*;
use super::super::Kernel;
use super::super::taskMgr;
use super::super::task::*;
use super::super::super::super::kernel_def::{StartRootContainer, StartExecProcess, StartSubContainerProcess};
use super::super::LOADER;
use super::super::SetWaitContainerfd;
use super::super::WaitContainerfd;
use super::super::IOURING;
use super::super::SHARESPACE;
use super::process::*;

pub fn ControllerProcessHandler() -> Result<()> {
    let task = Task::Current();
    loop {
        let fd = IOURING.SyncAccept(task, SHARESPACE.controlSock);
        taskMgr::CreateTask(ControlMsgHandler as u64, fd as *const u8, false);
    }
}

pub fn HandleSignal(signalArgs: &SignalArgs) {
    info!("get signal {:?}", &signalArgs);
    let task = Task::Current();
    match signalArgs.Mode {
        SignalDeliveryMode::DeliverToProcess => {
            match LOADER.Lock(task).unwrap().SignalProcess(signalArgs.CID.clone(), signalArgs.PID, signalArgs.Signo) {
                Err(e) => {
                    info!("signal DeliverToProcess fail with error {:?}", e);
                }
                Ok(())=> ()
            }
        }
        SignalDeliveryMode::DeliverToAllProcesses => {
            match LOADER.Lock(task).unwrap().SignalAll(signalArgs.Signo) {
                Err(e) => {
                    info!("signal DeliverToAllProcesses fail with error {:?}", e);
                }
                Ok(())=> ()
            }
        }
        SignalDeliveryMode::DeliverToForegroundProcessGroup => {
            match LOADER.Lock(task).unwrap().SignalForegroundProcessGroup(signalArgs.CID.clone(), signalArgs.PID, signalArgs.Signo) {
                Err(_e) => {
                    info!("signal DeliverToForegroundProcessGroup fail with error");
                    //todo: enable the error when ready
                    //info!("signal DeliverToForegroundProcessGroup fail with error {:?}", e);
                }
                Ok(())=> ()
            }
        }
    };

}

pub fn SignalHandler(_ :  *const u8) {
    let msg = SHARESPACE.signalArgs.lock().take();
    match msg {
        None => (),
        Some(msg) => {
            HandleSignal(&msg);
        }
    }

    CPULocal::SetPendingFreeStack(Task::Current().taskId);
    super::super::taskMgr::SwitchToNewTask();
}

pub fn ControlMsgHandler(fd: *const u8) {
    let fd = fd as i32;

    let task = Task::Current();
    let msg = {
        let mut buf: [u8; 8192] = [0; 8192];
        let addr = &mut buf[0] as * mut _ as u64;
        let ret = Kernel::HostSpace::ReadControlMsg(fd, addr, buf.len());

        if ret < 0 {
            return
        }

        let size = ret as usize;

        let msg : ControlMsg = serde_json::from_slice(&buf[0..size]).expect(&format!("LoadProcessKernel des fail size is {}", size));
        msg
    };

    match msg.payload {
        Payload::Pause => {
            let kernel = LOADER.Lock(task).unwrap().kernel.clone();
            kernel.Pause();
            WriteControlMsgResp(fd, &UCallResp::PauseResp);
        }
        Payload::Unpause => {
            let kernel = LOADER.Lock(task).unwrap().kernel.clone();
            kernel.Unpause();
            WriteControlMsgResp(fd, &UCallResp::UnpauseResp);
        }
        Payload::Ps(cid) => {
            let kernel = LOADER.Lock(task).unwrap().kernel.clone();
            let ps = Processes(&kernel, &cid);
            WriteControlMsgResp(fd, &UCallResp::PsResp(ps));
        }
        Payload::Signal(signalArgs) => {
            HandleSignal(&signalArgs);
            WriteControlMsgResp(fd, &UCallResp::SignalResp);
        }
        Payload::ContainerDestroy(cid) => {
            LOADER.Lock(task).unwrap().DestroyContainer(cid).unwrap();
            WriteControlMsgResp(fd, &UCallResp::ContainerDestroyResp);
        }
        Payload::RootContainerStart(_) => {
            WriteControlMsgResp(fd, &UCallResp::RootContainerStartResp);
            StartRootContainer(ptr::null());
        }
        Payload::ExecProcess(process) => {
            StartExecProcess(fd, process);
        }
        Payload::WaitContainer(cid) => {
            match LOADER.WaitContainer(cid) {
                Ok(exitStatus) => {
                    WriteControlMsgResp(fd, &UCallResp::WaitContainerResp(exitStatus));
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)));
                }
            }
        }
        Payload::WaitPid(waitpid) => {
            match LOADER.WaitPID(waitpid.cid, waitpid.pid, waitpid.clearStatus) {
                Ok(exitStatus) => {
                    WriteControlMsgResp(fd, &UCallResp::WaitPidResp(exitStatus));
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)));
                }
            }
        }
        Payload::CreateSubContainer(createArgs) => {
            match LOADER.CreateSubContainer(createArgs.cid, createArgs.fds) {
                Ok(()) => {
                    WriteControlMsgResp(fd, &UCallResp::CreateSubContainerResp);
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)));
                }
            }
        }
        Payload::StartSubContainer(startArgs) => {
            match LOADER.StartSubContainer(startArgs.process) {
                Ok((_, entry, userStackAddr, kernelStackAddr)) => {
                    WriteControlMsgResp(fd, &UCallResp::StartSubContainerResp);
                    StartSubContainerProcess(entry, userStackAddr, kernelStackAddr);
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)));
                }
            }

        }
        Payload::WaitAll => {
            SetWaitContainerfd(fd);
        }
    }

    // free curent task in the waitfn context
    CPULocal::SetPendingFreeStack(Task::Current().taskId);
    super::super::taskMgr::SwitchToNewTask();
}

pub fn WriteWaitAllResponse(cid: String, execId: String, status: i32) {
    let fd = WaitContainerfd();
    WriteControlMsgResp(fd, &UCallResp::WaitAllResp(WaitAllResp{
        cid,
        execId,
        status
    }));
}

pub fn WriteControlMsgResp(fd: i32, msg: &UCallResp) {
    let data : Vec<u8> = serde_json::to_vec(&msg).expect("LoadProcessKernel ser fail...");
    let addr = &data[0] as * const _ as u64;
    let len = data.len();

    Kernel::HostSpace::WriteControlMsgResp(fd, addr, len);
}