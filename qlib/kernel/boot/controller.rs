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
use alloc::vec::Vec;
use core::ptr;
use core::sync::atomic;

use crate::qlib::kernel::kernel::kernel::GetKernel;
use crate::qlib::kernel::Kernel::HostSpace;
//use crate::qlib::mem::list_allocator::*;
use super::super::super::super::kernel_def::{
    StartExecProcess, StartRootContainer, StartSubContainerProcess,
};
use super::super::super::common::*;
use super::super::super::control_msg::*;
use super::super::super::vcpu_mgr::*;
use super::super::task::*;
use super::super::taskMgr;
use super::super::Kernel;
use super::super::SetWaitContainerfd;
use super::super::WaitContainerfd;
use super::super::IOURING;
use super::super::LOADER;
use super::super::SHARESPACE;
use super::process::*;
use crate::qlib::linux::signal::*;
//use crate::qlib::kernel::vcpu::CPU_LOCAL;

pub fn ControllerProcessHandler() -> Result<()> {
    let task = Task::Current();
    loop {
        let fd = IOURING.SyncAccept(task, SHARESPACE.controlSock);
        taskMgr::CreateTask(ControlMsgHandler as u64, fd as *const u8, false);
    }
}

pub fn HandleSignal(signalArgs: &SignalArgs) {
    info!("HandleSignal: get signal {:?}", &signalArgs);

    // don't know why the winsz adjustment doesn't work
    // todo: fix this
    if signalArgs.Signo == 28 || signalArgs.Signo == 0 {
        return;
    }

    // SIGSTOP
    /*if signalArgs.Signo == 12 { //SIGSTOP.0 {
        //GetKernel().Pause();

        GetKernel().ClearFsCache();
        HostSpace::SwapOut();

        /*for vcpu in CPU_LOCAL.iter() {
            vcpu.AllocatorMut().Clear();
        }*/

        return;
    }*/

    {
        let hibernate_enabled = SHARESPACE.config.read().EnableHibernante;
        if hibernate_enabled {
            if signalArgs.Signo == SIGSTOP.0 || signalArgs.Signo == SIGUSR2.0 {
                if SHARESPACE.hibernatePause.load(atomic::Ordering::Relaxed) {
                    // if the sandbox has been paused, return
                    return;
                }
                GetKernel().Pause();
                GetKernel().ClearFsCache();
                HostSpace::SwapOut();
                SHARESPACE
                    .hibernatePause
                    .store(true, atomic::Ordering::SeqCst);
                return;
        
                /*for vcpu in CPU_LOCAL.iter() {
                    vcpu.AllocatorMut().Clear();
                }*/
            }
        
            if signalArgs.Signo == SIGCONT.0
                || signalArgs.Signo == SIGKILL.0
                || signalArgs.Signo == SIGINT.0
            {
                if SHARESPACE.hibernatePause.load(atomic::Ordering::Relaxed) {
                    SHARESPACE
                        .hibernatePause
                        .store(false, atomic::Ordering::SeqCst);
                    HostSpace::SwapIn();
                    GetKernel().Unpause();
                }
        
                if signalArgs.Signo == SIGCONT.0 {
                    return;
                }
            }
        }
    }



    let task = Task::Current();
    match signalArgs.Mode {
        SignalDeliveryMode::DeliverToProcess => {
            match LOADER.Lock(task).unwrap().SignalProcess(
                signalArgs.CID.clone(),
                signalArgs.PID,
                signalArgs.Signo,
            ) {
                Err(e) => {
                    info!("signal DeliverToProcess fail with error {:?}", e);
                }
                Ok(()) => (),
            }
        }
        SignalDeliveryMode::DeliverToAllProcesses => {
            match LOADER.Lock(task).unwrap().SignalAll(signalArgs.Signo) {
                Err(e) => {
                    info!("signal DeliverToAllProcesses fail with error {:?}", e);
                }
                Ok(()) => (),
            }
        }
        SignalDeliveryMode::DeliverToForegroundProcessGroup => {
            match LOADER.Lock(task).unwrap().SignalForegroundProcessGroup(
                signalArgs.CID.clone(),
                signalArgs.PID,
                signalArgs.Signo,
            ) {
                Err(_e) => {
                    info!("signal DeliverToForegroundProcessGroup fail with error");
                    //todo: enable the error when ready
                    //info!("signal DeliverToForegroundProcessGroup fail with error {:?}", e);
                }
                Ok(()) => (),
            }
        }
    };
}

pub fn ControlMsgHandler(fd: *const u8) {
    let fd = fd as i32;

    let task = Task::Current();
    let mut msg = ControlMsg::default();
    Kernel::HostSpace::ReadControlMsg(fd, &mut msg as *mut _ as u64);

    //info!("payload: {:?}", &msg.payload);
    //defer!(error!("payload handling ends"));
    match msg.payload {
        Payload::Pause => {
            let kernel = LOADER.Lock(task).unwrap().kernel.clone();
            kernel.Pause();
            WriteControlMsgResp(fd, &UCallResp::PauseResp, true);
        }
        Payload::Unpause => {
            let kernel = LOADER.Lock(task).unwrap().kernel.clone();
            kernel.Unpause();
            WriteControlMsgResp(fd, &UCallResp::UnpauseResp, true);
        }
        Payload::Ps(cid) => {
            let kernel = LOADER.Lock(task).unwrap().kernel.clone();
            let ps = Processes(&kernel, &cid);
            WriteControlMsgResp(fd, &UCallResp::PsResp(ps), true);
        }
        Payload::Signal(signalArgs) => {
            HandleSignal(&signalArgs);
            WriteControlMsgResp(fd, &UCallResp::SignalResp, true);
        }
        Payload::ContainerDestroy(cid) => {
            LOADER.Lock(task).unwrap().DestroyContainer(cid).unwrap();
            WriteControlMsgResp(fd, &UCallResp::ContainerDestroyResp, true);
        }
        Payload::RootContainerStart(_) => {
            WriteControlMsgResp(fd, &UCallResp::RootContainerStartResp, true);
            StartRootContainer(ptr::null());
        }
        Payload::ExecProcess(process) => {
            StartExecProcess(fd, process);
        }
        Payload::WaitContainer(cid) => match LOADER.WaitContainer(cid) {
            Ok(exitStatus) => {
                WriteControlMsgResp(fd, &UCallResp::WaitContainerResp(exitStatus), true);
            }
            Err(e) => {
                WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)), true);
            }
        },
        Payload::WaitPid(waitpid) => {
            match LOADER.WaitPID(waitpid.cid, waitpid.pid, waitpid.clearStatus) {
                Ok(exitStatus) => {
                    WriteControlMsgResp(fd, &UCallResp::WaitPidResp(exitStatus), true);
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)), true);
                }
            }
        }
        Payload::CreateSubContainer(createArgs) => {
            match LOADER.CreateSubContainer(createArgs.cid, createArgs.fds) {
                Ok(()) => {
                    WriteControlMsgResp(fd, &UCallResp::CreateSubContainerResp, true);
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)), true);
                }
            }
        }
        Payload::StartSubContainer(startArgs) => {
            match LOADER.StartSubContainer(startArgs.process) {
                Ok((_, entry, userStackAddr, kernelStackAddr)) => {
                    WriteControlMsgResp(fd, &UCallResp::StartSubContainerResp, true);
                    StartSubContainerProcess(entry, userStackAddr, kernelStackAddr);
                }
                Err(e) => {
                    WriteControlMsgResp(fd, &UCallResp::UCallRespErr(format!("{:?}", e)), true);
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
    WriteControlMsgResp(
        fd,
        &UCallResp::WaitAllResp(WaitAllResp {
            cid,
            execId,
            status,
        }),
        false,
    );
}

pub fn WriteControlMsgResp(fd: i32, msg: &UCallResp, close: bool) {
    let data: Vec<u8> = serde_json::to_vec(&msg).expect("LoadProcessKernel ser fail...");
    let addr = &data[0] as *const _ as u64;
    let len = data.len();

    Kernel::HostSpace::WriteControlMsgResp(fd, addr, len, close);
}
