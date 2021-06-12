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
use spin::Mutex;
use alloc::vec::Vec;
use core::{ptr};

use super::super::qlib::common::*;
use super::super::qlib::control_msg::*;
use super::super::Kernel;
use super::super::taskMgr;
use super::super::task::*;
use super::super::{StartRootContainer, StartExecProcess};
use super::super::{LOADER, KERNEL_STACK_ALLOCATOR};
use super::process::*;

lazy_static! {
    pub static ref MSG: Mutex<Option<ControlMsg>> = Mutex::new(None);
}

pub fn Run() -> Result<()> {
    let task = Task::Current();
    loop {
        let msg = ControlMsgCall()?;

        match msg.payload {
            Payload::Pause => {
                let kernel = LOADER.Lock(task).unwrap().kernel.clone();
                kernel.Pause();
                ControlMsgRet(msg.msgId, &UCallResp::PauseResp);
                continue;
            }
            Payload::Unpause => {
                let kernel = LOADER.Lock(task).unwrap().kernel.clone();
                kernel.Unpause();
                ControlMsgRet(msg.msgId, &UCallResp::UnpauseResp);
                continue;
            }
            Payload::Ps(cid) => {
                let kernel = LOADER.Lock(task).unwrap().kernel.clone();
                let ps = Processes(&kernel, &cid);
                ControlMsgRet(msg.msgId, &UCallResp::PsResp(ps));
                continue;
            }
            Payload::Signal(signalArgs) => {
                info!("get signal {:?}", &signalArgs);
                match signalArgs.Mode {
                    SignalDeliveryMode::DeliverToProcess => {
                        match LOADER.Lock(task).unwrap().SignalProcess(signalArgs.PID, signalArgs.Signo) {
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
                        match LOADER.Lock(task).unwrap().SignalForegroundProcessGroup(signalArgs.PID, signalArgs.Signo) {
                            Err(_e) => {
                                info!("signal DeliverToForegroundProcessGroup fail with error");
                                //todo: enable the error when ready
                                //info!("signal DeliverToForegroundProcessGroup fail with error {:?}", e);
                            }
                            Ok(())=> ()
                        }
                    }
                };

                ControlMsgRet(msg.msgId, &UCallResp::SignalResp);
                continue;
            }
            Payload::ContainerDestroy => {
                LOADER.Lock(task).unwrap().DestroyContainer()?;
                ControlMsgRet(msg.msgId, &UCallResp::ContainerDestroyResp);
                continue;
            }
            _ => ()
        }

        while MSG.lock().is_some() {
            taskMgr::Yield();
        }

        *MSG.lock() = Some(msg);

        taskMgr::CreateTask(ControlMsgHandler, ptr::null());
    }
}

pub fn ControlMsgHandler(_para: *const u8) {
    let msg = MSG.lock().take().expect("ControlMsgHandler: get none msg");

    match msg.payload {
        Payload::RootContainerStart(_) => {
            ControlMsgRet(msg.msgId, &UCallResp::RootContainerStartResp);
            StartRootContainer(ptr::null());
        }
        Payload::ExecProcess(process) => {
            StartExecProcess(msg.msgId, process);
        }
        Payload::WaitContainer => {
            match LOADER.WaitContainer() {
                Ok(exitStatus) => {
                    ControlMsgRet(msg.msgId, &UCallResp::WaitContainerResp(exitStatus));
                }
                Err(e) => {
                    ControlMsgRet(msg.msgId, &UCallResp::UCallRespErr(format!("{:?}", e)));
                }
            }
        }
        Payload::WaitPid(waitpid) => {
            match LOADER.WaitPID(waitpid.pid, waitpid.clearStatus) {
                Ok(exitStatus) => {
                    ControlMsgRet(msg.msgId, &UCallResp::WaitPidResp(exitStatus));
                }
                Err(e) => {
                    ControlMsgRet(msg.msgId, &UCallResp::UCallRespErr(format!("{:?}", e)));
                }
            }
        }
        _ => {
            panic!("ControlMsgHandler unsupported message {:?}", msg);
        }
    }


    let taskId = Task::Current().taskId;
    KERNEL_STACK_ALLOCATOR.Free(taskId).unwrap();
    //(*PAGE_ALLOCATOR).Free(taskId, DEFAULT_STACK_PAGES).unwrap();
    taskMgr::Wait();
}

pub fn ControlMsgCall() -> Result<ControlMsg> {
    let msg = {
        let mut buf: [u8; 8192] = [0; 8192];
        let addr = &mut buf[0] as * mut _ as u64;
        let size = Kernel::HostSpace::ControlMsgCall(addr, buf.len()) as usize;

        let msg : ControlMsg = serde_json::from_slice(&buf[0..size]).expect(&format!("LoadProcessKernel des fail size is {}", size));
        msg
    };

    return Ok(msg)
}

pub fn ControlMsgRet(msgId: u64, msg: &UCallResp) {
    let data : Vec<u8> = serde_json::to_vec(&msg).expect("LoadProcessKernel ser fail...");
    let addr = &data[0] as * const _ as u64;
    let len = data.len();

    Kernel::HostSpace::ControlMsgRet(msgId, addr, len);
}