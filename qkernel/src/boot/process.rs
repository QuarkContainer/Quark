// Copyright (c) 2021 QuarkSoft LLC
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

use super::super::qlib::control_msg::*;
use super::super::kernel::kernel::*;

pub fn Processes(k: &Kernel, containerID: &str) -> Vec<ProcessInfo> {
    let ts = k.TaskSet();
    let root = ts.Root();
    let tgs = root.ThreadGroups();

    let mut ret = Vec::new();

    for tg in tgs {
        let pid = root.IDOfThreadGroup(&tg);
        // If tg has already been reaped ignore it.
        if pid == 0 {
            continue;
        }

        let lead = tg.Leader().unwrap();

        if containerID.len() == 0 && containerID != &lead.ContainerID() {
            continue;
        }

        let mut ppid = 0;
        match lead.Parent() {
            None => (),
            Some(p) => {
                ppid = root.IDOfThreadGroup(&p.ThreadGroup())
            }
        }

        ret.push(ProcessInfo{
            UID:   lead.Credentials().lock().EffectiveKUID,
            PID:   pid,
            PPID:  ppid,
            STime: lead.StartTime().0,
            Utilization:     0,
            Time:  0,
            Cmd:   lead.Name(),
        })
    }

    return ret
}