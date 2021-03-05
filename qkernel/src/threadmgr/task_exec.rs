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

use super::task_stop::*;
use super::super::threadmgr::thread::*;

pub struct ExecStop {}

impl TaskStop for ExecStop {
    fn Type(&self) -> TaskStopType {
        return TaskStopType::EXECSTOP;
    }

    fn Killable(&self) -> bool {
        return true
    }
}

impl Thread {
    // promoteLocked makes t the leader of its thread group. If t is already the
    // thread group leader, promoteLocked is a no-op.
    //
    // Preconditions: All other tasks in t's thread group, including the existing
    // leader (if it is not t), have reached TaskExitZombie. The TaskSet mutex must
    // be locked for writing.
    pub fn promoteLocked(&self) {
        let t = self.clone();
        let tg = t.lock().tg.clone();
        let oldLeader = tg.lock().leader.Upgrade().unwrap();
        if t == oldLeader {
            return
        }

        // Swap the leader's TIDs with the execing task's. The latter will be
        // released when the old leader is reaped below.
        let mut pidns = tg.PIDNamespace();
        loop {
            let oldTID = *pidns.lock().tids.get(&t).unwrap();
            let leaderTID = *pidns.lock().tids.get(&oldLeader).unwrap();
            pidns.lock().tids.insert(oldLeader.clone(), oldTID);
            pidns.lock().tids.insert(t.clone(), leaderTID);
            pidns.lock().tasks.insert(oldTID, oldLeader.clone());
            pidns.lock().tasks.insert(leaderTID, t.clone());
            // Neither the ThreadGroup nor TGID change, so no need to
            // update ns.tgids.

            let temp = pidns.lock().parent.clone();
            if temp.is_none() {
                break;
            }

            pidns = temp.unwrap();
        }


        // Inherit the old leader's start time.
        let oldStartTime = oldLeader.StartTime();
        t.lock().startTime = oldStartTime;

        tg.lock().leader = t.Downgrade();

        // Reap the original leader. If it has a tracer, detach it instead of
        // waiting for it to acknowledge the original leader's death.
        oldLeader.lock().exitParentNotified = true;
        oldLeader.lock().exitParentAcked = true;
        oldLeader.exitNotifyLocked();
    }
}