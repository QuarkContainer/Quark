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

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::task::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::task_syscall::*;
use super::thread_group::*;

impl ThreadInternal {
    fn doStop(&self) {
        let task = Task::Current();
        self.blocker.WaitGroupWait(task, &self.stopCount);
    }
}

impl Task {
    pub fn RunApp(&mut self) -> TaskRunState {
        let t = self.Thread();

        //if the task has been interrupted
        if self.blocker.Interrupted() {
            // Checkpointing instructs tasks to stop by sending an interrupt, so we
            // must check for stops before entering runInterrupt (instead of
            // tail-calling it).
            return TaskRunState::RunInterrupt;
        }

        // We're about to switch to the application again. If there's still a
        // unhandled SyscallRestartErrno that wasn't translated to an EINTR,
        // restart the syscall that was interrupted. If there's a saved signal
        // mask, restore it. (Note that restoring the saved signal mask may unblock
        // a pending signal, causing another interruption, but that signal should
        // not interact with the interrupted syscall.)
        if self.haveSyscallReturn {
            let ret = self.Return();
            let (sre, ok) = SyscallRestartErrnoFromReturn(ret);
            if ok {
                if sre == SysErr::ERESTART_RESTARTBLOCK {
                    self.RestartSyscallWithRestartBlock();
                } else {
                    self.RestartSyscall();
                }
            }

            self.haveSyscallReturn = false;
        }

        let haveSavedSignalMask = t.lock().haveSavedSignalMask;
        if haveSavedSignalMask {
            let savedSignalMask = t.lock().savedSignalMask;
            t.SetSignalMask(savedSignalMask);
            if self.blocker.Interrupted() {
                return TaskRunState::RunInterrupt;
            }
        }

        return TaskRunState::RunSyscallRet;
    }
}

impl ThreadGroup {
    // WaitExited blocks until all task in tg have exited.
    pub fn WaitExited(&self, task: &Task) {
        let wg = self.lock().liveThreads.clone();
        task.blocker.WaitGroupWait(task, &wg);
    }
}