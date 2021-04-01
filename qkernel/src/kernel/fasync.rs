// Copyright (c) 2021 Quark Container Authors
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

use alloc::sync::Arc;
use spin::Mutex;
use core::ops::Deref;

use super::super::qlib::auth::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::linux::signal::*;
use super::super::task::*;
use super::super::SignalDef::*;
use super::super::threadmgr::processgroup::*;
use super::super::threadmgr::thread_group::*;
use super::super::threadmgr::thread::*;
use super::waiter::entry::*;
use super::waiter::*;

// FileAsync sends signals when the registered file is ready for IO.
#[derive(Default)]
pub struct FileAsyncInternal {
    pub e: WaitEntry,
    pub requester: Credentials,

    // Only one of the following is allowed to be non-nil.
    pub recipientPG: Option<ProcessGroup>,
    pub recipientTG: Option<ThreadGroup>,
    pub recipientT: Option<Thread>,
}

#[derive(Clone, Default)]
pub struct FileAsync(Arc<Mutex<FileAsyncInternal>>);

impl Deref for FileAsync {
    type Target = Arc<Mutex<FileAsyncInternal>>;

    fn deref(&self) -> &Arc<Mutex<FileAsyncInternal>> {
        &self.0
    }
}

impl FileAsync {
    pub fn Callback(&self) {
        let a = self.lock();

        /*match a.e.lock().context {
            WaitContext::None => return,
            _ =>(),
        }*/

        let mut t = a.recipientT.clone();
        let mut tg = a.recipientTG.clone();

        if a.recipientPG.is_some() {
            tg = Some(a.recipientPG.as_ref().unwrap().Originator());
        }

        if tg.is_some() {
            t = tg.unwrap().Leader();
        }

        if t.is_none() {
            // No recipient has been registered.
            return
        }

        let t = t.unwrap();

        let c = t.Credentials();

        let threadC = c.lock();
        let reqC = a.requester.lock();
        // Logic from sigio_perm in fs/fcntl.c.
        if reqC.EffectiveKUID.0 == 0 ||
            reqC.EffectiveKUID == threadC.SavedKUID ||
            reqC.EffectiveKUID == threadC.RealKUID ||
            reqC.RealKUID == threadC.SavedKUID ||
            reqC.RealKUID == threadC.RealKUID {
            t.SendSignal(&SignalInfoPriv(SIGIO.0)).unwrap();
        }
    }

    // Register sets the file which will be monitored for IO events.
    //
    // The file must not be currently registered.
    pub fn Register(&self, task: &Task, w: &Waitable) {
        let a = self.lock();

        match a.e.lock().context {
            WaitContext::None => (),
            _ => panic!("registering already registered file"),
        }

        a.e.lock().context = WaitContext::FileAsync(self.clone());
        w.EventRegister(task, &a.e, EVENT_IN | EVENT_OUT | EVENT_ERR | EVENT_HUP);
    }

    // Unregister stops monitoring a file.
    //
    // The file must be currently registered.
    pub fn Unregister(&self, task: &Task, w: &Waitable) {
        let a = self.lock();

        match a.e.lock().context {
            WaitContext::None => panic!("unregistering unregistered file"),
            _ =>(),
        }

        w.EventUnregister(task, &a.e);
        a.e.lock().context = WaitContext::None;
    }

    // Owner returns who is currently getting signals. All return values will be
    // nil if no one is set to receive signals.
    pub fn Owner(&self) -> (Option<Thread>, Option<ThreadGroup>, Option<ProcessGroup>) {
        let a = self.lock();
        return (a.recipientT.clone(), a.recipientTG.clone(), a.recipientPG.clone())
    }

    // SetOwnerTask sets the owner (who will receive signals) to a specified task.
    // Only this owner will receive signals.
    pub fn SetOwnerTask(&self, requester: &Task, recipient: Option<Thread>) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        a.recipientT = recipient;
        a.recipientTG = None;
        a.recipientPG = None;
    }

    // SetOwnerThreadGroup sets the owner (who will receive signals) to a specified
    // thread group. Only this owner will receive signals.
    pub fn SetOwnerThreadGroup(&self, requester: &Task, recipient: Option<ThreadGroup>) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        a.recipientT = None;
        a.recipientTG = recipient;
        a.recipientPG = None;
    }

    // SetOwnerProcessGroup sets the owner (who will receive signals) to a
    // specified process group. Only this owner will receive signals.
    pub fn SetOwnerProcessGroup(&self, requester: &Task, recipient: Option<ProcessGroup>) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        a.recipientT = None;
        a.recipientTG = None;
        a.recipientPG = recipient;
    }

    pub fn Unset(&self, requester: &Task) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        a.recipientT = None;
        a.recipientTG = None;
        a.recipientPG = None;
    }
}