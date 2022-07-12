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

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use core::ops::Deref;

use super::super::super::auth::*;
use super::super::super::linux::signal::*;
use super::super::super::common::*;
use super::super::super::linux_def::*;
use super::super::task::*;
use super::super::threadmgr::processgroup::*;
use super::super::threadmgr::thread::*;
use super::super::threadmgr::thread_group::*;
use super::super::SignalDef::*;
use super::waiter::entry::*;
use super::waiter::*;

lazy_static! {
    // Table to convert waiter event masks into si_band siginfo codes.
    // Taken from fs/fcntl.c:band_table.
    static ref BAND_TABLE : BTreeMap<EventMask, u64> = [
        (EVENT_IN,       LibcConst::EPOLLIN | LibcConst::EPOLLRDNORM),
        (EVENT_OUT,       LibcConst::EPOLLOUT | LibcConst::EPOLLWRNORM | LibcConst::EPOLLWRBAND),
        (EVENT_ERR,       LibcConst::EPOLLERR),
        (EVENT_PRI,       LibcConst::EPOLLPRI | LibcConst::EPOLLRDBAND),
        (EVENT_HUP,       LibcConst::EPOLLHUP | LibcConst::EPOLLERR),
    ].iter().cloned().collect();
}


#[derive(Clone)]
pub enum Recipient {
    None,
    PG(ProcessGroup),
    TG(ThreadGroup),
    Thread(Thread),
}

impl Default for Recipient {
    fn default() -> Self {
        return Recipient::None
    }
}

// FileAsync sends signals when the registered file is ready for IO.
#[derive(Default)]
pub struct FileAsyncInternal {
    pub e: WaitEntry,

    // fd is the file descriptor to notify about.
    // It is immutable, set at allocation time. This matches Linux semantics in
    // fs/fcntl.c:fasync_helper.
    // The fd value is passed to the signal recipient in siginfo.si_fd.
    pub fd: i32,

    pub requester: Credentials,

    pub registed: bool,
    // signal is the signal to deliver upon I/O being available.
    // The default value ("zero signal") means the default SIGIO signal will be
    // delivered.
    pub signal: Signal,

    pub recipient: Recipient,
    /*pub recipientPG: Option<ProcessGroup>,
    pub recipientTG: Option<ThreadGroup>,
    pub recipientT: Option<Thread>,*/
}

#[derive(Clone, Default)]
pub struct FileAsync(Arc<QMutex<FileAsyncInternal>>);

impl Deref for FileAsync {
    type Target = Arc<QMutex<FileAsyncInternal>>;

    fn deref(&self) -> &Arc<QMutex<FileAsyncInternal>> {
        &self.0
    }
}

impl FileAsync {
    pub fn New(fd: i32) -> Self {
        let intern = FileAsyncInternal {
            fd: fd,
            ..Default::default()
        };

        return Self(Arc::new(QMutex::new(intern)))
    }

    pub fn Callback(&self, mask: EventMask) {
        let a = self.lock();

        /*match a.e.lock().context {
            WaitContext::None => return,
            _ =>(),
        }*/

        let mut tg = None;
        let mut t = None;
        match &a.recipient {
            Recipient::PG(ref pg) => {
                tg = Some(pg.Originator());
            }
            Recipient::TG(threadgroup) => {
                tg = Some(threadgroup.clone());
            }
            Recipient::Thread(thread) => {
                t = Some(thread.clone());
            }
            Recipient::None => (),
        }

        if tg.is_some() {
            t = tg.unwrap().Leader();
        }

        if t.is_none() {
            // No recipient has been registered.
            return;
        }

        let t = t.unwrap();

        let c = t.Credentials();

        let threadC = c.lock();
        let reqC = a.requester.lock();

        // Logic from sigio_perm in fs/fcntl.c.
        let permCheck = reqC.EffectiveKUID.0 == 0
            || reqC.EffectiveKUID == threadC.SavedKUID
            || reqC.EffectiveKUID == threadC.RealKUID
            || reqC.RealKUID == threadC.SavedKUID
            || reqC.RealKUID == threadC.RealKUID;

        if !permCheck {
            return
        }

        let signalInfo = if a.signal.0 != 0 {
            let signalInfo = SignalInfoPriv(a.signal.0);
            let sigPoll = signalInfo.SigPoll();
            sigPoll.fd = a.fd;
            let mut band = 0;
            for (m, bandcode) in BAND_TABLE.iter() {
                if *m & mask != 0 {
                    band |= *bandcode;
                }
            }
            sigPoll.band = band as _;
            signalInfo
        } else {
            SignalInfoPriv(SIGIO.0)
        };

        t.SendSignal(&signalInfo).unwrap();
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
        w.EventRegister(task, &a.e, READABLE_EVENT | WRITEABLE_EVENT | EVENT_ERR | EVENT_HUP);
    }

    // Unregister stops monitoring a file.
    //
    // The file must be currently registered.
    pub fn Unregister(&self, task: &Task, w: &Waitable) {
        let a = self.lock();

        match a.e.lock().context {
            WaitContext::None => panic!("unregistering unregistered file"),
            _ => (),
        }

        w.EventUnregister(task, &a.e);
        a.e.lock().context = WaitContext::None;
    }

    // Owner returns who is currently getting signals. All return values will be
    // nil if no one is set to receive signals.
    pub fn Owner(&self) -> Recipient {
        let a = self.lock();
        return a.recipient.clone();
    }

    // SetOwnerTask sets the owner (who will receive signals) to a specified task.
    // Only this owner will receive signals.
    pub fn SetOwnerTask(&self, requester: &Task, recipient: Option<Thread>) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        match recipient {
            None => a.recipient = Recipient::None,
            Some(thread) => a.recipient = Recipient::Thread(thread)
        }
    }

    // SetOwnerThreadGroup sets the owner (who will receive signals) to a specified
    // thread group. Only this owner will receive signals.
    pub fn SetOwnerThreadGroup(&self, requester: &Task, recipient: Option<ThreadGroup>) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        match recipient {
            None => a.recipient = Recipient::None,
            Some(tg) => a.recipient = Recipient::TG(tg)
        }
    }

    // SetOwnerProcessGroup sets the owner (who will receive signals) to a
    // specified process group. Only this owner will receive signals.
    pub fn SetOwnerProcessGroup(&self, requester: &Task, recipient: Option<ProcessGroup>) {
        let mut a = self.lock();

        a.requester = requester.Creds();

        match recipient {
            None => a.recipient = Recipient::None,
            Some(pg) => a.recipient = Recipient::PG(pg)
        }
    }

    pub fn Unset(&self, requester: &Task) {
        let mut a = self.lock();

        a.requester = requester.Creds();
        a.recipient = Recipient::None;
    }

    // Signal returns which signal will be sent to the signal recipient.
    // A value of zero means the signal to deliver wasn't customized, which means
    // the default signal (SIGIO) will be delivered.
    pub fn Signal(&self) -> Signal {
        return self.lock().signal
    }

    // SetSignal overrides which signal to send when I/O is available.
    // The default behavior can be reset by specifying signal zero, which means
    // to send SIGIO.
    pub fn SetSignal(&self, signal: i32) -> Result<()> {
        if signal != 0 && !Signal(signal).IsValid() {
            return Err(Error::SysError(SysErr::EINVAL));
        }

        self.lock().signal = Signal(signal);
        return Ok(())
    }
}
