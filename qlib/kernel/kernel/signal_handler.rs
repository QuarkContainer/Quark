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

use alloc::collections::btree_map::BTreeMap;
use alloc::sync::Arc;
use ::qlib::mutex::*;
use core::ops::Deref;

use super::super::SignalDef::*;
use super::super::super::linux_def::*;

#[derive(Clone, Debug)]
pub struct SignalHandlersInternal {
    pub actions: BTreeMap<i32, SigAct>
}

impl Default for SignalHandlersInternal {
    fn default() -> Self {
        return Self {
            actions: BTreeMap::new(),
        }
    }
}

impl SignalHandlersInternal {
    pub fn GetAct(&mut self, sig: Signal) -> SigAct {
        match self.actions.get(&sig.0) {
            None => SigAct::default(),
            Some(act) => {
                *act
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SignalHandlers(Arc<QMutex<SignalHandlersInternal>>);

impl Deref for SignalHandlers {
    type Target = Arc<QMutex<SignalHandlersInternal>>;

    fn deref(&self) -> &Arc<QMutex<SignalHandlersInternal>> {
        &self.0
    }
}

impl SignalHandlers {
    pub fn Fork(&self) -> Self {
        let me = self.lock();
        let mut sh = SignalHandlersInternal::default();

        for (i, act) in &me.actions {
            sh.actions.insert(*i, *act);
        }

        return SignalHandlers(Arc::new(QMutex::new(sh)))
    }

    pub fn CopyForExec(&self) -> Self {
        let me = self.lock();
        let mut sh = SignalHandlersInternal::default();

        for (i, act) in &me.actions {
            if act.handler == SigAct::SIGNAL_ACT_IGNORE {
                sh.actions.insert(*i, SigAct {
                    handler: SigAct::SIGNAL_ACT_IGNORE,
                    ..Default::default()
                });
            }
        }

        return SignalHandlers(Arc::new(QMutex::new(sh)))
    }

    pub fn IsIgored(&self, sig: Signal) -> bool {
        match self.lock().actions.get(&sig.0) {
            None => false,
            Some(act) => {
                return act.handler == SigAct::SIGNAL_ACT_IGNORE;
            }
        }
    }

    pub fn DequeAct(&self, sig: Signal) -> SigAct {
        let mut me = self.lock();

        let act = match me.actions.get(&sig.0) {
            None => SigAct::default(),
            Some(act) => {
                *act
            }
        };

        if act.flags.IsResetHandler() {
            me.actions.remove(&sig.0);
        }

        return act;
    }

    pub fn SigAction(&mut self, sigNum: u64, action: u64, oldAction: u64) -> i64 {
        info!("SigAction sigNum is {} ********", sigNum);
        let mut me = self.lock();
        let sigNum = sigNum as i32;

        if oldAction != 0 {
            match me.actions.get(&sigNum) {
                None => unsafe {
                    *(oldAction as *mut SigAct) = SigAct::default();
                }
                Some(old) => unsafe {
                    *(oldAction as *mut SigAct) = *old;
                }
            }
        }

        if action != 0 {
            unsafe {
                let sigAct = *(action as *const SigAct);
                me.actions.insert(sigNum, sigAct);
            }
        }

        return 0;
    }

    pub fn GetAct(&mut self, sig: Signal) -> SigAct {
        let ret = self.lock().GetAct(sig);
        return ret;
    }
}