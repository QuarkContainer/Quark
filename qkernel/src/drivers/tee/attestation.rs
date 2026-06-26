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

use alloc::{boxed::Box, string::String, vec::Vec};
use lazy_static::lazy_static;
use core::cell::SyncUnsafeCell;
use spin::{Mutex, lazy::Lazy};
use log::*;
use crate::{qlib::{common::{Error, Result}, kernel::arch::tee::get_tee_type, linux_def::SysErr}, CCMode};

#[cfg(all(target_arch = "x86_64", feature = "snp"))]
use super::sev::attestation::SevAttestation;

pub type Challenge = Vec<u8>;
pub type Report = Vec<u8>;
pub type Response = String;

lazy_static! {
    pub static ref ATTESTATION_DRIVER: Mutex::<&'static mut AttestationDriver> = {
        static DRIVER: Lazy<SyncUnsafeCell<AttestationDriver>> =
            Lazy::new(lazy_cell_new::<AttestationDriver>);
        let _d = unsafe {
            &mut (*DRIVER.get())
        };

        _d.init();

        Mutex::<&mut AttestationDriver>::new(_d)
    };
}

fn lazy_cell_new<T>() -> SyncUnsafeCell<AttestationDriver>
    where T: core::default::Default {
    SyncUnsafeCell::new(AttestationDriver::default())
}

pub trait AttestationDriverT {
    fn init(&mut self) {}
    fn get_report(&mut self, _challenge: &Challenge) -> Result<Report> {
        todo!("Trait not implemented")
    }
    fn valid_challenge(&self, _challenge: &mut Challenge) -> bool {
        todo!("Trait not implemented")
    }
}

struct NoopAttestation;

impl AttestationDriverT for NoopAttestation {}

pub struct AttestationDriver {
    tee_attester: Box<dyn AttestationDriverT>,
    tee_type: CCMode,
}

impl Default for AttestationDriver {
    fn default() -> Self {
        let tee_type = get_tee_type();
        let tee_attester: Box<dyn AttestationDriverT> = match tee_type {
            #[cfg(all(target_arch = "x86_64", feature = "snp"))]
            CCMode::SevSnp => Box::new(SevAttestation::default()),
            _ => Box::new(NoopAttestation),
        };

        Self {
            tee_attester,
            tee_type,
        }
    }
}

unsafe impl Sync for AttestationDriver {}
unsafe impl Send for AttestationDriver {}

impl AttestationDriver {
    fn init(&mut self) {
        self.tee_attester.init();
    }

    pub fn get_report(&mut self, challenge: &mut Challenge) -> Result<Report> {
        debug!("VM: Cca-AtD - get report with challenge:{:?}", challenge);
        if self.challenge_valid_size(challenge) {
            self.tee_attester.get_report(challenge)
        } else {
            Err(Error::SysError(SysErr::EINVAL))
        }
    }

    pub fn challenge_valid_size(&self, challenge: &mut Challenge) -> bool {
        let len = challenge.len();
        let (min, max) = self.challenge_range();
        if len < min || len > max {
            return false;
        }
        if len < max {
            debug!("VM: padd challenge to valid length:{:?}->{:?}",len, max);
            challenge.resize(max, 0);
        }
        true
    }

    fn challenge_range(&self) -> (usize, usize) {
        match self.tee_type {
            #[cfg(all(target_arch = "x86_64", feature = "snp"))]
            CCMode::SevSnp => (0usize, 64usize),
            _ => (0, 0),
        }
    }
}
