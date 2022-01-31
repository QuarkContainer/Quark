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

use rand::prelude::*;
use rand_seeder::Seeder;
use rand_pcg::Pcg64;
use libc::*;
use std::slice;

use super::super::qlib::auxv::*;

pub struct RandGen {
    rng : Pcg64,
}

impl RandGen {
    pub fn Init() -> Self {
        let fakeRandom = false;

        if !fakeRandom {
            //use auxv AT_RANDOM as seed
            let auxvRandAddr = unsafe {
                getauxval(AuxVec::AT_RANDOM as u64)
            };

            let slice = unsafe {
                slice::from_raw_parts(auxvRandAddr as *mut u8, 16)
            };

            return RandGen {
                rng : Seeder::from(slice).make_rng(),
            }
        } else {
            error!("use fake random");
            let slice : [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

            return RandGen {
                rng : Seeder::from(slice).make_rng(),
            }
        }
    }

    pub fn Fill(&mut self, data: &mut [u8]) {
        self.rng.fill_bytes(data)
    }

    pub fn GetU64(&mut self) -> u64 {
        let mut res : u64 = 0;
        let ptr = &mut res as * mut u64 as * mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, 8) };
        self.Fill(slice);
        return res;
    }
}