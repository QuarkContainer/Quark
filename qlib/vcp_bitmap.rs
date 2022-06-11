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

use core::ops::Deref;
use spin::Mutex;

use super::common::*;
use super::linux_def::*;

pub struct VcpuBitmapIntern {
    pub bitmap: u64,
    pub waitingVcpu: i32,
}

pub struct VcpuBitmap {
    pub intern: Mutex<VcpuBitmapIntern>,
}

impl Deref for VcpuBitmap {
    type Target = Mutex<VcpuBitmapIntern>;

    fn deref(&self) -> &Mutex<VcpuBitmapIntern> {
        &self.intern
    }
}

impl Default for VcpuBitmap {
    fn default() -> Self {
        let intern = VcpuBitmapIntern {
            bitmap: 0,
            waitingVcpu: -1,
        };

        return Self {
            intern: Mutex::new(intern)
        }
    }
}

impl VcpuBitmap {
    // ******* the wake will be called in signal handler, NO heap memory allocation is allowed *******
    //ret: whether the target is waked
    pub fn Wakeup(&self, vcpId: usize) -> bool {
        assert!(vcpId < 64);
        {
            let mut intern = self.lock();
            if intern.bitmap == 0 {
                return false
            }

            intern.bitmap &= !(1<<vcpId);
            if intern.bitmap != 0 {
                return false
            } else {
                intern.waitingVcpu = -1;
            }
        }

        self.FutexWake();
        return true;
    }

    pub fn Bitmap(&self) -> u64 {
        return self.lock().bitmap;
    }

    pub fn InitWait(&self, vcpuId: i32, bitmap: u64) {
        assert!(vcpuId < 64);
        let mut intern = self.lock();

        assert!(intern.bitmap == 0 && intern.waitingVcpu == -1);
        intern.bitmap = bitmap;
        intern.waitingVcpu = vcpuId;
    }

    pub fn Wait(&self, vcpuId: i32, timeout: i32) -> Result<()> {
        assert!(vcpuId < 64);

        let ret = self.FutexWait(vcpuId, timeout);
        if ret == 0 || ret == -SysErr::EAGAIN as i64 {
            return Ok(())
        }

        return Err(Error::SysError(-ret as i32))
    }

    pub fn Clear(&self) {
        let mut intern = self.lock();
        intern.bitmap = 0;
        intern.waitingVcpu = -1;
    }

    pub fn FutexAddr(&self) -> u64 {
        return &self.lock().waitingVcpu as * const _ as u64
    }
}
