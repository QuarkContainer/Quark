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

use serde_json;
use alloc::slice;

use super::super::qlib::common::*;
use super::super::qlib::linux_def::*;
use super::super::qlib::ucall::*;
use super::super::task::*;
use super::super::fs::file::*;

pub struct UcallServer {
    pub file: File,
    pub fd : i32,
}

impl Drop for UcallServer {
    fn drop(&mut self) {
        let task = Task::Current();
        let file = task.RemoveFile(self.fd).unwrap();

        file.Flush(task).unwrap();
    }
}

impl UcallServer {
    pub fn New(fd: i32) -> Result<Self> {
        let file = Task::Current().GetFile(fd)?;
        return Ok(Self {
            file: file,
            fd: fd,
        })
    }

    pub fn ReadAll(&self, buf: &mut [u8]) -> Result<()> {
        let mut len = buf.len();
        while len > 0 {
            let iov = IoVec::NewFromAddr(&buf[buf.len() - len] as * const _ as u64, len);
            let mut iovs: [IoVec; 1] = [iov];

            let cnt = self.file.Readv(Task::Current(), &mut iovs)?;
            len -= cnt as usize;
        }

        return Ok(())
    }

    pub fn ReadLen(&self) -> Result<usize> {
        let mut len : u32 = 0;
        let ptr = &mut len as * mut _ as * mut u8;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, 4) };
        self.ReadAll(slice)?;
        return Ok(len as usize);
    }

    pub fn WriteAll(&self, buf: &[u8]) -> Result<()> {
        let mut len = buf.len();
        while len > 0 {
            let iov = IoVec::NewFromAddr(&buf[buf.len() - len] as * const _ as u64, len);
            let iovs: [IoVec; 1] = [iov];

            let cnt = self.file.Writev(Task::Current(), &iovs)?;
            len -= cnt as usize;
        }

        return Ok(())
    }

    pub fn WriteLen(&self, len: usize) -> Result<()> {
        let len = len as u32;
        let ptr = &len as * const _ as * const u8;
        let slice = unsafe { slice::from_raw_parts(ptr, 4) };
        return self.WriteAll(slice);
    }

    pub fn Process(&self) -> Result<()> {
        let len = self.ReadLen()?;
        let mut buf : [u8; UCALL_BUF_LEN] = [0; UCALL_BUF_LEN];

        assert!(len < UCALL_BUF_LEN, "UCallClient::Call resp is too long");
        self.ReadAll(&mut buf[0..len])?;
        let req : UCallReq = serde_json::from_slice(&buf[0..len]).map_err(|e|Error::Common(format!("UcallServer deser error is {:?}", e)))?;
        info!("get request {:?}", req);

        let resp = match req {
            UCallReq::RootContainerStart(req) => {
                super::super::StartRootProcess(&req.cid);
                UCallResp::RootContainerStartResp
            }
            UCallReq::ContainerStart(_req) => {

                UCallResp::ContainerStartResp
            }
        };

        let respVec = serde_json::to_vec(&resp).map_err(|e|Error::Common(format!("UCallClient ser error is {:?}", e)))?;
        self.WriteLen(respVec.len())?;
        self.WriteAll(&respVec)?;

        return Ok(())
    }
}