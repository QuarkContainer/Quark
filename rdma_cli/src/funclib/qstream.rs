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

use std::sync::Arc;
use std::io::Read;
use core::pin::Pin;
use core::task::Poll;
use core::task::Context;
use core::ops::Deref;
use alloc::collections::vec_deque::VecDeque;
use spin::Mutex;

use tokio::sync::Notify;
//use tokio_io::{AsyncRead, AsyncWrite};
use tokio::io::AsyncRead;
use tokio::io::ReadBuf;
use tokio::sync::Mutex as Lock;
//use bytes::Buf;

use crate::common::*;
use crate::RDMASvcClient;

use crate::funclib::uid::*;
use crate::qlib::common::*;
//use crate::qlib::linux_def::*;

#[derive(Clone)]
pub struct QStream(Arc<QStreamInner>);

impl Deref for QStream {
    type Target = Arc<QStreamInner>;

    fn deref(&self) -> &Arc<QStreamInner> {
        &self.0
    }
}

pub struct QStreamInner {
    pub dataSock: DataSock,
    pub canReadNotify: Arc<Notify>, // underlying socket has more data to read
    pub canWriteNotify: Arc<Notify>, // underlying socket has more space to write
    pub readLock: Lock<()>,
    pub writeLock: Lock<()>,
    pub rdmaCli: RDMASvcClient,
    pub outputBufs: Mutex<VecDeque<Vec<u8>>>,
}

impl QStreamInner {
    pub fn New(rdmaCli: RDMASvcClient, dataSock: DataSock) -> Self {
        return Self {
            dataSock: dataSock,
            canReadNotify: Arc::new(Notify::new()),
            canWriteNotify: Arc::new(Notify::new()),
            readLock: Lock::new(()),
            writeLock: Lock::new(()),
            rdmaCli: rdmaCli,
            outputBufs: Mutex::new(VecDeque::new()),
        }
    }

    pub async fn WriteBuf(&self, buf: &[u8]) -> Result<usize> {
        let _l = self.writeLock.lock().await;
        let count;
        loop {
            let (_trigger, cnt) = self.dataSock.sockBuff.writeBuf.lock().write(buf)?;

            if cnt == 0 {
                self.canWriteNotify.clone().notified().await;
            } else {
                count = cnt;
                break;
            }
        };

        let trigger = self.dataSock.sockBuff.writeBuf.lock().Produce(count);
        if trigger {
            let _ret = self.rdmaCli.write(*self.dataSock.channelId.lock());
        }

        return Ok(count);
    }

    pub async fn WriteAll(&self, buf: &[u8]) -> Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            let count = self.WriteBuf(&buf[offset..]).await?;
            offset += count;
        }

        return Ok(())
    }

    pub async fn ReadBuf(&self, buf: &mut [u8]) -> Result<usize> {
        let _l = self.readLock.lock().await;
        let count;
        loop {
            let (_trigger, cnt) = self.dataSock.sockBuff.readBuf.lock().read(buf)?;

            if cnt == 0 {
                self.canReadNotify.clone().notified().await;
            } else {
                count = cnt;
                break;
            }
        };
               
        let dataSize = self.dataSock.sockBuff.AddConsumeReadData(count as u64) as usize;
        let bufSize = self.dataSock.sockBuff.readBuf.lock().BufSize();
        if 2 * dataSize >= bufSize {
            let channelId = *self.dataSock.channelId.lock();
            self.rdmaCli.read(channelId)?;
        }
            
        return Ok(count)
    }

    pub async fn ReadAll(&self, buf: &mut [u8]) -> Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            let count = self.ReadBuf(&mut buf[offset..]).await?;
            offset += count;
        }

        return Ok(());
    }
}
