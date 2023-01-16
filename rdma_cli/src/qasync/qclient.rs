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
use spin::{Mutex, MutexGuard};
use std::collections::HashMap;

use tokio::io::AsyncReadExt;
use tokio::sync::Notify;
//use tokio_io::{AsyncRead, AsyncWrite};
use tokio::io::AsyncRead;
use tokio::io::ReadBuf;

use crate::common::*;
use crate::RDMASvcClient;

use crate::qasync::uid::*;
use crate::qasync::eventfd::*;
use crate::qasync::qstream::*;
use crate::qlib::common::*;
use crate::qlib::rdma_share::*;

pub struct QClientlib {
    pub rdmaCli: RDMASvcClient,
    pub notify: Arc<Notify>,
    pub eventfd: Mutex<EventFd>,

    // channel id --> QStream
    pub streams: Mutex<HashMap<u32, QStream>>,
}

impl QClientlib {
    pub async fn Process(&self) -> Result<()> {
        loop {
            let mut buf = [0; 8];
            self.eventfd.lock().read(&mut buf).await?;
            loop {
                let request = self.rdmaCli.cliShareRegion.lock().cq.Pop();

                match request {
                    Some(cq) => match cq.msg {
                        RDMARespMsg::RDMAConnect(_response) => {
                        }
                        _ =>()
                    }
                    None => {
                        break;
                    }
                }
            }
        }
    }
}