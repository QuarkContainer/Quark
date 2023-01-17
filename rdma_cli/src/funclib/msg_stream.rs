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

use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use super::qstream::*;
use super::message::*;
use crate::qlib::common::*;
use crate::qlib::bytestream::*;

pub enum MsgStream {
    QStream(QStream),
    QTcpStream(QTcpStream),
}

impl MsgStream {
    pub async fn ReadAll(&mut self, buf: &mut [u8]) -> Result<()> {
        match self {
            Self::QStream(s) => return s.ReadAll(buf).await,
            Self::QTcpStream(s) => return s.ReadAll(buf).await,
        }
    }

    pub async fn WriteAll(&mut self, buf: &[u8]) -> Result<()> {
        match self {
            Self::QStream(s) => return s.WriteAll(buf).await,
            Self::QTcpStream(s) => return s.WriteAll(buf).await,
        }
    }

    pub async fn ReadMsg(&mut self) -> Result<QMsg> {
        let mut lenBuf : [u8; 4] = [0; 4];
        self.ReadAll(&mut lenBuf).await?;
        let len = u32::from_le_bytes(lenBuf) as usize;

        let mut buf = Vec::with_capacity(len);
        buf.resize(len as usize, 0u8);

        self.ReadAll(&mut buf).await?;
        let msg = QMsg::Deserialize(&mut buf).unwrap();
        return Ok(msg)
    }

    pub async fn WriteMsg(&mut self, msg: &QMsg) -> Result<()> {
        let size = msg.Size() as u32;
        self.WriteAll(&size.to_le_bytes()).await?;
        
        let mut buf = Vec::with_capacity(size as usize);
        buf.resize(size as usize, 0u8); 

        msg.Serialize(&mut buf).unwrap();
        self.WriteAll(&buf).await?;
        
        return Ok(())
    }
}

pub struct QTcpStream(TcpStream);

impl QTcpStream {
    pub async fn ReadAll(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            let count = self.0.read(&mut buf[offset..]).await?;
            offset += count;
        }

        return Ok(())
    }

    pub async fn WriteAll(&mut self, buf: &[u8]) -> Result<()> {
        self.0.write_all(buf).await?;
        return Ok(())
    }

}


