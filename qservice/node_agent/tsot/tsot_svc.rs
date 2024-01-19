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
// limitations under

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::*;

use tokio::net::UnixListener;
use tokio::sync::Notify;

use qshare::common::*;

pub static SOCKET_PATH: &'static str = "/var/quarksvc-socket";

pub struct TsotSvc {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub listener: UnixListener,
}

impl TsotSvc {
    pub fn New() -> Result<Self> {
        let socket = Path::new(SOCKET_PATH);

        // Delete old socket if necessary
        if socket.exists() {
            std::fs::remove_file(&socket).unwrap();
        }

        let listener = UnixListener::bind(socket)?;

        return Ok(Self{
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),

            listener: listener,
        })
    }

    pub fn Close(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.closeNotify.notify_waiters();
    }
}