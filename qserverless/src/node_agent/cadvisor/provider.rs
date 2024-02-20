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

use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use core::ops::Deref;

use tokio::sync::Notify;
use tokio::time;

use qobjs::common::*;

use crate::CADVISOR_CLI;

use super::client::*;

#[derive(Debug)]
pub struct CadvisorInfoProviderInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

	pub nodeCAdvisorInfo: Mutex<Arc<NodeCAdvisorInfo>>,
	pub nodeInfoInterval: Duration,
	
}

#[derive(Clone, Debug)]
pub struct CadvisorInfoProvider(Arc<CadvisorInfoProviderInner>);

unsafe impl Send for CadvisorInfoProvider {}

impl Deref for CadvisorInfoProvider {
    type Target = Arc<CadvisorInfoProviderInner>;

    fn deref(&self) -> &Arc<CadvisorInfoProviderInner> {
        &self.0
    }
}

impl CadvisorInfoProvider {
	pub async fn New() -> Result<Self> {
		let info = Self::CollectCAdvisorInfo().await?;
		let inner = CadvisorInfoProviderInner {
			closeNotify: Arc::new(Notify::new()),
			stop: AtomicBool::new(false),
			nodeCAdvisorInfo: Mutex::new(Arc::new(info)),
			nodeInfoInterval: Duration::from_secs(10),
		};

		let provider = Self(Arc::new(inner));
		let clone = provider.clone();

		tokio::spawn(async move {
            clone.Process().await.unwrap();
        });

		return Ok(provider);
	}

	pub fn CAdvisorInfo(&self) -> Arc<NodeCAdvisorInfo> {
		return self.nodeCAdvisorInfo.lock().unwrap().clone();
	}

	pub fn Stop(&self) {
		self.closeNotify.notify_one();
	}

	pub async fn CollectCAdvisorInfo() -> Result<NodeCAdvisorInfo> {
		let machineInfo = CADVISOR_CLI.MachineInfo().await?;
		let versionInfo = CADVISOR_CLI.VersionInfo().await?;

		let info = NodeCAdvisorInfo {
			machineInfo: machineInfo,
			versionInfo: versionInfo,
		};

		return Ok(info)
	}

	pub async fn Process(&self) -> Result<()> {
		let mut interval = time::interval(self.nodeInfoInterval);
		loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, Ordering::SeqCst);
                    break;
                }
                _ = interval.tick() => {
					let info = Arc::new(Self::CollectCAdvisorInfo().await?);
                    *self.nodeCAdvisorInfo.lock().unwrap() = info;
                }
            }
        }
        
        return Ok(())
	}
}
