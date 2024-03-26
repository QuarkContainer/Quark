// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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

use std::str::FromStr;

use qshare::metastore::data_obj::*;
use qshare::metastore::informer::EventHandler;
use qshare::metastore::store::ThreadSafeStore;
use qshare::types::NodeInfo;

use crate::tsot::peer_mgr::PEER_MGR;

#[derive(Debug, Default, Clone)]
pub struct NodeMgr {}

impl EventHandler for NodeMgr {
    fn handle(&self, _store: &ThreadSafeStore, event: &DeltaEvent) {
        match &event.type_ {
            EventType::Added => {
                let obj = &event.obj;
                let nodeInfo: NodeInfo = serde_json::from_str(&obj.data).expect(&format!(
                    "NodeMgr::handle deserialize fail for {}",
                    &obj.data
                ));

                let peerIp = ipnetwork::Ipv4Network::from_str(&nodeInfo.nodeIp)
                    .unwrap()
                    .ip()
                    .into();
                let peerPort: u16 = nodeInfo.tsotSvcPort;
                let cidr = ipnetwork::Ipv4Network::from_str(&nodeInfo.cidr).unwrap();
                PEER_MGR
                    .AddPeer(peerIp, peerPort, cidr.ip().into())
                    .unwrap();
            }
            EventType::Deleted => {
                let obj: &DataObject = event.oldObj.as_ref().unwrap();
                let nodeInfo: NodeInfo = serde_json::from_str(&obj.data).expect(&format!(
                    "NodeMgr::handle deserialize fail for {}",
                    &obj.data
                ));

                let cidr = ipnetwork::Ipv4Network::from_str(&nodeInfo.cidr).unwrap();
                PEER_MGR.RemovePeer(cidr.ip().into()).unwrap();
            }
            _ => (),
        }
    }
}
