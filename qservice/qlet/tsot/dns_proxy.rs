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
// limitations under the License.

use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicBool, Mutex};
use std::sync::Arc;
use std::time::Duration;
use rand::Rng;
use tokio::sync::{mpsc, Notify};

use qshare::metastore::cacher_client::CacherClient;
use qshare::tsot_msg::*;
use crate::QLET_CONFIG;

use super::pod_broker::PodBroker;
use qshare::node::PodDef;

pub const DNS_POSTFIX : &'static str = "svc.cluster.local";


lazy_static::lazy_static! {
    pub static ref DNS_PROXY: DnsProxy = DnsProxy::New();
}

#[derive(Debug)]
pub struct DnsProxyReq {
    pub reqId: u16,
    pub podBroker: PodBroker,
    pub domains: Vec<String>
}

pub struct DnsProxy {
    pub closeNotify: Arc<Notify>,
    pub closed: AtomicBool,

    pub inputTx: mpsc::Sender<DnsProxyReq>,
    pub inputRx: Mutex<Option<mpsc::Receiver<DnsProxyReq>>>,

    pub stateSvcAddresses: Vec<String>
}

impl DnsProxy {
    pub fn New() -> Self {
        let (tx, rx) = mpsc::channel::<DnsProxyReq>(30);
        
        let mut stateSvcAddresses = Vec::new();
        if QLET_CONFIG.singleNodeModel {
            stateSvcAddresses.push(format!("http://127.0.0.1:{}", QLET_CONFIG.stateSvcPort));
        } else {
            for a in &QLET_CONFIG.stateSvcAddr {
                stateSvcAddresses.push(format!("http://{}", a));
            }
        }

        return Self {
            closeNotify: Arc::new(Notify::new()),
            closed: AtomicBool::new(false),

            inputTx: tx,
            inputRx: Mutex::new(Some(rx)),

            stateSvcAddresses: stateSvcAddresses
        }
    }

    pub fn Close(&self) {
        self.closeNotify.notify_waiters();
    }

    pub fn EnqMsg(&self, req: DnsProxyReq) {
        self.inputTx.try_send(req).unwrap();
    }

    pub async fn GetClient(&self) -> Option<CacherClient> {
        let size = self.stateSvcAddresses.len();
        let offset: usize = rand::thread_rng().gen_range(0..size);
        loop {
            for i in 0..size {
                let idx = (offset + i) % size;
                let addr = &self.stateSvcAddresses[idx];

                tokio::select! { 
                    out = CacherClient::New(addr.to_owned()) => {
                        match out {
                            Ok(client) => return Some(client),
                            Err(e) => {
                                error!("DnsProxy::GetClient fail to connect to {} with error {:?}", addr, e);
                            }
                        }
                    }
                    _ = self.closeNotify.notified() => {
                        self.closed.store(true, Ordering::SeqCst);
                        return None
                    }
                };
            }

            // retry after one second
            tokio::select! { 
                _ = tokio::time::sleep(Duration::from_millis(1000)) => {}
                _ = self.closeNotify.notified() => {
                    self.closed.store(true, Ordering::SeqCst);
                    return None
                }
            }
        }
    }

    pub async fn Process(&self) {
        let mut rx = self.inputRx.lock().unwrap().take().unwrap();
        let mut client = match self.GetClient().await {
            None => return,
            Some(c) => c
        };

        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.closed.store(false, Ordering::SeqCst);
                    break;
                }
                m = rx.recv() => {
                    match m {
                        None => (),
                        Some(m) => {
                            let m: DnsProxyReq = m;
                            let mut ips = Vec::new();
                            for domain in m.domains {
                                if domain.ends_with(DNS_POSTFIX) {
                                    let left = domain.strip_suffix(DNS_POSTFIX).unwrap();
                                    let split : Vec<&str> = left.split(".").collect();
                                    if split.len() != 4 {
                                        error!("get invalid domain {}/ {:?}/ {}", domain, &split, left);
                                        ips.push(0);
                                        continue;
                                    } 
                                    let tenant = split[2];
                                    let namespace = split[1];
                                    let name = split[0];
                                    match client.Get("pod", tenant, namespace, name, 0).await {
                                        Err(e) => {
                                            error!("DnsProxy::Process fail {:?}", e);
                                            client = match self.GetClient().await {
                                                None => return,
                                                Some(c) => c
                                            };
                                        }
                                        Ok(r) => {
                                            match r {
                                                None => ips.push(0),
                                                Some(obj) => {
                                                    let pod: PodDef = serde_json::from_str(&obj.data)
                                                    .expect(&format!("NodeMgr::handle deserialize fail for {}", &obj.data));
                                                    ips.push(pod.ipAddr);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if domain.len() == 0 {
                                        ips.push(0);
                                        continue;
                                    }
                                    match dns_lookup::lookup_host(&domain) {
                                        Err(_) => {
                                            ips.push(0);
                                        }
                                        Ok(ipAddresses) =>  {
                                            let mut hasIpv4 = false;
                                            for ip in ipAddresses {
                                                match ip {
                                                    std::net::IpAddr::V4(addr) => {
                                                        let octets = addr.octets();
                                                        let addr = ((octets[0] as u32) << 24) |
                                                            ((octets[1] as u32) << 16) |
                                                            ((octets[2] as u32) << 8) |
                                                            ((octets[3] as u32) << 0);

                                                            ips.push(addr);
                                                            hasIpv4 = true;
                                                            break;
                                                    }
                                                    _ => ()
                                                }
                                            }

                                            if !hasIpv4 {
                                                ips.push(0);
                                            }
                                        }
                                    };
                                }
                            } 

                            let mut dnsResp = DnsResp {
                                reqId: m.reqId,
                                count: ips.len(),
                                ..Default::default()
                            };

                            assert!(ips.len() <= 4);
                            for i in 0..ips.len() {
                                dnsResp.ips[i] = ips[i];
                            }


                            m.podBroker.EnqMsg(TsotMsg::DnsResp(dnsResp).into()).unwrap();
                        }


                    }
                }
            }
        }
    }

}


