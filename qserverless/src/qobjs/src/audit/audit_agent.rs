// Copyright (c) 2021 Quark Container Authors / 2014 The Kubernetes Authors
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
use std::sync::atomic::AtomicBool;
use core::ops::Deref;

use tokio::sync::{Notify, mpsc};

use crate::common::*;
use crate::audit::func_audit::*;

#[derive(Debug)]
pub struct CreateFunc {
    pub id: String,
    pub jobId: String,
    pub namespace: String,
    pub packageName: String,
    pub revision: i64,
    pub funcName: String,
    pub callerFuncId: String,
}

#[derive(Debug)]
pub struct AssignFunc {
    pub id: String,
    pub nodeId: String,
}

#[derive(Debug)]
pub struct FinishFunc {
    pub id: String,
    pub funcState: String,
}

#[derive(Debug)]
pub enum AuditEvent {
    CreateFunc(CreateFunc),
    AssignFunc(AssignFunc),
    FinishFunc(FinishFunc),
}

pub struct AuditAgentInner {
    pub closeNotify: Arc<Notify>,
    pub stop: AtomicBool,

    pub tx: mpsc::Sender<AuditEvent>,
    pub sqlAddr: String,
}


#[derive(Clone)]
pub struct AuditAgent(pub Arc<AuditAgentInner>);

impl Deref for AuditAgent {
    type Target = Arc<AuditAgentInner>;

    fn deref(&self) -> &Arc<AuditAgentInner> {
        &self.0
    }
}

impl AuditAgent {
    pub fn New(sqlAddr: &str) -> Self {
        let (tx, rx) = mpsc::channel::<AuditEvent>(100);
        let inner = AuditAgentInner {
            closeNotify: Arc::new(Notify::new()),
            stop: AtomicBool::new(false),
            tx: tx,
            sqlAddr: sqlAddr.to_owned()
        };

        let ret = AuditAgent(Arc::new(inner));
        let clone = ret.clone();

        tokio::spawn(async move {
            clone.Process(rx).await.unwrap();
        });

        return ret
    }

    pub fn CreateFunc(
        &self,
        id: &str, 
        jobId: &str, 
        namespace: &str,
        packageName: &str,
        revision: i64,
        funcName: &str,  
        callerFuncId: &str
    ) -> Result<()> {
        let cf = CreateFunc {
            id: id.to_owned(),
            jobId: jobId.to_owned(),
            namespace: namespace.to_owned(),
            packageName: packageName.to_owned(),
            revision: revision,
            funcName: funcName.to_owned(),
            callerFuncId: callerFuncId.to_owned()
        };

        self.Send(AuditEvent::CreateFunc(cf))?;
        return Ok(())
    }

    pub fn AssignFunc(
        &self,
        id: &str,
        nodeId: &str
    ) -> Result<()> {
        let cf = AssignFunc {
            id: id.to_owned(),
            nodeId: nodeId.to_owned(),
        };

        self.Send(AuditEvent::AssignFunc(cf))?;
        return Ok(())
    }

    pub fn FinishFunc(
        &self,
        id: &str, 
        funcState: &str
    ) -> Result<()> {
        let cf = FinishFunc {
            id: id.to_owned(),
            funcState: funcState.to_owned(),
        };

        self.Send(AuditEvent::FinishFunc(cf))?;
        return Ok(())
    }

    pub fn Send(&self, msg: AuditEvent) -> Result<()> {
        match self.tx.try_send(msg) {
            Ok(()) => return Ok(()),
            Err(e) => {
                return Err(Error::CommonError(format!("AuditAgent send message fail {:?}", e)))
            }
        }
    }

    pub async fn Process(&self, rx: mpsc::Receiver<AuditEvent>) -> Result<()> {
        let audit = SqlFuncAudit::New(&self.sqlAddr).await.unwrap();
        let mut rx = rx;
        loop {
            tokio::select! {
                _ = self.closeNotify.notified() => {
                    self.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                    break;
                }

                msg = rx.recv() => {
                    if let Some(msg) = msg {
                        match msg {
                            AuditEvent::CreateFunc(c) => {
                                audit.CreateFunc(&c.id, &c.jobId, &c.namespace, &c.packageName, c.revision, &c.funcName, &c.callerFuncId).await?;
                            }
                            AuditEvent::AssignFunc(c) => {
                                audit.AssignFunc(&c.id, &c.nodeId).await?;
                            }
                            AuditEvent::FinishFunc(c) => {
                                audit.FinishFunc(&c.id, &c.funcState).await?;
                            }
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        return Ok(())
    }
}