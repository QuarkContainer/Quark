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

use postgres::{Client, NoTls};

use crate::common::*;

pub trait FuncAudit {
    fn CreateFunc(&mut self, id: &str, jobId: &str, packageName: &str, callerFuncId: &str) -> Result<()>;
    fn FinishFunc(&mut self, id: &str, funcState: &str) -> Result<()>;
}

pub struct SqlFuncAudit {
    pub client: Client,
}

impl SqlFuncAudit {
    pub fn New(sqlSvcAddr: &str) -> Result<Self> {
        let client = Client::connect(sqlSvcAddr, NoTls)?;
        return Ok(Self {
            client: client
        })
    }
}

impl FuncAudit for SqlFuncAudit {
    fn CreateFunc(
        &mut self,
        id: &str, 
        jobId: &str, 
        packageName: &str, 
        callerFuncId: &str
    ) -> Result<()> {
        let sql = format!("insert into FuncAudit (id, jobId, packageName, callerFuncId, funcState, createTime) values \
            (uuid('{}'), uuid('{}'), '{}', uuid('{}'), 'Running', NOW())", 
            id, jobId, packageName, callerFuncId);
        println!("sql is {}", &sql);
        
        self.client.execute(&sql, &[])?;
        return Ok(())
    }

    fn FinishFunc(&mut self, id: &str, funcState: &str) -> Result<()> {
        let sql = format!("Update FuncAudit Set funcState = '{}', finishTime = NOW() where id = uuid('{}')", funcState, id);
        println!("sql is {}", &sql);
        self.client.execute(&sql, &[])?;
        return Ok(())
    }
}
