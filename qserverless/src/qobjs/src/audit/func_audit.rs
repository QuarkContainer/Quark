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

use sqlx::postgres::PgPoolOptions;
use sqlx::postgres::PgPool;

use crate::common::*;

#[async_trait::async_trait]
pub trait FuncAudit {
    async fn CreateFunc(&mut self, id: &str, jobId: &str, namespace: &str, packageName: &str, funcName: &str, callerFuncId: &str) -> Result<()>;
    async fn FinishFunc(&mut self, id: &str, funcState: &str) -> Result<()>;
}

pub struct SqlFuncAudit {
    pub pool: PgPool,
}

impl SqlFuncAudit {
    pub async fn New(sqlSvcAddr: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(sqlSvcAddr)
		    .await?;
        return Ok(Self {
            pool: pool
        })
    }
}

#[async_trait::async_trait]
impl FuncAudit for SqlFuncAudit {
    async fn CreateFunc(
        &mut self,
        id: &str, 
        jobId: &str, 
        namespace: &str,
        packageName: &str,
        funcName: &str, 
        callerFuncId: &str
    ) -> Result<()> {
        let query = "insert into FuncAudit (id, jobId, namespace, packageName, funcName, callerFuncId, funcState, createTime) values \
            (uuid($1), uuid($2), $3, $4, $5, $6, 'Running', NOW())";

        let _result = sqlx::query(query)
            .bind(id)
            .bind(jobId)
            .bind(namespace)
            .bind(packageName)
            .bind(funcName)
            .bind(callerFuncId)
            .execute(&self.pool)
            .await?;
            
        return Ok(())
    }

    async fn FinishFunc(&mut self, id: &str, funcState: &str) -> Result<()> {
        let query = "Update FuncAudit Set funcState = $1, finishTime = NOW() where id = uuid($2)";
        let _result = sqlx::query(query)
            .bind(funcState)
            .bind(id)
            .execute(&self.pool)
            .await?;

        return Ok(())
    }
}
