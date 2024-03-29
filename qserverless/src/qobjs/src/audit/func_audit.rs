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
use sqlx::postgres::PgConnectOptions;
use sqlx::Row;
use sqlx::ConnectOptions;

use crate::common::*;

pub const FuncStateCreated : &str = "created";
pub const FuncStateAssigned : &str = "assgined";
pub const FuncStateSuccess : &str = "sucsess";
pub const FuncStateFail : &str = "fail";

#[async_trait::async_trait]
pub trait FuncAudit {
    async fn CreateFunc(&self, id: &str, jobId: &str, namespace: &str, packageName: &str, revision: i64, funcName: &str, callerFuncId: &str) -> Result<()>;
    async fn AssignFunc(&self, id: &str, nodeId: &str) -> Result<()>;
    async fn FinishFunc(&self, id: &str, funcState: &str) -> Result<()>;
    async fn GetNode(&self, namespace: &str, funcId: &str) -> Result<String>;
}

#[derive(Debug)]
pub struct SqlFuncAudit {
    pub pool: PgPool,
}

impl SqlFuncAudit {
    pub async fn New(sqlSvcAddr: &str) -> Result<Self> {
        let url_parts = url::Url::parse(sqlSvcAddr).expect("Failed to parse URL");
        let username = url_parts.username();
        let password = url_parts.password().unwrap_or("");
        let host = url_parts.host_str().unwrap_or("localhost");
        let port = url_parts.port().unwrap_or(5432);
        let database = url_parts.path().trim_start_matches('/');

        let mut options = PgConnectOptions::new()
            .host(host)
            .port(port)
            .username(username)
            .password(password)
            .database(database);

        options.disable_statement_logging();

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;
        return Ok(Self {
            pool: pool
        })
    }
}

#[async_trait::async_trait]
impl FuncAudit for SqlFuncAudit {
    async fn CreateFunc(
        &self,
        id: &str, 
        jobId: &str, 
        namespace: &str,
        packageName: &str,
        revision: i64,
        funcName: &str, 
        callerFuncId: &str
    ) -> Result<()> {
        let query = "insert into FuncAudit (id, jobId, namespace, packageName, funcName, revision, callerFuncId, funcState, createTime) values \
            (uuid($1), uuid($2), $3, $4, $5, $6, $7, $8, NOW())";

        let _result = sqlx::query(query)
            .bind(id)
            .bind(jobId)
            .bind(namespace)
            .bind(packageName)
            .bind(funcName)
            .bind(revision)
            .bind(callerFuncId)
            .bind(FuncStateCreated.to_owned())
            .execute(&self.pool)
            .await?;
            
        return Ok(())
    }

    async fn AssignFunc(&self, id: &str, nodeId: &str) -> Result<()> {
        let query = "Update FuncAudit Set funcState = $1, nodeId=$2, assignedTime = NOW() where id = uuid($3)";
        let _result = sqlx::query(query)
            .bind(FuncStateAssigned)
            .bind(nodeId)
            .bind(id)
            .execute(&self.pool)
            .await?;

        return Ok(())
    }

    async fn FinishFunc(&self, id: &str, funcState: &str) -> Result<()> {
        let query = "Update FuncAudit Set funcState = $1, finishTime = NOW() where id = uuid($2)";
        let _result = sqlx::query(query)
            .bind(funcState)
            .bind(id)
            .execute(&self.pool)
            .await?;

        return Ok(())
    }

    async fn GetNode(&self, namespace: &str, funcId: &str) -> Result<String> {
        let query = "Select nodeid, namespace, funcstate from FuncAudit where id = uuid($1)";
        let rows = sqlx::query(query)
            .bind(funcId)
            .fetch_all(&self.pool)
            .await?;

        if rows.len() >= 1 {
            let row = &rows[0];
            let expecNamespace = row.get::<String, _>("namespace");
            let funcState = row.get::<String, _>("funcstate");
            let nodeId = row.get::<String, _>("nodeid");

            if &expecNamespace != namespace {
                return Err(Error::ENOENT(format!("SqlFuncAudit::GetNode has no audit for func {} with matched namespace {}", funcId, namespace)));
            }

            if funcState == FuncStateCreated {
                return Err(Error::ENOENT(format!("SqlFuncAudit::GetNode fail as func {} has not been assigned node", funcId)));
            }

            return Ok(nodeId)
        } else {
            return Err(Error::ENOENT(format!("SqlFuncAudit::GetNode has no audit for func {}", funcId)));
        }
    }
}
