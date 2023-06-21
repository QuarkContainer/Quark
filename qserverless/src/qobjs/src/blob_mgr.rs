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
use sqlx::Row;

use crate::common::*;

#[async_trait::async_trait]
pub trait BlobMgrTrait {
    async fn CreateBlob(&self, id: &str, data: &[u8]) -> Result<()>;
    async fn DropBlob(&self, id: &str) -> Result<()>;
    async fn ReadBlob(&self, id: &str) -> Result<Vec<u8>>;
}

pub struct SqlBlobMgr {
    pub pool: PgPool,
}

impl SqlBlobMgr {
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
impl BlobMgrTrait for SqlBlobMgr {
    async fn CreateBlob(&self, id: &str, data: &[u8]) -> Result<()> {
        let query = "insert into blobtbl (id, data, createTime) values \
            (uuid($1), $2, NOW())";
        let _result = sqlx::query(query)
            .bind(id)
            .bind(data)
            .execute(&self.pool)
            .await?;
            
        return Ok(())
    }
    
    async fn DropBlob(&self, id: &str) -> Result<()> {
        let query = "delete from blobtbl where id = uuid($1)";
        let _result = sqlx::query(query)
            .bind(id)
            .execute(&self.pool)
            .await?;
            
        return Ok(())
    }

    async fn ReadBlob(&self, id: &str) -> Result<Vec<u8>> {
        let query = "select data from blobtbl where id = uuid($1)";
        let data = sqlx::query(query)
            .bind(id)
            .fetch_one(&self.pool)
            .await?;

        let data : Vec<u8> = data.get("data");
            
        return Ok(data)
    }
}