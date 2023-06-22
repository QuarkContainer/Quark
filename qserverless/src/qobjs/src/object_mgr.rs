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
use sqlx::postgres::PgRow;

use crate::common::*;

#[async_trait::async_trait]
pub trait ObjectbMgrTrait {
    async fn PutObject(&self, namespace: &str, name: &str, data: &[u8]) -> Result<()>;
    async fn DeleteObject(&self, namespace: &str, name: &str) -> Result<()>;
    async fn ReadObject(&self, namespace: &str, name: &str) -> Result<Vec<u8>>;
    async fn ListObjects(&self, namespace: &str, prefix: &str) -> Result<Vec<ObjectMeta>>;
}

#[derive(Debug)]
pub struct SqlObjectMgr {
    pub pool: PgPool,
}

impl SqlObjectMgr {
    pub async fn New(blobDbAddr: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(blobDbAddr)
		    .await?;
        return Ok(Self {
            pool: pool
        })
    }
}

#[derive(Debug)]
pub struct ObjectMeta {
    pub name: String,
    pub size: i32,
}

#[async_trait::async_trait]
impl ObjectbMgrTrait for SqlObjectMgr {
    async fn PutObject(&self, namespace: &str, name: &str, data: &[u8]) -> Result<()> {
        self.DeleteObject(namespace, name).await.ok();
        let query = "insert into blobtbl (namespace, name, data, createTime) values \
            ($1, $2, $3, NOW())";
        let _result = sqlx::query(query)
            .bind(namespace)
            .bind(name)
            .bind(data)
            .execute(&self.pool)
            .await?;
            
        return Ok(())
    }
    
    async fn DeleteObject(&self, namespace: &str, name: &str) -> Result<()> {
        let query = "delete from blobtbl where namespace=$1 and name = $2";
        let _result = sqlx::query(query)
            .bind(namespace)
            .bind(name)
            .execute(&self.pool)
            .await?;
            
        return Ok(())
    }

    async fn ReadObject(&self, namespace: &str, name: &str) -> Result<Vec<u8>> {
        let query = "select data from blobtbl where namespace=$1 and name = $2";
        let data = sqlx::query(query)
            .bind(namespace)
            .bind(name)
            .fetch_one(&self.pool)
            .await?;

        let data : Vec<u8> = data.get("data");
            
        return Ok(data)
    }

    async fn ListObjects(&self, namespace: &str, prefix: &str) -> Result<Vec<ObjectMeta>> {
        let query = "select name, length(data) size from blobtbl where namespace=$1 and name like $2";
        let prefix = &format!("{}%", prefix);
        let select_query = sqlx::query(query);
        let objs: Vec<ObjectMeta> = select_query
            .bind(namespace)
            .bind(prefix)
            .map(|row: PgRow| ObjectMeta {
                name: row.get("name"),
                size: row.get("size"),
            })
		.fetch_all(&self.pool)
		.await?;

        return Ok(objs)
    }
}