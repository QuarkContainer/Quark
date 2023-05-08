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

use async_trait::async_trait;

use crate::common::*;
use crate::types::*;

#[async_trait]
pub trait CacheStore {
    async fn Create(&self, key: &str, obj: &DataObject) -> Result<DataObject>;
    async fn Update(
        &self,
        key: &str,
        expectedRev: i64,
        obj: &DataObject,
    ) -> Result<DataObject>;
    async fn Delete(&self, key: &str, expectedRev: i64) -> Result<i64>;
}