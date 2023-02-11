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

use crate::shared::common::*;

use super::etcd_store::*;
use super::selector::*;

#[derive(Debug)]
pub struct Continue {
    pub key: String,
    pub revision: i64,
}

impl Continue {
    pub fn Continue(&self, keyPrefix: &str) -> Result<(String, i64)> {
        if self.revision == 0 {
            return Err(Error::CommonError("continue key is not valid: incorrect encoded start resourceVersion (version meta.k8s.io/v1)".to_owned()));
        }

        if self.key.len() == 0 {
            return Err(Error::CommonError("continue key is not valid: encoded start key empty (version meta.k8s.io/v1)".to_owned()));
        }

        let mut key = self.key.clone();
        if !key.starts_with("/") {
            key = "/".to_owned() + &key;
        }

        return Ok((keyPrefix.to_string() + &key[1..], self.revision));
    }
}

#[derive(Debug)]
pub struct SelectionPredicate {
    pub label: Selector,
    pub field: Selector,
    pub limit: usize,
    pub continue_: Option<Continue>,
}

impl SelectionPredicate {
    pub fn Match(&self, obj: &DataObject) -> Result<bool> {
        if self.Empty() {
            return Ok(true);
        }

        let (labels, fields) = obj.Attributes();
        let mut matched = self.label.Match(&labels);
        if matched {
            matched = self.field.Match(&fields);
        }

        return Ok(matched);
    }

    pub fn Empty(&self) -> bool {
        return self.label.Empty() && self.field.Empty();
    }

    pub fn HasContinue(&self) -> bool {
        return self.continue_.is_some();
    }

    pub fn Continue(&self, keyPrefix: &str) -> Result<(String, i64)> {
        match &self.continue_ {
            None => return Err(Error::CommonError("SelectionPredicate has no continue".to_string())),
            Some(c) => return c.Continue(keyPrefix),
        }
    }
}

pub fn EncodeContinue(key: &str, keyPrefix: &str, revision: i64) -> Result<Continue> {
    let nextKey = match key.strip_prefix(keyPrefix) {
        Some(n) => return Err(Error::CommonError(format!("unable to encode next field: the key '{}' and key prefix '{}' do not match", n, keyPrefix))),
        None => key,
    };

    return Ok(Continue {
        revision: revision,
        key: nextKey.to_owned(),
    })
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RevisionMatch {
    NotOlderThan,
    Exact,
}

#[derive(Debug)]
pub struct ListOption {
    // revision provides a resource version constraint to apply to the list operation
	// as a "not older than" constraint: the result contains data at least as new as the provided
	// ResourceVersion. The newest available data is preferred, but any data not older than this
	// ResourceVersion may be served.
    pub revision: i64,

    // revisionMatch provides the rule for how the resource version constraint applies. If set
	// to the default value "" the legacy resource version semantic apply.
    pub revisionMatch: RevisionMatch,

    // Predicate provides the selection rules for the list operation.
    pub predicate: SelectionPredicate,
}