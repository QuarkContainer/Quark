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

use crate::common::*;

use super::data_obj::*;
use super::selector::*;

#[derive(Debug, Clone)]
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
            return Err(Error::CommonError(
                "continue key is not valid: encoded start key empty (version meta.k8s.io/v1)"
                    .to_owned(),
            ));
        }

        let mut key = self.key.clone();
        if !key.starts_with("/") {
            key = "/".to_owned() + &key;
        }

        return Ok((keyPrefix.to_string() + &key[1..], self.revision));
    }

    pub fn DeepCopy(&self) -> Self {
        return Self {
            key: self.key.clone(),
            revision: self.revision,
        };
    }
}

#[derive(Debug, Default)]
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

        let labels = obj.Labels();
        let mut matched = self.label.Match(&labels);
        if matched {
            let val = &obj.data;
            let jsonVal = match serde_json::from_str(&val) {
                Err(_) => return Ok(false),
                Ok(v) => v,
            };

            let mut attrs = match self.field.GetAttributes(&jsonVal) {
                None => return Ok(false),
                Some(l) => l,
            };

            for r in &self.field.0 {
                if &r.key == "metadata.name" {
                    attrs.insert("metadata.name".to_string(), obj.Name());
                } else if &r.key == "metadata.namespace" {
                    attrs.insert("metadata.namespace".to_string(), obj.Namespace());
                }
            }

            let attrs = attrs.into();
            matched = self.field.Match(&attrs);
        }

        return Ok(matched);
    }

    pub fn DeepCopy(&self) -> Self {
        return Self {
            label: self.label.DeepCopy(),
            field: self.field.DeepCopy(),
            limit: self.limit,
            continue_: match &self.continue_ {
                None => None,
                Some(c) => Some(c.DeepCopy()),
            },
        };
    }

    pub fn Empty(&self) -> bool {
        return self.label.Empty() && self.field.Empty();
    }

    pub fn HasContinue(&self) -> bool {
        return self.continue_.is_some();
    }

    pub fn Continue(&self, keyPrefix: &str) -> Result<(String, i64)> {
        match &self.continue_ {
            None => {
                return Err(Error::CommonError(
                    "SelectionPredicate has no continue".to_string(),
                ))
            }
            Some(c) => return c.Continue(keyPrefix),
        }
    }
}

pub fn EncodeContinue(key: &str, keyPrefix: &str, revision: i64) -> Result<Continue> {
    let nextKey = match key.strip_prefix(keyPrefix) {
        Some(n) => n,
        None => {
            return Err(Error::CommonError(format!(
                "unable to encode next field: the key '{}' and key prefix '{}' do not match",
                key, keyPrefix
            )))
        }
    };

    //let nextKey = key;
    return Ok(Continue {
        revision: revision,
        key: nextKey.to_owned(),
    });
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RevisionMatch {
    None,
    NotOlderThan,
    Exact,
}

impl Default for RevisionMatch {
    fn default() -> Self {
        return Self::None;
    }
}

#[derive(Debug, Default)]
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

impl ListOption {
    pub fn DeepCopy(&self) -> Self {
        return Self {
            revision: self.revision,
            revisionMatch: self.revisionMatch,
            predicate: self.predicate.DeepCopy(),
        };
    }
}
