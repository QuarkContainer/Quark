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

use std::collections::BTreeMap;

use crate::shared::common::*;

use super::validation::*;

pub enum SelectionOp {
    DoesNotExist,   // "!"
	Equals,         // "="
	DoubleEquals,   // "=="
	In,             // "in"
	NotEquals,      // "!="
	NotIn,          // "notin"
	Exists,         // "exists"
	GreaterThan,    // "gt"
	LessThan,       // "lt"
}

impl SelectionOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DoesNotExist => return "!",
            Self::Equals => return "=",
            Self::DoubleEquals => return "==",
            Self::In => return "in",
            Self::NotEquals => return "!=",
            Self::NotIn => return "notin",
            Self::Exists => return "exists",
            Self::GreaterThan => return "gt",
            Self::LessThan => return "lt",
        }
    }
}

pub struct Requirement {
    pub key: String,
    pub op: SelectionOp,
    pub strVal: Vec<String>
}

impl Requirement {
    // If any of these rules is violated, an error is returned:
    //  1. The operator can only be In, NotIn, Equals, DoubleEquals, Gt, Lt, NotEquals, Exists, or DoesNotExist.
    //  2. If the operator is In or NotIn, the values set must be non-empty.
    //  3. If the operator is Equals, DoubleEquals, or NotEquals, the values set must contain one value.
    //  4. If the operator is Exists or DoesNotExist, the value set must be empty.
    //  5. If the operator is Gt or Lt, the values set must contain only one value, which will be interpreted as an integer.
    //  6. The key is invalid due to its length, or sequence of characters. See validateLabelKey for more details.
    pub fn New(key: &str, op: SelectionOp, vals: Vec<String>) -> Result<Requirement> {
        IsValidLabelValue(key)?;
        match op {
            SelectionOp::In | SelectionOp::NotIn => {
                if vals.len() == 0 {
                    return Err(Error::CommonError("for 'in', 'notin' operators, values set can't be empty".to_owned()))
                }
            }
            SelectionOp::Equals | SelectionOp::DoubleEquals | SelectionOp::NotEquals => {
                if vals.len() != 1 {
                    return Err(Error::CommonError("exact-match compatibility requires one single value".to_owned()))
                }
            }
            SelectionOp::Exists | SelectionOp::DoesNotExist => {
                if vals.len() != 0 {
                    return Err(Error::CommonError("values set must be empty for exists and does not exist".to_owned()))
                }
            }
            SelectionOp::GreaterThan | SelectionOp::LessThan => {
                if vals.len() != 1 {
                    return Err(Error::CommonError("for 'Gt', 'Lt' operators, exactly one value is required".to_owned()))
                }

                for val in &vals {
                    match val.parse::<u64>() {
                        Err(_) => {
                            return Err(Error::CommonError("for 'Gt', 'Lt' operators, the value must be an integer".to_owned()))
                        }
                        _ => {}
                    }
                }

            }
        }

        return Ok(Requirement { key: key.to_owned(), op: op, strVal: vals })
    }

    pub fn HasValue(&self, val: &str) -> bool {
        for str in &self.strVal {
            if str == val {
                return true
            }
        }

        return false;
    }
}

#[derive(Debug, Default)]
pub struct Labels(BTreeMap<String, String>);

impl Labels {
    //!pub fn New(selector: &str)

    // String returns all labels listed as a human readable string.
    // Conveniently, exactly the format that ParseSelector takes.
    pub fn String(&self) -> String {
        let mut ret = "".to_owned();
        for (k, v) in &self.0 {
            if ret.len() != 0 {
                ret = ret + ",";
            }

            ret = ret + k + "=" + v;
        }

        return ret;
    }

    // Has returns whether the provided label exists in the map.
    pub fn Has(&self, label: &str) -> bool {
        return self.0.contains_key(label);
    }

    // Get returns the value in the map for the provided label.
    pub fn Get(&self, label: &str) -> Option<String> {
        match self.0.get(label) {
            None => return None,
            Some(v) => return Some(v.to_string()),
        }
    }

    // FormatLabels converts label map into plain string
    pub fn Format(&self) -> String {
        let l = self.String();
        if l.len() == 0 {
            return "<none>".to_owned();
        }
        return l;
    }

    // Conflicts takes 2 maps and returns true if there a key match between
    // the maps but the value doesn't match, and returns false in other cases
    pub fn Conflict(&self, labels: &Self) -> bool {
        let (small, big) = if self.0.len() < labels.0.len() {
            (self, labels)
        } else {
            (labels, self)
        };

        for (k, v) in &small.0 {
            match big.0.get(k) {
                None => return true,
                Some(val) => {
                    if v != val {
                        return true
                    }
                }
            }
        }

        return false;
    }

    // Merge combines given maps, and does not check for any conflicts
    // between the maps. In case of conflicts, second map (labels2) wins
    pub fn Merge(&self, labels: &Self) -> Self {
        let mut merged = Self::default();

        for (k, v) in &self.0 {
            merged.0.insert(k.to_string(), v.to_string());
        }

        for (k, v) in &labels.0 {
            merged.0.insert(k.to_string(), v.to_string());
        }

        return merged;
    }

    // Equals returns true if the given maps are equal
    pub fn Equals(&self, labels: &Self) -> bool {
        if self.0.len() != labels.0.len() {
            return false;
        }

        for (k, v) in &self.0 {
            match labels.0.get(k) {
                None => return false,
                Some(val) => {
                    if v != val {
                        return false
                    }
                }
            }
        }

        return true;
    }
}