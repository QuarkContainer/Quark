/* 
MIT License

Copyright (c) 2022 Alejandro Llanes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

// this is copy and modified from https://github.com/sombralibre/k8s-quantity-parser/blob/main/src/lib.rs

use std::collections::BTreeMap;

use regex::Regex;

use k8s_openapi::apimachinery::pkg::api::resource::Quantity;

use qobjs::common::*;

// CPU, in cores. (500m = .5 cores)
pub const ResourceCPU :&str = "cpu";
// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
pub const ResourceMemory: &str = "memory";
// Volume size, in bytes (e,g. 5Gi = 5GiB = 5 * 1024 * 1024 * 1024)
pub const ResourceStorage: &str = "storage";
// Local ephemeral storage, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
// The resource name for ResourceEphemeralStorage is alpha and it can change across releases.
pub const ResourceEphemeralStorage: &str = "ephemeral-storage";

#[derive(Debug, Default)]
pub struct QuarkResource {
    pub cpu: i64,
    pub memory: i64,
    pub storage: i64,
    pub ephemeralStorage: i64,
}

impl QuarkResource {
    pub fn ToQuantity(&self) -> BTreeMap<String, Quantity> {
        let mut ret = BTreeMap::new();
        ret.insert(ResourceCPU.to_string(), Quantity(format!("{}",self.cpu / 1000)));
        ret.insert(ResourceMemory.to_string(), Quantity(format!("{}",self.memory)));
        ret.insert(ResourceStorage.to_string(), Quantity(format!("{}",self.storage)));
        ret.insert(ResourceEphemeralStorage.to_string(), Quantity(format!("{}",self.ephemeralStorage)));
        return ret;
    }

    pub fn New(resources: &BTreeMap<String, Quantity>) -> Result<QuarkResource> {
        let cpu = match resources.get(ResourceCPU) {
            None => 0,
            Some(q) => QuantityToMilliCpus(q)?,
        };

        let memory = match resources.get(ResourceMemory) {
            None => 0,
            Some(q) => QuantityToBytes(q)?,
        };

        let storage = match resources.get(ResourceStorage) {
            None => 0,
            Some(q) => QuantityToBytes(q)?,
        };

        let ephemeralStorage = match resources.get(ResourceEphemeralStorage) {
            None => 0,
            Some(q) => QuantityToBytes(q)?,
        };

        return Ok(QuarkResource {
            cpu: cpu,
            memory: memory,
            storage: storage,
            ephemeralStorage: ephemeralStorage,
        })
    }

    pub fn Add(&mut self, r: &QuarkResource) {
        self.cpu += r.cpu;
        self.memory += r.memory;
        self.storage += r.storage;
        self.ephemeralStorage += r.ephemeralStorage;
    }
}

enum QuantityMemoryUnits {
    Ki,
    Mi,
    Gi,
    Ti,
    Pi,
    Ei,
    k,
    M,
    G,
    T,
    P,
    E,
    m,
    Invalid,
}

impl QuantityMemoryUnits {
    fn new(unit: &str) -> Self {
        match unit {
            "Ki" => Self::Ki,
            "Mi" => Self::Mi,
            "Gi" => Self::Gi,
            "Ti" => Self::Ti,
            "Pi" => Self::Pi,
            "Ei" => Self::Ei,
            "k" => Self::k,
            "M" => Self::M,
            "G" => Self::G,
            "T" => Self::T,
            "P" => Self::P,
            "E" => Self::E,
            "m" => Self::m,
            _ => Self::Invalid,
        }
    }
}

pub fn QuantityToMilliCpus(q: &Quantity) -> Result<i64> {
    let unit_str = &q.0;
    let rgx = Regex::new(r"([m]{1}$)")?;
    let cap = rgx.captures(unit_str);
    if cap.is_none() {
        return Ok(unit_str.parse::<i64>()? * 1000);
    };
    let mt = cap.unwrap().get(0).unwrap();
    let unit_str = unit_str.replace(mt.as_str(), "");
    Ok(unit_str.parse::<i64>()?)
} 

pub fn QuantityToBytes(q: &Quantity) -> Result<i64> {
    let unit_str = &q.0;
    let rgx = Regex::new(r"([[:alpha:]]{1,2}$)")?;
    let cap = rgx.captures(unit_str);

    if cap.is_none() {
        return Ok(unit_str.parse::<i64>()?);
    };

    // Is safe to use unwrap here, as the value is already checked.
    match cap.unwrap().get(0) {
        Some(m) => match QuantityMemoryUnits::new(m.as_str()) {
            QuantityMemoryUnits::Ki => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(amount * 1024)
            }
            QuantityMemoryUnits::Mi => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok((amount * 1024) * 1024)
            }
            QuantityMemoryUnits::Gi => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(((amount * 1024) * 1024) * 1024)
            }
            QuantityMemoryUnits::Ti => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok((((amount * 1024) * 1024) * 1024) * 1024)
            }
            QuantityMemoryUnits::Pi => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(((((amount * 1024) * 1024) * 1024) * 1024) * 1024)
            }
            QuantityMemoryUnits::Ei => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(
                    (((((amount * 1024) * 1024) * 1024) * 1024) * 1024) * 1024,
                )
            }
            QuantityMemoryUnits::k => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(amount * 1000)
            }
            QuantityMemoryUnits::M => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok((amount * 1000) * 1000)
            }
            QuantityMemoryUnits::G => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(((amount * 1000) * 1000) * 1000)
            }
            QuantityMemoryUnits::T => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok((((amount * 1000) * 1000) * 1000) * 1000)
            }
            QuantityMemoryUnits::P => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(((((amount * 1000) * 1000) * 1000) * 1000) * 1000)
            }
            QuantityMemoryUnits::E => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(
                    (((((amount * 1000) * 1000) * 1000) * 1000) * 1000) * 1000,
                )
            }
            QuantityMemoryUnits::m => {
                let unit_str = unit_str.replace(m.as_str(), "");
                let amount = unit_str.parse::<i64>()?;
                Ok(amount / 1000)
            }
            QuantityMemoryUnits::Invalid => Err(Error::CommonError("Invalid unit".to_string())),
        },
        None => Err(Error::CommonError("Invalid unit".to_string())),
    }
}

