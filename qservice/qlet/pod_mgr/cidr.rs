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

use core::ops::Deref;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::Mutex;

use qshare::common::*;

#[derive(Debug)]
pub struct CidrInner {
    pub addr: u32,
    pub mask: u32,
    pub minAddr: u32,
    pub maxAddr: u32,
    pub nextAddr: u32,
    pub allocated: BTreeSet<u32>,
}

#[derive(Debug, Clone)]
pub struct Cidr(Arc<Mutex<CidrInner>>);

impl Deref for Cidr {
    type Target = Arc<Mutex<CidrInner>>;

    fn deref(&self) -> &Arc<Mutex<CidrInner>> {
        &self.0
    }
}

impl Cidr {
    // for cidr (10.2.0.0/16), the addr is 10.2.0.0, the maskbits is 16
    pub fn New(addr: u32, maskbits: usize) -> Self {
        let mask: u32 = !((1 << maskbits) - 1);
        assert!(addr & !mask == 0);

        let minAddr = addr + 1; // we don't use the first addr
        let maxAddr = (addr + !mask) - 1; // we don't use the last addr

        let inner = CidrInner {
            addr: addr,
            mask: mask,
            minAddr: minAddr,
            maxAddr: maxAddr,
            nextAddr: minAddr,
            allocated: BTreeSet::new(),
        };

        return Self(Arc::new(Mutex::new(inner)));
    }

    pub fn Allocate(&self) -> Result<IpAddress> {
        let mut inner = self.lock().unwrap();
        if inner.allocated.len() == (inner.maxAddr - inner.minAddr + 1) as usize {
            return Err(Error::CommonError(
                "Cidr: the address are used up".to_owned(),
            ));
        }

        for current in inner.nextAddr..inner.maxAddr + 1 {
            if !inner.allocated.contains(&current) {
                inner.allocated.insert(current);
                if current == inner.maxAddr {
                    inner.nextAddr = inner.minAddr;
                } else {
                    inner.nextAddr = current + 1;
                }

                return Ok(IpAddress(current));
            }
        }

        for current in inner.minAddr..inner.nextAddr {
            if !inner.allocated.contains(&current) {
                inner.allocated.insert(current);
                inner.nextAddr = current + 1;

                return Ok(IpAddress(current));
            }
        }

        return Err(Error::CommonError(
            "Cidr: the address are used up".to_owned(),
        ));
    }

    pub fn Free(&self, addr: IpAddress) -> Result<()> {
        let mut inner = self.lock().unwrap();
        let exist = inner.allocated.remove(&addr.0);

        if !exist {
            return Err(Error::CommonError(format!(
                "Cidr: free an un-allocated address {:x?}",
                addr
            )));
        }

        return Ok(());
    }
}
