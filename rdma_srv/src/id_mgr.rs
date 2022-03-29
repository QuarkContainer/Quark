use super::qlib::common::{Error, Result};
use std::collections::HashSet;

pub struct IdMgr {
    pub set: HashSet<u32>,
    pub len: u32,
    pub start: u32,
}

impl IdMgr {
    pub fn Init(start: u32, len: u32) -> Self {
        return IdMgr {
            set: HashSet::new(),
            len: len,
            start: start,
        };
    }

    pub fn AllocId(&mut self) -> Result<u32> {
        if self.set.len() == self.len as usize {
            return Err(Error::NoEnoughSpace);
        }
        for i in self.start..(self.len + self.start) {
            if !self.set.contains(&i) {
                self.set.insert(i);
                return Ok(i);
            }
        }
        return Err(Error::NoData);
    }

    pub fn Remove(&mut self, i: u32) {
        self.set.remove(&i);
    }

    pub fn AddCapacity(&mut self, i: u32) {
        self.len += i;
    }
}
