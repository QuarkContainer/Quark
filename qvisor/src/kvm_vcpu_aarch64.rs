use super::qlib::common::Result;
use crate::KVMVcpu;

impl KVMVcpu {
    pub fn run(&self, tgid: i32) -> Result<()> {
        Ok(())
    }

    pub fn dump(&self) -> Result<()> {
        Ok(())
    }

    pub fn Signal(&self, signal: i32) {
    }
}