use super::qlib::common::Result;
use crate::{KVMVcpu, kvm_vcpu::SetExitSignal};
use crate::qlib::singleton::Singleton;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Default)]
pub struct KVMVcpuInit {
    target: u32,
    features: [u32;7],
}

lazy_static! {
    pub static ref KVM_VCPU_INIT: Singleton<KVMVcpuInit> = Singleton::<KVMVcpuInit>::New();
}

impl KVMVcpu {
    pub fn run(&self, tgid: i32) -> Result<()> {
        SetExitSignal();
        Ok(())
    }

    pub fn dump(&self) -> Result<()> {
        Ok(())
    }

    pub fn Signal(&self, signal: i32) {}
    

    fn setup_long_mode(&self) -> Result<()> {
        Ok(())
    }
}
