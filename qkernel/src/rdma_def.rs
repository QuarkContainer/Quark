use super::qlib::linux_def::*;
use super::qlib::rdma_svc_cli::*;

impl RDMASvcClient {
    pub fn wakeupSvc(&self) {
        super::Kernel::HostSpace::EventfdWrite(self.srvEventFd);
    }

    pub fn CreateSocket(&self) -> i64 {
        super::Kernel::HostSpace::UnblockedSocket(AFType::AF_INET, 1, 0)
    }
}
