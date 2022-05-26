use super::qlib::rdma_svc_cli::*;

impl RDMASvcClient {
    pub fn wakeupSvc(&self) {
        super::Kernel::HostSpace::EventfdWrite(self.srvEventFd);
    }
}
