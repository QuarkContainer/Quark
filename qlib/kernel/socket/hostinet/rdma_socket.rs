use alloc::sync::Arc;

use super::super::super::super::common::*;
use super::super::super::super::linux_def::*;
use super::super::super::super::qmsg::qcall::*;
use super::super::super::super::socket_buf::*;
use super::super::super::task::*;
use super::super::super::Kernel::HostSpace;
//use super::super::super::kernel::waiter::*;

pub struct RDMA {}

impl RDMA {
    pub fn Accept(fd: i32, acceptQueue: &AcceptQueue) -> Result<AcceptItem> {
        let (trigger, ai) = acceptQueue.lock().DeqSocket();
        if trigger {
            HostSpace::RDMANotify(fd, RDMANotifyType::Accept);
        }

        return ai
    }

    pub fn Read(task: &Task, fd: i32, buf: Arc<SocketBuff>, dsts: &mut [IoVec]) -> Result<i64> {
        let (trigger, cnt) = buf.Readv(task, dsts)?;
        if !RDMA_ENABLE {
            if trigger {
                HostSpace::RDMANotify(fd, RDMANotifyType::Read);
            }
        } else {
            let dataSize = buf.AddConsumeReadData(cnt as u64) as usize;//buf.readBuf.lock().AvailableDataSize();
            let bufSize = buf.readBuf.lock().BufSize();
            if 2 * dataSize >= bufSize {
                HostSpace::RDMANotify(fd, RDMANotifyType::RDMARead);
            }
        }


        return Ok(cnt as i64)
    }

    //todo: put ops: &SocketOperations in the write request to make the socket won't be closed before write is finished
    pub fn Write(task: &Task, fd: i32, buf: Arc<SocketBuff>, srcs: &[IoVec]/*, ops: &SocketOperations*/) -> Result<i64> {
        //debug!("Write::1");
        let (count, writeBuf) = buf.Writev(task, srcs)?;
        //debug!("Write::2, count: {}", count);
        if writeBuf.is_some() {
            if RDMA_ENABLE {
                debug!("Write::3, count: {}", count);
                HostSpace::RDMANotify(fd, RDMANotifyType::RDMAWrite);
            } else {
                HostSpace::RDMANotify(fd, RDMANotifyType::Write);
            }
        }

        return Ok(count as i64)
    }
}