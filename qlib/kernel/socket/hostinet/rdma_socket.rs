use alloc::sync::Arc;

use crate::qlib::kernel::GlobalIOMgr;

use super::super::super::super::common::*;
use super::super::super::super::fileinfo::SockInfo;
use super::super::super::super::kernel::GlobalRDMASvcCli;
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

        return ai;
    }

    pub fn Read(task: &Task, fd: i32, buf: Arc<SocketBuff>, dsts: &mut [IoVec]) -> Result<i64> {
        // let (trigger, cnt) = buf.RDMAReadv(task, dsts)?;
        let (trigger, cnt) = buf.Readv(task, dsts)?;
        if !RDMA_ENABLE {
            if trigger {
                HostSpace::RDMANotify(fd, RDMANotifyType::Read);
            }
        } else {
            let dataSize = buf.AddConsumeReadData(cnt as u64) as usize;
            let bufSize = buf.readBuf.lock().BufSize();
            if 2 * dataSize >= bufSize {
                // HostSpace::RDMANotify(fd, RDMANotifyType::RDMARead);
                let fdInfo = GlobalIOMgr().GetByHost(fd).unwrap();
                let fdInfoLock = fdInfo.lock();
                let sockInfo = fdInfoLock.sockInfo.lock().clone();

                match sockInfo {
                    SockInfo::RDMADataSocket(rdmaDataScoket) => {
                        let _ret = GlobalRDMASvcCli().read(rdmaDataScoket.channelId);
                    }
                    _ => {
                        panic!("")
                    }
                }
            }
        }

        return Ok(cnt as i64);
    }

    //todo: put ops: &SocketOperations in the write request to make the socket won't be closed before write is finished
    pub fn Write(
        task: &Task,
        fd: i32,
        buf: Arc<SocketBuff>,
        srcs: &[IoVec], /*, ops: &SocketOperations*/
    ) -> Result<i64> {
        // let (count, writeBuf) = buf.RDMAWritev(task, srcs)?;
        let (count, writeBuf) = buf.Writev(task, srcs)?;
        if writeBuf.is_some() {
            if RDMA_ENABLE {
                // HostSpace::RDMANotify(fd, RDMANotifyType::RDMAWrite);
                let fdInfo = GlobalIOMgr().GetByHost(fd).unwrap();
                let fdInfoLock = fdInfo.lock();
                let sockInfo = fdInfoLock.sockInfo.lock().clone();

                match sockInfo {
                    SockInfo::RDMADataSocket(rdmaDataScoket) => {
                        let _ret = GlobalRDMASvcCli().write(rdmaDataScoket.channelId);
                    }
                    _ => {
                        panic!("")
                    }
                }
            } else {
                HostSpace::RDMANotify(fd, RDMANotifyType::Write);
            }
        }

        return Ok(count as i64);
    }
}
