use core::ops::Deref;
use core::sync::atomic;
use core::sync::atomic::AtomicU64;
use rdmaffi;
use spin::Mutex;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::mem::MaybeUninit;
use std::ptr;

use super::qlib::common::*;
use super::qlib::linux_def::*;
use super::rdma_srv::RDMA_SRV;
// use super::qlib::kernel::TSC;
//use super::super::super::IO_MGR;

use lazy_static::lazy_static;

lazy_static! {
    pub static ref RDMA: RDMAContext = RDMAContext::default();
    static ref RDMAUID: AtomicU64 = AtomicU64::new(1);
}

pub fn NewUID() -> u64 {
    return RDMAUID.fetch_add(1, atomic::Ordering::SeqCst);
}

#[derive(Default, Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Gid {
    raw: [u8; 16],
}

impl Gid {
    /// Expose the subnet_prefix component of the `Gid` as a u64. This is
    /// equivalent to accessing the `global.subnet_prefix` component of the
    /// `rdmaffi::ibv_gid` union.
    #[allow(dead_code)]
    fn subnet_prefix(&self) -> u64 {
        u64::from_be_bytes(self.raw[..8].try_into().unwrap())
    }

    /// Expose the interface_id component of the `Gid` as a u64. This is
    /// equivalent to accessing the `global.interface_id` component of the
    /// `rdmaffi::ibv_gid` union.
    #[allow(dead_code)]
    fn interface_id(&self) -> u64 {
        u64::from_be_bytes(self.raw[8..].try_into().unwrap())
    }
}

impl From<rdmaffi::ibv_gid> for Gid {
    fn from(gid: rdmaffi::ibv_gid) -> Self {
        Self {
            raw: unsafe { gid.raw },
        }
    }
}

impl From<Gid> for rdmaffi::ibv_gid {
    fn from(mut gid: Gid) -> Self {
        *gid.as_mut()
    }
}

impl AsRef<rdmaffi::ibv_gid> for Gid {
    fn as_ref(&self) -> &rdmaffi::ibv_gid {
        unsafe { &*self.raw.as_ptr().cast::<rdmaffi::ibv_gid>() }
    }
}

impl AsMut<rdmaffi::ibv_gid> for Gid {
    fn as_mut(&mut self) -> &mut rdmaffi::ibv_gid {
        unsafe { &mut *self.raw.as_mut_ptr().cast::<rdmaffi::ibv_gid>() }
    }
}

pub struct IBContext(pub *mut rdmaffi::ibv_context);

impl Drop for IBContext {
    fn drop(&mut self) {
        if self.0 as u64 != 0 {
            // to close ibv_context
        }
    }
}

impl Default for IBContext {
    fn default() -> Self {
        return Self(0 as _);
    }
}

impl IBContext {
    pub fn New(deviceName: &str) -> Self {
        // look for device
        let mut deviceNumber = 0;
        // println!("IBContext::New, 1");
        let device_list = unsafe { rdmaffi::ibv_get_device_list(&mut deviceNumber as *mut _) };
        if device_list.is_null() {
            // TODO: clean up
            panic!("ibv_get_device_list failed: {}", errno::errno().0);
        }

        // println!("IBContext::New, deviceNumber: {}", deviceNumber);
        if deviceNumber == 0 {
            // TODO: clean up
            panic!("IB device is not found");
        }

        let devices = unsafe {
            use std::slice;
            slice::from_raw_parts_mut(device_list, deviceNumber as usize)
        };

        // println!("IBContext::New, len: {}", devices.len());

        let mut device = devices[0];

        if deviceName.len() != 0 {
            let mut found = false;

            for i in 0..devices.len() {
                let cur = unsafe { rdmaffi::ibv_get_device_name(devices[i]) };
                let cur = unsafe { std::ffi::CStr::from_ptr(cur) };
                let cur = cur.to_str().unwrap();
                if deviceName.eq(cur) {
                    device = devices[i];
                    found = true;
                    break;
                }
            }

            if !found {
                panic!("Could not found IB device with name: {}", deviceName);
            }
        }

        let context = unsafe { rdmaffi::ibv_open_device(device) };
        if context.is_null() {
            panic!("Failed to open IB device error");
        }

        // println!("ibv_open_device succeeded");
        /* We are now done with device list, free it */
        unsafe { rdmaffi::ibv_free_device_list(device_list) };

        return Self(context);
    }

    pub fn QueryPort(&self, ibPort: u8) -> PortAttr {
        let mut port_attr = rdmaffi::ibv_port_attr {
            state: rdmaffi::ibv_port_state::IBV_PORT_NOP,
            max_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            active_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            gid_tbl_len: 0,
            port_cap_flags: 0,
            max_msg_sz: 0,
            bad_pkey_cntr: 0,
            qkey_viol_cntr: 0,
            pkey_tbl_len: 0,
            lid: 0,
            sm_lid: 0,
            lmc: 0,
            max_vl_num: 0,
            sm_sl: 0,
            subnet_timeout: 0,
            init_type_reply: 0,
            active_width: 0,
            active_speed: 0,
            phys_state: 0,
            link_layer: 0,
            flags: 0,
            port_cap_flags2: 0,
        };

        /* query port properties */
        if unsafe { rdmaffi::___ibv_query_port(self.0, ibPort, &mut port_attr) } != 0 {
            // TODO: cleanup
            panic!("___ibv_query_port on port {} failed\n", ibPort);
        }

        return PortAttr(port_attr);
    }

    pub fn AllocProtectionDomain(&self) -> ProtectionDomain {
        let pd = unsafe { rdmaffi::ibv_alloc_pd(self.0) };
        if pd.is_null() {
            // TODO: cleanup
            panic!("ibv_alloc_pd failed\n");
        }

        return ProtectionDomain(pd);
    }

    pub fn CreateCompleteChannel(&self) -> CompleteChannel {
        let completionChannel = unsafe { rdmaffi::ibv_create_comp_channel(self.0) };
        if completionChannel.is_null() {
            // TODO: cleanup
            panic!("ibv_create_comp_channel failed\n");
        }
        return CompleteChannel(completionChannel);
    }

    pub fn CreateCompleteQueue(&self, cc: &CompleteChannel) -> CompleteQueue {
        let cq = unsafe { rdmaffi::ibv_create_cq(self.0, 2000, ptr::null_mut(), cc.0, 0) };

        if cq.is_null() {
            // TODO: cleanup
            panic!("ibv_create_cq failed\n");
        }

        unsafe {
            let _ret = rdmaffi::ibv_req_notify_cq(cq, 0);
            // println!("ibv_req_notify_cq, ret: {}", ret);
        }

        return CompleteQueue(cq);
    }

    pub fn QueryGid(&self, ibPort: u8) -> Gid {
        let mut gid = Gid::default();
        let ok = unsafe { rdmaffi::ibv_query_gid(self.0, ibPort, 0, gid.as_mut()) };

        if ok != 0 {
            panic!("ibv_query_gid failed: {}\n", errno::errno().0);
        }

        return gid;
    }
}

pub struct PortAttr(pub rdmaffi::ibv_port_attr);
impl Deref for PortAttr {
    type Target = rdmaffi::ibv_port_attr;

    fn deref(&self) -> &rdmaffi::ibv_port_attr {
        &self.0
    }
}

impl Default for PortAttr {
    fn default() -> Self {
        let attr = rdmaffi::ibv_port_attr {
            state: rdmaffi::ibv_port_state::IBV_PORT_NOP,
            max_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            active_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            gid_tbl_len: 0,
            port_cap_flags: 0,
            max_msg_sz: 0,
            bad_pkey_cntr: 0,
            qkey_viol_cntr: 0,
            pkey_tbl_len: 0,
            lid: 0,
            sm_lid: 0,
            lmc: 0,
            max_vl_num: 0,
            sm_sl: 0,
            subnet_timeout: 0,
            init_type_reply: 0,
            active_width: 0,
            active_speed: 0,
            phys_state: 0,
            link_layer: 0,
            flags: 0,
            port_cap_flags2: 0,
        };

        return Self(attr);
    }
}

pub struct ProtectionDomain(pub *mut rdmaffi::ibv_pd);

impl Drop for ProtectionDomain {
    fn drop(&mut self) {}
}

impl Default for ProtectionDomain {
    fn default() -> Self {
        return Self(0 as _);
    }
}

pub struct CompleteChannel(pub *mut rdmaffi::ibv_comp_channel);
impl Drop for CompleteChannel {
    fn drop(&mut self) {}
}

impl Default for CompleteChannel {
    fn default() -> Self {
        return Self(0 as _);
    }
}

pub struct CompleteQueue(pub *mut rdmaffi::ibv_cq);
impl Drop for CompleteQueue {
    fn drop(&mut self) {}
}

impl Default for CompleteQueue {
    fn default() -> Self {
        return Self(0 as _);
    }
}

#[derive(Default)]
pub struct RDMAContextIntern {
    //device_attr: rdmaffi::ibv_device_attr,
    /* Device attributes */
    portAttr: PortAttr,               /* IB port attributes */
    ibContext: IBContext,             /* device handle */
    protectDomain: ProtectionDomain,  /* PD handle */
    completeChannel: CompleteChannel, /* io completion channel */
    completeQueue: CompleteQueue,     /* CQ handle */
    ccfd: i32,                        // complete channel fd
    ibPort: u8,
    gid: Gid,
}

impl RDMAContextIntern {
    pub fn New(deviceName: &str, ibPort: u8) -> Self {
        let ibContext = IBContext::New(deviceName);
        let portAttr = ibContext.QueryPort(ibPort);
        let protectDomain = ibContext.AllocProtectionDomain();
        let completeChannel = ibContext.CreateCompleteChannel();
        let ccfd = unsafe { (*completeChannel.0).fd };

        // start to monitor the complete channel
        //IO_MGR.AddRDMAContext(ccfd);
        //IO_MGR.AddWait(ccfd, EVENT_READ);
        // println!("RDMA ccfd: {}", ccfd);

        let completeQueue = ibContext.CreateCompleteQueue(&completeChannel);
        let gid = ibContext.QueryGid(ibPort);

        // unblock complete channel fd
        //TODO: unblock fd
        //super::super::VMSpace::UnblockFd(ccfd);

        // unsafe {
        //     let fd = ccfd;
        //     let flags = libc::fcntl(fd, Cmd::F_GETFL, 0);
        //     let ret = libc::fcntl(fd, Cmd::F_SETFL, flags | Flags::O_NONBLOCK);
        //     assert!(ret == 0, "UnblockFd fail");
        // }

        return Self {
            portAttr: portAttr,
            ibContext: ibContext,
            protectDomain: protectDomain,
            completeChannel: completeChannel,
            ccfd: ccfd,
            completeQueue: completeQueue,
            ibPort: ibPort,
            gid: gid,
        };
    }
}

#[derive(Default)]
pub struct RDMAContext(Mutex<RDMAContextIntern>);

unsafe impl Send for RDMAContext {}
unsafe impl Sync for RDMAContext {}

impl Deref for RDMAContext {
    type Target = Mutex<RDMAContextIntern>;

    fn deref(&self) -> &Mutex<RDMAContextIntern> {
        &self.0
    }
}

pub const MAX_SEND_WR: u32 = 100;
pub const MAX_RECV_WR: u32 = 8192;
pub const MAX_SEND_SGE: u32 = 1;
pub const MAX_RECV_SGE: u32 = 1;

impl RDMAContext {
    pub fn Init(&self, deviceName: &str, ibPort: u8) {
        *self.0.lock() = RDMAContextIntern::New(deviceName, ibPort);
    }

    pub fn Lid(&self) -> u16 {
        let context = self.lock();
        return context.portAttr.0.lid;
    }

    pub fn Gid(&self) -> Gid {
        let context = self.lock();
        return context.gid;
    }

    pub fn CreateAddressHandler(&self, port_num: u8, lid: u16, gid: Gid) -> Result<AddressHandler> {
        let mut ah_attr = rdmaffi::ibv_ah_attr {
            grh: rdmaffi::ibv_global_route {
                dgid: rdmaffi::ibv_gid::from(gid),
                flow_label: 0,
                sgid_index: 0,
                hop_limit: 1,
                traffic_class: 0,
            },
            dlid: lid,
            sl: 0,
            src_path_bits: 0,
            static_rate: 0,
            is_global: 1,
            port_num,
        };

        // let mut ah_attr = unsafe { MaybeUninit::<rdmaffi::ibv_ah_attr>::zeroed().assume_init() };
        // ah_attr.grh.dgid = rdmaffi::ibv_gid::from(gid);
        // ah_attr.grh.hop_limit = 5;
        // ah_attr.is_global = 1;
        // ah_attr.port_num = port_num;
        let context = self.lock();
        let ah = unsafe { rdmaffi::ibv_create_ah(context.protectDomain.0, &mut ah_attr as *mut _) };
        if ah.is_null() {
            error!("CreateAddressHandler, errorno: {}", errno::errno().0);
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(AddressHandler(Mutex::new(ah)));
    }

    // Create Queue Pair
    pub fn CreateQueuePair(&self, qp_type: u32) -> Result<QueuePair> {
        // println!("CreateQueuePair 1");
        let context = self.lock();
        //create queue pair
        let mut qp_init_attr = rdmaffi::ibv_qp_init_attr {
            // TODO: offset(0), may need find some different value
            qp_context: 0 as *mut _,
            send_cq: context.completeQueue.0 as *const _ as *mut _,
            recv_cq: context.completeQueue.0 as *const _ as *mut _,
            srq: ptr::null::<rdmaffi::ibv_srq>() as *mut _,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 8192, //MAX_SEND_WR,
                max_recv_wr: 8192, //MAX_RECV_WR,
                max_send_sge: MAX_SEND_SGE,
                max_recv_sge: MAX_RECV_SGE,
                max_inline_data: 0,
            },
            qp_type,
            sq_sig_all: 0,
        };

        let qp =
            unsafe { rdmaffi::ibv_create_qp(context.protectDomain.0, &mut qp_init_attr as *mut _) };
        if qp.is_null() {
            // println!("errorno: {}", errno::errno().0);
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(QueuePair(Mutex::new(qp)));
    }

    pub fn CreateRCQueuePair(&self) -> Result<QueuePair> {
        self.CreateQueuePair(rdmaffi::ibv_qp_type::IBV_QPT_RC)
    }

    pub fn CreateUDQueuePair(&self) -> Result<QueuePair> {
        self.CreateQueuePair(rdmaffi::ibv_qp_type::IBV_QPT_UD)
    }

    pub fn CreateMemoryRegion(&self, addr: u64, size: usize) -> Result<MemoryRegion> {
        let context = self.lock();
        let access = rdmaffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE
            | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ
            | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_ATOMIC;

        let mr = unsafe {
            rdmaffi::ibv_reg_mr(
                context.protectDomain.0,
                addr as *mut _,
                size,
                access.0 as i32,
            )
        };

        if mr.is_null() {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(MemoryRegion(mr));
    }

    pub fn CompleteQueue(&self) -> *mut rdmaffi::ibv_cq {
        return self.lock().completeQueue.0;
    }

    pub fn CompleteChannel(&self) -> *mut rdmaffi::ibv_comp_channel {
        return self.lock().completeChannel.0;
    }

    pub fn CompleteChannelFd(&self) -> i32 {
        let fd = self.lock().ccfd;
        // println!("XXXX, fd: {} ", fd);
        return fd;
    }

    pub fn PollCompletionQueueAndProcess(
        &self,
        channels: &mut HashMap<u32, HashSet<u32>>,
    ) -> usize {
        // println!("PollCompletionQueueAndProcess");
        let mut wc = rdmaffi::ibv_wc {
            //TODO: find a better way to initialize
            wr_id: 0,
            status: rdmaffi::ibv_wc_status::IBV_WC_SUCCESS,
            opcode: rdmaffi::ibv_wc_opcode::IBV_WC_BIND_MW,
            vendor_err: 0,
            byte_len: 0,
            imm_data_invalidated_rkey_union: rdmaffi::imm_data_invalidated_rkey_union_t {
                imm_data: 0,
            }, //TODO: need double check
            qp_num: 0,
            src_qp: 0,
            wc_flags: 0,
            pkey_index: 0,
            slid: 0,
            sl: 0,
            dlid_path_bits: 0,
        };

        let mut count = 0;

        loop {
            // let poll_result = unsafe { rdmaffi::ibv_poll_cq(self.CompleteQueue(), 2, &mut wc) };
            // let wc_ptr: *const rdmaffi::ibv_wc = &wc;
            // if poll_result == 2 {
            //     count += 2;
            //     //self.ProcessWC(&wc);
            //     self.ProcessWC(unsafe {&(*wc_ptr.offset(0))});
            //     self.ProcessWC(unsafe {&(*wc_ptr.offset(1))});
            // } else if poll_result == 1 {
            //     // if count > 0 {
            //     //     error!("PollCompletionQueueAndProcess: processed wcs: {}", count);
            //     // }
            //     count += 1;
            //     self.ProcessWC(unsafe {&(*wc_ptr.offset(0))});
            //     return count;
            // } else if poll_result == 0 {
            //     return count;
            // } else {
            //     // debug!("Error to query CQ!")
            //     // break;
            // }

            let poll_result = unsafe { rdmaffi::ibv_poll_cq(self.CompleteQueue(), 1, &mut wc) };
            if poll_result == 1 {
                count += 1;
                self.ProcessWC(&wc, channels);
            } else if poll_result == 0 {
                // if count > 0 {
                //     error!("PollCompletionQueueAndProcess: processed wcs: {}", count);
                // }
                return count;
            } else {
                // debug!("Error to query CQ!")
                // break;
            }
        }
    }

    // pub fn HandleCQEvent(&self) -> Result<()> {
    //     Ok(())
    // }

    pub fn HandleCQEvent(&self) -> Result<()> {
        let mut cq_ptr: *mut rdmaffi::ibv_cq = ptr::null_mut();
        let mut cq_context: *mut std::os::raw::c_void = ptr::null_mut();
        let ret = unsafe {
            rdmaffi::ibv_get_cq_event(
                self.CompleteChannel(),
                &mut cq_ptr, //&mut self.CompleteQueue(),
                &mut cq_context,
            )
        };

        if ret != 0 {
            //// debug!("Failed to get next CQ event");
            return Ok(());
        }

        let ret1 = unsafe { rdmaffi::ibv_req_notify_cq(self.CompleteQueue(), 0) };
        if ret1 != 0 {
            // TODO: should keep call here?
        }

        unsafe { rdmaffi::ibv_ack_cq_events(cq_ptr, 1) };
        Ok(())
    }

    pub fn PollCompletion(&self) -> Result<()> {
        let mut wc = rdmaffi::ibv_wc {
            //TODO: find a better way to initialize
            wr_id: 0,
            status: rdmaffi::ibv_wc_status::IBV_WC_SUCCESS,
            opcode: rdmaffi::ibv_wc_opcode::IBV_WC_BIND_MW,
            vendor_err: 0,
            byte_len: 0,
            imm_data_invalidated_rkey_union: rdmaffi::imm_data_invalidated_rkey_union_t {
                imm_data: 0,
            }, //TODO: need double check
            qp_num: 0,
            src_qp: 0,
            wc_flags: 0,
            pkey_index: 0,
            slid: 0,
            sl: 0,
            dlid_path_bits: 0,
        };

        let mut cq_ptr: *mut rdmaffi::ibv_cq = ptr::null_mut();
        let mut cq_context: *mut std::os::raw::c_void = ptr::null_mut();
        let ret = unsafe {
            rdmaffi::ibv_get_cq_event(
                self.CompleteChannel(),
                &mut cq_ptr, //&mut self.CompleteQueue(),
                &mut cq_context,
            )
        };

        if ret != 0 {
            // debug!("Failed to get next CQ event");
        }

        let ret1 = unsafe { rdmaffi::ibv_req_notify_cq(self.CompleteQueue(), 0) };
        if ret1 != 0 {
            // TODO: should keep call here?
        }

        loop {
            let poll_result = unsafe { rdmaffi::ibv_poll_cq(self.CompleteQueue(), 1, &mut wc) };
            if poll_result > 0 {
                self.ProcessWC(&wc, &mut HashMap::new());
            } else if poll_result == 0 {
                break;
            } else {
                // debug!("Error to query CQ!")
                // break;
            }
        }

        unsafe { rdmaffi::ibv_ack_cq_events(cq_ptr, 1) };
        Ok(())

        // let ret1 = unsafe { rdmaffi::ibv_req_notify_cq(self.CompleteQueue(), 0) };
        // if ret1 != 0 {
        //     // TODO: should keep call here?
        //     // debug!("Couldn't request CQ notification\n");
        // }

        // loop {
        //     loop {
        //         let poll_result = unsafe { rdmaffi::ibv_poll_cq(self.CompleteQueue(), 1, &mut wc) };
        //         if poll_result > 0 {
        //             self.ProcessWC(&wc);
        //         } else if poll_result == 0 {
        //             break;
        //         } else {
        //             // debug!("Error to query CQ!")
        //             // break;
        //         }
        //     }

        //     let mut cq_ptr: *mut rdmaffi::ibv_cq = ptr::null_mut();
        //     let mut cq_context: *mut std::os::raw::c_void = ptr::null_mut();
        //     let ret = unsafe {
        //         rdmaffi::ibv_get_cq_event(
        //             self.CompleteChannel(),
        //             &mut cq_ptr, //&mut self.CompleteQueue(),
        //             &mut cq_context,
        //         )
        //     };

        //     let mut ret1 = unsafe { rdmaffi::ibv_req_notify_cq(self.CompleteQueue(), 0) };
        //     if ret1 != 0 {
        //         // TODO: should keep call here?
        //     }

        //     if ret == -1 {
        //         return Ok(());
        //     }
        //     //TODO: potnetial improvemnt to ack in batch
        //     unsafe { rdmaffi::ibv_ack_cq_events(cq_ptr, 1) };
        //     ret1 = unsafe { rdmaffi::ibv_req_notify_cq(self.CompleteQueue(), 0) };
        //     if ret1 != 0 {
        //         // TODO: should keep call here?
        //     }
        // }
    }

    // call back for
    pub fn ProcessWC(&self, wc: &rdmaffi::ibv_wc, channels: &mut HashMap<u32, HashSet<u32>>) {
        // println!("ProcessWC 1");
        let wrid = WorkRequestId(wc.wr_id);
        let _fd = wrid.Fd();

        // match typ {
        //     WorkRequestType::WriteImm => {
        //         // debug!("ProcessWC: WriteImm, opcode: {}", wc.opcode);
        //         IO_MGR.ProcessRDMAWriteImmFinish(fd);
        //     }
        //     WorkRequestType::Recv => {
        //         let imm = unsafe { wc.imm_data_invalidated_rkey_union.imm_data };
        //         let immData = ImmData(imm);
        //         // debug!("ProcessWC: readCount: {}, writeCount: {}, opcode: {}", immData.ReadCount(), immData.WriteCount(), wc.opcode);
        //         IO_MGR.ProcessRDMARecvWriteImm(
        //             fd,
        //             immData.WriteCount() as _,
        //             immData.ReadCount() as _,
        //         );
        //     }
        // }
        // println!(
        //     "ProcessWC: wrid: {}, qp_num: {}, op_code: {}, status: {}",
        //     wc.wr_id, wc.qp_num, wc.opcode, wc.status
        // );
        if wc.status != rdmaffi::ibv_wc_status::IBV_WC_SUCCESS {
            error!(
                "ProcessWC::1, work reqeust failed with status: {}, id: {}",
                wc.status, wc.wr_id
            );
        }
        if wc.opcode == rdmaffi::ibv_wc_opcode::IBV_WC_RDMA_WRITE {
            // debug!(
            //     "ProcessWC::2, writeIMM status: {}, id: {}",
            //     wc.status, wc.wr_id
            // );
            //IO_MGR.ProcessRDMAWriteImmFinish(fd);
            // channels.insert(wc.wr_id as u32);
            // let channelId = wc.wr_id as u32 & 0x7FFFFFFF;
            // if channelId != 0 {
            //     if channels.contains_key(&wc.qp_num) {
            //         channels.get_mut(&wc.qp_num).unwrap().insert(channelId);
            //     }
            //     else {
            //         channels.insert(wc.qp_num, vec![channelId].into_iter().collect());
            //     }
            // }
            // println!("qq1: IBV_WC_RDMA_WRITE");
            // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc());
            // let len = RDMA_SRV.timestamps.lock().len();
            // let mut i = 1;
            // let mut v1 = RDMA_SRV.timestamps.lock()[0];
            // let mut v2 = RDMA_SRV.timestamps.lock()[1];
            // println!("qq, Handle connect request time is: {}", RDMA_SRV.timestamps.lock()[len - 1] - v1);
            // loop {
            //     println!("{}", v2 - v1);
            //     i += 1;
            //     if i == len {
            //         break;
            //     }
            //     v1 = v2;
            //     v2 = RDMA_SRV.timestamps.lock()[i];            
            // }
            // RDMA_SRV.timestamps.lock().clear();

            RDMA_SRV.ProcessRDMAWriteImmFinish(wc.wr_id as u32, wc.qp_num);
        } else if wc.opcode == rdmaffi::ibv_wc_opcode::IBV_WC_RECV_RDMA_WITH_IMM {
            let imm = unsafe { wc.imm_data_invalidated_rkey_union.imm_data };
            // error!("ProcessWC. received len: {}, qpNum: {}", wc.byte_len, wc.qp_num);
            let immData = ImmData(imm);
            // debug!(
            //     "ProcessWC::2, recv len:{}, writelen: {}, status: {}, id: 0x{:x}",
            //     wc.byte_len,
            //     immData.ReadCount(),
            //     wc.status,
            //     wc.wr_id
            // );
            //IO_MGR.ProcessRDMARecvWriteImm(fd, wc.byte_len as _, immData.ReadCount() as _);
            // debug!("ProcessWC imm: {}", imm);
            // channels.insert(imm);
            let channelId = imm & 0x7FFFFFFF;
            if channelId != 0 {
                if channels.contains_key(&wc.qp_num) {
                    channels.get_mut(&wc.qp_num).unwrap().insert(channelId);
                } else {
                    channels.insert(wc.qp_num, vec![channelId].into_iter().collect());
                }
            }
            // println!("qq1: IBV_WC_RECV_RDMA_WITH_IMM");
            // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc());
            RDMA_SRV.ProcessRDMARecvWriteImm(immData.ReadCount() as _, wc.qp_num, wc.byte_len as _);
        } else if wc.opcode == rdmaffi::ibv_wc_opcode::IBV_WC_RECV {
            // error!("ProcessWC::3");
            RDMA_SRV.ProcessRDMARecv(wc.qp_num, wc.wr_id, wc.byte_len);
        }else if wc.opcode == rdmaffi::ibv_wc_opcode::IBV_WC_SEND {
            // error!("ProcessWC::4");
            RDMA_SRV.ProcessRDMASend(wc.wr_id);
        } else {
            // debug!("ProcessWC::5, opcode: {}, wr_id: {}", wc.opcode, wc.wr_id);
        }
    }
}

pub struct ImmData(pub u32);

impl ImmData {
    pub fn New(readCount: usize) -> Self {
        return Self(readCount as u32);
    }

    pub fn ReadCount(&self) -> u32 {
        return self.0;
    }

    // pub fn WriteCount(&self) -> u16 {
    //     return ((self.0 >> 16) & 0xffff) as u16;
    // }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
#[repr(u32)]
pub enum WorkRequestType {
    WriteImm,
    Recv,
}

pub struct WorkRequestId(pub u64);

impl WorkRequestId {
    pub fn New(fd: i32) -> Self {
        return Self(((fd as u64) << 32) | (NewUID() as u32 as u64));
    }

    pub fn Fd(&self) -> i32 {
        ((self.0 >> 32) & 0xffff_ffff) as i32
    }

    // pub fn Type(&self) -> WorkRequestType {
    //     let val = self.0 & 0xffff_ffff;
    //     if val == 0 {
    //         return WorkRequestType::WriteImm;
    //     } else {
    //         assert!(val == 1);
    //         return WorkRequestType::Recv;
    //     }
    // }
}

pub struct AddressHandler(pub Mutex<*mut rdmaffi::ibv_ah>);

impl Default for AddressHandler {
    fn default() -> Self {
        return Self(Mutex::new(0 as _));
    }
}

unsafe impl Send for AddressHandler {}
unsafe impl Sync for AddressHandler {}

impl Drop for AddressHandler {
    fn drop(&mut self) {}
}

impl AddressHandler {
    pub fn Data(&self) -> *mut rdmaffi::ibv_ah {
        return *self.0.lock();
    }
}

pub struct QueuePair(pub Mutex<*mut rdmaffi::ibv_qp>);

impl Default for QueuePair {
    fn default() -> Self {
        return Self(Mutex::new(0 as _));
    }
}

unsafe impl Send for QueuePair {}
unsafe impl Sync for QueuePair {}

impl Drop for QueuePair {
    fn drop(&mut self) {}
}

impl QueuePair {
    pub fn Data(&self) -> *mut rdmaffi::ibv_qp {
        return *self.0.lock();
    }

    pub fn qpNum(&self) -> u32 {
        return unsafe { (*self.Data()).qp_num };
    }

    pub fn WriteImm(
        &self,
        wrId: u64,
        laddr: u64,
        len: u32,
        lkey: u32,
        raddr: u64,
        rkey: u32,
        imm: u32,
    ) -> Result<()> {
        let opcode = rdmaffi::ibv_wr_opcode::IBV_WR_RDMA_WRITE_WITH_IMM;
        let mut sge = rdmaffi::ibv_sge {
            addr: laddr,
            length: len,
            lkey: lkey,
        };

        let mut sw = rdmaffi::ibv_send_wr {
            wr_id: wrId,
            next: ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
            opcode: opcode,
            send_flags: rdmaffi::ibv_send_flags::IBV_SEND_SIGNALED.0,
            imm_data_invalidated_rkey_union: rdmaffi::imm_data_invalidated_rkey_union_t {
                imm_data: imm,
            }, //TODO: need double check
            qp_type: rdmaffi::qp_type_t {
                xrc: rdmaffi::xrc_t { remote_srqn: 0 },
            },
            wr: rdmaffi::wr_t {
                rdma: rdmaffi::rdma_t {
                    //TODO: this is not needed when opcode is IBV_WR_SEND
                    remote_addr: raddr,
                    rkey: rkey,
                },
            },
            bind_mw_tso_union: rdmaffi::bind_mw_tso_union_t {
                //TODO: need a better init solution
                tso: rdmaffi::tso_t {
                    hdr: ptr::null_mut(),
                    hdr_sz: 0,
                    mss: 0,
                },
            },
        };

        let mut bad_wr: *mut rdmaffi::ibv_send_wr = ptr::null_mut();

        // println!("qq1: WriteImm, before post_send");
        // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc());
        let rc = unsafe { rdmaffi::ibv_post_send(self.Data(), &mut sw, &mut bad_wr) };
        // println!("qq1: WriteImm, after post_send");
        // RDMA_SRV.timestamps.lock().push(TSC.Rdtsc());
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }
        return Ok(());
    }

    pub fn PostRecv(&self, wrId: u64, addr: u64, lkey: u32, length: u32) -> Result<()> {
        let mut sge = rdmaffi::ibv_sge { addr, length, lkey };
        let mut rw = rdmaffi::ibv_recv_wr {
            wr_id: wrId,
            next: ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
        };
        let mut bad_wr: *mut rdmaffi::ibv_recv_wr = ptr::null_mut();
        let rc = unsafe { rdmaffi::ibv_post_recv(self.Data(), &mut rw, &mut bad_wr) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        // println!("QP::PostRecv");

        return Ok(());
    }

    pub fn PostSendUDQP(
        &self,
        ah: &AddressHandler,
        remote_qpn: u32,
        wrId: u64,
        laddr: u64,
        len: u32,
        lkey: u32,
    ) -> Result<()> {
        // error!(
        //     "PostSendUDQP, remote_qpn: {}, wrId: {}, laddr: 0x{:x}, len: {}, lkey: {}",
        //     remote_qpn, wrId, laddr, len, lkey
        // );
        let opcode = rdmaffi::ibv_wr_opcode::IBV_WR_SEND;
        let mut sge = rdmaffi::ibv_sge {
            addr: laddr,
            length: len,
            lkey: lkey,
        };

        //TODO: delete!
        let mut sw = rdmaffi::ibv_send_wr {
            wr_id: wrId,
            next: ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
            opcode: opcode,
            send_flags: rdmaffi::ibv_send_flags::IBV_SEND_SIGNALED.0,
            imm_data_invalidated_rkey_union: rdmaffi::imm_data_invalidated_rkey_union_t {
                imm_data: 0,
            },
            qp_type: rdmaffi::qp_type_t {
                xrc: rdmaffi::xrc_t { remote_srqn: 0 },
            },
            wr: rdmaffi::wr_t {
                ud: rdmaffi::ud_t {
                    ah: ah.Data(),
                    remote_qpn,
                    remote_qkey: 0x11111111,
                },
            },
            bind_mw_tso_union: rdmaffi::bind_mw_tso_union_t {
                tso: rdmaffi::tso_t {
                    hdr: ptr::null_mut(),
                    hdr_sz: 0,
                    mss: 0,
                },
            },
        };

        let mut bad_wr: *mut rdmaffi::ibv_send_wr = ptr::null_mut();

        let rc = unsafe { rdmaffi::ibv_post_send(self.Data(), &mut sw, &mut bad_wr) };
        if rc != 0 {
            error!("PostSendUDQP, rc: {}", rc);
            return Err(Error::SysError(errno::errno().0));
        }
        return Ok(());
    }

    pub fn SetupRCQP(
        &self,
        context: &RDMAContext,
        remote_qpn: u32,
        dlid: u16,
        dgid: Gid,
    ) -> Result<()> {
        self.ToInitRCQP(context)?;
        self.ToRtrRCQP(context, remote_qpn, dlid, dgid)?;
        self.ToRtsRCQP()?;
        return Ok(());
    }

    pub fn SetupUDQP(&self, context: &RDMAContext) -> Result<()> {
        self.ToInitUDQP(context)?;
        self.ToRtrUDQP(context)?;
        self.ToRtsUDQP()?;
        return Ok(());
    }

    pub fn ToInitRCQP(&self, context: &RDMAContext) -> Result<()> {
        let mut attr = rdmaffi::ibv_qp_attr {
            qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            cur_qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            path_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            path_mig_state: rdmaffi::ibv_mig_state::IBV_MIG_ARMED,
            qkey: 0,
            rq_psn: 0,
            sq_psn: 0,
            dest_qp_num: 0,
            qp_access_flags: 0,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 0,
                max_recv_wr: 0,
                max_send_sge: 0,
                max_recv_sge: 0,
                max_inline_data: 0,
            },
            ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            alt_ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            pkey_index: 0,
            alt_pkey_index: 0,
            en_sqd_async_notify: 0,
            sq_draining: 0,
            max_rd_atomic: 0,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 0,
            port_num: 0,
            timeout: 0,
            retry_cnt: 0,
            rnr_retry: 0,
            alt_port_num: 0,
            alt_timeout: 0,
            rate_limit: 0,
        };

        attr.qp_state = rdmaffi::ibv_qp_state::IBV_QPS_INIT;
        attr.port_num = context.lock().ibPort;
        attr.pkey_index = 0;
        let qp_access_flags = rdmaffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
            | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ
            | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE;
        attr.qp_access_flags = qp_access_flags.0;
        let flags = rdmaffi::ibv_qp_attr_mask::IBV_QP_STATE
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_PORT
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.Data(), &mut attr, flags.0 as i32) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToInitUDQP(&self, context: &RDMAContext) -> Result<()> {
        let mut attr = rdmaffi::ibv_qp_attr {
            qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            cur_qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            path_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            path_mig_state: rdmaffi::ibv_mig_state::IBV_MIG_ARMED,
            qkey: 0x11111111,
            rq_psn: 0,
            sq_psn: 0,
            dest_qp_num: 0,
            qp_access_flags: 0,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 0,
                max_recv_wr: 0,
                max_send_sge: 0,
                max_recv_sge: 0,
                max_inline_data: 0,
            },
            ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            alt_ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            pkey_index: 0,
            alt_pkey_index: 0,
            en_sqd_async_notify: 0,
            sq_draining: 0,
            max_rd_atomic: 0,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 0,
            port_num: 0,
            timeout: 0,
            retry_cnt: 0,
            rnr_retry: 0,
            alt_port_num: 0,
            alt_timeout: 0,
            rate_limit: 0,
        };

        attr.qp_state = rdmaffi::ibv_qp_state::IBV_QPS_INIT;
        attr.port_num = context.lock().ibPort;
        attr.pkey_index = 0;
        // let qp_access_flags = rdmaffi::ibv_access_flags::IBV_ACCESS_LOCAL_WRITE
        //     | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_READ
        //     | rdmaffi::ibv_access_flags::IBV_ACCESS_REMOTE_WRITE;
        // attr.qp_access_flags = qp_access_flags.0;
        let flags = rdmaffi::ibv_qp_attr_mask::IBV_QP_STATE
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_PKEY_INDEX
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_PORT
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_QKEY;
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS;
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.Data(), &mut attr, flags.0 as i32) };
        if rc != 0 {
            error!("ToInitUDQP, rc: {}", rc);
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToRtrRCQP(
        &self,
        context: &RDMAContext,
        remote_qpn: u32,
        dlid: u16,
        dgid: Gid,
    ) -> Result<()> {
        let mut attr = rdmaffi::ibv_qp_attr {
            qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            cur_qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            path_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            path_mig_state: rdmaffi::ibv_mig_state::IBV_MIG_ARMED,
            qkey: 0,
            rq_psn: 0,
            sq_psn: 0,
            dest_qp_num: 0,
            qp_access_flags: 0,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 0,
                max_recv_wr: 0,
                max_send_sge: 0,
                max_recv_sge: 0,
                max_inline_data: 0,
            },
            ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            alt_ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            pkey_index: 0,
            alt_pkey_index: 0,
            en_sqd_async_notify: 0,
            sq_draining: 0,
            max_rd_atomic: 0,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 0,
            port_num: 0,
            timeout: 0,
            retry_cnt: 0,
            rnr_retry: 0,
            alt_port_num: 0,
            alt_timeout: 0,
            rate_limit: 0,
        };

        attr.qp_state = rdmaffi::ibv_qp_state::IBV_QPS_RTR;
        attr.path_mtu = rdmaffi::ibv_mtu::IBV_MTU_4096;
        attr.dest_qp_num = remote_qpn;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 0x12;
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = dlid;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = context.lock().ibPort;
        let gid_idx = 0;

        // todo: configure with Qingqu
        //if gid_idx >= 0 {
        {
            attr.ah_attr.is_global = 1;
            attr.ah_attr.port_num = 1;
            // memcpy (&attr.ah_attr.grh.dgid, dgid, 16);
            attr.ah_attr.grh.dgid = rdmaffi::ibv_gid::from(dgid);
            attr.ah_attr.grh.flow_label = 0;
            attr.ah_attr.grh.hop_limit = 1;
            attr.ah_attr.grh.sgid_index = gid_idx;
            attr.ah_attr.grh.traffic_class = 0;
        }

        let flags = rdmaffi::ibv_qp_attr_mask::IBV_QP_STATE
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_AV
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_PATH_MTU
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_DEST_QPN
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_RQ_PSN
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.Data(), &mut attr, flags.0 as i32) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToRtrUDQP(&self, context: &RDMAContext) -> Result<()> {
        let mut attr = rdmaffi::ibv_qp_attr {
            qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            cur_qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            path_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            path_mig_state: rdmaffi::ibv_mig_state::IBV_MIG_ARMED,
            qkey: 0,
            rq_psn: 0,
            sq_psn: 0,
            dest_qp_num: 0,
            qp_access_flags: 0,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 0,
                max_recv_wr: 0,
                max_send_sge: 0,
                max_recv_sge: 0,
                max_inline_data: 0,
            },
            ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            alt_ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            pkey_index: 0,
            alt_pkey_index: 0,
            en_sqd_async_notify: 0,
            sq_draining: 0,
            max_rd_atomic: 0,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 0,
            port_num: 0,
            timeout: 0,
            retry_cnt: 0,
            rnr_retry: 0,
            alt_port_num: 0,
            alt_timeout: 0,
            rate_limit: 0,
        };

        attr.qp_state = rdmaffi::ibv_qp_state::IBV_QPS_RTR;
        attr.path_mtu = rdmaffi::ibv_mtu::IBV_MTU_4096;
        attr.dest_qp_num = 0;
        attr.rq_psn = 0;
        attr.max_dest_rd_atomic = 1;
        attr.min_rnr_timer = 0x12;
        attr.ah_attr.is_global = 0;
        attr.ah_attr.dlid = 0;
        attr.ah_attr.sl = 0;
        attr.ah_attr.src_path_bits = 0;
        attr.ah_attr.port_num = context.lock().ibPort;
        // let gid_idx = 0;

        // todo: configure with Qingqu
        //if gid_idx >= 0 {
        // {
        //     attr.ah_attr.is_global = 1;
        //     attr.ah_attr.port_num = 1;
        //     // memcpy (&attr.ah_attr.grh.dgid, dgid, 16);
        //     attr.ah_attr.grh.dgid = rdmaffi::ibv_gid::from(dgid);
        //     attr.ah_attr.grh.flow_label = 0;
        //     attr.ah_attr.grh.hop_limit = 1;
        //     attr.ah_attr.grh.sgid_index = gid_idx;
        //     attr.ah_attr.grh.traffic_class = 0;
        // }

        let flags = rdmaffi::ibv_qp_attr_mask::IBV_QP_STATE;
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_AV
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_PATH_MTU
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_DEST_QPN
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_RQ_PSN
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC
        // | rdmaffi::ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER;
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.Data(), &mut attr, flags.0 as i32) };
        if rc != 0 {
            error!("ToRtrUDQP, rc: {}", rc);
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToRtsRCQP(&self) -> Result<()> {
        let mut attr = rdmaffi::ibv_qp_attr {
            qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            cur_qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            path_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            path_mig_state: rdmaffi::ibv_mig_state::IBV_MIG_ARMED,
            qkey: 0,
            rq_psn: 0,
            sq_psn: 0,
            dest_qp_num: 0,
            qp_access_flags: 0,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 0,
                max_recv_wr: 0,
                max_send_sge: 0,
                max_recv_sge: 0,
                max_inline_data: 0,
            },
            ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            alt_ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            pkey_index: 0,
            alt_pkey_index: 0,
            en_sqd_async_notify: 0,
            sq_draining: 0,
            max_rd_atomic: 0,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 0,
            port_num: 0,
            timeout: 0,
            retry_cnt: 0,
            rnr_retry: 0,
            alt_port_num: 0,
            alt_timeout: 0,
            rate_limit: 0,
        };

        attr.qp_state = rdmaffi::ibv_qp_state::IBV_QPS_RTS;
        attr.timeout = 0x12;
        attr.retry_cnt = 6;
        attr.rnr_retry = 0;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        let flags = rdmaffi::ibv_qp_attr_mask::IBV_QP_STATE
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_TIMEOUT
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_RETRY_CNT
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_RNR_RETRY
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_SQ_PSN
            | rdmaffi::ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC;
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.Data(), &mut attr, flags.0 as i32) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToRtsUDQP(&self) -> Result<()> {
        let mut attr = rdmaffi::ibv_qp_attr {
            qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            cur_qp_state: rdmaffi::ibv_qp_state::IBV_QPS_INIT,
            path_mtu: rdmaffi::ibv_mtu::IBV_MTU_1024,
            path_mig_state: rdmaffi::ibv_mig_state::IBV_MIG_ARMED,
            qkey: 0,
            rq_psn: 0,
            sq_psn: 0,
            dest_qp_num: 0,
            qp_access_flags: 0,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: 0,
                max_recv_wr: 0,
                max_send_sge: 0,
                max_recv_sge: 0,
                max_inline_data: 0,
            },
            ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            alt_ah_attr: rdmaffi::ibv_ah_attr {
                grh: rdmaffi::ibv_global_route {
                    dgid: *Gid::default().as_mut(), //TODO: need recheck
                    flow_label: 0,
                    sgid_index: 0,
                    hop_limit: 0,
                    traffic_class: 0,
                },
                dlid: 0,
                sl: 0,
                src_path_bits: 0,
                static_rate: 0,
                is_global: 0,
                port_num: 0,
            },
            pkey_index: 0,
            alt_pkey_index: 0,
            en_sqd_async_notify: 0,
            sq_draining: 0,
            max_rd_atomic: 0,
            max_dest_rd_atomic: 0,
            min_rnr_timer: 0,
            port_num: 0,
            timeout: 0,
            retry_cnt: 0,
            rnr_retry: 0,
            alt_port_num: 0,
            alt_timeout: 0,
            rate_limit: 0,
        };

        attr.qp_state = rdmaffi::ibv_qp_state::IBV_QPS_RTS;
        attr.timeout = 0x12;
        attr.retry_cnt = 6;
        attr.rnr_retry = 0;
        attr.sq_psn = 0;
        attr.max_rd_atomic = 1;
        let flags =
            rdmaffi::ibv_qp_attr_mask::IBV_QP_STATE | rdmaffi::ibv_qp_attr_mask::IBV_QP_SQ_PSN;

        let rc = unsafe { rdmaffi::ibv_modify_qp(self.Data(), &mut attr, flags.0 as i32) };
        if rc != 0 {
            error!("ToRtsUDQP, rc: {}", rc);
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }
}

pub struct MemoryRegion(pub *mut rdmaffi::ibv_mr);
impl Drop for MemoryRegion {
    fn drop(&mut self) {
        unsafe {
            if self.0 as *const _ as u64 != 0 {
                let _ret = rdmaffi::ibv_dereg_mr(self.0);
            }
        }
    }
}

impl Default for MemoryRegion {
    fn default() -> Self {
        return Self(0 as _);
    }
}

impl MemoryRegion {
    pub fn LKey(&self) -> u32 {
        return unsafe { (*self.0).lkey };
    }

    pub fn RKey(&self) -> u32 {
        return unsafe { (*self.0).rkey };
    }
}

unsafe impl Send for MemoryRegion {}
unsafe impl Sync for MemoryRegion {}
