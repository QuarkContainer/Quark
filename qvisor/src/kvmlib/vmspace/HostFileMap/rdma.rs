use rdmaffi;
use core::ops::Deref;
use spin::Mutex;
use std::ptr;
use std::convert::TryInto;

use super::super::super::qlib::common::*;

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


pub struct IBContext (pub * mut rdmaffi::ibv_context);

impl Drop for IBContext {
    fn drop(&mut self) {}
}

impl IBContext {
    pub fn New(deviceName: &str) -> Self {
        // look for device
        let mut deviceNumber = 0;
        let device_list = unsafe { rdmaffi::ibv_get_device_list(&mut deviceNumber as *mut _) };
        if device_list.is_null() {
            // TODO: clean up
            panic!("ibv_get_device_list failed: {}", errno::errno().0);
        }

        if deviceNumber == 0 {
            // TODO: clean up
            panic!("IB device is not found");
        }

        let devices = unsafe {
            use std::slice;
            slice::from_raw_parts_mut(device_list, deviceNumber as usize)
        };

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

        info!("ibv_open_device succeeded");
        /* We are now done with device list, free it */
        unsafe { rdmaffi::ibv_free_device_list(device_list) };

        return Self(context)
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

        return PortAttr(port_attr)
    }

    pub fn AllocProtectionDomain(&self) -> ProtectionDomain {
        let pd = unsafe { rdmaffi::ibv_alloc_pd(self.0) };
        if pd.is_null() {
            // TODO: cleanup
            panic!("ibv_alloc_pd failed\n");
        }

        return ProtectionDomain(pd)
    }

    pub fn CreateCompleteChannel(&self) -> CompleteChannel {
        let completionChannel = unsafe { rdmaffi::ibv_create_comp_channel(self.0) };
        if completionChannel.is_null() {
            // TODO: cleanup
            panic!("ibv_create_comp_channel failed\n");
        }
        return CompleteChannel(completionChannel)
    }

    pub fn CreateCompleteQueue(&self, cc: &CompleteChannel) -> CompleteQueue {
        let cq = unsafe {
            rdmaffi::ibv_create_cq(self.0, 1, ptr::null_mut(), cc.0, 0)
        };

        if cq.is_null() {
            // TODO: cleanup
            panic!("ibv_create_cq failed\n");
        }

        return CompleteQueue(cq)
    }

    pub fn QueryGid(&self, ibPort: u8) -> Gid {
        let mut gid = Gid::default();
        let ok = unsafe {
            rdmaffi::ibv_query_gid(self.0, ibPort, 0, gid.as_mut())
        };

        if ok != 0 {
            panic!("ibv_query_gid failed: {}\n", errno::errno().0);
        }

        return gid
    }
}

pub struct PortAttr(pub rdmaffi::ibv_port_attr);
impl Deref for PortAttr {
    type Target = rdmaffi::ibv_port_attr;

    fn deref(&self) -> &rdmaffi::ibv_port_attr {
        &self.0
    }
}

pub struct ProtectionDomain(pub * mut rdmaffi::ibv_pd);

impl Drop for ProtectionDomain {
    fn drop(&mut self) {}
}

pub struct CompleteChannel(pub * mut rdmaffi::ibv_comp_channel);
impl Drop for CompleteChannel {
    fn drop(&mut self) {}
}

pub struct CompleteQueue(pub * mut rdmaffi::ibv_cq);
impl Drop for CompleteQueue {
    fn drop(&mut self) {}
}



pub struct RDMAContextIntern {
    //device_attr: rdmaffi::ibv_device_attr,
    /* Device attributes */
    portAttr: PortAttr,     /* IB port attributes */
    ibContext: IBContext, /* device handle */
    protectDomain: ProtectionDomain, /* PD handle */
    completeChannel: CompleteChannel, /* io completion channel */
    completeQueue: CompleteQueue, /* CQ handle */
    ibPort: u8,
    gid: Gid,
}

impl RDMAContextIntern {
    pub fn New(deviceName: &str, ibPort: u8) -> Self {
        let ibContext = IBContext::New(deviceName);
        let portAttr = ibContext.QueryPort(ibPort);
        let protectDomain = ibContext.AllocProtectionDomain();
        let completeChannel = ibContext.CreateCompleteChannel();
        let completeQueue = ibContext.CreateCompleteQueue(&completeChannel);
        let gid = ibContext.QueryGid(ibPort);

        return Self {
            portAttr: portAttr,
            ibContext: ibContext,
            protectDomain: protectDomain,
            completeChannel: completeChannel,
            completeQueue: completeQueue,
            ibPort: ibPort,
            gid: gid
        }
    }
}

pub struct RDMAContext(Mutex<RDMAContextIntern>);

impl Deref for RDMAContext {
    type Target = Mutex<RDMAContextIntern>;

    fn deref(&self) -> &Mutex<RDMAContextIntern> {
        &self.0
    }
}

pub const MAX_SEND_WR: u32 = 1;
pub const MAX_RECV_WR: u32 = 1;
pub const MAX_SEND_SGE: u32 = 1;
pub const MAX_RECV_SGE: u32 = 1;

impl RDMAContext {
    pub fn CreateQueuePair(&self) -> Result<QueuePair> {
        let context = self.lock();
        //create queue pair
        let mut qp_init_attr = rdmaffi::ibv_qp_init_attr {
            // TODO: offset(0), may need find some different value
            qp_context: 0 as *mut _,
            send_cq: context.completeQueue.0 as *const _ as *mut _,
            recv_cq: context.completeQueue.0 as *const _ as *mut _,
            srq: ptr::null::<rdmaffi::ibv_srq>() as *mut _,
            cap: rdmaffi::ibv_qp_cap {
                max_send_wr: MAX_SEND_WR,
                max_recv_wr: MAX_RECV_WR,
                max_send_sge: MAX_SEND_SGE,
                max_recv_sge: MAX_RECV_SGE,
                max_inline_data: 0,
            },
            qp_type: rdmaffi::ibv_qp_type::IBV_QPT_RC,
            sq_sig_all: 0,
        };

        let qp = unsafe { rdmaffi::ibv_create_qp(context.protectDomain.0, &mut qp_init_attr as *mut _) };
        if qp.is_null() {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(QueuePair(qp))
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

        return Ok(MemoryRegion(mr))
    }

}
pub struct QueuePair(pub *mut rdmaffi::ibv_qp);

impl Drop for QueuePair {
    fn drop(&mut self) {}
}

impl QueuePair {
    pub fn ToInit(&self, context: &RDMAContext) -> Result<()> {
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
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.0, &mut attr, flags.0 as i32) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToRtr(&self, context: &RDMAContext, remote_qpn: u32, dlid: u16, dgid: Gid) -> Result<()> {
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
        attr.path_mtu = rdmaffi::ibv_mtu::IBV_MTU_256;
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
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.0, &mut attr, flags.0 as i32) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }

    pub fn ToRts(&self) -> Result<()> {
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
        let rc = unsafe { rdmaffi::ibv_modify_qp(self.0, &mut attr, flags.0 as i32) };
        if rc != 0 {
            return Err(Error::SysError(errno::errno().0));
        }

        return Ok(());
    }
}

pub struct MemoryRegion(pub *mut rdmaffi::ibv_mr);
impl Drop for MemoryRegion {
    fn drop(&mut self) {}
}


#[derive(Clone)]
pub struct CMConData {
    addr: u64,     /* Buffer address */
    rkey: u32,     /* Remote key */
    qp_num: u32,   /* QP number */
    lid: u16,      /* LID of the IB port */
    gid: [u8; 16], /* gid */
}
