// Copyright (c) 2021 Quark Container Authors / 2018 The gVisor Authors.
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

use crate::qlib::mutex::*;
use alloc::collections::btree_map::BTreeMap;
use alloc::string::ToString;
use alloc::sync::Arc;

use super::super::super::super::super::super::auth::*;
use super::super::super::super::super::super::common::*;
use super::super::super::super::super::super::linux_def::*;
use super::super::super::super::super::task::*;
use super::super::super::super::attr::*;
use super::super::super::super::dirent::*;
use super::super::super::super::file::*;
use super::super::super::super::flags::*;
use super::super::super::super::inode::*;
use super::super::super::super::mount::*;
use super::super::super::super::ramfs::dir::*;
use super::super::super::dir_proc::*;
use super::super::super::inode::*;

pub struct Ipv4Node {}

impl DirDataNode for Ipv4Node {
    fn Lookup(&self, d: &Dir, task: &Task, dir: &Inode, name: &str) -> Result<Dirent> {
        return d.Lookup(task, dir, name);
    }

    fn GetFile(
        &self,
        d: &Dir,
        task: &Task,
        dir: &Inode,
        dirent: &Dirent,
        flags: FileFlags,
    ) -> Result<File> {
        return d.GetFile(task, dir, dirent, flags);
    }
}

pub fn NewIpv4(task: &Task, msrc: &Arc<QMutex<MountSource>>) -> Inode {
    let mut contents = BTreeMap::new();
    contents.insert("tcp_sack".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("ip_forward".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("ip_local_port_range".to_string(), NewStaticProcInode(task, msrc, &Arc::new("32768	60999\n".as_bytes().to_vec())));
    contents.insert("ip_local_reserved_ports".to_string(), NewStaticProcInode(task, msrc, &Arc::new("\n".as_bytes().to_vec())));
    contents.insert("ipfrag_time".to_string(), NewStaticProcInode(task, msrc, &Arc::new("30\n".as_bytes().to_vec())));
    contents.insert("ip_nonlocal_bind".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("ip_no_pmtu_disc".to_string(), NewStaticProcInode(task, msrc, &Arc::new("1\n".as_bytes().to_vec())));
    contents.insert("tcp_allowed_congestion_control".to_string(), NewStaticProcInode(task, msrc, &Arc::new("\n".as_bytes().to_vec())));
    contents.insert("tcp_available_congestion_control".to_string(), NewStaticProcInode(task, msrc, &Arc::new("reno\n".as_bytes().to_vec())));
    contents.insert("tcp_congestion_control".to_string(), NewStaticProcInode(task, msrc, &Arc::new("reno\n".as_bytes().to_vec())));

    contents.insert("tcp_base_mss".to_string(), NewStaticProcInode(task, msrc, &Arc::new("1280\n".as_bytes().to_vec())));
    contents.insert("tcp_dsack".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_early_retrans".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_fack".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_fastopen".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_fastopen_key".to_string(), NewStaticProcInode(task, msrc, &Arc::new("\n".as_bytes().to_vec())));
    contents.insert("tcp_invalid_ratelimit".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_keepalive_intvl".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_keepalive_probes".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_keepalive_time".to_string(), NewStaticProcInode(task, msrc, &Arc::new("7200\n".as_bytes().to_vec())));
    contents.insert("tcp_mtu_probing".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_no_metrics_save".to_string(), NewStaticProcInode(task, msrc, &Arc::new("1\n".as_bytes().to_vec())));
    contents.insert("tcp_probe_interval".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_probe_threshold".to_string(), NewStaticProcInode(task, msrc, &Arc::new("0\n".as_bytes().to_vec())));
    contents.insert("tcp_retries1".to_string(), NewStaticProcInode(task, msrc, &Arc::new("3\n".as_bytes().to_vec())));
    contents.insert("tcp_retries2".to_string(), NewStaticProcInode(task, msrc, &Arc::new("15\n".as_bytes().to_vec())));
    contents.insert("tcp_rfc1337".to_string(), NewStaticProcInode(task, msrc, &Arc::new("1\n".as_bytes().to_vec())));
    contents.insert("tcp_slow_start_after_idle".to_string(), NewStaticProcInode(task, msrc, &Arc::new("1\n".as_bytes().to_vec())));
    contents.insert("tcp_synack_retries".to_string(), NewStaticProcInode(task, msrc, &Arc::new("5\n".as_bytes().to_vec())));
    contents.insert("tcp_syn_retries".to_string(), NewStaticProcInode(task, msrc, &Arc::new("3\n".as_bytes().to_vec())));
    contents.insert("tcp_timestamps".to_string(), NewStaticProcInode(task, msrc, &Arc::new("1\n".as_bytes().to_vec())));

    let ipv4Dir = DirNode {
        dir: Dir::New(
            task,
            contents,
            &ROOT_OWNER,
            &FilePermissions::FromMode(FileMode(0o0555)),
        ),
        data: Ipv4Node {},
    };

    return NewProcInode(&Arc::new(ipv4Dir), msrc, InodeType::SpecialDirectory, None);
}
