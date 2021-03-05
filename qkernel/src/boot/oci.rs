// Copyright (c) 2021 QuarkSoft LLC
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

#![allow(non_camel_case_types)]

use alloc::string::String;
use alloc::collections::btree_map::BTreeMap;
use alloc::vec::Vec;

fn is_false(b: &bool) -> bool {
    !b
}

#[derive(Debug)]
pub struct Platform {
    pub os: String,
    pub arch: String,
}

#[derive(Default, PartialEq, Debug)]
pub struct Box {
    pub height: u64,
    pub width: u64,
}

fn is_default<T: Default + PartialEq>(b: &T) -> bool {
    *b == T::default()
}

#[derive(Debug)]
pub struct User {
    pub uid: u32,
    pub gid: u32,
    pub additional_gids: Vec<u32>,
    pub username: String,
}

// this converts directly to the correct int
#[derive(Debug, Clone, Copy)]
pub enum LinuxRlimitType {
    RLIMIT_CPU,
    // CPU time in sec
    RLIMIT_FSIZE,
    // Maximum filesize
    RLIMIT_DATA,
    // max data size
    RLIMIT_STACK,
    // max stack size
    RLIMIT_CORE,
    // max core file size
    RLIMIT_RSS,
    // max resident set size
    RLIMIT_NPROC,
    // max number of processes
    RLIMIT_NOFILE,
    // max number of open files
    RLIMIT_MEMLOCK,
    // max locked-in-memory address space
    RLIMIT_AS,
    // address space limit
    RLIMIT_LOCKS,
    // maximum file locks held
    RLIMIT_SIGPENDING,
    // max number of pending signals
    RLIMIT_MSGQUEUE,
    // maximum bytes in POSIX mqueues
    RLIMIT_NICE,
    // max nice prio allowed to raise to
    RLIMIT_RTPRIO,
    // maximum realtime priority
    RLIMIT_RTTIME,
    // timeout for RT tasks in us
}

#[derive(Debug)]
pub struct LinuxRlimit {
    pub typ: LinuxRlimitType,
    pub hard: u64,
    pub soft: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum LinuxCapabilityType {
    CAP_CHOWN,
    CAP_DAC_OVERRIDE,
    CAP_DAC_READ_SEARCH,
    CAP_FOWNER,
    CAP_FSETID,
    CAP_KILL,
    CAP_SETGID,
    CAP_SETUID,
    CAP_SETPCAP,
    CAP_LINUX_IMMUTABLE,
    CAP_NET_BIND_SERVICE,
    CAP_NET_BROADCAST,
    CAP_NET_ADMIN,
    CAP_NET_RAW,
    CAP_IPC_LOCK,
    CAP_IPC_OWNER,
    CAP_SYS_MODULE,
    CAP_SYS_RAWIO,
    CAP_SYS_CHROOT,
    CAP_SYS_PTRACE,
    CAP_SYS_PACCT,
    CAP_SYS_ADMIN,
    CAP_SYS_BOOT,
    CAP_SYS_NICE,
    CAP_SYS_RESOURCE,
    CAP_SYS_TIME,
    CAP_SYS_TTY_CONFIG,
    CAP_MKNOD,
    CAP_LEASE,
    CAP_AUDIT_WRITE,
    CAP_AUDIT_CONTROL,
    CAP_SETFCAP,
    CAP_MAC_OVERRIDE,
    CAP_MAC_ADMIN,
    CAP_SYSLOG,
    CAP_WAKE_ALARM,
    CAP_BLOCK_SUSPEND,
    CAP_AUDIT_READ,
}

#[derive(Debug)]
pub struct LinuxCapabilities {
    pub bounding: Vec<LinuxCapabilityType>,
    pub effective: Vec<LinuxCapabilityType>,
    pub inheritable: Vec<LinuxCapabilityType>,
    pub permitted: Vec<LinuxCapabilityType>,
    pub ambient: Vec<LinuxCapabilityType>,
}

#[derive(Debug)]
pub struct Process {
    pub terminal: bool,
    pub console_size: Box,
    pub user: User,
    pub args: Vec<String>,
    pub env: Vec<String>,
    pub cwd: String,
    pub capabilities: Option<LinuxCapabilities>,
    pub rlimits: Vec<LinuxRlimit>,
    pub no_new_privileges: bool,
    pub apparmor_profile: String,
    pub selinux_label: String,
}

#[derive(Debug)]
pub struct Root {
    pub path: String,
    pub readonly: bool,
}

#[derive(Debug, Clone)]
pub struct Mount {
    pub destination: String,
    pub typ: String,
    pub source: String,
    pub options: Vec<String>,
}

#[derive(Debug)]
pub struct Hook {
    pub path: String,
    pub args: Vec<String>,
    pub env: Vec<String>,
    pub timeout: Option<i64>,
}

#[derive(Debug)]
pub struct Hooks {
    pub prestart: Vec<Hook>,
    pub poststart: Vec<Hook>,
    pub poststop: Vec<Hook>,
}

#[derive(Debug, Clone)]
pub struct LinuxIDMapping {
    pub host_id: u32,
    pub container_id: u32,
    pub size: u32,
}

// a is for LinuxDeviceCgroup
#[derive(Debug, Clone, Copy)]
pub enum LinuxDeviceType {
    b,
    c,
    u,
    p,
    a,
}

impl Default for LinuxDeviceType {
    fn default() -> LinuxDeviceType {
        LinuxDeviceType::a
    }
}

#[derive(Debug)]
pub struct LinuxDeviceCgroup {
    pub allow: bool,
    pub typ: LinuxDeviceType,
    pub major: Option<i64>,
    pub minor: Option<i64>,
    pub access: String,
}

#[derive(Debug)]
pub struct LinuxMemory {
    pub limit: Option<i64>,
    pub reservation: Option<i64>,
    pub swap: Option<i64>,
    pub kernel: Option<i64>,
    pub kernel_tcp: Option<i64>,
    pub swappiness: Option<u64>,
}

#[derive(Debug)]
pub struct LinuxCPU {
    pub shares: Option<u64>,
    pub quota: Option<i64>,
    pub period: Option<u64>,
    pub realtime_runtime: Option<i64>,
    pub realtime_period: Option<u64>,
    pub cpus: String,
    pub mems: String,
}

#[derive(Debug)]
pub struct LinuxPids {
    pub limit: i64,
}

#[derive(Debug)]
pub struct LinuxWeightDevice {
    pub major: i64,
    pub minor: i64,
    pub weight: Option<u16>,
    pub leaf_weight: Option<u16>,
}

#[derive(Debug)]
pub struct LinuxThrottleDevice {
    pub major: i64,
    pub minor: i64,
    pub rate: u64,
}

#[derive(Debug)]
pub struct LinuxBlockIO {
    pub weight: Option<u16>,
    pub leaf_weight: Option<u16>,
    pub weight_device: Vec<LinuxWeightDevice>,
    pub throttle_read_bps_device: Vec<LinuxThrottleDevice>,
    pub throttle_write_bps_device: Vec<LinuxThrottleDevice>,
    pub throttle_read_iops_device: Vec<LinuxThrottleDevice>,
    pub throttle_write_iops_device: Vec<LinuxThrottleDevice>,
}

#[derive(Debug)]
pub struct LinuxHugepageLimit {
    pub page_size: String,
    pub limit: i64,
}


#[derive(Debug)]
pub struct LinuxInterfacePriority {
    pub name: String,
    pub priority: u32,
}

#[derive(Debug)]
pub struct LinuxNetwork {
    pub class_id: Option<u32>,
    pub priorities: Vec<LinuxInterfacePriority>,
}

#[derive(Default, Debug)]
pub struct LinuxResources {
    pub devices: Vec<LinuxDeviceCgroup>,
    // NOTE: spec uses a pointer here, so perhaps this should be an Option, but
    //       false == unset so we don't bother.
    pub disable_oom_killer: bool,
    // NOTE: spec refers to this as an isize but the range is -1000 to 1000, so
    //       an i32 seems just fine
    pub oom_score_adj: Option<i32>,
    pub memory: Option<LinuxMemory>,
    pub cpu: Option<LinuxCPU>,
    pub pids: Option<LinuxPids>,
    pub block_io: Option<LinuxBlockIO>,
    pub hugepage_limits: Vec<LinuxHugepageLimit>,
    pub network: Option<LinuxNetwork>,
}

#[derive(Debug, Clone, Copy)]
pub enum LinuxNamespaceType {
    mount = 0x00020000,
    /* New mount namespace group */
    cgroup = 0x02000000,
    /* New cgroup namespace */
    uts = 0x04000000,
    /* New utsname namespace */
    ipc = 0x08000000,
    /* New ipc namespace */
    user = 0x10000000,
    /* New user namespace */
    pid = 0x20000000,
    /* New pid namespace */
    network = 0x40000000,
    /* New network namespace */
}

#[derive(Debug)]
pub struct LinuxNamespace {
    pub typ: LinuxNamespaceType,
    pub path: String,
}

#[derive(Debug)]
pub struct LinuxDevice {
    pub path: String,
    pub typ: LinuxDeviceType,
    pub major: u64,
    pub minor: u64,
    pub file_mode: Option<u32>,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum LinuxSeccompAction {
    SCMP_ACT_KILL = 0x00000000,
    SCMP_ACT_TRAP = 0x00030000,
    SCMP_ACT_ERRNO = 0x00050001,
    /* ERRNO + EPERM */
    SCMP_ACT_TRACE = 0x7ff00001,
    /* TRACE + EPERM */
    SCMP_ACT_ALLOW = 0x7fff0000,
}

#[derive(Debug, Clone, Copy)]
pub enum Arch {
    SCMP_ARCH_NATIVE = 0x00000000,
    SCMP_ARCH_X86 = 0x40000003,
    SCMP_ARCH_X86_64 = 0xc000003e,
    SCMP_ARCH_X32 = 0x4000003e,
    SCMP_ARCH_ARM = 0x40000028,
    SCMP_ARCH_AARCH64 = 0xc00000b7,
    SCMP_ARCH_MIPS = 0x00000008,
    SCMP_ARCH_MIPS64 = 0x80000008,
    SCMP_ARCH_MIPS64N32 = 0xa0000008,
    SCMP_ARCH_MIPSEL = 0x40000008,
    SCMP_ARCH_MIPSEL64 = 0xc0000008,
    SCMP_ARCH_MIPSEL64N32 = 0xe0000008,
    SCMP_ARCH_PPC = 0x00000014,
    SCMP_ARCH_PPC64 = 0x80000015,
    SCMP_ARCH_PPC64LE = 0xc0000015,
    SCMP_ARCH_S390 = 0x00000016,
    SCMP_ARCH_S390X = 0x80000016,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum LinuxSeccompOperator {
    SCMP_CMP_NE = 1,
    /* not equal */
    SCMP_CMP_LT = 2,
    /* less than */
    SCMP_CMP_LE = 3,
    /* less than or equal */
    SCMP_CMP_EQ = 4,
    /* equal */
    SCMP_CMP_GE = 5,
    /* greater than or equal */
    SCMP_CMP_GT = 6,
    /* greater than */
    SCMP_CMP_MASKED_EQ = 7,
    /* masked equality */
}

#[derive(Debug)]
pub struct LinuxSeccompArg {
    pub index: usize,
    pub value: u64,
    pub value_two: u64,
    pub op: LinuxSeccompOperator,
}

#[derive(Debug)]
pub struct LinuxSyscall {
    // old version used name
    pub name: String,
    pub names: Vec<String>,
    pub action: LinuxSeccompAction,
    pub args: Vec<LinuxSeccompArg>,
}

#[derive(Debug)]
pub struct LinuxSeccomp {
    pub default_action: LinuxSeccompAction,
    pub architectures: Vec<Arch>,
    pub syscalls: Vec<LinuxSyscall>,
}

#[derive(Debug)]
pub struct Linux {
    pub uid_mappings: Vec<LinuxIDMapping>,
    pub gid_mappings: Vec<LinuxIDMapping>,
    pub sysctl: BTreeMap<String, String>,
    pub resources: Option<LinuxResources>,
    pub cgroups_path: String,
    pub namespaces: Vec<LinuxNamespace>,
    pub devices: Vec<LinuxDevice>,
    pub seccomp: Option<LinuxSeccomp>,
    pub rootfs_propagation: String,
    pub masked_paths: Vec<String>,
    pub readonly_paths: Vec<String>,
    pub mount_label: String,
}

// NOTE: Solaris and Windows are ignored for the moment
pub type Solaris = Value;
pub type Windows = Value;
pub type Value = i32;


#[derive(Debug)]
pub struct Spec {
    pub version: String,
    // NOTE: Platform was removed, but keeping it as an option
    //       to support older docker versions
    pub platform: Option<Platform>,
    //pub process: Process,
    pub root: Root,
    pub hostname: String,
    pub mounts: Vec<Mount>,
    pub hooks: Option<Hooks>,
    pub annotations: BTreeMap<String, String>,
    pub linux: Option<Linux>,
    pub solaris: Option<Solaris>,
    pub windows: Option<Windows>,
}

#[derive(Debug)]
pub struct State {
    pub version: String,
    pub id: String,
    pub status: String,
    pub pid: i32,
    pub bundle: String,
    pub annotations: BTreeMap<String, String>,
}
