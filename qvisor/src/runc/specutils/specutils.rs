use alloc::collections::btree_set::BTreeSet;
use alloc::string::ToString;
use alloc::vec::Vec;
use libc::*;
use std::fs;
use std::fs::File;

use super::super::super::qlib::auth::cap_set::*;
use super::super::super::qlib::common::*;
use super::super::super::qlib::linux_def::*;
use super::super::super::qlib::path::*;
use super::super::oci::*;
use super::fs::*;

pub const EXE_PATH: &str = "/proc/self/exe";

pub fn ReadLink(path: &str) -> Result<String> {
    let p = match fs::read_link(path) {
        Err(e) => return Err(Error::SysError(e.raw_os_error().unwrap())),
        Ok(p) => p.into_os_string().into_string().unwrap(),
    };

    return Ok(p);
}

// ContainerdContainerTypeAnnotation is the OCI annotation set by
// containerd to indicate whether the container to create should have
// its own sandbox or a container within an existing sandbox.
const CONTAINERD_CONTAINER_TYPE_ANNOTATION: &str = "io.kubernetes.cri.container-type";
// ContainerdContainerTypeContainer is the container type value
// indicating the container should be created in an existing sandbox.
const CONTAINERD_CONTAINER_TYPE_CONTAINER: &str = "container";
// ContainerdContainerTypeSandbox is the container type value
// indicating the container should be created in a new sandbox.
const CONTAINERD_CONTAINER_TYPE_SANDBOX: &str = "sandbox";

// ContainerdSandboxIDAnnotation is the OCI annotation set to indicate
// which sandbox the container should be created in when the container
// is not the first container in the sandbox.
const CONTAINERD_SANDBOX_IDANNOTATION: &str = "io.kubernetes.cri.sandbox-id";

// ValidateSpec validates that the spec is compatible with qvisor.
pub fn ValidateSpec(spec: &Spec) -> Result<()> {
    // Mandatory fields.
    if spec.process.args.len() == 0 {
        return Err(Error::Common(format!(
            "Spec.Process.Arg must be defined: {:?}",
            spec.process
        )));
    }

    if spec.root.path.len() == 0 {
        return Err(Error::Common(format!(
            "Spec.Root.Path must be defined: {:?}",
            spec.root
        )));
    }

    // Unsupported fields.
    if spec.solaris.is_some() {
        return Err(Error::Common(format!(
            "Spec.solaris is not supported: {:?}",
            spec
        )));
    }

    if spec.windows.is_some() {
        return Err(Error::Common(format!(
            "Spec.windows is not supported: {:?}",
            spec
        )));
    }

    if spec.process.selinux_label.len() > 0 {
        return Err(Error::Common(format!(
            "SELinux is not supported: {:?}",
            spec.process.selinux_label
        )));
    }

    // Docker uses AppArmor by default, so just log that it's being ignored.
    if spec.process.apparmor_profile.len() != 0 {
        info!(
            "AppArmor profile {:?} is being ignored",
            spec.process.apparmor_profile
        )
    }

    if spec.linux.is_some() && spec.linux.as_ref().unwrap().seccomp.is_some() {
        info!("Seccomp spec {:?} is being ignored", spec.linux)
    }

    if spec.linux.is_some() && spec.linux.as_ref().unwrap().rootfs_propagation.len() != 0 {
        ValidateRootfsPropagation(&spec.linux.as_ref().unwrap().rootfs_propagation)?;
    }

    for m in &spec.mounts {
        ValidateMount(m)?;
    }

    // Two annotations are use by containerd to support multi-container pods.
    //   "io.kubernetes.cri.container-type"
    //   "io.kubernetes.cri.sandbox-id"

    let (containerType, hasContainerType) =
        match spec.annotations.get(CONTAINERD_CONTAINER_TYPE_ANNOTATION) {
            None => ("".to_string(), false),
            Some(typ) => (typ.to_string(), true),
        };

    let hasSandboxID = spec
        .annotations
        .contains_key(CONTAINERD_SANDBOX_IDANNOTATION);

    if containerType.as_str() == "CONTAINERD_CONTAINER_TYPE_CONTAINER" && !hasSandboxID {
        return Err(Error::Common(format!(
            "spec has container-type of {:?}, but no sandbox ID set",
            containerType
        )));
    }

    if !hasContainerType || containerType.as_str() == CONTAINERD_CONTAINER_TYPE_SANDBOX {
        return Ok(());
    }

    return Err(Error::Common(format!(
        "unknown container-type: {:?}",
        containerType
    )));
}

// absPath turns the given path into an absolute path (if it is not already
// absolute) by prepending the base path.
pub fn AbsPath(base: &str, rel: &str) -> String {
    if IsAbs(rel) {
        return rel.to_string();
    }

    return Join(base, rel);
}

// OpenSpec opens an OCI runtime spec from the given bundle directory.
pub fn OpenSpec(bundleDir: &str) -> Result<Spec> {
    let path = Join(bundleDir, "config.json");
    return Spec::load(&path)
        .map_err(|e| Error::IOError(format!("can't load config.json is {:?}", e)));
}

pub fn Capabilities(enableRaw: bool, specCaps: &Option<LinuxCapabilities>) -> TaskCaps {
    // Strip CAP_NET_RAW from all capability sets if necessary.
    let mut skipSet = BTreeSet::new();
    if !enableRaw {
        skipSet.insert(Capability::CAP_NET_RAW);
    }

    let mut caps = TaskCaps::default();
    if specCaps.is_some() {
        let specCaps = specCaps.as_ref().unwrap();
        caps.BoundingCaps = CapsFromSpec(&specCaps.bounding[..], &skipSet);
        caps.EffectiveCaps = CapsFromSpec(&specCaps.effective[..], &skipSet);
        caps.InheritableCaps = CapsFromSpec(&specCaps.inheritable[..], &skipSet);
        caps.PermittedCaps = CapsFromSpec(&specCaps.permitted[..], &skipSet);
    }

    return caps;
}

// Capabilities takes in spec and returns a TaskCapabilities corresponding to
// the spec.
pub fn CapsFromSpec(caps: &[LinuxCapabilityType], skipSet: &BTreeSet<u64>) -> CapSet {
    let mut capVec = Vec::new();

    for c in caps {
        let c = *c as u64;
        if skipSet.contains(&c) {
            continue;
        }

        capVec.push(c);
    }

    return CapSet::NewWithCaps(&capVec);
}

// IsSupportedDevMount returns true if the mount is a supported /dev mount.
// Only mount that does not conflict with runsc default /dev mount is
// supported.
pub fn IsSupportedDevMount(m: Mount) -> bool {
    let existingDevices = [
        "/dev/fd",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
        "/dev/null",
        "/dev/zero",
        "/dev/full",
        "/dev/random",
        "/dev/urandom",
        "/dev/shm",
        "/dev/pts",
        "/dev/ptmx",
    ];

    let dst = Clean(&m.destination);
    if dst.as_str() == "/dev" {
        // OCI spec uses many different mounts for the things inside of '/dev'. We
        // have a single mount at '/dev' that is always mounted, regardless of
        // whether it was asked for, as the spec says we SHOULD.
        return false;
    }

    for dev in &existingDevices {
        if dst.as_str() == *dev || HasPrefix(&dst, &(dev.to_string() + "/")) {
            return false;
        }
    }

    return true;
}

// ShouldCreateSandbox returns true if the spec indicates that a new sandbox
// should be created for the container. If false, the container should be
// started in an existing sandbox.
pub fn ShouldCreateSandbox(spec: &Spec) -> bool {
    match spec.annotations.get(CONTAINERD_CONTAINER_TYPE_ANNOTATION) {
        None => return true,
        Some(t) => return t == CONTAINERD_CONTAINER_TYPE_SANDBOX,
    }
}

// SandboxID returns the ID of the sandbox to join and whether an ID was found
// in the spec.
pub fn SandboxID(spec: &Spec) -> Option<String> {
    return match spec.annotations.get(CONTAINERD_SANDBOX_IDANNOTATION) {
        None => None,
        Some(s) => Some(s.to_string()),
    };
}

pub fn MkdirAll(dst: &str) -> Result<()> {
    return fs::create_dir_all(dst)
        .map_err(|e| Error::IOError(format!("Mkdir({:?}) failed: {:?}", dst, e)));
}

// Mount creates the mount point and calls Mount with the given flags.
pub fn Mount(src: &str, dst: &str, typ: &str, flags: u32) -> Result<()> {
    // Create the mount point inside. The type must be the same as the
    // source (file or directory).
    let isDir;

    if typ == "proc" {
        isDir = true;
    } else {
        let fi = fs::metadata(src)
            .map_err(|e| Error::IOError(format!("Stat({:?}) failed: {:?}", src, e)))?;
        isDir = fi.is_dir();
    }

    if isDir {
        // Create the destination directory.
        // fs::create_dir_all(dst).map_err(|e| Error::IOError(format!("Mkdir({:?}) failed: {:?}", dst, e)))?;
        MkdirAll(dst)?;
    } else {
        // Create the parent destination directory.
        let parent = Dir(dst);
        //fs::create_dir_all(&parent).map_err(|e| Error::IOError(format!("Mkdir({:?}) failed: {:?}", dst, e)))?;
        MkdirAll(&parent)?;
        File::create(dst)
            .map_err(|e| Error::IOError(format!("Open({:?}) failed: {:?}", dst, e)))?;
    }

    let ret = unsafe {
        mount(
            src as *const _ as *const c_char,
            dst as *const _ as *const c_char,
            typ as *const _ as *const c_char,
            flags as u64,
            0 as *const c_void,
        )
    };

    if ret == 0 {
        return Ok(());
    }

    return Err(Error::SysError(-ret as i32));
}

// ContainsStr returns true if 'str' is inside 'strs'.
pub fn ContainsStr(strs: &[&str], str: &str) -> bool {
    for s in strs {
        if *s == str {
            return true;
        }
    }

    return false;
}
