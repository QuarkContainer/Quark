use super::super::oci::Spec;
use super::super::super::qlib::common::Result;
use super::super::super::qlib::path::{IsAbs, Join};
use super::super::super::namespace::Util;
use std::fs::create_dir_all;

const DEFAULT_QUARK_SANDBOX_ROOT_PATH: &str = "/var/lib/quark/"; 

pub struct FsImageMounter {
    pub rootPath: String,
    pub sandboxId: String
}

impl FsImageMounter {
    pub fn NewWithRootPath(sandboxId: &str, rootPath: &str) -> Self {
        return FsImageMounter{
            rootPath: rootPath.to_string(),
            sandboxId: sandboxId.to_string(),
        };
    }


    pub fn New(sandboxId: &str) -> Self {
        return FsImageMounter{
            rootPath: DEFAULT_QUARK_SANDBOX_ROOT_PATH.to_string(),
            sandboxId: sandboxId.to_string(),
        };
    }

    fn sandboxRoot(&self) -> String {
        return Join(&self.rootPath, &self.sandboxId)
    }

    // This method mount the fs image specified in spec into the quark sandbox path and made available to qkernel
    // TODO: still not sure if this will be observable from inside... Let's do it first
    pub fn MountContainerFs(&self, bundleDir: &str, spec: &Spec, containerId: &str) -> Result<()> {
        let rbindFlags = libc::MS_REC | libc::MS_BIND;
        let rootSpec = spec.root.path.as_str();
        let containerFsRootSource = if IsAbs(rootSpec) {
            rootSpec.to_string()
        } else {
            Join(bundleDir, rootSpec)
        };
        let containerFsRootTarget = Join(&self.sandboxRoot(), containerId);
        match create_dir_all(&containerFsRootTarget) {
            Ok(()) => (),
            Err(_e) => panic!("failed to create dir to mount root for container {}", containerId)
        };

        info!("start subcontainer: mounting {} to {}", &containerFsRootSource, &containerFsRootTarget);
        let ret = Util::Mount(&containerFsRootSource, &containerFsRootTarget, "", rbindFlags, "");
        if  ret < 0 {
            panic!("MountContainerFs: mount container rootfs fail, error is {}", ret);
        }
        return Ok(());
    }
}