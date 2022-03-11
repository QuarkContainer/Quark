/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


// copy from container_shim::util
use oci_spec::runtime::Spec;
use std::fs::{rename, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;

//use serde::{Deserialize, Serialize};

use shim_proto::shim::oci::Options;
use containerd_shim::protos::protobuf::well_known_types::Timestamp;


use super::super::super::qlib::common::*;

const OPTIONS_FILE_NAME: &str = "options.json";
const RUNTIME_FILE_NAME: &str = "runtime";

// Define JsonOptions here for Json serialize and deserialize
// as rust-protobuf hasn't released serde_derive feature,
// see https://github.com/stepancheg/rust-protobuf/#serde_derive-support
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JsonOptions {
    #[serde(default)]
    pub no_pivot_root: bool,
    #[serde(default)]
    pub no_new_keyring: bool,
    pub shim_cgroup: ::std::string::String,
    #[serde(default)]
    pub io_uid: u32,
    #[serde(default)]
    pub io_gid: u32,
    pub binary_name: ::std::string::String,
    pub root: ::std::string::String,
    pub criu_path: ::std::string::String,
    #[serde(default)]
    pub systemd_cgroup: bool,
    pub criu_image_path: ::std::string::String,
    pub criu_work_path: ::std::string::String,
}

impl From<Options> for JsonOptions {
    fn from(o: Options) -> Self {
        Self {
            no_pivot_root: o.no_pivot_root,
            no_new_keyring: o.no_new_keyring,
            shim_cgroup: o.shim_cgroup,
            io_uid: o.io_uid,
            io_gid: o.io_gid,
            binary_name: o.binary_name,
            root: o.root,
            criu_path: o.criu_path,
            systemd_cgroup: o.systemd_cgroup,
            criu_image_path: o.criu_image_path,
            criu_work_path: o.criu_work_path,
        }
    }
}

pub fn read_file_to_str<P: AsRef<Path>>(filename: P) -> Result<String> {
    let mut file = File::open(&filename).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
    let mut content: String = String::new();
    file.read_to_string(&mut content).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
    Ok(content)
}

pub fn read_options(bundle: &str) -> Result<JsonOptions> {
    let path = Path::new(bundle).join(OPTIONS_FILE_NAME);
    let opts_str = read_file_to_str(path)?;
    let json_opt: JsonOptions = serde_json::from_str(&opts_str).map_err(|e| Error::SerdeJson(format!("io::error is {:?}", e)))?;
    Ok(json_opt)
}

pub fn read_runtime(bundle: &str) -> Result<String> {
    let path = Path::new(bundle).join(RUNTIME_FILE_NAME);
    read_file_to_str(path)
}

pub fn read_address() -> Result<String> {
    let path = Path::new("address");
    read_file_to_str(path)
}

pub fn read_pid_from_file(pid_path: &Path) -> Result<i32> {
    let pid_str = read_file_to_str(pid_path)?;
    let pid = pid_str.parse::<i32>().expect("read_pid_from_file fail");
    Ok(pid)
}

pub fn write_str_to_path(filename: &Path, s: &str) -> Result<()> {
    let file = filename
        .file_name()
        .ok_or_else(|| Error::InvalidArgument(String::from("pid path illegal")))?;
    let tmp_path = filename
        .parent()
        .map(|x| x.join(format!(".{}", file.to_str().unwrap_or(""))))
        .ok_or_else(|| Error::InvalidArgument(String::from("failed to create tmp path")))?;
    let tmp_path = tmp_path
        .to_str()
        .ok_or_else(|| Error::InvalidArgument(String::from("failed to get path")))?;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tmp_path)
        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
    f.write_all(s.as_bytes())
        .map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
    rename(tmp_path, filename).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))?;
    Ok(())
}

pub fn write_options(bundle: &str, opt: &Options) -> Result<()> {
    let json_opt = JsonOptions::from(opt.to_owned());
    let opts_str = serde_json::to_string(&json_opt).map_err(|e| Error::SerdeJson(format!("SerdeJson::error is {:?}", e)))?;
    let path = Path::new(bundle).join(OPTIONS_FILE_NAME);
    write_str_to_path(path.as_path(), opts_str.as_str())
}

pub fn write_runtime(bundle: &str, binary_name: &str) -> Result<()> {
    let path = Path::new(bundle).join(RUNTIME_FILE_NAME);
    write_str_to_path(path.as_path(), binary_name)
}

pub fn write_address(address: &str) -> Result<()> {
    let path = Path::new("address");
    write_str_to_path(path, address)
}

pub fn read_spec_from_file(bundle: &str) -> Result<Spec> {
    let path = Path::new(bundle).join("config.json");
    Spec::load(path).map_err(|e| Error::IOError(format!("io::error is {:?}", e)))
}

pub fn get_timestamp() -> Result<Timestamp> {
    let mut timestamp = Timestamp::new();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("get_timestamp fail");
    timestamp.set_seconds(now.as_secs() as i64);
    timestamp.set_nanos(now.subsec_nanos() as i32);
    Ok(timestamp)
}

pub trait IntoOption
    where
        Self: Sized,
{
    fn none_if<F>(self, callback: F) -> Option<Self>
        where
            F: Fn(&Self) -> bool,
    {
        if callback(&self) {
            None
        } else {
            Some(self)
        }
    }
}

impl<T> IntoOption for T {}

/// A helper to help remove temperate file or dir when it became useless
pub struct HelperRemoveFile {
    path: String,
}

impl HelperRemoveFile {
    pub fn new(path: String) -> Self {
        Self { path }
    }
}
impl Drop for HelperRemoveFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path)
            .unwrap_or_else(|e| warn!("remove dir {} error: {}", &self.path, e));
    }
}

