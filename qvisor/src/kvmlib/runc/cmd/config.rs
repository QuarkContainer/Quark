// Copyright (c) 2021 Quark Container Authors
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

use alloc::string::String;
use simplelog::LevelFilter;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum DebugLevel {
    Debug,
    Info,
    Trace,
}

impl DebugLevel {
    pub fn FromLevelFilter(level: &LevelFilter) -> Self {
        match level {
            LevelFilter::Debug => Self::Debug,
            LevelFilter::Info => Self::Info,
            _ => Self::Trace,
        }
    }

    pub fn ToLevelFilter(&self) -> LevelFilter {
        match self {
            DebugLevel::Debug => LevelFilter::Debug,
            DebugLevel::Info => LevelFilter::Info,
            DebugLevel::Trace => LevelFilter::Trace,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GlobalConfig {
    // RootDir is the runtime root directory.
    pub RootDir: String,

    pub DebugLevel: DebugLevel,

    // DebugLog is the path to log debug information to, if not empty.
    pub DebugLog: String,

    // FileAccess indicates how the filesystem is accessed.
    pub FileAccess: FileAccessType,

    // Network indicates what type of network to use.
    pub Network: NetworkType,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        return Self {
            RootDir: String::default(),
            DebugLevel: DebugLevel::Info,
            DebugLog: String::default(),
            FileAccess: FileAccessType::default(),
            Network: NetworkType::default(),
        }
    }
}

impl GlobalConfig {
    pub fn Copy(&self) -> Self {
        return Self {
            RootDir: self.RootDir.to_string(),
            DebugLevel: self.DebugLevel,
            DebugLog: self.DebugLog.to_string(),
            FileAccess: self.FileAccess,
            Network: self.Network,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub enum FileAccessType {
    FileAccessShared,
    FileAccessCached,
}

impl Default for FileAccessType {
    fn default() -> FileAccessType {
        FileAccessType::FileAccessShared
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
pub enum NetworkType {
    // NetworkHost redirects network related syscalls to the host network.
    NetworkHost,

    // NetworkNone sets up just loopback using netstack.
    NetworkNone,
}

impl Default for NetworkType {
    fn default() -> NetworkType {
        NetworkType::NetworkHost
    }
}
