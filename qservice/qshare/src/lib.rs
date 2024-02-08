// Copyright (c) 2023 Quark Container Authors / 2018 The gVisor Authors.
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

#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]

#[macro_use]
extern crate log;

pub mod common;
pub mod config;
pub mod consts;
pub mod types;
pub mod k8s_util;
pub mod databuf;
pub mod metastore;

pub mod crictl {
    include!("pb_gen/runtime.v1alpha2.rs");
}

pub mod tsot {
    include!("pb_gen/tsot.rs");
}

pub mod na {
    include!("pb_gen/na.rs");
}

pub mod tsot_cni {
    include!("pb_gen/tsot_cni.rs");
}

pub mod qmeta {
    include!("pb_gen/qmeta.rs");
}


pub use k8s_openapi::api::core::v1 as k8s;
pub use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta as ObjectMeta;