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
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub fn BuildProto(protoFile: &str) -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("./src/pb_gen")
        .compile(&[protoFile], &["."])
        .unwrap_or_else(|e| panic!("protobuf compile error: {}", e));
    tonic_build::compile_protos(protoFile)?;

    return Ok(());
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    BuildProto("./proto/tsot.proto")?;
    BuildProto("./proto/v1.proto")?;
    BuildProto("./proto/v1alpha2.proto")?;
    BuildProto("./proto/qobjs.proto")?;

    return Ok(());
}