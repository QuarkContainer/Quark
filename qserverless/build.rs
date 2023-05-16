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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "./proto/qobjs.proto";
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("./src/qobjs/src/pb_gen")
        .compile(&[proto_file], &["."])
        .unwrap_or_else(|e| panic!("protobuf compile error: {}", e));
    tonic_build::compile_protos(proto_file)?;

    let nm_proto_file = "./proto/nm.proto";
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("./src/qobjs/src/pb_gen")
        .compile(&[nm_proto_file], &["."])
        .unwrap_or_else(|e| panic!("protobuf compile error: {}", e));
    tonic_build::compile_protos(nm_proto_file)?;

    /*let v1_proto_file = "./proto/v1.proto";
    tonic_build::configure()
    .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
    //.build_server(true)
    .build_client(true)
    .out_dir("./src/qobjs/src/pb_gen")
    .compile(&[v1_proto_file], &["."])
    .unwrap_or_else(|e| panic!("protobuf compile error: {}", e));
    tonic_build::compile_protos(v1_proto_file)?;*/

    let v1alpha2_proto_file = "./proto/v1alpha2.proto";
    tonic_build::configure()
    .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
    //.build_server(true)
    .build_client(true)
    .out_dir("./src/qobjs/src/pb_gen")
    .compile(&[v1alpha2_proto_file], &["."])
    .unwrap_or_else(|e| panic!("protobuf compile error: {}", e));
    tonic_build::compile_protos(v1alpha2_proto_file)?;

    Ok(())
}
