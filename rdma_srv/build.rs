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

// rdma_cli needs same file from rdma_srv. Crating a soft link.
// cd rdma_cli;ln -s ../rdma_srv/build.rs build.rs

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../svc_client/quarkcmsvc.proto")?;
    Ok(())
}
