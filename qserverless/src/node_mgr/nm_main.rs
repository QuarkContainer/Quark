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

#![allow(dead_code)]
#![allow(non_snake_case)]
#[allow(non_camel_case_types)]

extern crate reqwest;


use qobjs::common::Result as QResult;

#[tokio::main]
async fn main() -> QResult<()> {
    /*
    //cadvisor::client::Client::Test().await?;
    let client = cadvisor::client::Client::Init();
    //println!("machine is {:#?}", client.MachineInfo().await?);
    //println!("versioninfo is {:#?}", client.VersionInfo().await?);
    println!("versioninfo is {:#?}", client.GetInfo().await?);
*/
    Ok(())
}


