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


use cni_plugin::*;

#[macro_use]
extern crate log;
extern crate simple_logging;



fn main() {
    log4rs::init_file("/etc/quark/cni_logging_config.yaml", Default::default()).unwrap();

    let input = Cni::load().into_inputs().unwrap();
    error!("plugins is {:#?}", &input);

    let cni_version = input.config.cni_version.clone(); // for error
	
    match input.command {
        Command::Add => {
            let res = reply::SuccessReply {
                cni_version,
                interfaces: Default::default(),
                ips: vec![
                    reply::Ip {
                        address: "10.1.0.4".parse().unwrap(),
                        gateway: None,
                        interface: None
                    }
                ],
                // ips: Default::default(),
                routes: Default::default(),
                dns: Default::default(),
                specific: Default::default(),
            };
        
            reply::reply(res);
        }
        _ => {
            let res = reply::SuccessReply {
                cni_version,
                interfaces: Default::default(),
                ips: Default::default(),
                routes: Default::default(),
                dns: Default::default(),
                specific: Default::default(),
            };
        
            reply::reply(res);
        }
    }
}