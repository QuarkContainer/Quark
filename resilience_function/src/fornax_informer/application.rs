// Copyright (c) 2023 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
use fornax_openapi::fornax_serverless::pkg::apis::core::v1::{self as fornax, Application};
use futures_util::StreamExt;

#[derive(Debug)]
pub struct ApplicationInformer {
    watched_resource_version: i32,
}

impl ApplicationInformer {
    fn new() -> ApplicationInformer {
        ApplicationInformer {
            watched_resource_version: 0,
        }
    }

    async fn start(
        &mut self,
        namespace: &str,
        options: fornax_openapi::WatchOptional<'_>,
        event_handler: fn(event: meta::WatchEvent<Application>) -> Application,
    ) -> Result<(), std::string::String> {
        let mut client = fornax_openapi::Client::new("application_informer");
        let watch_events = match fornax::Application::watch(namespace, options) {
            Ok((request, response_body)) => {
                let watch_events = client.get_multiple_values(request, response_body);
                watch_events
            }
            Err(e) => return Err(e.to_string()),
        };

        futures_util::pin_mut!(watch_events);
        let mut watch_events = watch_events.filter_map(|watch_event| {
            let app = match watch_event {
                (fornax_openapi::WatchResponse::Ok(event), _) => event,
                (_, status_code) => panic!("watch panic: {status_code}"),
            };
            std::future::ready(Some(app))
        });
        loop {
            match watch_events.next().await {
                Some(event) => {
                    let app = event_handler(event);
                    self.watched_resource_version = app
                        .metadata
                        .resource_version
                        .map_or(0, |r| r.parse::<i32>().unwrap())
                }
                None => (),
            }
        }
    }
}
