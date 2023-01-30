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

static mut add_app_events: i32 = 0;
static mut del_app_events: i32 = 0;
static mut upd_app_events: i32 = 0;

static mut add_session_events: i32 = 0;
static mut del_session_events: i32 = 0;
static mut upd_session_events: i32 = 0;

// #[tokio::test]
async fn application_informer() {
    use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
    use fornax_openapi::fornax_serverless::pkg::apis::core::v1 as fornax;

    fn event_handler(app_event: meta::WatchEvent<fornax::Application>) -> Option<String> {
        unsafe {
            let rv = match app_event {
                meta::WatchEvent::Added(app) => {
                    add_app_events += 1;
                    println!("add application = {:?}", app.metadata.name);
                    app.metadata.resource_version.unwrap()
                }
                meta::WatchEvent::Deleted(app) => {
                    del_app_events += 1;
                    println!("delete application = {:?}", app.metadata.name);
                    app.metadata.resource_version.unwrap()
                }
                meta::WatchEvent::Modified(app) => {
                    upd_app_events += 1;
                    println!("modify application = {:?}", app.metadata.name);
                    app.metadata.resource_version.unwrap()
                }
                _ => return None,
            };
            println!(
                "received application events, add events {}, upd events {},  del events {}",
                add_app_events, upd_app_events, del_app_events,
            );
            Some(rv)
        }
    }

    let mut informer = super::new_application_informer(event_handler);
    match informer
        .start(
            "".to_string(),
            fornax_openapi::ListOptional::default(),
            fornax_openapi::WatchOptional::default(),
        )
        .await
    {
        Ok(()) => println!("informer return"),
        Err(msg) => panic!("got error msg {}", msg),
    };
}

#[tokio::test]
async fn session_informer() {
    use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
    use fornax_openapi::fornax_serverless::pkg::apis::core::v1 as fornax;

    fn event_handler(app_event: meta::WatchEvent<fornax::ApplicationSession>) -> Option<String> {
        unsafe {
            let rv = match app_event {
                meta::WatchEvent::Added(app) => {
                    add_session_events += 1;
                    println!("add session = {:?}", app.metadata.name);
                    app.metadata.resource_version.unwrap()
                }
                meta::WatchEvent::Deleted(app) => {
                    del_session_events += 1;
                    println!("delete session = {:?}", app.metadata.name);
                    app.metadata.resource_version.unwrap()
                }
                meta::WatchEvent::Modified(app) => {
                    upd_session_events += 1;
                    println!("modify session = {:?}", app.metadata.name);
                    app.metadata.resource_version.unwrap()
                }
                _ => return None,
            };
            println!(
                "received session events, add events {}, upd events {},  del events {}",
                add_session_events, upd_session_events, del_session_events
            );
            Some(rv)
        }
    }

    let mut list_optional = fornax_openapi::ListOptional::default();
    list_optional.limit = Some(2000);
    let mut informer = super::new_application_session_informer(event_handler);
    match informer
        .start(
            "".to_string(),
            list_optional,
            fornax_openapi::WatchOptional::default(),
        )
        .await
    {
        Ok(()) => println!("informer return"),
        Err(msg) => panic!("got error msg {}", msg),
    };
}
