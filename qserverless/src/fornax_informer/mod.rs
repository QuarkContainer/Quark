pub mod test;

use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
use fornax_openapi::fornax_serverless::pkg::apis::core::v1::{Application, ApplicationSession};
use fornax_openapi::http::{Request, StatusCode};
use fornax_openapi::serde::Deserialize;
use fornax_openapi::{
    ListOptional, ListResponse, ListableResource, RequestError, ResponseBody, WatchOptional,
    WatchResponse,
};
use futures_util::StreamExt;

pub fn new_application_session_informer(
    event_handler: fn(event: meta::WatchEvent<ApplicationSession>) -> Option<String>,
) -> Informer<ApplicationSession> {
    Informer {
        latest_resource_version: None,
        list: ApplicationSession::list,
        watch: ApplicationSession::watch,
        event_handler,
    }
}

pub fn new_application_informer(
    event_handler: fn(event: meta::WatchEvent<Application>) -> Option<String>,
) -> Informer<Application> {
    Informer {
        latest_resource_version: None,
        list: Application::list,
        watch: Application::watch,
        event_handler,
    }
}

#[allow(dead_code)]
pub struct Informer<T: for<'de> Deserialize<'de> + ListableResource + fornax_openapi::Metadata> {
    latest_resource_version: Option<String>,
    list: fn(
        namespace: &str,
        options: ListOptional,
    ) -> Result<
        (
            Request<Vec<u8>>,
            fn(StatusCode) -> ResponseBody<ListResponse<T>>,
        ),
        RequestError,
    >,
    watch: fn(
        namespace: &str,
        options: WatchOptional,
    ) -> Result<
        (
            Request<Vec<u8>>,
            fn(StatusCode) -> ResponseBody<WatchResponse<T>>,
        ),
        RequestError,
    >,
    event_handler: fn(event: meta::WatchEvent<T>) -> Option<String>,
}

impl<T: for<'de> Deserialize<'de> + ListableResource + fornax_openapi::Metadata> Informer<T> {
    pub async fn start(
        &mut self,
        namespace: String,
        list_options: ListOptional<'_>,
        watch_options: WatchOptional<'_>,
    ) -> Result<(), String> {
        // list
        let mut client = match fornax_openapi::Client::new() {
            Ok(c) => c,
            Err(msg) => return Err(format!("can not new fornax client, error:{msg}")),
        };
        let mut has_more = true;
        while has_more {
            let mut rv: String = String::new();
            let mut options = list_options.clone();
            options.limit = Some(500);
            options.resource_version = match self.latest_resource_version.clone() {
                Some(s) => {
                    s.clone_into(&mut rv);
                    Some(&rv)
                }
                None => None,
            };
            let list_events = match (self.list)(namespace.as_ref(), options) {
                Ok((request, response_body)) => {
                    let _events = match client.get_single_value(request, response_body).await {
                        Ok((ListResponse::Ok(list), _)) => list,
                        Ok((_, status_code)) => {
                            return Err(format!("api error, status code: {status_code}"))
                        }
                        Err(msg) => return Err(format!("api error, unknown error: {msg}")),
                    };
                    _events
                }
                Err(e) => return Err(e.to_string()),
            };
            let mut iter = list_events.items.into_iter();
            // peak first one to see if has more data
            has_more = match iter.next() {
                Some(e) => {
                    (self.event_handler)(meta::WatchEvent::Added(e))
                        .map(|s| self.latest_resource_version = Some(s));
                    true
                }
                None => false,
            };
            while let Some(e) = iter.next() {
                (self.event_handler)(meta::WatchEvent::Added(e))
                    .map(|s| self.latest_resource_version = Some(s));
            }
        }

        // watch
        let mut rv: String = String::new();
        let mut options = watch_options.clone();
        options.resource_version = match self.latest_resource_version.clone() {
            Some(s) => {
                s.clone_into(&mut rv);
                Some(&rv)
            }
            None => None,
        };
        let mut client = match fornax_openapi::Client::new() {
            Ok(c) => c,
            Err(msg) => return Err(format!("can not new fornax client, error:{msg}")),
        };
        let watch_events = match (self.watch)(namespace.as_ref(), options) {
            Ok((request, response_body)) => {
                let watch_events = client.get_multiple_values(request, response_body);
                watch_events
            }
            Err(e) => return Err(e.to_string()),
        };

        futures_util::pin_mut!(watch_events);
        loop {
            match watch_events.next().await {
                Some(r) => match r {
                    Ok((fornax_openapi::WatchResponse::Ok(e), _)) => {
                        (self.event_handler)(e).map(|s| self.latest_resource_version = Some(s));
                    }
                    Ok((_, status_code)) => {
                        return Err(format!("watch error, status code: {status_code}"))
                    }
                    Err(msg) => return Err(format!("watch unexpected error: {msg}")),
                },
                None => (),
            }
        }
    }
}
