#[tokio::test]
async fn watch_sessions() {
    use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
    use fornax_openapi::fornax_serverless::pkg::apis::core::v1 as fornax;
    use futures_util::StreamExt;

    println!("GWJ, starting watch sessions");
    let mut client = fornax_openapi::Client::new("watch_sessions");

    let (request, response_body) =
        fornax::ApplicationSession::watch("", Default::default())
            .expect("couldn't watch application");
    let watch_events = client.get_multiple_values(request, response_body);
    futures_util::pin_mut!(watch_events);

    let mut watch_events = watch_events.filter_map(|event| {
        let app = match event {
            (fornax_openapi::WatchResponse::Ok(event), _) => event,
            (other, status_code) => panic!("{other:?} {status_code}"),
        };
        std::future::ready(Some(app))
    });

    let mut add_events = 0;
    let mut del_events = 0;
    let mut upd_events = 0;
    loop {
        let session_event = watch_events
            .next()
            .await
            .expect("couldn't find apiserver app");

        match session_event {
            meta::WatchEvent::Added(session) => {
                add_events += 1;
                println!("GWJ, add session = {:?}", session.metadata.name);
            }
            meta::WatchEvent::Deleted(session) => {
                del_events += 1;
                println!("GWJ, delete session = {:?}", session.metadata.name);
            }
            meta::WatchEvent::Modified(session) => {
                upd_events += 1;
                println!("GWJ, modify session = {:?}", session.metadata.name);
            }
            _ => {
                println!("GWJ, unknown event type = {:?}", session_event);
            }
        }
        println!(
            "GWJ, received session events, add events {}, upd events {},  del events {}",
            add_events, upd_events, del_events
        );
    }
}
