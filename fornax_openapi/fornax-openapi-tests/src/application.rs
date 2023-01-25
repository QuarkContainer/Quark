use futures_util::StreamExt;
#[tokio::test]
async fn create_list_delete() {
    use fornax_openapi::api::core::v1;
    use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
    use fornax_openapi::fornax_serverless::pkg::apis::core::v1 as fornax;

    let mut client = fornax_openapi::Client::new("application");

    // Create deployment with container that uses alpine:3.6
    let app_spec = fornax::ApplicationSpec {
        config_data: None,
        //
        /// runtime image and resource requirement of a application container
        containers: Some(vec![v1::Container {
            ..Default::default()
        }]),

        /// application scaling policy
        scaling_policy: Some(fornax::ScalingPolicy {
            burst: Some(10),
            idle_session_num_threshold: Some(fornax::IdelSessionNumThreshold {
                high: Some(10),
                low: Some(0),
            }),
            idle_session_percent_threshold: None,
            maximum_instance: Some(10),
            minimum_instance: Some(0),
            scaling_policy_type: Some("abc".to_owned()),
        }),

        /// container will use grpc session service on node agent to start application session
        using_node_session_service: Some(true),
    };
    let app = fornax::Application {
        metadata: meta::ObjectMeta {
            name: Some("echo0".to_owned()),
            namespace: Some("fornaxtest".to_owned()),
            ..Default::default()
        },
        spec: Some(app_spec),
        ..Default::default()
    };
    let (request, response_body) =
        fornax::Application::create("fornaxtest", &app, Default::default())
            .expect("couldn't create deployment");
    match client.get_single_value(request, response_body).await {
        (fornax_openapi::CreateResponse::Created(_), _) => (),
        (other, status_code) => panic!("{other:?} {status_code}"),
    }

    let (request, response_body) = fornax::Application::list("fornaxtest", Default::default())
        .expect("couldn't list applications");
    let application_list = match client.get_single_value(request, response_body).await {
        (fornax_openapi::ListResponse::Ok(list), _) => list,
        (other, status_code) => panic!("{other:?} {status_code}"),
    };
    assert_eq!(fornax_openapi::kind(&application_list), "ApplicationList");

    // assert_eq!(application_list.items.len(), 1);
    application_list.items.into_iter().for_each(|app| {
        assert_eq!(app.metadata.name.as_ref().unwrap(), "echo0");
        println!("application = {:?}", app)
    });

    // Delete deployment
    let (request, response_body) =
        fornax::Application::delete("echo0", "fornaxtest", Default::default())
            .expect("couldn't delete deployment");
    match client.get_single_value(request, response_body).await {
        (
            fornax_openapi::DeleteResponse::OkStatus(_)
            | fornax_openapi::DeleteResponse::OkValue(_),
            _,
        ) => (),
        (other, status_code) => panic!("{other:?} {status_code}"),
    }
}

// #[tokio::test]
async fn watch_applications() {
    use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
    use fornax_openapi::fornax_serverless::pkg::apis::core::v1 as fornax;

    let mut client = fornax_openapi::Client::new("watch_applications");

    let (request, response_body) = fornax::Application::watch("", Default::default())
        .expect("couldn't watch application");
    let watch_events = client.get_multiple_values(request, response_body);
    futures_util::pin_mut!(watch_events);

    let mut watch_events = watch_events.filter_map(|watch_event| {
        let app = match watch_event {
            (fornax_openapi::WatchResponse::Ok(event), _) => event,
            (other, status_code) => panic!("{other:?} {status_code}"),
        };
        std::future::ready(Some(app))
    });

    let mut add_events = 0;
    let mut del_events = 0;
    let mut upd_events = 0;
    loop {
        let app_event = watch_events
            .next()
            .await
            .expect("couldn't find apiserver app");
        match app_event {
            meta::WatchEvent::Added(app) => {
                add_events +=1;
                println!("add application = {:?}", app.metadata.name);
            }
            meta::WatchEvent::Deleted(app) => {
                del_events +=1;
                println!("delete application = {:?}", app.metadata.name);
            }
            meta::WatchEvent::Modified(app) => {
                upd_events +=1;
                println!("modify application = {:?}", app.metadata.name);
            }
            _ => {
                println!("unknown event type = {:?}", app_event);
            }
        }
        println!(
            "received application events, add events {}, upd events {},  del events {}",
            add_events, upd_events, del_events
        );
    }
}

