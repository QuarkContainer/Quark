// // use fornax_openapi::serde_json;
//
// use crate::serde_json;
// use fornax_openapi::apimachinery::pkg::apis::meta::v1 as meta;
// use fornax_openapi::fornax_serverless::pkg::apis::core::v1 as fornax;
//
// #[tokio::test]
// async fn deployment() {
//     let mut client = crate::Client::new("patch-deployment");
//
//     // Create deployment with container that uses alpine:3.6
//     let deployment_spec = fornax::ApplicationSpec {
//     };
//     let deployment = fornax::Application {
//         metadata: meta::ObjectMeta {
//             name: Some("k8s-openapi-tests-patch-deployment".to_owned()),
//             ..Default::default()
//         },
//         spec: Some(deployment_spec),
//         ..Default::default()
//     };
//     let (request, response_body) =
//         fornax::Application::create("fornaxtest", &deployment, Default::default())
//             .expect("couldn't create deployment");
//     match client.get_single_value(request, response_body).await {
//         (fornax_openapi::CreateResponse::Created(_), _) => (),
//         (other, status_code) => panic!("{other:?} {status_code}"),
//     }
//
//     // Use JSON patch to patch deployment with alpine:3.7 container
//     let patch = meta::Patch::Json(vec![
//         serde_json::Value::Object(
//             [
//                 (
//                     "op".to_owned(),
//                     serde_json::Value::String("test".to_owned()),
//                 ),
//                 (
//                     "path".to_owned(),
//                     serde_json::Value::String("/spec/template/spec/containers/0/image".to_owned()),
//                 ),
//                 (
//                     "value".to_owned(),
//                     serde_json::Value::String("alpine:3.6".to_owned()),
//                 ),
//             ]
//             .into_iter()
//             .collect(),
//         ),
//         serde_json::Value::Object(
//             [
//                 (
//                     "op".to_owned(),
//                     serde_json::Value::String("replace".to_owned()),
//                 ),
//                 (
//                     "path".to_owned(),
//                     serde_json::Value::String("/spec/template/spec/containers/0/image".to_owned()),
//                 ),
//                 (
//                     "value".to_owned(),
//                     serde_json::Value::String("alpine:3.7".to_owned()),
//                 ),
//             ]
//             .into_iter()
//             .collect(),
//         ),
//     ]);
//     patch_and_assert_container_has_image(&mut client, &patch, "alpine:3.7").await;
//
//     // Use merge patch to patch deployment with alpine:3.8 container
//     let patch = fornax::Application {
//         spec: Some(fornax::ApplicationSpec {
//             // config_data: 
//             //
//             // template: api::PodTemplateSpec {
//             //     spec: Some(api::PodSpec {
//             //         containers: vec![api::Container {
//             //             name: "k8s-openapi-tests-patch-deployment".to_owned(),
//             //             image: "alpine:3.8".to_owned().into(),
//             //             ..Default::default()
//             //         }],
//             //         ..Default::default()
//             //     }),
//             //     ..Default::default()
//             // },
//             // ..Default::default()
//         }),
//         ..Default::default()
//     };
//     let patch = meta::Patch::Merge(serde_json::to_value(&patch).expect("couldn't create patch"));
//     patch_and_assert_container_has_image(&mut client, &patch, "alpine:3.8").await;
//
//     // Use strategic merge patch to patch deployment with alpine:3.9 container
//     let patch = fornax::Application {
//         spec: Some(fornax::ApplicationSpec {
//             // template: api::PodTemplateSpec {
//             //     spec: Some(api::PodSpec {
//             //         containers: vec![api::Container {
//             //             name: "k8s-openapi-tests-patch-deployment".to_owned(),
//             //             image: "alpine:3.9".to_owned().into(),
//             //             ..Default::default()
//             //         }],
//             //         ..Default::default()
//             //     }),
//             //     ..Default::default()
//             // },
//             // ..Default::default()
//         }),
//         ..Default::default()
//     };
//     let patch =
//         meta::Patch::StrategicMerge(serde_json::to_value(&patch).expect("couldn't create patch"));
//     patch_and_assert_container_has_image(&mut client, &patch, "alpine:3.9").await;
//
// }
