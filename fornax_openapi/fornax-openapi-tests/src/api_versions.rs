#[tokio::test]
async fn list() {
    let mut client = fornax_openapi::Client::new().expect("can not create client");

    let (request, response_body) =
        fornax_openapi::get_api_versions().expect("couldn't get API versions");
    let api_versions = match client.get_single_value(request, response_body).await {
        Ok((fornax_openapi::GetAPIVersionsResponse::Ok(api_versions), _)) => api_versions,
        Ok((other, status_code)) => panic!("{other:?} {status_code}"),
        Err(msg) => panic!("fatal error: {msg}"),
    };

    assert_eq!(fornax_openapi::kind(&api_versions), "APIGroupList");
}
