#[tokio::test]
async fn list() {
    let mut client = fornax_openapi::Client::new("api_versions-list");

    let (request, response_body) =
        fornax_openapi::get_api_versions().expect("couldn't get API versions");
    let api_versions = match client.get_single_value(request, response_body).await {
        (fornax_openapi::GetAPIVersionsResponse::Ok(api_versions), _) => api_versions,
        (other, status_code) => panic!("{other:?} {status_code}"),
    };

    assert_eq!(fornax_openapi::kind(&api_versions), "APIGroupList");
}
