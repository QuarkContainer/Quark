pub mod fornax_serverless;

#[cfg(feature = "api")]
mod create_optional;
#[cfg(feature = "api")]
pub use self::create_optional::CreateOptional;

#[cfg(feature = "api")]
mod create_response;
#[cfg(feature = "api")]
pub use self::create_response::CreateResponse;

#[cfg(feature = "api")]
mod delete_optional;
#[cfg(feature = "api")]
pub use self::delete_optional::DeleteOptional;

#[cfg(feature = "api")]
mod delete_response;
#[cfg(feature = "api")]
pub use self::delete_response::DeleteResponse;

mod list;
pub use self::list::List;

#[cfg(feature = "api")]
mod list_optional;
#[cfg(feature = "api")]
pub use self::list_optional::ListOptional;

#[cfg(feature = "api")]
mod list_response;
#[cfg(feature = "api")]
pub use self::list_response::ListResponse;

#[cfg(feature = "api")]
mod patch_optional;
#[cfg(feature = "api")]
pub use self::patch_optional::PatchOptional;

#[cfg(feature = "api")]
mod patch_response;
#[cfg(feature = "api")]
pub use self::patch_response::PatchResponse;

#[cfg(feature = "api")]
mod replace_optional;
#[cfg(feature = "api")]
pub use self::replace_optional::ReplaceOptional;

#[cfg(feature = "api")]
mod replace_response;
#[cfg(feature = "api")]
pub use self::replace_response::ReplaceResponse;

#[cfg(feature = "api")]
mod watch_optional;
#[cfg(feature = "api")]
pub use self::watch_optional::WatchOptional;

#[cfg(feature = "api")]
mod watch_response;
#[cfg(feature = "api")]
pub use self::watch_response::WatchResponse;

pub mod api;

pub mod apimachinery;

// Generated from operation getAPIVersions

/// get available API versions
///
/// Use the returned [`crate::ResponseBody`]`<`[`GetAPIVersionsResponse`]`>` constructor, or [`GetAPIVersionsResponse`] directly, to parse the HTTP response.
#[cfg(feature = "api")]
pub fn get_api_versions(
) -> Result<(crate::http::Request<Vec<u8>>, fn(crate::http::StatusCode) -> crate::ResponseBody<GetAPIVersionsResponse>), crate::RequestError> {
    let __url = "/apis/".to_owned();

    let __request = crate::http::Request::get(__url);
    let __body = vec![];
    match __request.body(__body) {
        Ok(request) => Ok((request, crate::ResponseBody::new)),
        Err(err) => Err(crate::RequestError::Http(err)),
    }
}

/// Use `<GetAPIVersionsResponse as Response>::try_from_parts` to parse the HTTP response body of [`get_api_versions`]
#[cfg(feature = "api")]
#[derive(Debug)]
pub enum GetAPIVersionsResponse {
    Ok(crate::apimachinery::pkg::apis::meta::v1::APIGroupList),
    Other(Result<Option<crate::serde_json::Value>, crate::serde_json::Error>),
}

#[cfg(feature = "api")]
impl crate::Response for GetAPIVersionsResponse {
    fn try_from_parts(status_code: crate::http::StatusCode, buf: &[u8]) -> Result<(Self, usize), crate::ResponseError> {
        match status_code {
            crate::http::StatusCode::OK => {
                let result = match crate::serde_json::from_slice(buf) {
                    Ok(value) => value,
                    Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                    Err(err) => return Err(crate::ResponseError::Json(err)),
                };
                Ok((GetAPIVersionsResponse::Ok(result), buf.len()))
            },
            _ => {
                let (result, read) =
                    if buf.is_empty() {
                        (Ok(None), 0)
                    }
                    else {
                        match crate::serde_json::from_slice(buf) {
                            Ok(value) => (Ok(Some(value)), buf.len()),
                            Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                            Err(err) => (Err(err), 0),
                        }
                    };
                Ok((GetAPIVersionsResponse::Other(result), read))
            },
        }
    }
}

// Generated from operation getCodeVersion

/// get the code version
///
/// Use the returned [`crate::ResponseBody`]`<`[`GetCodeVersionResponse`]`>` constructor, or [`GetCodeVersionResponse`] directly, to parse the HTTP response.
#[cfg(feature = "api")]
pub fn get_code_version(
) -> Result<(crate::http::Request<Vec<u8>>, fn(crate::http::StatusCode) -> crate::ResponseBody<GetCodeVersionResponse>), crate::RequestError> {
    let __url = "/version/".to_owned();

    let __request = crate::http::Request::get(__url);
    let __body = vec![];
    match __request.body(__body) {
        Ok(request) => Ok((request, crate::ResponseBody::new)),
        Err(err) => Err(crate::RequestError::Http(err)),
    }
}

/// Use `<GetCodeVersionResponse as Response>::try_from_parts` to parse the HTTP response body of [`get_code_version`]
#[cfg(feature = "api")]
#[derive(Debug)]
pub enum GetCodeVersionResponse {
    Ok(crate::apimachinery::pkg::version::Info),
    Other(Result<Option<crate::serde_json::Value>, crate::serde_json::Error>),
}

#[cfg(feature = "api")]
impl crate::Response for GetCodeVersionResponse {
    fn try_from_parts(status_code: crate::http::StatusCode, buf: &[u8]) -> Result<(Self, usize), crate::ResponseError> {
        match status_code {
            crate::http::StatusCode::OK => {
                let result = match crate::serde_json::from_slice(buf) {
                    Ok(value) => value,
                    Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                    Err(err) => return Err(crate::ResponseError::Json(err)),
                };
                Ok((GetCodeVersionResponse::Ok(result), buf.len()))
            },
            _ => {
                let (result, read) =
                    if buf.is_empty() {
                        (Ok(None), 0)
                    }
                    else {
                        match crate::serde_json::from_slice(buf) {
                            Ok(value) => (Ok(Some(value)), buf.len()),
                            Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                            Err(err) => (Err(err), 0),
                        }
                    };
                Ok((GetCodeVersionResponse::Other(result), read))
            },
        }
    }
}

// Generated from operation getCoreFornaxServerlessCentaurusinfraIoAPIGroup

/// get information of a group
///
/// Use the returned [`crate::ResponseBody`]`<`[`GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse`]`>` constructor, or [`GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse`] directly, to parse the HTTP response.
#[cfg(feature = "api")]
pub fn get_core_fornax_serverless_centaurusinfra_io_api_group(
) -> Result<(crate::http::Request<Vec<u8>>, fn(crate::http::StatusCode) -> crate::ResponseBody<GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse>), crate::RequestError> {
    let __url = "/apis/core.fornax-serverless.centaurusinfra.io/".to_owned();

    let __request = crate::http::Request::get(__url);
    let __body = vec![];
    match __request.body(__body) {
        Ok(request) => Ok((request, crate::ResponseBody::new)),
        Err(err) => Err(crate::RequestError::Http(err)),
    }
}

/// Use `<GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse as Response>::try_from_parts` to parse the HTTP response body of [`get_core_fornax_serverless_centaurusinfra_io_api_group`]
#[cfg(feature = "api")]
#[derive(Debug)]
pub enum GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse {
    Ok(crate::apimachinery::pkg::apis::meta::v1::APIGroup),
    Other(Result<Option<crate::serde_json::Value>, crate::serde_json::Error>),
}

#[cfg(feature = "api")]
impl crate::Response for GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse {
    fn try_from_parts(status_code: crate::http::StatusCode, buf: &[u8]) -> Result<(Self, usize), crate::ResponseError> {
        match status_code {
            crate::http::StatusCode::OK => {
                let result = match crate::serde_json::from_slice(buf) {
                    Ok(value) => value,
                    Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                    Err(err) => return Err(crate::ResponseError::Json(err)),
                };
                Ok((GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse::Ok(result), buf.len()))
            },
            _ => {
                let (result, read) =
                    if buf.is_empty() {
                        (Ok(None), 0)
                    }
                    else {
                        match crate::serde_json::from_slice(buf) {
                            Ok(value) => (Ok(Some(value)), buf.len()),
                            Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                            Err(err) => (Err(err), 0),
                        }
                    };
                Ok((GetCoreFornaxServerlessCentaurusinfraIoAPIGroupResponse::Other(result), read))
            },
        }
    }
}

// Generated from operation getCoreFornaxServerlessCentaurusinfraIoV1APIResources

/// get available resources
///
/// Use the returned [`crate::ResponseBody`]`<`[`GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse`]`>` constructor, or [`GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse`] directly, to parse the HTTP response.
#[cfg(feature = "api")]
pub fn get_core_fornax_serverless_centaurusinfra_io_v1_api_resources(
) -> Result<(crate::http::Request<Vec<u8>>, fn(crate::http::StatusCode) -> crate::ResponseBody<GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse>), crate::RequestError> {
    let __url = "/apis/core.fornax-serverless.centaurusinfra.io/v1/".to_owned();

    let __request = crate::http::Request::get(__url);
    let __body = vec![];
    match __request.body(__body) {
        Ok(request) => Ok((request, crate::ResponseBody::new)),
        Err(err) => Err(crate::RequestError::Http(err)),
    }
}

/// Use `<GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse as Response>::try_from_parts` to parse the HTTP response body of [`get_core_fornax_serverless_centaurusinfra_io_v1_api_resources`]
#[cfg(feature = "api")]
#[derive(Debug)]
pub enum GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse {
    Ok(crate::apimachinery::pkg::apis::meta::v1::APIResourceList),
    Other(Result<Option<crate::serde_json::Value>, crate::serde_json::Error>),
}

#[cfg(feature = "api")]
impl crate::Response for GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse {
    fn try_from_parts(status_code: crate::http::StatusCode, buf: &[u8]) -> Result<(Self, usize), crate::ResponseError> {
        match status_code {
            crate::http::StatusCode::OK => {
                let result = match crate::serde_json::from_slice(buf) {
                    Ok(value) => value,
                    Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                    Err(err) => return Err(crate::ResponseError::Json(err)),
                };
                Ok((GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse::Ok(result), buf.len()))
            },
            _ => {
                let (result, read) =
                    if buf.is_empty() {
                        (Ok(None), 0)
                    }
                    else {
                        match crate::serde_json::from_slice(buf) {
                            Ok(value) => (Ok(Some(value)), buf.len()),
                            Err(err) if err.is_eof() => return Err(crate::ResponseError::NeedMoreData),
                            Err(err) => (Err(err), 0),
                        }
                    };
                Ok((GetCoreFornaxServerlessCentaurusinfraIoV1APIResourcesResponse::Other(result), read))
            },
        }
    }
}
