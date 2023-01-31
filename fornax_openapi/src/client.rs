#![deny(clippy::all, clippy::pedantic)]
#![allow(
    clippy::default_trait_access,
    clippy::let_and_return,
    clippy::let_underscore_drop,
    clippy::let_unit_value,
    clippy::too_many_lines
)]

use crate::http;
use http::StatusCode;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::Stream;
use futures_io::AsyncRead;
use futures_util::{StreamExt, TryStreamExt};

#[derive(Debug)]
pub struct Client {
    inner: reqwest::Client,
    server: http::Uri,
}

impl Client {
    pub fn new() -> Result<Self, String> {
        let kubeconfig_file = if std::env::var("KUBECONFIG").is_ok() {
            match std::fs::File::open(std::env::var("KUBECONFIG").unwrap()) {
                Ok(d) => d,
                Err(_) => return Err("couldn't open kube config".to_string()),
            }
        } else {
            let mut kubeconfig_path = match dirs::home_dir() {
                Some(d) => d,
                None => return Err("couldn't open home dir".to_string()),
            };
            kubeconfig_path.push(".kube");
            kubeconfig_path.push("config");
            match std::fs::File::open(kubeconfig_path) {
                Ok(d) => d,
                Err(_) => return Err("couldn't open kube config".to_string()),
            }
        };
        let kubeconfig: KubeConfig =
            match serde_yaml::from_reader(std::io::BufReader::new(kubeconfig_file)) {
                Ok(d) => d,
                Err(_) => return Err("couldn't parse kube config".to_string()),
            };

        let context = std::env::var("K8S_CONTEXT").unwrap_or(kubeconfig.current_context);

        let KubeConfigContext {
            cluster: cluster_name,
            user,
        } = kubeconfig
            .contexts
            .into_iter()
            .find(|c| c.name == context)
            .unwrap_or_else(|| panic!("couldn't find context named {context}"))
            .context;

        let KubeConfigCluster {
            insecure_skip_tls_verify,
            server,
            certificate_authority,
        } = match kubeconfig
            .clusters
            .into_iter()
            .find(|c| c.name == cluster_name)
        {
            Some(e) => e.cluster,
            None => return Err("couldn't find cluster named {clusterName}".to_string()),
        };

        let server: http::Uri = match server.parse() {
            Ok(e) => e,
            Err(_) => return Err("couldn't parse server URL".to_string()),
        };
        if let Some(path_and_query) = server.path_and_query() {
            assert_eq!(
                path_and_query, "/",
                "server URL {server} has path and query {path_and_query}"
            );
        }

        if insecure_skip_tls_verify.unwrap_or(false) {
            let inner = match reqwest::Client::builder()
                .danger_accept_invalid_certs(true)
                .build()
            {
                Ok(e) => e,
                Err(_) => return Err("can not create client".to_string()),
            };
            Ok(Client { inner, server })
        } else {
            let ca_certificate = {
                let ca_cert_pem = match certificate_authority {
                    Some(CertificateAuthority::File(path)) => match std::fs::read(path) {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't read CA certificate file".to_string()),
                    },
                    Some(CertificateAuthority::Inline(data)) => match base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        data,
                    ) {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't parse CA certificate file".to_string()),
                    },
                    None => return Err("ca_certificate is empty".to_string()),
                };
                let c = match reqwest::Certificate::from_pem(&ca_cert_pem) {
                    Ok(e) => e,
                    Err(_) => return Err("couldn't create CA certificate".to_string()),
                };
                c
            };

            let KubeConfigUser {
                client_certificate,
                client_key,
                username: _,
            } = match kubeconfig.users.into_iter().find(|u| u.name == user) {
                Some(e) => e.user,
                None => return Err("couldn't find user named {user}".to_string()),
            };

            // reqwest::Identity supports from_pem, which is implemented using rustls to parse the PEM.
            // This also requires the reqwest::Client to be built with use_rustls_tls(), otherwise the Identity is ignored.
            //
            // However, the client then fails to connect to kind clusters anyway, because kind clusters listen on 127.0.0.1
            // and hyper-rustls doesn't support connecting to IPs. Ref: https://github.com/ctz/hyper-rustls/issues/84
            //
            // So we need to use the native-tls backend, and thus Identity::from_pkcs12_der
            let client_tls_identity = {
                let public_key_pem = match client_certificate {
                    Some(ClientCertificate::File(path)) => match std::fs::read(path) {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't read client certificate file".to_string()),
                    },
                    Some(ClientCertificate::Inline(data)) => match base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        data,
                    ) {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't read client certificate file".to_string()),
                    },
                    None => return Err("client_certificate is empty".to_string()),
                };
                let public_key = match openssl::x509::X509::from_pem(&public_key_pem) {
                    Ok(e) => e,
                    Err(_) => return Err("couldn't parse client certificate".to_string()),
                };

                let private_key_pem = match client_key {
                    Some(ClientKey::File(path)) => match std::fs::read(path) {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't read client key file".to_string()),
                    },
                    Some(ClientKey::Inline(data)) => match base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        data,
                    ) {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't parse client key file".to_string()),
                    },
                    None => return Err("client_certificate is empty".to_string()),
                };
                let private_key = match openssl::pkey::PKey::private_key_from_pem(&private_key_pem)
                {
                    Ok(e) => e,
                    Err(_) => return Err("couldn't parse client key file".to_string()),
                };

                let pkcs12 = match openssl::pkcs12::Pkcs12::builder().build(
                    "",
                    "admin",
                    &private_key,
                    &public_key,
                ) {
                    Ok(e) => match e.to_der() {
                        Ok(e) => e,
                        Err(_) => return Err("couldn't construct client identity".to_string()),
                    },
                    Err(_) => return Err("couldn't construct client identity".to_string()),
                };

                let tls_identity = match reqwest::Identity::from_pkcs12_der(&pkcs12, "") {
                    Ok(e) => e,
                    Err(_) => return Err("couldn't construct client identity".to_string()),
                };
                tls_identity
            };
            let inner = match reqwest::Client::builder()
                .use_native_tls()
                .add_root_certificate(ca_certificate)
                .identity(client_tls_identity)
                .build()
            {
                Ok(e) => e,
                Err(_) => return Err("couldn't create client".to_string()),
            };
            Ok(Client { inner, server })
        }
    }

    pub async fn get_single_value<R>(
        &mut self,
        request: http::Request<Vec<u8>>,
        response_body: fn(http::StatusCode) -> crate::ResponseBody<R>,
    ) -> Result<(R, http::StatusCode), String>
    where
        R: crate::Response,
    {
        let stream = self.get_multiple_values(request, response_body);
        futures_util::pin_mut!(stream);
        stream
            .next()
            .await
            .map_or(Err("unexpected EOF".to_string()), |s| s)
    }

    pub fn get_multiple_values<'a, R>(
        &'a mut self,
        request: http::Request<Vec<u8>>,
        response_body: fn(http::StatusCode) -> crate::ResponseBody<R>,
    ) -> impl Stream<Item = Result<(R, http::StatusCode), String>> + 'a
    where
        R: crate::Response + 'a,
    {
        MultipleValuesStream::ExecutingRequest {
            f: self.execute(request),
            response_body,
        }
    }

    async fn execute(
        &mut self,
        request: http::Request<Vec<u8>>,
    ) -> Result<ClientResponse<'_, impl AsyncRead>, String> {
        let (path, method, body, content_type) = {
            let content_type = request
                .headers()
                .get(http::header::CONTENT_TYPE)
                .map(|value| {
                    value
                        .to_str()
                        .expect("Content-Type header is not set to valid utf-8 string")
                        .to_owned()
                });

            let (parts, body) = request.into_parts();
            let mut url: http::uri::Parts = parts.uri.into();
            let path = match url.path_and_query.take() {
                Some(s) => s,
                None => return Err("request doesn't have path and query".to_string()),
            };

            (path, parts.method, body, content_type)
        };

        let mut url: http::uri::Parts = self.server.clone().into();
        url.path_and_query = Some(path);
        let url = match http::Uri::from_parts(url) {
            Ok(s) => s,
            Err(_) => return Err("couldn't parse URL".to_string()),
        };

        let request = self.inner.request(method, url.to_string());
        let request = if let Some(content_type) = content_type {
            request.header(http::header::CONTENT_TYPE, content_type)
        } else {
            request
        };
        let response = match request.body(body).send().await {
            Ok(s) => s,
            Err(_) => return Err("couldn't send HTTP request".to_string()),
        };
        let response_status_code = response.status();

        let response = response
            .bytes_stream()
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
            .into_async_read();

        let mut a = Vec::new();
        a.push(response_status_code);
        Ok(ClientResponse {
            status_code: response_status_code,
            body: ClientResponseBody::Response(response),
        })
    }
}

#[derive(Debug)]
#[pin_project::pin_project]
struct ClientResponse<'a, TResponse> {
    status_code: http::StatusCode,
    #[pin]
    body: ClientResponseBody<'a, TResponse>,
}

#[pin_project::pin_project(project = ClientResponseBodyProj)]
#[derive(Debug)]
enum ClientResponseBody<'a, TResponse> {
    Response(#[pin] TResponse),
    Error(&'a mut Vec<StatusCode>),
}

impl<'a, TResponse> AsyncRead for ClientResponse<'a, TResponse>
where
    TResponse: AsyncRead,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.project().body.project() {
            ClientResponseBodyProj::Response(response) => response.poll_read(cx, buf).map(|read| {
                let read = read?;
                Ok(read)
            }),
            ClientResponseBodyProj::Error(_) => Poll::Pending,
        }
    }
}

#[pin_project::pin_project(project = MultipleValuesStreamProj)]
enum MultipleValuesStream<'a, TResponseFuture, TResponse, R> {
    ExecutingRequest {
        #[pin]
        f: TResponseFuture,
        response_body: fn(http::StatusCode) -> crate::ResponseBody<R>,
    },
    Response {
        #[pin]
        response: ClientResponse<'a, TResponse>,
        response_body: crate::ResponseBody<R>,
        buf: Box<[u8; 4096]>,
    },
}

impl<'a, TResponseFuture, TResponse, R> Stream
    for MultipleValuesStream<'a, TResponseFuture, TResponse, R>
where
    TResponseFuture: Future<Output = Result<ClientResponse<'a, TResponse>, String>>,
    ClientResponse<'a, TResponse>: AsyncRead,
    R: crate::Response,
{
    type Item = Result<(R, http::StatusCode), String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.as_mut().project() {
                MultipleValuesStreamProj::ExecutingRequest { f, response_body } => {
                    match f.poll(cx) {
                        Poll::Ready(response) => match response {
                            Ok(response) => {
                                let response_body = response_body(response.status_code);
                                let buf = Box::new([0_u8; 4096]);
                                self.set(MultipleValuesStream::Response {
                                    response,
                                    response_body,
                                    buf,
                                });
                            }
                            Err(msg) => {
                                return Poll::Ready(Some(Err(format!(
                                    "get error message: {}",
                                    msg
                                ))))
                            }
                        },
                        Poll::Pending => return Poll::Pending,
                    }
                }

                MultipleValuesStreamProj::Response {
                    mut response,
                    response_body,
                    buf,
                } => loop {
                    let poll = match response_body.parse() {
                        Ok(value) => Poll::Ready(Some((value, response_body.status_code))),
                        Err(crate::ResponseError::NeedMoreData) => Poll::Pending,
                        Err(err) => panic!("{err}"),
                    };

                    match poll {
                        Poll::Pending => {
                            match response.as_mut().poll_read(cx, &mut buf[..]) {
                                Poll::Pending => return Poll::Pending,
                                Poll::Ready(Ok(0)) if response_body.is_empty() => {
                                    return Poll::Pending
                                }
                                Poll::Ready(Ok(0)) => {
                                    return Poll::Ready(Some(Err("unexpected EOF".to_string())))
                                }
                                Poll::Ready(Ok(read)) => response_body.append_slice(&buf[..read]),
                                Poll::Ready(Err(err)) => {
                                    return Poll::Ready(Some(Err(err.to_string())))
                                }
                            };
                        }
                        Poll::Ready(r) => match r {
                            Some(r) => return Poll::Ready(Some(Ok(r))),
                            None => return Poll::Pending,
                        },
                    }
                },
            }
        }
    }
}

#[derive(crate::serde::Deserialize)]
struct KubeConfig {
    clusters: Vec<KubeConfigClusterEntry>,
    contexts: Vec<KubeConfigContextEntry>,
    #[serde(rename = "current-context")]
    current_context: String,
    users: Vec<KubeConfigUserEntry>,
}

#[derive(serde::Deserialize)]
struct KubeConfigClusterEntry {
    cluster: KubeConfigCluster,
    name: String,
}

#[derive(serde::Deserialize)]
struct KubeConfigCluster {
    #[serde(default)]
    certificate_authority: Option<CertificateAuthority>,

    #[serde(default)]
    #[serde(rename = "insecure-skip-tls-verify")]
    insecure_skip_tls_verify: Option<bool>,

    server: String,
}

#[derive(crate::serde::Deserialize)]
enum CertificateAuthority {
    #[serde(rename = "certificate-authority")]
    File(std::path::PathBuf),
    #[serde(rename = "certificate-authority-data")]
    Inline(String),
}

#[derive(crate::serde::Deserialize)]
struct KubeConfigContextEntry {
    context: KubeConfigContext,
    name: String,
}

#[derive(crate::serde::Deserialize)]
struct KubeConfigContext {
    cluster: String,
    user: String,
}

#[derive(crate::serde::Deserialize)]
struct KubeConfigUserEntry {
    name: String,
    user: KubeConfigUser,
}

#[derive(crate::serde::Deserialize)]
struct KubeConfigUser {
    username: String,

    #[serde(default)]
    #[serde(flatten)]
    client_certificate: Option<ClientCertificate>,

    #[serde(default)]
    #[serde(flatten)]
    client_key: Option<ClientKey>,
}

#[derive(crate::serde::Deserialize)]
enum ClientCertificate {
    #[serde(rename = "client-certificate")]
    File(std::path::PathBuf),
    #[serde(rename = "client-certificate-data")]
    Inline(String),
}

#[derive(crate::serde::Deserialize)]
enum ClientKey {
    #[serde(rename = "client-key")]
    File(std::path::PathBuf),
    #[serde(rename = "client-key-data")]
    Inline(String),
}

mod bytestring {
    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: crate::serde::Deserializer<'de>,
    {
        let s: String = crate::serde::Deserialize::deserialize(deserializer)?;
        Ok(s.into_bytes())
    }

    pub(super) fn serialize<S>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: crate::serde::Serializer,
    {
        let s = std::str::from_utf8(bytes).expect("bytes are not valid utf-8");
        crate::serde::Serialize::serialize(&s, serializer)
    }
}

mod methodstring {
    use super::http;

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<http::Method, D::Error>
    where
        D: crate::serde::Deserializer<'de>,
    {
        struct Visitor;

        impl<'de> crate::serde::de::Visitor<'de> for Visitor {
            type Value = http::Method;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an HTTP method name")
            }

            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: crate::serde::de::Error,
            {
                v.parse().map_err(crate::serde::de::Error::custom)
            }
        }

        deserializer.deserialize_str(Visitor)
    }

    pub(super) fn serialize<S>(method: &http::Method, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: crate::serde::Serializer,
    {
        crate::serde::Serialize::serialize(&method.to_string(), serializer)
    }
}
