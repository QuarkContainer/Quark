#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeAgentRespMsg {
    #[prost(oneof = "node_agent_resp_msg::MessageBody", tags = "100, 200")]
    pub message_body: ::core::option::Option<node_agent_resp_msg::MessageBody>,
}
/// Nested message and enum types in `NodeAgentRespMsg`.
pub mod node_agent_resp_msg {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum MessageBody {
        #[prost(message, tag = "100")]
        NodeAgentResp(super::NodeAgentResp),
        #[prost(message, tag = "200")]
        NodeAgentStreamMsg(super::NodeAgentStreamMsg),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeAgentReq {
    #[prost(uint64, tag = "1")]
    pub request_id: u64,
    #[prost(oneof = "node_agent_req::MessageBody", tags = "100, 200, 300, 400")]
    pub message_body: ::core::option::Option<node_agent_req::MessageBody>,
}
/// Nested message and enum types in `NodeAgentReq`.
pub mod node_agent_req {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum MessageBody {
        #[prost(message, tag = "100")]
        NodeConfigReq(super::NodeConfigReq),
        #[prost(message, tag = "200")]
        CreatePodReq(super::CreatePodReq),
        #[prost(message, tag = "300")]
        TerminatePodReq(super::TerminatePodReq),
        #[prost(message, tag = "400")]
        ReadFuncLogReq(super::ReadFuncLogReq),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeAgentResp {
    #[prost(uint64, tag = "1")]
    pub request_id: u64,
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
    #[prost(oneof = "node_agent_resp::MessageBody", tags = "100, 200, 300, 400")]
    pub message_body: ::core::option::Option<node_agent_resp::MessageBody>,
}
/// Nested message and enum types in `NodeAgentResp`.
pub mod node_agent_resp {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum MessageBody {
        #[prost(message, tag = "100")]
        NodeConfigResp(super::NodeConfigResp),
        #[prost(message, tag = "200")]
        CreatePodResp(super::CreatePodResp),
        #[prost(message, tag = "300")]
        TerminatePodResp(super::TerminatePodResp),
        #[prost(message, tag = "400")]
        ReadFuncLogResp(super::ReadFuncLogResp),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadFuncLogReq {
    #[prost(string, tag = "1")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub func_name: ::prost::alloc::string::String,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
    #[prost(uint32, tag = "4")]
    pub len: u32,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadFuncLogResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub content: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeConfigReq {
    #[prost(string, tag = "1")]
    pub cluster_domain: ::prost::alloc::string::String,
    /// k8s.io.api.core.v1.Node node = 2;
    ///
    /// repeated k8s.io.api.core.v1.Pod daemonPods = 3;
    /// repeated string daemonPods = 3;
    #[prost(string, tag = "2")]
    pub node: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeConfigResp {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePodReq {
    #[prost(string, tag = "1")]
    pub pod: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub config_map: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePodResp {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TerminatePodReq {
    #[prost(string, tag = "1")]
    pub pod_id: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TerminatePodResp {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeAgentStreamMsg {
    #[prost(oneof = "node_agent_stream_msg::EventBody", tags = "100, 200, 300")]
    pub event_body: ::core::option::Option<node_agent_stream_msg::EventBody>,
}
/// Nested message and enum types in `NodeAgentStreamMsg`.
pub mod node_agent_stream_msg {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum EventBody {
        #[prost(message, tag = "100")]
        NodeRegister(super::NodeRegister),
        #[prost(message, tag = "200")]
        NodeUpdate(super::NodeUpdate),
        #[prost(message, tag = "300")]
        PodEvent(super::PodEvent),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeRegister {
    #[prost(int64, tag = "2")]
    pub revision: i64,
    /// k8s::Node json
    #[prost(string, tag = "3")]
    pub node: ::prost::alloc::string::String,
    /// Vec<k8s::Pod>
    #[prost(string, repeated, tag = "4")]
    pub pods: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeUpdate {
    #[prost(int64, tag = "2")]
    pub revision: i64,
    /// k8s::Node json
    #[prost(string, tag = "3")]
    pub node: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodEvent {
    #[prost(enumeration = "EventType", tag = "1")]
    pub event_type: i32,
    #[prost(int64, tag = "2")]
    pub revision: i64,
    /// k8s::Node json
    #[prost(string, tag = "3")]
    pub pod: ::prost::alloc::string::String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum EventType {
    Add = 0,
    Update = 2,
    Delete = 3,
}
impl EventType {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            EventType::Add => "Add",
            EventType::Update => "Update",
            EventType::Delete => "Delete",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "Add" => Some(Self::Add),
            "Update" => Some(Self::Update),
            "Delete" => Some(Self::Delete),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod node_agent_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct NodeAgentServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl NodeAgentServiceClient<tonic::transport::Channel> {
        /// Attempt to create a new client by connecting to a given endpoint.
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> NodeAgentServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::Error: Into<StdError>,
        T::ResponseBody: Body<Data = Bytes> + Send + 'static,
        <T::ResponseBody as Body>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_origin(inner: T, origin: Uri) -> Self {
            let inner = tonic::client::Grpc::with_origin(inner, origin);
            Self { inner }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> NodeAgentServiceClient<InterceptedService<T, F>>
        where
            F: tonic::service::Interceptor,
            T::ResponseBody: Default,
            T: tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
                Response = http::Response<
                    <T as tonic::client::GrpcService<tonic::body::BoxBody>>::ResponseBody,
                >,
            >,
            <T as tonic::codegen::Service<
                http::Request<tonic::body::BoxBody>,
            >>::Error: Into<StdError> + Send + Sync,
        {
            NodeAgentServiceClient::new(InterceptedService::new(inner, interceptor))
        }
        /// Compress requests with the given encoding.
        ///
        /// This requires the server to support it otherwise it might respond with an
        /// error.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.send_compressed(encoding);
            self
        }
        /// Enable decompressing responses.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.inner = self.inner.accept_compressed(encoding);
            self
        }
        pub async fn stream_process(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::NodeAgentRespMsg>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::NodeAgentReq>>,
            tonic::Status,
        > {
            self.inner
                .ready()
                .await
                .map_err(|e| {
                    tonic::Status::new(
                        tonic::Code::Unknown,
                        format!("Service was not ready: {}", e.into()),
                    )
                })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/nm.NodeAgentService/StreamProcess",
            );
            self.inner.streaming(request.into_streaming_request(), path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod node_agent_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with NodeAgentServiceServer.
    #[async_trait]
    pub trait NodeAgentService: Send + Sync + 'static {
        /// Server streaming response type for the StreamProcess method.
        type StreamProcessStream: futures_core::Stream<
                Item = Result<super::NodeAgentReq, tonic::Status>,
            >
            + Send
            + 'static;
        async fn stream_process(
            &self,
            request: tonic::Request<tonic::Streaming<super::NodeAgentRespMsg>>,
        ) -> Result<tonic::Response<Self::StreamProcessStream>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct NodeAgentServiceServer<T: NodeAgentService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: NodeAgentService> NodeAgentServiceServer<T> {
        pub fn new(inner: T) -> Self {
            Self::from_arc(Arc::new(inner))
        }
        pub fn from_arc(inner: Arc<T>) -> Self {
            let inner = _Inner(inner);
            Self {
                inner,
                accept_compression_encodings: Default::default(),
                send_compression_encodings: Default::default(),
            }
        }
        pub fn with_interceptor<F>(
            inner: T,
            interceptor: F,
        ) -> InterceptedService<Self, F>
        where
            F: tonic::service::Interceptor,
        {
            InterceptedService::new(Self::new(inner), interceptor)
        }
        /// Enable decompressing requests with the given encoding.
        #[must_use]
        pub fn accept_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.accept_compression_encodings.enable(encoding);
            self
        }
        /// Compress responses with the given encoding, if the client supports it.
        #[must_use]
        pub fn send_compressed(mut self, encoding: CompressionEncoding) -> Self {
            self.send_compression_encodings.enable(encoding);
            self
        }
    }
    impl<T, B> tonic::codegen::Service<http::Request<B>> for NodeAgentServiceServer<T>
    where
        T: NodeAgentService,
        B: Body + Send + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/nm.NodeAgentService/StreamProcess" => {
                    #[allow(non_camel_case_types)]
                    struct StreamProcessSvc<T: NodeAgentService>(pub Arc<T>);
                    impl<
                        T: NodeAgentService,
                    > tonic::server::StreamingService<super::NodeAgentRespMsg>
                    for StreamProcessSvc<T> {
                        type Response = super::NodeAgentReq;
                        type ResponseStream = T::StreamProcessStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                tonic::Streaming<super::NodeAgentRespMsg>,
                            >,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).stream_process(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = StreamProcessSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            );
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => {
                    Box::pin(async move {
                        Ok(
                            http::Response::builder()
                                .status(200)
                                .header("grpc-status", "12")
                                .header("content-type", "application/grpc")
                                .body(empty_body())
                                .unwrap(),
                        )
                    })
                }
            }
        }
    }
    impl<T: NodeAgentService> Clone for NodeAgentServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: NodeAgentService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: NodeAgentService> tonic::server::NamedService for NodeAgentServiceServer<T> {
        const NAME: &'static str = "nm.NodeAgentService";
    }
}
