#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentMsg {
    #[prost(uint64, tag = "1")]
    pub msg_id: u64,
    #[prost(oneof = "func_agent_msg::EventBody", tags = "100, 200, 300, 400")]
    pub event_body: ::core::option::Option<func_agent_msg::EventBody>,
}
/// Nested message and enum types in `FuncAgentMsg`.
pub mod func_agent_msg {
    #[derive(serde::Serialize, serde::Deserialize)]
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum EventBody {
        #[prost(message, tag = "100")]
        FuncPodRegisterReq(super::FuncPodRegisterReq),
        #[prost(message, tag = "200")]
        FuncPodRegisterResp(super::FuncPodRegisterResp),
        #[prost(message, tag = "300")]
        FuncAgentCallReq(super::FuncAgentCallReq),
        #[prost(message, tag = "400")]
        FuncAgentCallResp(super::FuncAgentCallResp),
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodRegisterReq {
    #[prost(string, tag = "1")]
    pub instance_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodRegisterResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentCallReq {
    #[prost(string, tag = "1")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub package: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub func_name: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub parameters: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentCallResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub resp: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncSvcMsg {
    #[prost(uint64, tag = "1")]
    pub msg_id: u64,
    #[prost(
        oneof = "func_svc_msg::EventBody",
        tags = "100, 200, 300, 400, 500, 600, 700, 800"
    )]
    pub event_body: ::core::option::Option<func_svc_msg::EventBody>,
}
/// Nested message and enum types in `FuncSvcMsg`.
pub mod func_svc_msg {
    #[derive(serde::Serialize, serde::Deserialize)]
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum EventBody {
        #[prost(message, tag = "100")]
        FuncAgentRegisterReq(super::FuncAgentRegisterReq),
        #[prost(message, tag = "200")]
        FuncAgentRegisterResp(super::FuncAgentRegisterResp),
        #[prost(message, tag = "300")]
        FuncPodConnReq(super::FuncPodConnReq),
        #[prost(message, tag = "400")]
        FuncPodConnResp(super::FuncPodConnResp),
        #[prost(message, tag = "500")]
        FuncPodDisconnReq(super::FuncPodDisconnReq),
        #[prost(message, tag = "600")]
        FuncPodDisconnResp(super::FuncPodDisconnResp),
        #[prost(message, tag = "700")]
        FuncSvcCallReq(super::FuncSvcCallReq),
        #[prost(message, tag = "800")]
        FuncSvcCallResp(super::FuncSvcCallResp),
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentRegisterReq {
    #[prost(string, tag = "1")]
    pub node_id: ::prost::alloc::string::String,
    /// func calls waiting for response
    #[prost(message, repeated, tag = "2")]
    pub func_calls: ::prost::alloc::vec::Vec<FuncSvcCallReq>,
    /// func pods running on the node
    #[prost(message, repeated, tag = "3")]
    pub func_pods: ::prost::alloc::vec::Vec<FuncPodStatus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentRegisterResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodConnReq {
    #[prost(string, tag = "1")]
    pub node_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodConnResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodDisconnReq {
    #[prost(string, tag = "1")]
    pub pod_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodDisconnResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncSvcCallReq {
    #[prost(string, tag = "1")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub package_name: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub func_name: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub parameters: ::prost::alloc::string::String,
    #[prost(int32, tag = "5")]
    pub priority: i32,
    #[prost(message, optional, tag = "6")]
    pub createtime: ::core::option::Option<Timestamp>,
    /// when funcCall is process by a funcPod, this is the NodeId
    #[prost(string, tag = "7")]
    pub callee_node_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncSvcCallResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub resp: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodStatus {
    #[prost(string, tag = "1")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub package_name: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub pod_name: ::prost::alloc::string::String,
    #[prost(enumeration = "FuncPodState", tag = "4")]
    pub state: i32,
    /// when pod is running, the funccall id
    #[prost(string, tag = "5")]
    pub func_name: ::prost::alloc::string::String,
    ///   when pod is running, the funccall caller id
    #[prost(string, tag = "6")]
    pub func_caller_node_id: ::prost::alloc::string::String,
    /// k8s::pod
    #[prost(string, tag = "7")]
    pub pod: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Timestamp {
    /// Represents seconds of UTC time since Unix epoch
    /// 1970-01-01T00:00:00Z. Must be from 0001-01-01T00:00:00Z to
    /// 9999-12-31T23:59:59Z inclusive.
    #[prost(int64, tag = "1")]
    pub seconds: i64,
    /// Non-negative fractions of a second at nanosecond resolution. Negative
    /// second values with fractions must still have non-negative nanos values
    /// that count forward in time. Must be from 0 to 999,999,999
    /// inclusive.
    #[prost(int32, tag = "2")]
    pub nanos: i32,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum FuncPodState {
    Creating = 0,
    Keepalive = 2,
    Running = 3,
}
impl FuncPodState {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            FuncPodState::Creating => "Creating",
            FuncPodState::Keepalive => "Keepalive",
            FuncPodState::Running => "Running",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "Creating" => Some(Self::Creating),
            "Keepalive" => Some(Self::Keepalive),
            "Running" => Some(Self::Running),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod func_agent_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct FuncAgentServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FuncAgentServiceClient<tonic::transport::Channel> {
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
    impl<T> FuncAgentServiceClient<T>
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
        ) -> FuncAgentServiceClient<InterceptedService<T, F>>
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
            FuncAgentServiceClient::new(InterceptedService::new(inner, interceptor))
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
            request: impl tonic::IntoStreamingRequest<Message = super::FuncAgentMsg>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::FuncAgentMsg>>,
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
                "/func.FuncAgentService/StreamProcess",
            );
            self.inner.streaming(request.into_streaming_request(), path, codec).await
        }
        pub async fn func_call(
            &mut self,
            request: impl tonic::IntoRequest<super::FuncAgentCallReq>,
        ) -> Result<tonic::Response<super::FuncAgentCallResp>, tonic::Status> {
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
                "/func.FuncAgentService/FuncCall",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
    }
}
/// Generated client implementations.
pub mod func_svc_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct FuncSvcServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FuncSvcServiceClient<tonic::transport::Channel> {
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
    impl<T> FuncSvcServiceClient<T>
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
        ) -> FuncSvcServiceClient<InterceptedService<T, F>>
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
            FuncSvcServiceClient::new(InterceptedService::new(inner, interceptor))
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
            request: impl tonic::IntoStreamingRequest<Message = super::FuncSvcMsg>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::FuncSvcMsg>>,
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
                "/func.FuncSvcService/StreamProcess",
            );
            self.inner.streaming(request.into_streaming_request(), path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod func_agent_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with FuncAgentServiceServer.
    #[async_trait]
    pub trait FuncAgentService: Send + Sync + 'static {
        /// Server streaming response type for the StreamProcess method.
        type StreamProcessStream: futures_core::Stream<
                Item = Result<super::FuncAgentMsg, tonic::Status>,
            >
            + Send
            + 'static;
        async fn stream_process(
            &self,
            request: tonic::Request<tonic::Streaming<super::FuncAgentMsg>>,
        ) -> Result<tonic::Response<Self::StreamProcessStream>, tonic::Status>;
        async fn func_call(
            &self,
            request: tonic::Request<super::FuncAgentCallReq>,
        ) -> Result<tonic::Response<super::FuncAgentCallResp>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct FuncAgentServiceServer<T: FuncAgentService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: FuncAgentService> FuncAgentServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for FuncAgentServiceServer<T>
    where
        T: FuncAgentService,
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
                "/func.FuncAgentService/StreamProcess" => {
                    #[allow(non_camel_case_types)]
                    struct StreamProcessSvc<T: FuncAgentService>(pub Arc<T>);
                    impl<
                        T: FuncAgentService,
                    > tonic::server::StreamingService<super::FuncAgentMsg>
                    for StreamProcessSvc<T> {
                        type Response = super::FuncAgentMsg;
                        type ResponseStream = T::StreamProcessStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                tonic::Streaming<super::FuncAgentMsg>,
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
                "/func.FuncAgentService/FuncCall" => {
                    #[allow(non_camel_case_types)]
                    struct FuncCallSvc<T: FuncAgentService>(pub Arc<T>);
                    impl<
                        T: FuncAgentService,
                    > tonic::server::UnaryService<super::FuncAgentCallReq>
                    for FuncCallSvc<T> {
                        type Response = super::FuncAgentCallResp;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FuncAgentCallReq>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).func_call(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = FuncCallSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            );
                        let res = grpc.unary(method, req).await;
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
    impl<T: FuncAgentService> Clone for FuncAgentServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: FuncAgentService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: FuncAgentService> tonic::server::NamedService for FuncAgentServiceServer<T> {
        const NAME: &'static str = "func.FuncAgentService";
    }
}
/// Generated server implementations.
pub mod func_svc_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with FuncSvcServiceServer.
    #[async_trait]
    pub trait FuncSvcService: Send + Sync + 'static {
        /// Server streaming response type for the StreamProcess method.
        type StreamProcessStream: futures_core::Stream<
                Item = Result<super::FuncSvcMsg, tonic::Status>,
            >
            + Send
            + 'static;
        async fn stream_process(
            &self,
            request: tonic::Request<tonic::Streaming<super::FuncSvcMsg>>,
        ) -> Result<tonic::Response<Self::StreamProcessStream>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct FuncSvcServiceServer<T: FuncSvcService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: FuncSvcService> FuncSvcServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for FuncSvcServiceServer<T>
    where
        T: FuncSvcService,
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
                "/func.FuncSvcService/StreamProcess" => {
                    #[allow(non_camel_case_types)]
                    struct StreamProcessSvc<T: FuncSvcService>(pub Arc<T>);
                    impl<
                        T: FuncSvcService,
                    > tonic::server::StreamingService<super::FuncSvcMsg>
                    for StreamProcessSvc<T> {
                        type Response = super::FuncSvcMsg;
                        type ResponseStream = T::StreamProcessStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::FuncSvcMsg>>,
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
    impl<T: FuncSvcService> Clone for FuncSvcServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: FuncSvcService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: FuncSvcService> tonic::server::NamedService for FuncSvcServiceServer<T> {
        const NAME: &'static str = "func.FuncSvcService";
    }
}
