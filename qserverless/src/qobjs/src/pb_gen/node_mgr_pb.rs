#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FornaxCoreMessage {
    #[prost(message, optional, tag = "1")]
    pub node_identifier: ::core::option::Option<NodeIdentifier>,
    #[prost(enumeration = "MessageType", tag = "2")]
    pub message_type: i32,
    #[prost(
        oneof = "fornax_core_message::MessageBody",
        tags = "100, 200, 201, 202, 203, 204, 300, 301, 302, 303"
    )]
    pub message_body: ::core::option::Option<fornax_core_message::MessageBody>,
}
/// Nested message and enum types in `FornaxCoreMessage`.
pub mod fornax_core_message {
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum MessageBody {
        #[prost(message, tag = "100")]
        FornaxCoreConfiguration(super::FornaxCoreConfiguration),
        #[prost(message, tag = "200")]
        NodeConfiguration(super::NodeConfiguration),
        #[prost(message, tag = "201")]
        NodeRegistry(super::NodeRegistry),
        #[prost(message, tag = "202")]
        NodeReady(super::NodeReady),
        #[prost(message, tag = "203")]
        NodeState(super::NodeState),
        #[prost(message, tag = "204")]
        NodeFullSync(super::NodeFullSync),
        #[prost(message, tag = "300")]
        PodCreate(super::PodCreate),
        #[prost(message, tag = "301")]
        PodTerminate(super::PodTerminate),
        #[prost(message, tag = "302")]
        PodHibernate(super::PodHibernate),
        #[prost(message, tag = "303")]
        PodState(super::PodState),
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FornaxCore {
    #[prost(string, tag = "1")]
    pub ip: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub identifier: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FornaxCoreConfiguration {
    #[prost(message, optional, tag = "1")]
    pub primary: ::core::option::Option<FornaxCore>,
    #[prost(message, repeated, tag = "2")]
    pub standbys: ::prost::alloc::vec::Vec<FornaxCore>,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeIdentifier {
    #[prost(string, tag = "1")]
    pub ip: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub identifier: ::prost::alloc::string::String,
}
/// node register with fornax core, wait for a configuration message to initialize it
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeRegistry {
    #[prost(int64, tag = "1")]
    pub node_revision: i64,
    /// k8s.io.api.core.v1.Node node = 2;
    #[prost(string, tag = "2")]
    pub node: ::prost::alloc::string::String,
}
/// fornax core send node configuration to node to initialize using this configuration before tell fornax it's ready
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeConfiguration {
    #[prost(string, tag = "1")]
    pub cluster_domain: ::prost::alloc::string::String,
    /// k8s.io.api.core.v1.Node node = 2;
    ///
    /// repeated k8s.io.api.core.v1.Pod daemonPods = 3;
    /// repeated string daemonPods = 3;
    #[prost(string, tag = "2")]
    pub node: ::prost::alloc::string::String,
}
/// node report back to fornax core, it's ready for take pod
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeReady {
    #[prost(int64, tag = "1")]
    pub node_revision: i64,
    /// k8s.io.api.core.v1.Node node = 2;
    #[prost(string, tag = "2")]
    pub node: ::prost::alloc::string::String,
    #[prost(message, repeated, tag = "3")]
    pub pod_states: ::prost::alloc::vec::Vec<PodState>,
}
/// node report back full state to fornax core
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeState {
    #[prost(int64, tag = "1")]
    pub node_revision: i64,
    /// k8s.io.api.core.v1.Node node = 2;
    #[prost(string, tag = "2")]
    pub node: ::prost::alloc::string::String,
    #[prost(message, repeated, tag = "3")]
    pub pod_states: ::prost::alloc::vec::Vec<PodState>,
}
/// fornax core ask node to send its full state if node revision are not same between fornax core and node
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NodeFullSync {}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodState {
    #[prost(int64, tag = "1")]
    pub node_revision: i64,
    #[prost(enumeration = "pod_state::State", tag = "2")]
    pub state: i32,
    /// k8s.io.api.core.v1.Pod pod = 3;
    #[prost(string, tag = "3")]
    pub pod: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "4")]
    pub resource: ::core::option::Option<PodResource>,
}
/// Nested message and enum types in `PodState`.
pub mod pod_state {
    #[derive(
        Clone,
        Copy,
        Debug,
        PartialEq,
        Eq,
        Hash,
        PartialOrd,
        Ord,
        ::prost::Enumeration
    )]
    #[repr(i32)]
    pub enum State {
        Creating = 0,
        Standby = 10,
        Activating = 20,
        Running = 30,
        Evacuating = 40,
        Terminating = 50,
        Terminated = 60,
    }
    impl State {
        /// String value of the enum field names used in the ProtoBuf definition.
        ///
        /// The values are not transformed in any way and thus are considered stable
        /// (if the ProtoBuf definition does not change) and safe for programmatic use.
        pub fn as_str_name(&self) -> &'static str {
            match self {
                State::Creating => "Creating",
                State::Standby => "Standby",
                State::Activating => "Activating",
                State::Running => "Running",
                State::Evacuating => "Evacuating",
                State::Terminating => "Terminating",
                State::Terminated => "Terminated",
            }
        }
        /// Creates an enum from field names used in the ProtoBuf definition.
        pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
            match value {
                "Creating" => Some(Self::Creating),
                "Standby" => Some(Self::Standby),
                "Activating" => Some(Self::Activating),
                "Running" => Some(Self::Running),
                "Evacuating" => Some(Self::Evacuating),
                "Terminating" => Some(Self::Terminating),
                "Terminated" => Some(Self::Terminated),
                _ => None,
            }
        }
    }
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodResource {
    /// k8s.io.api.core.v1.ResourceQuotaStatus resourceQuotaStatus = 1;
    ///
    /// repeated k8s.io.api.core.v1.AttachedVolume volumes = 2;
    /// repeated k8s.io.api.core.v1.AttachedVolume volumes = 2;
    #[prost(string, tag = "1")]
    pub resource_quota_status: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodCreate {
    #[prost(string, tag = "1")]
    pub pod_identifier: ::prost::alloc::string::String,
    /// k8s.io.api.core.v1.Pod pod = 2;
    #[prost(string, tag = "2")]
    pub pod: ::prost::alloc::string::String,
    /// k8s.io.api.core.v1.ConfigMap configMap = 3;
    #[prost(string, tag = "3")]
    pub config_map: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodTerminate {
    #[prost(string, tag = "1")]
    pub pod_identifier: ::prost::alloc::string::String,
}
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodHibernate {
    #[prost(string, tag = "1")]
    pub pod_identifier: ::prost::alloc::string::String,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum MessageType {
    Unspecified = 0,
    FornaxCoreConfiguration = 100,
    NodeConfiguration = 200,
    NodeRegister = 201,
    NodeReady = 202,
    NodeState = 203,
    NodeFullSync = 204,
    PodCreate = 300,
    PodTerminate = 301,
    PodHibernate = 302,
    PodState = 303,
}
impl MessageType {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            MessageType::Unspecified => "UNSPECIFIED",
            MessageType::FornaxCoreConfiguration => "FORNAX_CORE_CONFIGURATION",
            MessageType::NodeConfiguration => "NODE_CONFIGURATION",
            MessageType::NodeRegister => "NODE_REGISTER",
            MessageType::NodeReady => "NODE_READY",
            MessageType::NodeState => "NODE_STATE",
            MessageType::NodeFullSync => "NODE_FULL_SYNC",
            MessageType::PodCreate => "POD_CREATE",
            MessageType::PodTerminate => "POD_TERMINATE",
            MessageType::PodHibernate => "POD_HIBERNATE",
            MessageType::PodState => "POD_STATE",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "UNSPECIFIED" => Some(Self::Unspecified),
            "FORNAX_CORE_CONFIGURATION" => Some(Self::FornaxCoreConfiguration),
            "NODE_CONFIGURATION" => Some(Self::NodeConfiguration),
            "NODE_REGISTER" => Some(Self::NodeRegister),
            "NODE_READY" => Some(Self::NodeReady),
            "NODE_STATE" => Some(Self::NodeState),
            "NODE_FULL_SYNC" => Some(Self::NodeFullSync),
            "POD_CREATE" => Some(Self::PodCreate),
            "POD_TERMINATE" => Some(Self::PodTerminate),
            "POD_HIBERNATE" => Some(Self::PodHibernate),
            "POD_STATE" => Some(Self::PodState),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod fornax_core_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct FornaxCoreServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FornaxCoreServiceClient<tonic::transport::Channel> {
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
    impl<T> FornaxCoreServiceClient<T>
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
        ) -> FornaxCoreServiceClient<InterceptedService<T, F>>
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
            FornaxCoreServiceClient::new(InterceptedService::new(inner, interceptor))
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
        pub async fn get_message(
            &mut self,
            request: impl tonic::IntoRequest<super::NodeIdentifier>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::FornaxCoreMessage>>,
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
                "/node_mgr_pb.FornaxCoreService/getMessage",
            );
            self.inner.server_streaming(request.into_request(), path, codec).await
        }
        pub async fn put_message(
            &mut self,
            request: impl tonic::IntoRequest<super::FornaxCoreMessage>,
        ) -> Result<tonic::Response<()>, tonic::Status> {
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
                "/node_mgr_pb.FornaxCoreService/putMessage",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod fornax_core_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with FornaxCoreServiceServer.
    #[async_trait]
    pub trait FornaxCoreService: Send + Sync + 'static {
        /// Server streaming response type for the getMessage method.
        type getMessageStream: futures_core::Stream<
                Item = Result<super::FornaxCoreMessage, tonic::Status>,
            >
            + Send
            + 'static;
        async fn get_message(
            &self,
            request: tonic::Request<super::NodeIdentifier>,
        ) -> Result<tonic::Response<Self::getMessageStream>, tonic::Status>;
        async fn put_message(
            &self,
            request: tonic::Request<super::FornaxCoreMessage>,
        ) -> Result<tonic::Response<()>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct FornaxCoreServiceServer<T: FornaxCoreService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: FornaxCoreService> FornaxCoreServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for FornaxCoreServiceServer<T>
    where
        T: FornaxCoreService,
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
                "/node_mgr_pb.FornaxCoreService/getMessage" => {
                    #[allow(non_camel_case_types)]
                    struct getMessageSvc<T: FornaxCoreService>(pub Arc<T>);
                    impl<
                        T: FornaxCoreService,
                    > tonic::server::ServerStreamingService<super::NodeIdentifier>
                    for getMessageSvc<T> {
                        type Response = super::FornaxCoreMessage;
                        type ResponseStream = T::getMessageStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::NodeIdentifier>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).get_message(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = getMessageSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = tonic::server::Grpc::new(codec)
                            .apply_compression_config(
                                accept_compression_encodings,
                                send_compression_encodings,
                            );
                        let res = grpc.server_streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/node_mgr_pb.FornaxCoreService/putMessage" => {
                    #[allow(non_camel_case_types)]
                    struct putMessageSvc<T: FornaxCoreService>(pub Arc<T>);
                    impl<
                        T: FornaxCoreService,
                    > tonic::server::UnaryService<super::FornaxCoreMessage>
                    for putMessageSvc<T> {
                        type Response = ();
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::FornaxCoreMessage>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).put_message(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = putMessageSvc(inner);
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
    impl<T: FornaxCoreService> Clone for FornaxCoreServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: FornaxCoreService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: FornaxCoreService> tonic::server::NamedService
    for FornaxCoreServiceServer<T> {
        const NAME: &'static str = "node_mgr_pb.FornaxCoreService";
    }
}
