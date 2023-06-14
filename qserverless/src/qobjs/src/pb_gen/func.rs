#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobSvcReq {
    #[prost(uint64, tag = "1")]
    pub msg_id: u64,
    #[prost(oneof = "blob_svc_req::EventBody", tags = "501, 505, 507, 513, 515")]
    pub event_body: ::core::option::Option<blob_svc_req::EventBody>,
}
/// Nested message and enum types in `BlobSvcReq`.
pub mod blob_svc_req {
    #[derive(serde::Serialize, serde::Deserialize)]
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum EventBody {
        #[prost(message, tag = "501")]
        BlobOpenReq(super::BlobOpenReq),
        #[prost(message, tag = "505")]
        BlobReadReq(super::BlobReadReq),
        #[prost(message, tag = "507")]
        BlobSeekReq(super::BlobSeekReq),
        #[prost(message, tag = "513")]
        BlobCloseReq(super::BlobCloseReq),
        #[prost(message, tag = "515")]
        BlobDeleteReq(super::BlobDeleteReq),
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobSvcResp {
    #[prost(uint64, tag = "1")]
    pub msg_id: u64,
    #[prost(oneof = "blob_svc_resp::EventBody", tags = "502, 506, 508, 514, 516")]
    pub event_body: ::core::option::Option<blob_svc_resp::EventBody>,
}
/// Nested message and enum types in `BlobSvcResp`.
pub mod blob_svc_resp {
    #[derive(serde::Serialize, serde::Deserialize)]
    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum EventBody {
        #[prost(message, tag = "502")]
        BlobOpenResp(super::BlobOpenResp),
        #[prost(message, tag = "506")]
        BlobReadResp(super::BlobReadResp),
        #[prost(message, tag = "508")]
        BlobSeekResp(super::BlobSeekResp),
        #[prost(message, tag = "514")]
        BlobCloseResp(super::BlobCloseResp),
        #[prost(message, tag = "516")]
        BlobDeleteResp(super::BlobDeleteResp),
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentMsg {
    #[prost(uint64, tag = "1")]
    pub msg_id: u64,
    #[prost(
        oneof = "func_agent_msg::EventBody",
        tags = "100, 200, 300, 400, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 513, 514, 515, 516"
    )]
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
        #[prost(message, tag = "501")]
        BlobOpenReq(super::BlobOpenReq),
        #[prost(message, tag = "502")]
        BlobOpenResp(super::BlobOpenResp),
        #[prost(message, tag = "503")]
        BlobCreateReq(super::BlobCreateReq),
        #[prost(message, tag = "504")]
        BlobCreateResp(super::BlobCreateResp),
        #[prost(message, tag = "505")]
        BlobReadReq(super::BlobReadReq),
        #[prost(message, tag = "506")]
        BlobReadResp(super::BlobReadResp),
        #[prost(message, tag = "507")]
        BlobSeekReq(super::BlobSeekReq),
        #[prost(message, tag = "508")]
        BlobSeekResp(super::BlobSeekResp),
        #[prost(message, tag = "509")]
        BlobWriteReq(super::BlobWriteReq),
        #[prost(message, tag = "510")]
        BlobWriteResp(super::BlobWriteResp),
        #[prost(message, tag = "513")]
        BlobCloseReq(super::BlobCloseReq),
        #[prost(message, tag = "514")]
        BlobCloseResp(super::BlobCloseResp),
        #[prost(message, tag = "515")]
        BlobDeleteReq(super::BlobDeleteReq),
        #[prost(message, tag = "516")]
        BlobDeleteResp(super::BlobDeleteResp),
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobOpenReq {
    #[prost(string, tag = "2")]
    pub svc_addr: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub name: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobOpenResp {
    #[prost(uint64, tag = "2")]
    pub id: u64,
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub name: ::prost::alloc::string::String,
    #[prost(uint64, tag = "5")]
    pub size: u64,
    #[prost(string, tag = "6")]
    pub checksum: ::prost::alloc::string::String,
    #[prost(message, optional, tag = "7")]
    pub create_time: ::core::option::Option<Timestamp>,
    #[prost(message, optional, tag = "8")]
    pub last_access_time: ::core::option::Option<Timestamp>,
    #[prost(string, tag = "9")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobDeleteReq {
    #[prost(string, tag = "2")]
    pub svc_addr: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub name: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobDeleteResp {
    #[prost(string, tag = "1")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobCreateReq {
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub name: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobCreateResp {
    #[prost(uint64, tag = "2")]
    pub id: u64,
    #[prost(string, tag = "3")]
    pub svc_addr: ::prost::alloc::string::String,
    #[prost(string, tag = "9")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobReadReq {
    #[prost(uint64, tag = "2")]
    pub id: u64,
    #[prost(uint64, tag = "3")]
    pub len: u64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobReadResp {
    #[prost(bytes = "vec", tag = "3")]
    pub data: ::prost::alloc::vec::Vec<u8>,
    #[prost(string, tag = "4")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobSeekReq {
    #[prost(uint64, tag = "2")]
    pub id: u64,
    #[prost(int64, tag = "3")]
    pub pos: i64,
    #[prost(uint32, tag = "4")]
    pub seek_type: u32,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobSeekResp {
    #[prost(uint64, tag = "2")]
    pub offset: u64,
    #[prost(string, tag = "3")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobCloseReq {
    #[prost(uint64, tag = "2")]
    pub id: u64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobCloseResp {
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobWriteReq {
    #[prost(uint64, tag = "2")]
    pub id: u64,
    #[prost(bytes = "vec", tag = "3")]
    pub data: ::prost::alloc::vec::Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobWriteResp {
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobSealReq {
    #[prost(uint64, tag = "2")]
    pub id: u64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BlobSealResp {
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodRegisterReq {
    #[prost(string, tag = "1")]
    pub func_pod_id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub package_name: ::prost::alloc::string::String,
    /// client mode pod will only send func call request, can't serve func call request
    #[prost(bool, tag = "4")]
    pub client_mode: bool,
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
    pub id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub package_name: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub func_name: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub parameters: ::prost::alloc::string::String,
    #[prost(uint64, tag = "6")]
    pub priority: u64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncAgentCallResp {
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub resp: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncSvcMsg {
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
    /// func calls from the node
    #[prost(message, repeated, tag = "2")]
    pub caller_calls: ::prost::alloc::vec::Vec<FuncSvcCallReq>,
    /// func calls processing in the node
    #[prost(message, repeated, tag = "3")]
    pub callee_calls: ::prost::alloc::vec::Vec<FuncSvcCallReq>,
    /// func pods running on the node
    #[prost(message, repeated, tag = "4")]
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
    #[prost(string, tag = "2")]
    pub func_pod_id: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub package_name: ::prost::alloc::string::String,
    #[prost(bool, tag = "5")]
    pub client_mode: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodConnResp {
    #[prost(string, tag = "1")]
    pub func_pod_id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodDisconnReq {
    #[prost(string, tag = "1")]
    pub func_pod_id: ::prost::alloc::string::String,
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
    pub id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub package_name: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub func_name: ::prost::alloc::string::String,
    #[prost(string, tag = "5")]
    pub parameters: ::prost::alloc::string::String,
    #[prost(uint64, tag = "6")]
    pub priority: u64,
    #[prost(message, optional, tag = "7")]
    pub createtime: ::core::option::Option<Timestamp>,
    #[prost(string, tag = "8")]
    pub caller_node_id: ::prost::alloc::string::String,
    #[prost(string, tag = "9")]
    pub caller_pod_id: ::prost::alloc::string::String,
    /// when funcCall is process by a funcPod, this is the NodeId
    #[prost(string, tag = "10")]
    pub callee_node_id: ::prost::alloc::string::String,
    #[prost(string, tag = "11")]
    pub callee_pod_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncSvcCallResp {
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub error: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub resp: ::prost::alloc::string::String,
    #[prost(string, tag = "8")]
    pub caller_node_id: ::prost::alloc::string::String,
    #[prost(string, tag = "9")]
    pub caller_pod_id: ::prost::alloc::string::String,
    /// when funcCall is process by a funcPod, this is the NodeId
    #[prost(string, tag = "10")]
    pub callee_node_id: ::prost::alloc::string::String,
    #[prost(string, tag = "11")]
    pub callee_pod_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FuncPodStatus {
    #[prost(string, tag = "1")]
    pub func_pod_id: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub namespace: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub package_name: ::prost::alloc::string::String,
    #[prost(enumeration = "FuncPodState", tag = "4")]
    pub state: i32,
    /// when the pod is running the funcCallId
    #[prost(string, tag = "5")]
    pub func_call_id: ::prost::alloc::string::String,
    #[prost(bool, tag = "6")]
    pub client_mode: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Timestamp {
    #[prost(uint64, tag = "1")]
    pub seconds: u64,
    #[prost(uint32, tag = "2")]
    pub nanos: u32,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum FuncPodState {
    Idle = 0,
    Running = 1,
}
impl FuncPodState {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            FuncPodState::Idle => "Idle",
            FuncPodState::Running => "Running",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "Idle" => Some(Self::Idle),
            "Running" => Some(Self::Running),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod blob_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    #[derive(Debug, Clone)]
    pub struct BlobServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl BlobServiceClient<tonic::transport::Channel> {
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
    impl<T> BlobServiceClient<T>
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
        ) -> BlobServiceClient<InterceptedService<T, F>>
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
            BlobServiceClient::new(InterceptedService::new(inner, interceptor))
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
            request: impl tonic::IntoStreamingRequest<Message = super::BlobSvcReq>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::BlobSvcResp>>,
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
                "/func.BlobService/StreamProcess",
            );
            self.inner.streaming(request.into_streaming_request(), path, codec).await
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
pub mod blob_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with BlobServiceServer.
    #[async_trait]
    pub trait BlobService: Send + Sync + 'static {
        /// Server streaming response type for the StreamProcess method.
        type StreamProcessStream: futures_core::Stream<
                Item = Result<super::BlobSvcResp, tonic::Status>,
            >
            + Send
            + 'static;
        async fn stream_process(
            &self,
            request: tonic::Request<tonic::Streaming<super::BlobSvcReq>>,
        ) -> Result<tonic::Response<Self::StreamProcessStream>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct BlobServiceServer<T: BlobService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: BlobService> BlobServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for BlobServiceServer<T>
    where
        T: BlobService,
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
                "/func.BlobService/StreamProcess" => {
                    #[allow(non_camel_case_types)]
                    struct StreamProcessSvc<T: BlobService>(pub Arc<T>);
                    impl<
                        T: BlobService,
                    > tonic::server::StreamingService<super::BlobSvcReq>
                    for StreamProcessSvc<T> {
                        type Response = super::BlobSvcResp;
                        type ResponseStream = T::StreamProcessStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::BlobSvcReq>>,
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
    impl<T: BlobService> Clone for BlobServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: BlobService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: BlobService> tonic::server::NamedService for BlobServiceServer<T> {
        const NAME: &'static str = "func.BlobService";
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
