#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodRegister {
    #[prost(string, tag = "1")]
    pub pod_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodConnect {
    #[prost(uint64, tag = "1")]
    pub msg_id: u64,
    #[prost(uint32, tag = "2")]
    pub addr: u32,
    #[prost(uint32, tag = "3")]
    pub port: u32,
}
