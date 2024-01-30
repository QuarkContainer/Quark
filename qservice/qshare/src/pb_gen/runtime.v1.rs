#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VersionRequest {
    /// Version of the kubelet runtime API.
    #[prost(string, tag = "1")]
    pub version: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct VersionResponse {
    /// Version of the kubelet runtime API.
    #[prost(string, tag = "1")]
    pub version: ::prost::alloc::string::String,
    /// Name of the container runtime.
    #[prost(string, tag = "2")]
    pub runtime_name: ::prost::alloc::string::String,
    /// Version of the container runtime. The string must be
    /// semver-compatible.
    #[prost(string, tag = "3")]
    pub runtime_version: ::prost::alloc::string::String,
    /// API version of the container runtime. The string must be
    /// semver-compatible.
    #[prost(string, tag = "4")]
    pub runtime_api_version: ::prost::alloc::string::String,
}
/// DNSConfig specifies the DNS servers and search domains of a sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DnsConfig {
    /// List of DNS servers of the cluster.
    #[prost(string, repeated, tag = "1")]
    pub servers: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// List of DNS search domains of the cluster.
    #[prost(string, repeated, tag = "2")]
    pub searches: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// List of DNS options. See <https://linux.die.net/man/5/resolv.conf>
    /// for all available options.
    #[prost(string, repeated, tag = "3")]
    pub options: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}
/// PortMapping specifies the port mapping configurations of a sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PortMapping {
    /// Protocol of the port mapping.
    #[prost(enumeration = "Protocol", tag = "1")]
    pub protocol: i32,
    /// Port number within the container. Default: 0 (not specified).
    #[prost(int32, tag = "2")]
    pub container_port: i32,
    /// Port number on the host. Default: 0 (not specified).
    #[prost(int32, tag = "3")]
    pub host_port: i32,
    /// Host IP.
    #[prost(string, tag = "4")]
    pub host_ip: ::prost::alloc::string::String,
}
/// Mount specifies a host volume to mount into a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Mount {
    /// Path of the mount within the container.
    #[prost(string, tag = "1")]
    pub container_path: ::prost::alloc::string::String,
    /// Path of the mount on the host. If the hostPath doesn't exist, then runtimes
    /// should report error. If the hostpath is a symbolic link, runtimes should
    /// follow the symlink and mount the real destination to container.
    #[prost(string, tag = "2")]
    pub host_path: ::prost::alloc::string::String,
    /// If set, the mount is read-only.
    #[prost(bool, tag = "3")]
    pub readonly: bool,
    /// If set, the mount needs SELinux relabeling.
    #[prost(bool, tag = "4")]
    pub selinux_relabel: bool,
    /// Requested propagation mode.
    #[prost(enumeration = "MountPropagation", tag = "5")]
    pub propagation: i32,
}
/// IDMapping describes host to container ID mappings for a pod sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IdMapping {
    /// HostId is the id on the host.
    #[prost(uint32, tag = "1")]
    pub host_id: u32,
    /// ContainerId is the id in the container.
    #[prost(uint32, tag = "2")]
    pub container_id: u32,
    /// Length is the size of the range to map.
    #[prost(uint32, tag = "3")]
    pub length: u32,
}
/// UserNamespace describes the intended user namespace configuration for a pod sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UserNamespace {
    /// Mode is the NamespaceMode for this UserNamespace.
    /// Note: NamespaceMode for UserNamespace currently supports only POD and NODE, not CONTAINER OR TARGET.
    #[prost(enumeration = "NamespaceMode", tag = "1")]
    pub mode: i32,
    /// Uids specifies the UID mappings for the user namespace.
    #[prost(message, repeated, tag = "2")]
    pub uids: ::prost::alloc::vec::Vec<IdMapping>,
    /// Gids specifies the GID mappings for the user namespace.
    #[prost(message, repeated, tag = "3")]
    pub gids: ::prost::alloc::vec::Vec<IdMapping>,
}
/// NamespaceOption provides options for Linux namespaces.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NamespaceOption {
    /// Network namespace for this container/sandbox.
    /// Note: There is currently no way to set CONTAINER scoped network in the Kubernetes API.
    /// Namespaces currently set by the kubelet: POD, NODE
    #[prost(enumeration = "NamespaceMode", tag = "1")]
    pub network: i32,
    /// PID namespace for this container/sandbox.
    /// Note: The CRI default is POD, but the v1.PodSpec default is CONTAINER.
    /// The kubelet's runtime manager will set this to CONTAINER explicitly for v1 pods.
    /// Namespaces currently set by the kubelet: POD, CONTAINER, NODE, TARGET
    #[prost(enumeration = "NamespaceMode", tag = "2")]
    pub pid: i32,
    /// IPC namespace for this container/sandbox.
    /// Note: There is currently no way to set CONTAINER scoped IPC in the Kubernetes API.
    /// Namespaces currently set by the kubelet: POD, NODE
    #[prost(enumeration = "NamespaceMode", tag = "3")]
    pub ipc: i32,
    /// Target Container ID for NamespaceMode of TARGET. This container must have been
    /// previously created in the same pod. It is not possible to specify different targets
    /// for each namespace.
    #[prost(string, tag = "4")]
    pub target_id: ::prost::alloc::string::String,
    /// UsernsOptions for this pod sandbox.
    /// The Kubelet picks the user namespace configuration to use for the pod sandbox.  The mappings
    /// are specified as part of the UserNamespace struct.  If the struct is nil, then the POD mode
    /// must be assumed.  This is done for backward compatibility with older Kubelet versions that
    /// do not set a user namespace.
    #[prost(message, optional, tag = "5")]
    pub userns_options: ::core::option::Option<UserNamespace>,
}
/// Int64Value is the wrapper of int64.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Int64Value {
    /// The value.
    #[prost(int64, tag = "1")]
    pub value: i64,
}
/// LinuxSandboxSecurityContext holds linux security configuration that will be
/// applied to a sandbox. Note that:
/// 1) It does not apply to containers in the pods.
/// 2) It may not be applicable to a PodSandbox which does not contain any running
///     process.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxSandboxSecurityContext {
    /// Configurations for the sandbox's namespaces.
    /// This will be used only if the PodSandbox uses namespace for isolation.
    #[prost(message, optional, tag = "1")]
    pub namespace_options: ::core::option::Option<NamespaceOption>,
    /// Optional SELinux context to be applied.
    #[prost(message, optional, tag = "2")]
    pub selinux_options: ::core::option::Option<SeLinuxOption>,
    /// UID to run sandbox processes as, when applicable.
    #[prost(message, optional, tag = "3")]
    pub run_as_user: ::core::option::Option<Int64Value>,
    /// GID to run sandbox processes as, when applicable. run_as_group should only
    /// be specified when run_as_user is specified; otherwise, the runtime MUST error.
    #[prost(message, optional, tag = "8")]
    pub run_as_group: ::core::option::Option<Int64Value>,
    /// If set, the root filesystem of the sandbox is read-only.
    #[prost(bool, tag = "4")]
    pub readonly_rootfs: bool,
    /// List of groups applied to the first process run in the sandbox, in
    /// addition to the sandbox's primary GID.
    #[prost(int64, repeated, tag = "5")]
    pub supplemental_groups: ::prost::alloc::vec::Vec<i64>,
    /// Indicates whether the sandbox will be asked to run a privileged
    /// container. If a privileged container is to be executed within it, this
    /// MUST be true.
    /// This allows a sandbox to take additional security precautions if no
    /// privileged containers are expected to be run.
    #[prost(bool, tag = "6")]
    pub privileged: bool,
    /// Seccomp profile for the sandbox.
    #[prost(message, optional, tag = "9")]
    pub seccomp: ::core::option::Option<SecurityProfile>,
    /// AppArmor profile for the sandbox.
    #[prost(message, optional, tag = "10")]
    pub apparmor: ::core::option::Option<SecurityProfile>,
    /// Seccomp profile for the sandbox, candidate values are:
    /// * runtime/default: the default profile for the container runtime
    /// * unconfined: unconfined profile, ie, no seccomp sandboxing
    /// * localhost/<full-path-to-profile>: the profile installed on the node.
    ///    <full-path-to-profile> is the full path of the profile.
    /// Default: "", which is identical with unconfined.
    #[deprecated]
    #[prost(string, tag = "7")]
    pub seccomp_profile_path: ::prost::alloc::string::String,
}
/// A security profile which can be used for sandboxes and containers.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SecurityProfile {
    /// Indicator which `ProfileType` should be applied.
    #[prost(enumeration = "security_profile::ProfileType", tag = "1")]
    pub profile_type: i32,
    /// Indicates that a pre-defined profile on the node should be used.
    /// Must only be set if `ProfileType` is `Localhost`.
    /// For seccomp, it must be an absolute path to the seccomp profile.
    /// For AppArmor, this field is the AppArmor `<profile name>/`
    #[prost(string, tag = "2")]
    pub localhost_ref: ::prost::alloc::string::String,
}
/// Nested message and enum types in `SecurityProfile`.
pub mod security_profile {
    /// Available profile types.
    #[derive(serde::Serialize, serde::Deserialize)]
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
    pub enum ProfileType {
        /// The container runtime default profile should be used.
        RuntimeDefault = 0,
        /// Disable the feature for the sandbox or the container.
        Unconfined = 1,
        /// A pre-defined profile on the node should be used.
        Localhost = 2,
    }
    impl ProfileType {
        /// String value of the enum field names used in the ProtoBuf definition.
        ///
        /// The values are not transformed in any way and thus are considered stable
        /// (if the ProtoBuf definition does not change) and safe for programmatic use.
        pub fn as_str_name(&self) -> &'static str {
            match self {
                ProfileType::RuntimeDefault => "RuntimeDefault",
                ProfileType::Unconfined => "Unconfined",
                ProfileType::Localhost => "Localhost",
            }
        }
        /// Creates an enum from field names used in the ProtoBuf definition.
        pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
            match value {
                "RuntimeDefault" => Some(Self::RuntimeDefault),
                "Unconfined" => Some(Self::Unconfined),
                "Localhost" => Some(Self::Localhost),
                _ => None,
            }
        }
    }
}
/// LinuxPodSandboxConfig holds platform-specific configurations for Linux
/// host platforms and Linux-based containers.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxPodSandboxConfig {
    /// Parent cgroup of the PodSandbox.
    /// The cgroupfs style syntax will be used, but the container runtime can
    /// convert it to systemd semantics if needed.
    #[prost(string, tag = "1")]
    pub cgroup_parent: ::prost::alloc::string::String,
    /// LinuxSandboxSecurityContext holds sandbox security attributes.
    #[prost(message, optional, tag = "2")]
    pub security_context: ::core::option::Option<LinuxSandboxSecurityContext>,
    /// Sysctls holds linux sysctls config for the sandbox.
    #[prost(map = "string, string", tag = "3")]
    pub sysctls: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Optional overhead represents the overheads associated with this sandbox
    #[prost(message, optional, tag = "4")]
    pub overhead: ::core::option::Option<LinuxContainerResources>,
    /// Optional resources represents the sum of container resources for this sandbox
    #[prost(message, optional, tag = "5")]
    pub resources: ::core::option::Option<LinuxContainerResources>,
}
/// PodSandboxMetadata holds all necessary information for building the sandbox name.
/// The container runtime is encouraged to expose the metadata associated with the
/// PodSandbox in its user interface for better user experience. For example,
/// the runtime can construct a unique PodSandboxName based on the metadata.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxMetadata {
    /// Pod name of the sandbox. Same as the pod name in the Pod ObjectMeta.
    #[prost(string, tag = "1")]
    pub name: ::prost::alloc::string::String,
    /// Pod UID of the sandbox. Same as the pod UID in the Pod ObjectMeta.
    #[prost(string, tag = "2")]
    pub uid: ::prost::alloc::string::String,
    /// Pod namespace of the sandbox. Same as the pod namespace in the Pod ObjectMeta.
    #[prost(string, tag = "3")]
    pub namespace: ::prost::alloc::string::String,
    /// Attempt number of creating the sandbox. Default: 0.
    #[prost(uint32, tag = "4")]
    pub attempt: u32,
}
/// PodSandboxConfig holds all the required and optional fields for creating a
/// sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxConfig {
    /// Metadata of the sandbox. This information will uniquely identify the
    /// sandbox, and the runtime should leverage this to ensure correct
    /// operation. The runtime may also use this information to improve UX, such
    /// as by constructing a readable name.
    #[prost(message, optional, tag = "1")]
    pub metadata: ::core::option::Option<PodSandboxMetadata>,
    /// Hostname of the sandbox. Hostname could only be empty when the pod
    /// network namespace is NODE.
    #[prost(string, tag = "2")]
    pub hostname: ::prost::alloc::string::String,
    /// Path to the directory on the host in which container log files are
    /// stored.
    /// By default the log of a container going into the LogDirectory will be
    /// hooked up to STDOUT and STDERR. However, the LogDirectory may contain
    /// binary log files with structured logging data from the individual
    /// containers. For example, the files might be newline separated JSON
    /// structured logs, systemd-journald journal files, gRPC trace files, etc.
    /// E.g.,
    ///      PodSandboxConfig.LogDirectory = `/var/log/pods/<podUID>/`
    ///      ContainerConfig.LogPath = `containerName/Instance#.log`
    #[prost(string, tag = "3")]
    pub log_directory: ::prost::alloc::string::String,
    /// DNS config for the sandbox.
    #[prost(message, optional, tag = "4")]
    pub dns_config: ::core::option::Option<DnsConfig>,
    /// Port mappings for the sandbox.
    #[prost(message, repeated, tag = "5")]
    pub port_mappings: ::prost::alloc::vec::Vec<PortMapping>,
    /// Key-value pairs that may be used to scope and select individual resources.
    #[prost(map = "string, string", tag = "6")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map that may be set by the kubelet to store and
    /// retrieve arbitrary metadata. This will include any annotations set on a
    /// pod through the Kubernetes API.
    ///
    /// Annotations MUST NOT be altered by the runtime; the annotations stored
    /// here MUST be returned in the PodSandboxStatus associated with the pod
    /// this PodSandboxConfig creates.
    ///
    /// In general, in order to preserve a well-defined interface between the
    /// kubelet and the container runtime, annotations SHOULD NOT influence
    /// runtime behaviour.
    ///
    /// Annotations can also be useful for runtime authors to experiment with
    /// new features that are opaque to the Kubernetes APIs (both user-facing
    /// and the CRI). Whenever possible, however, runtime authors SHOULD
    /// consider proposing new typed fields for any new features instead.
    #[prost(map = "string, string", tag = "7")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Optional configurations specific to Linux hosts.
    #[prost(message, optional, tag = "8")]
    pub linux: ::core::option::Option<LinuxPodSandboxConfig>,
    /// Optional configurations specific to Windows hosts.
    #[prost(message, optional, tag = "9")]
    pub windows: ::core::option::Option<WindowsPodSandboxConfig>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunPodSandboxRequest {
    /// Configuration for creating a PodSandbox.
    #[prost(message, optional, tag = "1")]
    pub config: ::core::option::Option<PodSandboxConfig>,
    /// Named runtime configuration to use for this PodSandbox.
    /// If the runtime handler is unknown, this request should be rejected.  An
    /// empty string should select the default handler, equivalent to the
    /// behavior before this feature was added.
    /// See <https://git.k8s.io/enhancements/keps/sig-node/585-runtime-class>
    #[prost(string, tag = "2")]
    pub runtime_handler: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunPodSandboxResponse {
    /// ID of the PodSandbox to run.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StopPodSandboxRequest {
    /// ID of the PodSandbox to stop.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StopPodSandboxResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemovePodSandboxRequest {
    /// ID of the PodSandbox to remove.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemovePodSandboxResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStatusRequest {
    /// ID of the PodSandbox for which to retrieve status.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
    /// Verbose indicates whether to return extra information about the pod sandbox.
    #[prost(bool, tag = "2")]
    pub verbose: bool,
}
/// PodIP represents an ip of a Pod
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodIp {
    /// an ip is a string representation of an IPv4 or an IPv6
    #[prost(string, tag = "1")]
    pub ip: ::prost::alloc::string::String,
}
/// PodSandboxNetworkStatus is the status of the network for a PodSandbox.
/// Currently ignored for pods sharing the host networking namespace.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxNetworkStatus {
    /// IP address of the PodSandbox.
    #[prost(string, tag = "1")]
    pub ip: ::prost::alloc::string::String,
    /// list of additional ips (not inclusive of PodSandboxNetworkStatus.Ip) of the PodSandBoxNetworkStatus
    #[prost(message, repeated, tag = "2")]
    pub additional_ips: ::prost::alloc::vec::Vec<PodIp>,
}
/// Namespace contains paths to the namespaces.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Namespace {
    /// Namespace options for Linux namespaces.
    #[prost(message, optional, tag = "2")]
    pub options: ::core::option::Option<NamespaceOption>,
}
/// LinuxSandboxStatus contains status specific to Linux sandboxes.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxPodSandboxStatus {
    /// Paths to the sandbox's namespaces.
    #[prost(message, optional, tag = "1")]
    pub namespaces: ::core::option::Option<Namespace>,
}
/// PodSandboxStatus contains the status of the PodSandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStatus {
    /// ID of the sandbox.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// Metadata of the sandbox.
    #[prost(message, optional, tag = "2")]
    pub metadata: ::core::option::Option<PodSandboxMetadata>,
    /// State of the sandbox.
    #[prost(enumeration = "PodSandboxState", tag = "3")]
    pub state: i32,
    /// Creation timestamp of the sandbox in nanoseconds. Must be > 0.
    #[prost(int64, tag = "4")]
    pub created_at: i64,
    /// Network contains network status if network is handled by the runtime.
    #[prost(message, optional, tag = "5")]
    pub network: ::core::option::Option<PodSandboxNetworkStatus>,
    /// Linux-specific status to a pod sandbox.
    #[prost(message, optional, tag = "6")]
    pub linux: ::core::option::Option<LinuxPodSandboxStatus>,
    /// Labels are key-value pairs that may be used to scope and select individual resources.
    #[prost(map = "string, string", tag = "7")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map holding arbitrary metadata.
    /// Annotations MUST NOT be altered by the runtime; the value of this field
    /// MUST be identical to that of the corresponding PodSandboxConfig used to
    /// instantiate the pod sandbox this status represents.
    #[prost(map = "string, string", tag = "8")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// runtime configuration used for this PodSandbox.
    #[prost(string, tag = "9")]
    pub runtime_handler: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStatusResponse {
    /// Status of the PodSandbox.
    #[prost(message, optional, tag = "1")]
    pub status: ::core::option::Option<PodSandboxStatus>,
    /// Info is extra information of the PodSandbox. The key could be arbitrary string, and
    /// value should be in json format. The information could include anything useful for
    /// debug, e.g. network namespace for linux container based container runtime.
    /// It should only be returned non-empty when Verbose is true.
    #[prost(map = "string, string", tag = "2")]
    pub info: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
/// PodSandboxStateValue is the wrapper of PodSandboxState.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStateValue {
    /// State of the sandbox.
    #[prost(enumeration = "PodSandboxState", tag = "1")]
    pub state: i32,
}
/// PodSandboxFilter is used to filter a list of PodSandboxes.
/// All those fields are combined with 'AND'
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxFilter {
    /// ID of the sandbox.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// State of the sandbox.
    #[prost(message, optional, tag = "2")]
    pub state: ::core::option::Option<PodSandboxStateValue>,
    /// LabelSelector to select matches.
    /// Only api.MatchLabels is supported for now and the requirements
    /// are ANDed. MatchExpressions is not supported yet.
    #[prost(map = "string, string", tag = "3")]
    pub label_selector: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListPodSandboxRequest {
    /// PodSandboxFilter to filter a list of PodSandboxes.
    #[prost(message, optional, tag = "1")]
    pub filter: ::core::option::Option<PodSandboxFilter>,
}
/// PodSandbox contains minimal information about a sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandbox {
    /// ID of the PodSandbox.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// Metadata of the PodSandbox.
    #[prost(message, optional, tag = "2")]
    pub metadata: ::core::option::Option<PodSandboxMetadata>,
    /// State of the PodSandbox.
    #[prost(enumeration = "PodSandboxState", tag = "3")]
    pub state: i32,
    /// Creation timestamps of the PodSandbox in nanoseconds. Must be > 0.
    #[prost(int64, tag = "4")]
    pub created_at: i64,
    /// Labels of the PodSandbox.
    #[prost(map = "string, string", tag = "5")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map holding arbitrary metadata.
    /// Annotations MUST NOT be altered by the runtime; the value of this field
    /// MUST be identical to that of the corresponding PodSandboxConfig used to
    /// instantiate this PodSandbox.
    #[prost(map = "string, string", tag = "6")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// runtime configuration used for this PodSandbox.
    #[prost(string, tag = "7")]
    pub runtime_handler: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListPodSandboxResponse {
    /// List of PodSandboxes.
    #[prost(message, repeated, tag = "1")]
    pub items: ::prost::alloc::vec::Vec<PodSandbox>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStatsRequest {
    /// ID of the pod sandbox for which to retrieve stats.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStatsResponse {
    #[prost(message, optional, tag = "1")]
    pub stats: ::core::option::Option<PodSandboxStats>,
}
/// PodSandboxStatsFilter is used to filter the list of pod sandboxes to retrieve stats for.
/// All those fields are combined with 'AND'.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStatsFilter {
    /// ID of the pod sandbox.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// LabelSelector to select matches.
    /// Only api.MatchLabels is supported for now and the requirements
    /// are ANDed. MatchExpressions is not supported yet.
    #[prost(map = "string, string", tag = "2")]
    pub label_selector: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListPodSandboxStatsRequest {
    /// Filter for the list request.
    #[prost(message, optional, tag = "1")]
    pub filter: ::core::option::Option<PodSandboxStatsFilter>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListPodSandboxStatsResponse {
    /// Stats of the pod sandbox.
    #[prost(message, repeated, tag = "1")]
    pub stats: ::prost::alloc::vec::Vec<PodSandboxStats>,
}
/// PodSandboxAttributes provides basic information of the pod sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxAttributes {
    /// ID of the pod sandbox.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// Metadata of the pod sandbox.
    #[prost(message, optional, tag = "2")]
    pub metadata: ::core::option::Option<PodSandboxMetadata>,
    /// Key-value pairs that may be used to scope and select individual resources.
    #[prost(map = "string, string", tag = "3")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map holding arbitrary metadata.
    /// Annotations MUST NOT be altered by the runtime; the value of this field
    /// MUST be identical to that of the corresponding PodSandboxStatus used to
    /// instantiate the PodSandbox this status represents.
    #[prost(map = "string, string", tag = "4")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
/// PodSandboxStats provides the resource usage statistics for a pod.
/// The linux or windows field will be populated depending on the platform.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PodSandboxStats {
    /// Information of the pod.
    #[prost(message, optional, tag = "1")]
    pub attributes: ::core::option::Option<PodSandboxAttributes>,
    /// Stats from linux.
    #[prost(message, optional, tag = "2")]
    pub linux: ::core::option::Option<LinuxPodSandboxStats>,
    /// Stats from windows.
    #[prost(message, optional, tag = "3")]
    pub windows: ::core::option::Option<WindowsPodSandboxStats>,
}
/// LinuxPodSandboxStats provides the resource usage statistics for a pod sandbox on linux.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxPodSandboxStats {
    /// CPU usage gathered for the pod sandbox.
    #[prost(message, optional, tag = "1")]
    pub cpu: ::core::option::Option<CpuUsage>,
    /// Memory usage gathered for the pod sandbox.
    #[prost(message, optional, tag = "2")]
    pub memory: ::core::option::Option<MemoryUsage>,
    /// Network usage gathered for the pod sandbox
    #[prost(message, optional, tag = "3")]
    pub network: ::core::option::Option<NetworkUsage>,
    /// Stats pertaining to processes in the pod sandbox.
    #[prost(message, optional, tag = "4")]
    pub process: ::core::option::Option<ProcessUsage>,
    /// Stats of containers in the measured pod sandbox.
    #[prost(message, repeated, tag = "5")]
    pub containers: ::prost::alloc::vec::Vec<ContainerStats>,
}
/// WindowsPodSandboxStats provides the resource usage statistics for a pod sandbox on windows
///
/// TODO: Add stats relevant to windows.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowsPodSandboxStats {}
/// NetworkUsage contains data about network resources.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NetworkUsage {
    /// The time at which these stats were updated.
    #[prost(int64, tag = "1")]
    pub timestamp: i64,
    /// Stats for the default network interface.
    #[prost(message, optional, tag = "2")]
    pub default_interface: ::core::option::Option<NetworkInterfaceUsage>,
    /// Stats for all found network interfaces, excluding the default.
    #[prost(message, repeated, tag = "3")]
    pub interfaces: ::prost::alloc::vec::Vec<NetworkInterfaceUsage>,
}
/// NetworkInterfaceUsage contains resource value data about a network interface.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NetworkInterfaceUsage {
    /// The name of the network interface.
    #[prost(string, tag = "1")]
    pub name: ::prost::alloc::string::String,
    /// Cumulative count of bytes received.
    #[prost(message, optional, tag = "2")]
    pub rx_bytes: ::core::option::Option<UInt64Value>,
    /// Cumulative count of receive errors encountered.
    #[prost(message, optional, tag = "3")]
    pub rx_errors: ::core::option::Option<UInt64Value>,
    /// Cumulative count of bytes transmitted.
    #[prost(message, optional, tag = "4")]
    pub tx_bytes: ::core::option::Option<UInt64Value>,
    /// Cumulative count of transmit errors encountered.
    #[prost(message, optional, tag = "5")]
    pub tx_errors: ::core::option::Option<UInt64Value>,
}
/// ProcessUsage are stats pertaining to processes.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ProcessUsage {
    /// The time at which these stats were updated.
    #[prost(int64, tag = "1")]
    pub timestamp: i64,
    /// Number of processes.
    #[prost(message, optional, tag = "2")]
    pub process_count: ::core::option::Option<UInt64Value>,
}
/// ImageSpec is an internal representation of an image.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageSpec {
    /// Container's Image field (e.g. imageID or imageDigest).
    #[prost(string, tag = "1")]
    pub image: ::prost::alloc::string::String,
    /// Unstructured key-value map holding arbitrary metadata.
    /// ImageSpec Annotations can be used to help the runtime target specific
    /// images in multi-arch images.
    #[prost(map = "string, string", tag = "2")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KeyValue {
    #[prost(string, tag = "1")]
    pub key: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub value: ::prost::alloc::string::String,
}
/// LinuxContainerResources specifies Linux specific configuration for
/// resources.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxContainerResources {
    /// CPU CFS (Completely Fair Scheduler) period. Default: 0 (not specified).
    #[prost(int64, tag = "1")]
    pub cpu_period: i64,
    /// CPU CFS (Completely Fair Scheduler) quota. Default: 0 (not specified).
    #[prost(int64, tag = "2")]
    pub cpu_quota: i64,
    /// CPU shares (relative weight vs. other containers). Default: 0 (not specified).
    #[prost(int64, tag = "3")]
    pub cpu_shares: i64,
    /// Memory limit in bytes. Default: 0 (not specified).
    #[prost(int64, tag = "4")]
    pub memory_limit_in_bytes: i64,
    /// OOMScoreAdj adjusts the oom-killer score. Default: 0 (not specified).
    #[prost(int64, tag = "5")]
    pub oom_score_adj: i64,
    /// CpusetCpus constrains the allowed set of logical CPUs. Default: "" (not specified).
    #[prost(string, tag = "6")]
    pub cpuset_cpus: ::prost::alloc::string::String,
    /// CpusetMems constrains the allowed set of memory nodes. Default: "" (not specified).
    #[prost(string, tag = "7")]
    pub cpuset_mems: ::prost::alloc::string::String,
    /// List of HugepageLimits to limit the HugeTLB usage of container per page size. Default: nil (not specified).
    #[prost(message, repeated, tag = "8")]
    pub hugepage_limits: ::prost::alloc::vec::Vec<HugepageLimit>,
    /// Unified resources for cgroup v2. Default: nil (not specified).
    /// Each key/value in the map refers to the cgroup v2.
    /// e.g. "memory.max": "6937202688" or "io.weight": "default 100".
    #[prost(map = "string, string", tag = "9")]
    pub unified: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Memory swap limit in bytes. Default 0 (not specified).
    #[prost(int64, tag = "10")]
    pub memory_swap_limit_in_bytes: i64,
}
/// HugepageLimit corresponds to the file`hugetlb.<hugepagesize>.limit_in_byte` in container level cgroup.
/// For example, `PageSize=1GB`, `Limit=1073741824` means setting `1073741824` bytes to hugetlb.1GB.limit_in_bytes.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HugepageLimit {
    /// The value of PageSize has the format <size><unit-prefix>B (2MB, 1GB),
    /// and must match the <hugepagesize> of the corresponding control file found in `hugetlb.<hugepagesize>.limit_in_bytes`.
    /// The values of <unit-prefix> are intended to be parsed using base 1024("1KB" = 1024, "1MB" = 1048576, etc).
    #[prost(string, tag = "1")]
    pub page_size: ::prost::alloc::string::String,
    /// limit in bytes of hugepagesize HugeTLB usage.
    #[prost(uint64, tag = "2")]
    pub limit: u64,
}
/// SELinuxOption are the labels to be applied to the container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SeLinuxOption {
    #[prost(string, tag = "1")]
    pub user: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub role: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub r#type: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub level: ::prost::alloc::string::String,
}
/// Capability contains the container capabilities to add or drop
/// Dropping a capability will drop it from all sets.
/// If a capability is added to only the add_capabilities list then it gets added to permitted,
/// inheritable, effective and bounding sets, i.e. all sets except the ambient set.
/// If a capability is added to only the add_ambient_capabilities list then it gets added to all sets, i.e permitted
/// inheritable, effective, bounding and ambient sets.
/// If a capability is added to add_capabilities and add_ambient_capabilities lists then it gets added to all sets, i.e.
/// permitted, inheritable, effective, bounding and ambient sets.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Capability {
    /// List of capabilities to add.
    #[prost(string, repeated, tag = "1")]
    pub add_capabilities: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// List of capabilities to drop.
    #[prost(string, repeated, tag = "2")]
    pub drop_capabilities: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// List of ambient capabilities to add.
    #[prost(string, repeated, tag = "3")]
    pub add_ambient_capabilities: ::prost::alloc::vec::Vec<
        ::prost::alloc::string::String,
    >,
}
/// LinuxContainerSecurityContext holds linux security configuration that will be applied to a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxContainerSecurityContext {
    /// Capabilities to add or drop.
    #[prost(message, optional, tag = "1")]
    pub capabilities: ::core::option::Option<Capability>,
    /// If set, run container in privileged mode.
    /// Privileged mode is incompatible with the following options. If
    /// privileged is set, the following features MAY have no effect:
    /// 1. capabilities
    /// 2. selinux_options
    /// 4. seccomp
    /// 5. apparmor
    ///
    /// Privileged mode implies the following specific options are applied:
    /// 1. All capabilities are added.
    /// 2. Sensitive paths, such as kernel module paths within sysfs, are not masked.
    /// 3. Any sysfs and procfs mounts are mounted RW.
    /// 4. AppArmor confinement is not applied.
    /// 5. Seccomp restrictions are not applied.
    /// 6. The device cgroup does not restrict access to any devices.
    /// 7. All devices from the host's /dev are available within the container.
    /// 8. SELinux restrictions are not applied (e.g. label=disabled).
    #[prost(bool, tag = "2")]
    pub privileged: bool,
    /// Configurations for the container's namespaces.
    /// Only used if the container uses namespace for isolation.
    #[prost(message, optional, tag = "3")]
    pub namespace_options: ::core::option::Option<NamespaceOption>,
    /// SELinux context to be optionally applied.
    #[prost(message, optional, tag = "4")]
    pub selinux_options: ::core::option::Option<SeLinuxOption>,
    /// UID to run the container process as. Only one of run_as_user and
    /// run_as_username can be specified at a time.
    #[prost(message, optional, tag = "5")]
    pub run_as_user: ::core::option::Option<Int64Value>,
    /// GID to run the container process as. run_as_group should only be specified
    /// when run_as_user or run_as_username is specified; otherwise, the runtime
    /// MUST error.
    #[prost(message, optional, tag = "12")]
    pub run_as_group: ::core::option::Option<Int64Value>,
    /// User name to run the container process as. If specified, the user MUST
    /// exist in the container image (i.e. in the /etc/passwd inside the image),
    /// and be resolved there by the runtime; otherwise, the runtime MUST error.
    #[prost(string, tag = "6")]
    pub run_as_username: ::prost::alloc::string::String,
    /// If set, the root filesystem of the container is read-only.
    #[prost(bool, tag = "7")]
    pub readonly_rootfs: bool,
    /// List of groups applied to the first process run in the container, in
    /// addition to the container's primary GID.
    #[prost(int64, repeated, tag = "8")]
    pub supplemental_groups: ::prost::alloc::vec::Vec<i64>,
    /// no_new_privs defines if the flag for no_new_privs should be set on the
    /// container.
    #[prost(bool, tag = "11")]
    pub no_new_privs: bool,
    /// masked_paths is a slice of paths that should be masked by the container
    /// runtime, this can be passed directly to the OCI spec.
    #[prost(string, repeated, tag = "13")]
    pub masked_paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// readonly_paths is a slice of paths that should be set as readonly by the
    /// container runtime, this can be passed directly to the OCI spec.
    #[prost(string, repeated, tag = "14")]
    pub readonly_paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Seccomp profile for the container.
    #[prost(message, optional, tag = "15")]
    pub seccomp: ::core::option::Option<SecurityProfile>,
    /// AppArmor profile for the container.
    #[prost(message, optional, tag = "16")]
    pub apparmor: ::core::option::Option<SecurityProfile>,
    /// AppArmor profile for the container, candidate values are:
    /// * runtime/default: equivalent to not specifying a profile.
    /// * unconfined: no profiles are loaded
    /// * localhost/<profile_name>: profile loaded on the node
    ///     (localhost) by name. The possible profile names are detailed at
    ///     <https://gitlab.com/apparmor/apparmor/-/wikis/AppArmor_Core_Policy_Reference>
    #[deprecated]
    #[prost(string, tag = "9")]
    pub apparmor_profile: ::prost::alloc::string::String,
    /// Seccomp profile for the container, candidate values are:
    /// * runtime/default: the default profile for the container runtime
    /// * unconfined: unconfined profile, ie, no seccomp sandboxing
    /// * localhost/<full-path-to-profile>: the profile installed on the node.
    ///    <full-path-to-profile> is the full path of the profile.
    /// Default: "", which is identical with unconfined.
    #[deprecated]
    #[prost(string, tag = "10")]
    pub seccomp_profile_path: ::prost::alloc::string::String,
}
/// LinuxContainerConfig contains platform-specific configuration for
/// Linux-based containers.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LinuxContainerConfig {
    /// Resources specification for the container.
    #[prost(message, optional, tag = "1")]
    pub resources: ::core::option::Option<LinuxContainerResources>,
    /// LinuxContainerSecurityContext configuration for the container.
    #[prost(message, optional, tag = "2")]
    pub security_context: ::core::option::Option<LinuxContainerSecurityContext>,
}
/// WindowsSandboxSecurityContext holds platform-specific configurations that will be
/// applied to a sandbox.
/// These settings will only apply to the sandbox container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowsSandboxSecurityContext {
    /// User name to run the container process as. If specified, the user MUST
    /// exist in the container image and be resolved there by the runtime;
    /// otherwise, the runtime MUST return error.
    #[prost(string, tag = "1")]
    pub run_as_username: ::prost::alloc::string::String,
    /// The contents of the GMSA credential spec to use to run this container.
    #[prost(string, tag = "2")]
    pub credential_spec: ::prost::alloc::string::String,
    /// Indicates whether the container requested to run as a HostProcess container.
    #[prost(bool, tag = "3")]
    pub host_process: bool,
}
/// WindowsPodSandboxConfig holds platform-specific configurations for Windows
/// host platforms and Windows-based containers.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowsPodSandboxConfig {
    /// WindowsSandboxSecurityContext holds sandbox security attributes.
    #[prost(message, optional, tag = "1")]
    pub security_context: ::core::option::Option<WindowsSandboxSecurityContext>,
}
/// WindowsContainerSecurityContext holds windows security configuration that will be applied to a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowsContainerSecurityContext {
    /// User name to run the container process as. If specified, the user MUST
    /// exist in the container image and be resolved there by the runtime;
    /// otherwise, the runtime MUST return error.
    #[prost(string, tag = "1")]
    pub run_as_username: ::prost::alloc::string::String,
    /// The contents of the GMSA credential spec to use to run this container.
    #[prost(string, tag = "2")]
    pub credential_spec: ::prost::alloc::string::String,
    /// Indicates whether a container is to be run as a HostProcess container.
    #[prost(bool, tag = "3")]
    pub host_process: bool,
}
/// WindowsContainerConfig contains platform-specific configuration for
/// Windows-based containers.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowsContainerConfig {
    /// Resources specification for the container.
    #[prost(message, optional, tag = "1")]
    pub resources: ::core::option::Option<WindowsContainerResources>,
    /// WindowsContainerSecurityContext configuration for the container.
    #[prost(message, optional, tag = "2")]
    pub security_context: ::core::option::Option<WindowsContainerSecurityContext>,
}
/// WindowsContainerResources specifies Windows specific configuration for
/// resources.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WindowsContainerResources {
    /// CPU shares (relative weight vs. other containers). Default: 0 (not specified).
    #[prost(int64, tag = "1")]
    pub cpu_shares: i64,
    /// Number of CPUs available to the container. Default: 0 (not specified).
    #[prost(int64, tag = "2")]
    pub cpu_count: i64,
    /// Specifies the portion of processor cycles that this container can use as a percentage times 100.
    #[prost(int64, tag = "3")]
    pub cpu_maximum: i64,
    /// Memory limit in bytes. Default: 0 (not specified).
    #[prost(int64, tag = "4")]
    pub memory_limit_in_bytes: i64,
    /// Specifies the size of the rootfs / scratch space in bytes to be configured for this container. Default: 0 (not specified).
    #[prost(int64, tag = "5")]
    pub rootfs_size_in_bytes: i64,
}
/// ContainerMetadata holds all necessary information for building the container
/// name. The container runtime is encouraged to expose the metadata in its user
/// interface for better user experience. E.g., runtime can construct a unique
/// container name based on the metadata. Note that (name, attempt) is unique
/// within a sandbox for the entire lifetime of the sandbox.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerMetadata {
    /// Name of the container. Same as the container name in the PodSpec.
    #[prost(string, tag = "1")]
    pub name: ::prost::alloc::string::String,
    /// Attempt number of creating the container. Default: 0.
    #[prost(uint32, tag = "2")]
    pub attempt: u32,
}
/// Device specifies a host device to mount into a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Device {
    /// Path of the device within the container.
    #[prost(string, tag = "1")]
    pub container_path: ::prost::alloc::string::String,
    /// Path of the device on the host.
    #[prost(string, tag = "2")]
    pub host_path: ::prost::alloc::string::String,
    /// Cgroups permissions of the device, candidates are one or more of
    /// * r - allows container to read from the specified device.
    /// * w - allows container to write to the specified device.
    /// * m - allows container to create device files that do not yet exist.
    #[prost(string, tag = "3")]
    pub permissions: ::prost::alloc::string::String,
}
/// ContainerConfig holds all the required and optional fields for creating a
/// container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerConfig {
    /// Metadata of the container. This information will uniquely identify the
    /// container, and the runtime should leverage this to ensure correct
    /// operation. The runtime may also use this information to improve UX, such
    /// as by constructing a readable name.
    #[prost(message, optional, tag = "1")]
    pub metadata: ::core::option::Option<ContainerMetadata>,
    /// Image to use.
    #[prost(message, optional, tag = "2")]
    pub image: ::core::option::Option<ImageSpec>,
    /// Command to execute (i.e., entrypoint for docker)
    #[prost(string, repeated, tag = "3")]
    pub command: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Args for the Command (i.e., command for docker)
    #[prost(string, repeated, tag = "4")]
    pub args: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Current working directory of the command.
    #[prost(string, tag = "5")]
    pub working_dir: ::prost::alloc::string::String,
    /// List of environment variable to set in the container.
    #[prost(message, repeated, tag = "6")]
    pub envs: ::prost::alloc::vec::Vec<KeyValue>,
    /// Mounts for the container.
    #[prost(message, repeated, tag = "7")]
    pub mounts: ::prost::alloc::vec::Vec<Mount>,
    /// Devices for the container.
    #[prost(message, repeated, tag = "8")]
    pub devices: ::prost::alloc::vec::Vec<Device>,
    /// Key-value pairs that may be used to scope and select individual resources.
    /// Label keys are of the form:
    ///      label-key ::= prefixed-name | name
    ///      prefixed-name ::= prefix '/' name
    ///      prefix ::= DNS_SUBDOMAIN
    ///      name ::= DNS_LABEL
    #[prost(map = "string, string", tag = "9")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map that may be used by the kubelet to store and
    /// retrieve arbitrary metadata.
    ///
    /// Annotations MUST NOT be altered by the runtime; the annotations stored
    /// here MUST be returned in the ContainerStatus associated with the container
    /// this ContainerConfig creates.
    ///
    /// In general, in order to preserve a well-defined interface between the
    /// kubelet and the container runtime, annotations SHOULD NOT influence
    /// runtime behaviour.
    #[prost(map = "string, string", tag = "10")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Path relative to PodSandboxConfig.LogDirectory for container to store
    /// the log (STDOUT and STDERR) on the host.
    /// E.g.,
    ///      PodSandboxConfig.LogDirectory = `/var/log/pods/<podUID>/`
    ///      ContainerConfig.LogPath = `containerName/Instance#.log`
    ///
    /// WARNING: Log management and how kubelet should interface with the
    /// container logs are under active discussion in
    /// <https://issues.k8s.io/24677.> There *may* be future change of direction
    /// for logging as the discussion carries on.
    #[prost(string, tag = "11")]
    pub log_path: ::prost::alloc::string::String,
    /// Variables for interactive containers, these have very specialized
    /// use-cases (e.g. debugging).
    #[prost(bool, tag = "12")]
    pub stdin: bool,
    #[prost(bool, tag = "13")]
    pub stdin_once: bool,
    #[prost(bool, tag = "14")]
    pub tty: bool,
    /// Configuration specific to Linux containers.
    #[prost(message, optional, tag = "15")]
    pub linux: ::core::option::Option<LinuxContainerConfig>,
    /// Configuration specific to Windows containers.
    #[prost(message, optional, tag = "16")]
    pub windows: ::core::option::Option<WindowsContainerConfig>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateContainerRequest {
    /// ID of the PodSandbox in which the container should be created.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
    /// Config of the container.
    #[prost(message, optional, tag = "2")]
    pub config: ::core::option::Option<ContainerConfig>,
    /// Config of the PodSandbox. This is the same config that was passed
    /// to RunPodSandboxRequest to create the PodSandbox. It is passed again
    /// here just for easy reference. The PodSandboxConfig is immutable and
    /// remains the same throughout the lifetime of the pod.
    #[prost(message, optional, tag = "3")]
    pub sandbox_config: ::core::option::Option<PodSandboxConfig>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreateContainerResponse {
    /// ID of the created container.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartContainerRequest {
    /// ID of the container to start.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartContainerResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StopContainerRequest {
    /// ID of the container to stop.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Timeout in seconds to wait for the container to stop before forcibly
    /// terminating it. Default: 0 (forcibly terminate the container immediately)
    #[prost(int64, tag = "2")]
    pub timeout: i64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StopContainerResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoveContainerRequest {
    /// ID of the container to remove.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoveContainerResponse {}
/// ContainerStateValue is the wrapper of ContainerState.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStateValue {
    /// State of the container.
    #[prost(enumeration = "ContainerState", tag = "1")]
    pub state: i32,
}
/// ContainerFilter is used to filter containers.
/// All those fields are combined with 'AND'
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerFilter {
    /// ID of the container.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// State of the container.
    #[prost(message, optional, tag = "2")]
    pub state: ::core::option::Option<ContainerStateValue>,
    /// ID of the PodSandbox.
    #[prost(string, tag = "3")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
    /// LabelSelector to select matches.
    /// Only api.MatchLabels is supported for now and the requirements
    /// are ANDed. MatchExpressions is not supported yet.
    #[prost(map = "string, string", tag = "4")]
    pub label_selector: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListContainersRequest {
    #[prost(message, optional, tag = "1")]
    pub filter: ::core::option::Option<ContainerFilter>,
}
/// Container provides the runtime information for a container, such as ID, hash,
/// state of the container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Container {
    /// ID of the container, used by the container runtime to identify
    /// a container.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// ID of the sandbox to which this container belongs.
    #[prost(string, tag = "2")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
    /// Metadata of the container.
    #[prost(message, optional, tag = "3")]
    pub metadata: ::core::option::Option<ContainerMetadata>,
    /// Spec of the image.
    #[prost(message, optional, tag = "4")]
    pub image: ::core::option::Option<ImageSpec>,
    /// Reference to the image in use. For most runtimes, this should be an
    /// image ID.
    #[prost(string, tag = "5")]
    pub image_ref: ::prost::alloc::string::String,
    /// State of the container.
    #[prost(enumeration = "ContainerState", tag = "6")]
    pub state: i32,
    /// Creation time of the container in nanoseconds.
    #[prost(int64, tag = "7")]
    pub created_at: i64,
    /// Key-value pairs that may be used to scope and select individual resources.
    #[prost(map = "string, string", tag = "8")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map holding arbitrary metadata.
    /// Annotations MUST NOT be altered by the runtime; the value of this field
    /// MUST be identical to that of the corresponding ContainerConfig used to
    /// instantiate this Container.
    #[prost(map = "string, string", tag = "9")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListContainersResponse {
    /// List of containers.
    #[prost(message, repeated, tag = "1")]
    pub containers: ::prost::alloc::vec::Vec<Container>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStatusRequest {
    /// ID of the container for which to retrieve status.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Verbose indicates whether to return extra information about the container.
    #[prost(bool, tag = "2")]
    pub verbose: bool,
}
/// ContainerStatus represents the status of a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStatus {
    /// ID of the container.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// Metadata of the container.
    #[prost(message, optional, tag = "2")]
    pub metadata: ::core::option::Option<ContainerMetadata>,
    /// Status of the container.
    #[prost(enumeration = "ContainerState", tag = "3")]
    pub state: i32,
    /// Creation time of the container in nanoseconds.
    #[prost(int64, tag = "4")]
    pub created_at: i64,
    /// Start time of the container in nanoseconds. Default: 0 (not specified).
    #[prost(int64, tag = "5")]
    pub started_at: i64,
    /// Finish time of the container in nanoseconds. Default: 0 (not specified).
    #[prost(int64, tag = "6")]
    pub finished_at: i64,
    /// Exit code of the container. Only required when finished_at != 0. Default: 0.
    #[prost(int32, tag = "7")]
    pub exit_code: i32,
    /// Spec of the image.
    #[prost(message, optional, tag = "8")]
    pub image: ::core::option::Option<ImageSpec>,
    /// Reference to the image in use. For most runtimes, this should be an
    /// image ID
    #[prost(string, tag = "9")]
    pub image_ref: ::prost::alloc::string::String,
    /// Brief CamelCase string explaining why container is in its current state.
    #[prost(string, tag = "10")]
    pub reason: ::prost::alloc::string::String,
    /// Human-readable message indicating details about why container is in its
    /// current state.
    #[prost(string, tag = "11")]
    pub message: ::prost::alloc::string::String,
    /// Key-value pairs that may be used to scope and select individual resources.
    #[prost(map = "string, string", tag = "12")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map holding arbitrary metadata.
    /// Annotations MUST NOT be altered by the runtime; the value of this field
    /// MUST be identical to that of the corresponding ContainerConfig used to
    /// instantiate the Container this status represents.
    #[prost(map = "string, string", tag = "13")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Mounts for the container.
    #[prost(message, repeated, tag = "14")]
    pub mounts: ::prost::alloc::vec::Vec<Mount>,
    /// Log path of container.
    #[prost(string, tag = "15")]
    pub log_path: ::prost::alloc::string::String,
    /// Resource limits configuration of the container.
    #[prost(message, optional, tag = "16")]
    pub resources: ::core::option::Option<ContainerResources>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStatusResponse {
    /// Status of the container.
    #[prost(message, optional, tag = "1")]
    pub status: ::core::option::Option<ContainerStatus>,
    /// Info is extra information of the Container. The key could be arbitrary string, and
    /// value should be in json format. The information could include anything useful for
    /// debug, e.g. pid for linux container based container runtime.
    /// It should only be returned non-empty when Verbose is true.
    #[prost(map = "string, string", tag = "2")]
    pub info: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
/// ContainerResources holds resource limits configuration for a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerResources {
    /// Resource limits configuration specific to Linux container.
    #[prost(message, optional, tag = "1")]
    pub linux: ::core::option::Option<LinuxContainerResources>,
    /// Resource limits configuration specific to Windows container.
    #[prost(message, optional, tag = "2")]
    pub windows: ::core::option::Option<WindowsContainerResources>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateContainerResourcesRequest {
    /// ID of the container to update.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Resource configuration specific to Linux containers.
    #[prost(message, optional, tag = "2")]
    pub linux: ::core::option::Option<LinuxContainerResources>,
    /// Resource configuration specific to Windows containers.
    #[prost(message, optional, tag = "3")]
    pub windows: ::core::option::Option<WindowsContainerResources>,
    /// Unstructured key-value map holding arbitrary additional information for
    /// container resources updating. This can be used for specifying experimental
    /// resources to update or other options to use when updating the container.
    #[prost(map = "string, string", tag = "4")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateContainerResourcesResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecSyncRequest {
    /// ID of the container.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Command to execute.
    #[prost(string, repeated, tag = "2")]
    pub cmd: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Timeout in seconds to stop the command. Default: 0 (run forever).
    #[prost(int64, tag = "3")]
    pub timeout: i64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecSyncResponse {
    /// Captured command stdout output.
    #[prost(bytes = "vec", tag = "1")]
    pub stdout: ::prost::alloc::vec::Vec<u8>,
    /// Captured command stderr output.
    #[prost(bytes = "vec", tag = "2")]
    pub stderr: ::prost::alloc::vec::Vec<u8>,
    /// Exit code the command finished with. Default: 0 (success).
    #[prost(int32, tag = "3")]
    pub exit_code: i32,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecRequest {
    /// ID of the container in which to execute the command.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Command to execute.
    #[prost(string, repeated, tag = "2")]
    pub cmd: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Whether to exec the command in a TTY.
    #[prost(bool, tag = "3")]
    pub tty: bool,
    /// Whether to stream stdin.
    /// One of `stdin`, `stdout`, and `stderr` MUST be true.
    #[prost(bool, tag = "4")]
    pub stdin: bool,
    /// Whether to stream stdout.
    /// One of `stdin`, `stdout`, and `stderr` MUST be true.
    #[prost(bool, tag = "5")]
    pub stdout: bool,
    /// Whether to stream stderr.
    /// One of `stdin`, `stdout`, and `stderr` MUST be true.
    /// If `tty` is true, `stderr` MUST be false. Multiplexing is not supported
    /// in this case. The output of stdout and stderr will be combined to a
    /// single stream.
    #[prost(bool, tag = "6")]
    pub stderr: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecResponse {
    /// Fully qualified URL of the exec streaming server.
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttachRequest {
    /// ID of the container to which to attach.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Whether to stream stdin.
    /// One of `stdin`, `stdout`, and `stderr` MUST be true.
    #[prost(bool, tag = "2")]
    pub stdin: bool,
    /// Whether the process being attached is running in a TTY.
    /// This must match the TTY setting in the ContainerConfig.
    #[prost(bool, tag = "3")]
    pub tty: bool,
    /// Whether to stream stdout.
    /// One of `stdin`, `stdout`, and `stderr` MUST be true.
    #[prost(bool, tag = "4")]
    pub stdout: bool,
    /// Whether to stream stderr.
    /// One of `stdin`, `stdout`, and `stderr` MUST be true.
    /// If `tty` is true, `stderr` MUST be false. Multiplexing is not supported
    /// in this case. The output of stdout and stderr will be combined to a
    /// single stream.
    #[prost(bool, tag = "5")]
    pub stderr: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AttachResponse {
    /// Fully qualified URL of the attach streaming server.
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PortForwardRequest {
    /// ID of the container to which to forward the port.
    #[prost(string, tag = "1")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
    /// Port to forward.
    #[prost(int32, repeated, tag = "2")]
    pub port: ::prost::alloc::vec::Vec<i32>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PortForwardResponse {
    /// Fully qualified URL of the port-forward streaming server.
    #[prost(string, tag = "1")]
    pub url: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageFilter {
    /// Spec of the image.
    #[prost(message, optional, tag = "1")]
    pub image: ::core::option::Option<ImageSpec>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListImagesRequest {
    /// Filter to list images.
    #[prost(message, optional, tag = "1")]
    pub filter: ::core::option::Option<ImageFilter>,
}
/// Basic information about a container image.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Image {
    /// ID of the image.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// Other names by which this image is known.
    #[prost(string, repeated, tag = "2")]
    pub repo_tags: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Digests by which this image is known.
    #[prost(string, repeated, tag = "3")]
    pub repo_digests: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
    /// Size of the image in bytes. Must be > 0.
    #[prost(uint64, tag = "4")]
    pub size: u64,
    /// UID that will run the command(s). This is used as a default if no user is
    /// specified when creating the container. UID and the following user name
    /// are mutually exclusive.
    #[prost(message, optional, tag = "5")]
    pub uid: ::core::option::Option<Int64Value>,
    /// User name that will run the command(s). This is used if UID is not set
    /// and no user is specified when creating container.
    #[prost(string, tag = "6")]
    pub username: ::prost::alloc::string::String,
    /// ImageSpec for image which includes annotations
    #[prost(message, optional, tag = "7")]
    pub spec: ::core::option::Option<ImageSpec>,
    /// Recommendation on whether this image should be exempt from garbage collection.
    /// It must only be treated as a recommendation -- the client can still request that the image be deleted,
    /// and the runtime must oblige.
    #[prost(bool, tag = "8")]
    pub pinned: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListImagesResponse {
    /// List of images.
    #[prost(message, repeated, tag = "1")]
    pub images: ::prost::alloc::vec::Vec<Image>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageStatusRequest {
    /// Spec of the image.
    #[prost(message, optional, tag = "1")]
    pub image: ::core::option::Option<ImageSpec>,
    /// Verbose indicates whether to return extra information about the image.
    #[prost(bool, tag = "2")]
    pub verbose: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageStatusResponse {
    /// Status of the image.
    #[prost(message, optional, tag = "1")]
    pub image: ::core::option::Option<Image>,
    /// Info is extra information of the Image. The key could be arbitrary string, and
    /// value should be in json format. The information could include anything useful
    /// for debug, e.g. image config for oci image based container runtime.
    /// It should only be returned non-empty when Verbose is true.
    #[prost(map = "string, string", tag = "2")]
    pub info: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
/// AuthConfig contains authorization information for connecting to a registry.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AuthConfig {
    #[prost(string, tag = "1")]
    pub username: ::prost::alloc::string::String,
    #[prost(string, tag = "2")]
    pub password: ::prost::alloc::string::String,
    #[prost(string, tag = "3")]
    pub auth: ::prost::alloc::string::String,
    #[prost(string, tag = "4")]
    pub server_address: ::prost::alloc::string::String,
    /// IdentityToken is used to authenticate the user and get
    /// an access token for the registry.
    #[prost(string, tag = "5")]
    pub identity_token: ::prost::alloc::string::String,
    /// RegistryToken is a bearer token to be sent to a registry
    #[prost(string, tag = "6")]
    pub registry_token: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PullImageRequest {
    /// Spec of the image.
    #[prost(message, optional, tag = "1")]
    pub image: ::core::option::Option<ImageSpec>,
    /// Authentication configuration for pulling the image.
    #[prost(message, optional, tag = "2")]
    pub auth: ::core::option::Option<AuthConfig>,
    /// Config of the PodSandbox, which is used to pull image in PodSandbox context.
    #[prost(message, optional, tag = "3")]
    pub sandbox_config: ::core::option::Option<PodSandboxConfig>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PullImageResponse {
    /// Reference to the image in use. For most runtimes, this should be an
    /// image ID or digest.
    #[prost(string, tag = "1")]
    pub image_ref: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoveImageRequest {
    /// Spec of the image to remove.
    #[prost(message, optional, tag = "1")]
    pub image: ::core::option::Option<ImageSpec>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RemoveImageResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NetworkConfig {
    /// CIDR to use for pod IP addresses. If the CIDR is empty, runtimes
    /// should omit it.
    #[prost(string, tag = "1")]
    pub pod_cidr: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RuntimeConfig {
    #[prost(message, optional, tag = "1")]
    pub network_config: ::core::option::Option<NetworkConfig>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateRuntimeConfigRequest {
    #[prost(message, optional, tag = "1")]
    pub runtime_config: ::core::option::Option<RuntimeConfig>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateRuntimeConfigResponse {}
/// RuntimeCondition contains condition information for the runtime.
/// There are 2 kinds of runtime conditions:
/// 1. Required conditions: Conditions are required for kubelet to work
/// properly. If any required condition is unmet, the node will be not ready.
/// The required conditions include:
///    * RuntimeReady: RuntimeReady means the runtime is up and ready to accept
///    basic containers e.g. container only needs host network.
///    * NetworkReady: NetworkReady means the runtime network is up and ready to
///    accept containers which require container network.
/// 2. Optional conditions: Conditions are informative to the user, but kubelet
/// will not rely on. Since condition type is an arbitrary string, all conditions
/// not required are optional. These conditions will be exposed to users to help
/// them understand the status of the system.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RuntimeCondition {
    /// Type of runtime condition.
    #[prost(string, tag = "1")]
    pub r#type: ::prost::alloc::string::String,
    /// Status of the condition, one of true/false. Default: false.
    #[prost(bool, tag = "2")]
    pub status: bool,
    /// Brief CamelCase string containing reason for the condition's last transition.
    #[prost(string, tag = "3")]
    pub reason: ::prost::alloc::string::String,
    /// Human-readable message indicating details about last transition.
    #[prost(string, tag = "4")]
    pub message: ::prost::alloc::string::String,
}
/// RuntimeStatus is information about the current status of the runtime.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RuntimeStatus {
    /// List of current observed runtime conditions.
    #[prost(message, repeated, tag = "1")]
    pub conditions: ::prost::alloc::vec::Vec<RuntimeCondition>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StatusRequest {
    /// Verbose indicates whether to return extra information about the runtime.
    #[prost(bool, tag = "1")]
    pub verbose: bool,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StatusResponse {
    /// Status of the Runtime.
    #[prost(message, optional, tag = "1")]
    pub status: ::core::option::Option<RuntimeStatus>,
    /// Info is extra information of the Runtime. The key could be arbitrary string, and
    /// value should be in json format. The information could include anything useful for
    /// debug, e.g. plugins used by the container runtime.
    /// It should only be returned non-empty when Verbose is true.
    #[prost(map = "string, string", tag = "2")]
    pub info: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageFsInfoRequest {}
/// UInt64Value is the wrapper of uint64.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UInt64Value {
    /// The value.
    #[prost(uint64, tag = "1")]
    pub value: u64,
}
/// FilesystemIdentifier uniquely identify the filesystem.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FilesystemIdentifier {
    /// Mountpoint of a filesystem.
    #[prost(string, tag = "1")]
    pub mountpoint: ::prost::alloc::string::String,
}
/// FilesystemUsage provides the filesystem usage information.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FilesystemUsage {
    /// Timestamp in nanoseconds at which the information were collected. Must be > 0.
    #[prost(int64, tag = "1")]
    pub timestamp: i64,
    /// The unique identifier of the filesystem.
    #[prost(message, optional, tag = "2")]
    pub fs_id: ::core::option::Option<FilesystemIdentifier>,
    /// UsedBytes represents the bytes used for images on the filesystem.
    /// This may differ from the total bytes used on the filesystem and may not
    /// equal CapacityBytes - AvailableBytes.
    #[prost(message, optional, tag = "3")]
    pub used_bytes: ::core::option::Option<UInt64Value>,
    /// InodesUsed represents the inodes used by the images.
    /// This may not equal InodesCapacity - InodesAvailable because the underlying
    /// filesystem may also be used for purposes other than storing images.
    #[prost(message, optional, tag = "4")]
    pub inodes_used: ::core::option::Option<UInt64Value>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageFsInfoResponse {
    /// Information of image filesystem(s).
    #[prost(message, repeated, tag = "1")]
    pub image_filesystems: ::prost::alloc::vec::Vec<FilesystemUsage>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStatsRequest {
    /// ID of the container for which to retrieve stats.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStatsResponse {
    /// Stats of the container.
    #[prost(message, optional, tag = "1")]
    pub stats: ::core::option::Option<ContainerStats>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListContainerStatsRequest {
    /// Filter for the list request.
    #[prost(message, optional, tag = "1")]
    pub filter: ::core::option::Option<ContainerStatsFilter>,
}
/// ContainerStatsFilter is used to filter containers.
/// All those fields are combined with 'AND'
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStatsFilter {
    /// ID of the container.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// ID of the PodSandbox.
    #[prost(string, tag = "2")]
    pub pod_sandbox_id: ::prost::alloc::string::String,
    /// LabelSelector to select matches.
    /// Only api.MatchLabels is supported for now and the requirements
    /// are ANDed. MatchExpressions is not supported yet.
    #[prost(map = "string, string", tag = "3")]
    pub label_selector: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListContainerStatsResponse {
    /// Stats of the container.
    #[prost(message, repeated, tag = "1")]
    pub stats: ::prost::alloc::vec::Vec<ContainerStats>,
}
/// ContainerAttributes provides basic information of the container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerAttributes {
    /// ID of the container.
    #[prost(string, tag = "1")]
    pub id: ::prost::alloc::string::String,
    /// Metadata of the container.
    #[prost(message, optional, tag = "2")]
    pub metadata: ::core::option::Option<ContainerMetadata>,
    /// Key-value pairs that may be used to scope and select individual resources.
    #[prost(map = "string, string", tag = "3")]
    pub labels: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
    /// Unstructured key-value map holding arbitrary metadata.
    /// Annotations MUST NOT be altered by the runtime; the value of this field
    /// MUST be identical to that of the corresponding ContainerConfig used to
    /// instantiate the Container this status represents.
    #[prost(map = "string, string", tag = "4")]
    pub annotations: ::std::collections::HashMap<
        ::prost::alloc::string::String,
        ::prost::alloc::string::String,
    >,
}
/// ContainerStats provides the resource usage statistics for a container.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerStats {
    /// Information of the container.
    #[prost(message, optional, tag = "1")]
    pub attributes: ::core::option::Option<ContainerAttributes>,
    /// CPU usage gathered from the container.
    #[prost(message, optional, tag = "2")]
    pub cpu: ::core::option::Option<CpuUsage>,
    /// Memory usage gathered from the container.
    #[prost(message, optional, tag = "3")]
    pub memory: ::core::option::Option<MemoryUsage>,
    /// Usage of the writable layer.
    #[prost(message, optional, tag = "4")]
    pub writable_layer: ::core::option::Option<FilesystemUsage>,
}
/// CpuUsage provides the CPU usage information.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CpuUsage {
    /// Timestamp in nanoseconds at which the information were collected. Must be > 0.
    #[prost(int64, tag = "1")]
    pub timestamp: i64,
    /// Cumulative CPU usage (sum across all cores) since object creation.
    #[prost(message, optional, tag = "2")]
    pub usage_core_nano_seconds: ::core::option::Option<UInt64Value>,
    /// Total CPU usage (sum of all cores) averaged over the sample window.
    /// The "core" unit can be interpreted as CPU core-nanoseconds per second.
    #[prost(message, optional, tag = "3")]
    pub usage_nano_cores: ::core::option::Option<UInt64Value>,
}
/// MemoryUsage provides the memory usage information.
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MemoryUsage {
    /// Timestamp in nanoseconds at which the information were collected. Must be > 0.
    #[prost(int64, tag = "1")]
    pub timestamp: i64,
    /// The amount of working set memory in bytes.
    #[prost(message, optional, tag = "2")]
    pub working_set_bytes: ::core::option::Option<UInt64Value>,
    /// Available memory for use. This is defined as the memory limit - workingSetBytes.
    #[prost(message, optional, tag = "3")]
    pub available_bytes: ::core::option::Option<UInt64Value>,
    /// Total memory in use. This includes all memory regardless of when it was accessed.
    #[prost(message, optional, tag = "4")]
    pub usage_bytes: ::core::option::Option<UInt64Value>,
    /// The amount of anonymous and swap cache memory (includes transparent hugepages).
    #[prost(message, optional, tag = "5")]
    pub rss_bytes: ::core::option::Option<UInt64Value>,
    /// Cumulative number of minor page faults.
    #[prost(message, optional, tag = "6")]
    pub page_faults: ::core::option::Option<UInt64Value>,
    /// Cumulative number of major page faults.
    #[prost(message, optional, tag = "7")]
    pub major_page_faults: ::core::option::Option<UInt64Value>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReopenContainerLogRequest {
    /// ID of the container for which to reopen the log.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReopenContainerLogResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CheckpointContainerRequest {
    /// ID of the container to be checkpointed.
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Location of the checkpoint archive used for export
    #[prost(string, tag = "2")]
    pub location: ::prost::alloc::string::String,
    /// Timeout in seconds for the checkpoint to complete.
    /// Timeout of zero means to use the CRI default.
    /// Timeout > 0 means to use the user specified timeout.
    #[prost(int64, tag = "3")]
    pub timeout: i64,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CheckpointContainerResponse {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetEventsRequest {}
#[derive(serde::Serialize, serde::Deserialize)]
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ContainerEventResponse {
    /// ID of the container
    #[prost(string, tag = "1")]
    pub container_id: ::prost::alloc::string::String,
    /// Type of the container event
    #[prost(enumeration = "ContainerEventType", tag = "2")]
    pub container_event_type: i32,
    /// Creation timestamp of this event
    #[prost(int64, tag = "3")]
    pub created_at: i64,
    /// ID of the sandbox container
    #[prost(message, optional, tag = "4")]
    pub pod_sandbox_metadata: ::core::option::Option<PodSandboxMetadata>,
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Protocol {
    Tcp = 0,
    Udp = 1,
    Sctp = 2,
}
impl Protocol {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            Protocol::Tcp => "TCP",
            Protocol::Udp => "UDP",
            Protocol::Sctp => "SCTP",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "TCP" => Some(Self::Tcp),
            "UDP" => Some(Self::Udp),
            "SCTP" => Some(Self::Sctp),
            _ => None,
        }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum MountPropagation {
    /// No mount propagation ("private" in Linux terminology).
    PropagationPrivate = 0,
    /// Mounts get propagated from the host to the container ("rslave" in Linux).
    PropagationHostToContainer = 1,
    /// Mounts get propagated from the host to the container and from the
    /// container to the host ("rshared" in Linux).
    PropagationBidirectional = 2,
}
impl MountPropagation {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            MountPropagation::PropagationPrivate => "PROPAGATION_PRIVATE",
            MountPropagation::PropagationHostToContainer => {
                "PROPAGATION_HOST_TO_CONTAINER"
            }
            MountPropagation::PropagationBidirectional => "PROPAGATION_BIDIRECTIONAL",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "PROPAGATION_PRIVATE" => Some(Self::PropagationPrivate),
            "PROPAGATION_HOST_TO_CONTAINER" => Some(Self::PropagationHostToContainer),
            "PROPAGATION_BIDIRECTIONAL" => Some(Self::PropagationBidirectional),
            _ => None,
        }
    }
}
/// A NamespaceMode describes the intended namespace configuration for each
/// of the namespaces (Network, PID, IPC) in NamespaceOption. Runtimes should
/// map these modes as appropriate for the technology underlying the runtime.
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum NamespaceMode {
    /// A POD namespace is common to all containers in a pod.
    /// For example, a container with a PID namespace of POD expects to view
    /// all of the processes in all of the containers in the pod.
    Pod = 0,
    /// A CONTAINER namespace is restricted to a single container.
    /// For example, a container with a PID namespace of CONTAINER expects to
    /// view only the processes in that container.
    Container = 1,
    /// A NODE namespace is the namespace of the Kubernetes node.
    /// For example, a container with a PID namespace of NODE expects to view
    /// all of the processes on the host running the kubelet.
    Node = 2,
    /// TARGET targets the namespace of another container. When this is specified,
    /// a target_id must be specified in NamespaceOption and refer to a container
    /// previously created with NamespaceMode CONTAINER. This containers namespace
    /// will be made to match that of container target_id.
    /// For example, a container with a PID namespace of TARGET expects to view
    /// all of the processes that container target_id can view.
    Target = 3,
}
impl NamespaceMode {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            NamespaceMode::Pod => "POD",
            NamespaceMode::Container => "CONTAINER",
            NamespaceMode::Node => "NODE",
            NamespaceMode::Target => "TARGET",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "POD" => Some(Self::Pod),
            "CONTAINER" => Some(Self::Container),
            "NODE" => Some(Self::Node),
            "TARGET" => Some(Self::Target),
            _ => None,
        }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum PodSandboxState {
    SandboxReady = 0,
    SandboxNotready = 1,
}
impl PodSandboxState {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            PodSandboxState::SandboxReady => "SANDBOX_READY",
            PodSandboxState::SandboxNotready => "SANDBOX_NOTREADY",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "SANDBOX_READY" => Some(Self::SandboxReady),
            "SANDBOX_NOTREADY" => Some(Self::SandboxNotready),
            _ => None,
        }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ContainerState {
    ContainerCreated = 0,
    ContainerRunning = 1,
    ContainerExited = 2,
    ContainerUnknown = 3,
}
impl ContainerState {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            ContainerState::ContainerCreated => "CONTAINER_CREATED",
            ContainerState::ContainerRunning => "CONTAINER_RUNNING",
            ContainerState::ContainerExited => "CONTAINER_EXITED",
            ContainerState::ContainerUnknown => "CONTAINER_UNKNOWN",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "CONTAINER_CREATED" => Some(Self::ContainerCreated),
            "CONTAINER_RUNNING" => Some(Self::ContainerRunning),
            "CONTAINER_EXITED" => Some(Self::ContainerExited),
            "CONTAINER_UNKNOWN" => Some(Self::ContainerUnknown),
            _ => None,
        }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ContainerEventType {
    /// Container created
    ContainerCreatedEvent = 0,
    /// Container started
    ContainerStartedEvent = 1,
    /// Container stopped
    ContainerStoppedEvent = 2,
    /// Container deleted
    ContainerDeletedEvent = 3,
}
impl ContainerEventType {
    /// String value of the enum field names used in the ProtoBuf definition.
    ///
    /// The values are not transformed in any way and thus are considered stable
    /// (if the ProtoBuf definition does not change) and safe for programmatic use.
    pub fn as_str_name(&self) -> &'static str {
        match self {
            ContainerEventType::ContainerCreatedEvent => "CONTAINER_CREATED_EVENT",
            ContainerEventType::ContainerStartedEvent => "CONTAINER_STARTED_EVENT",
            ContainerEventType::ContainerStoppedEvent => "CONTAINER_STOPPED_EVENT",
            ContainerEventType::ContainerDeletedEvent => "CONTAINER_DELETED_EVENT",
        }
    }
    /// Creates an enum from field names used in the ProtoBuf definition.
    pub fn from_str_name(value: &str) -> ::core::option::Option<Self> {
        match value {
            "CONTAINER_CREATED_EVENT" => Some(Self::ContainerCreatedEvent),
            "CONTAINER_STARTED_EVENT" => Some(Self::ContainerStartedEvent),
            "CONTAINER_STOPPED_EVENT" => Some(Self::ContainerStoppedEvent),
            "CONTAINER_DELETED_EVENT" => Some(Self::ContainerDeletedEvent),
            _ => None,
        }
    }
}
/// Generated client implementations.
pub mod runtime_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    /// Runtime service defines the public APIs for remote container runtimes
    #[derive(Debug, Clone)]
    pub struct RuntimeServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl RuntimeServiceClient<tonic::transport::Channel> {
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
    impl<T> RuntimeServiceClient<T>
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
        ) -> RuntimeServiceClient<InterceptedService<T, F>>
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
            RuntimeServiceClient::new(InterceptedService::new(inner, interceptor))
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
        /// Version returns the runtime name, runtime version, and runtime API version.
        pub async fn version(
            &mut self,
            request: impl tonic::IntoRequest<super::VersionRequest>,
        ) -> Result<tonic::Response<super::VersionResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/Version",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// RunPodSandbox creates and starts a pod-level sandbox. Runtimes must ensure
        /// the sandbox is in the ready state on success.
        pub async fn run_pod_sandbox(
            &mut self,
            request: impl tonic::IntoRequest<super::RunPodSandboxRequest>,
        ) -> Result<tonic::Response<super::RunPodSandboxResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/RunPodSandbox",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// StopPodSandbox stops any running process that is part of the sandbox and
        /// reclaims network resources (e.g., IP addresses) allocated to the sandbox.
        /// If there are any running containers in the sandbox, they must be forcibly
        /// terminated.
        /// This call is idempotent, and must not return an error if all relevant
        /// resources have already been reclaimed. kubelet will call StopPodSandbox
        /// at least once before calling RemovePodSandbox. It will also attempt to
        /// reclaim resources eagerly, as soon as a sandbox is not needed. Hence,
        /// multiple StopPodSandbox calls are expected.
        pub async fn stop_pod_sandbox(
            &mut self,
            request: impl tonic::IntoRequest<super::StopPodSandboxRequest>,
        ) -> Result<tonic::Response<super::StopPodSandboxResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/StopPodSandbox",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// RemovePodSandbox removes the sandbox. If there are any running containers
        /// in the sandbox, they must be forcibly terminated and removed.
        /// This call is idempotent, and must not return an error if the sandbox has
        /// already been removed.
        pub async fn remove_pod_sandbox(
            &mut self,
            request: impl tonic::IntoRequest<super::RemovePodSandboxRequest>,
        ) -> Result<tonic::Response<super::RemovePodSandboxResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/RemovePodSandbox",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// PodSandboxStatus returns the status of the PodSandbox. If the PodSandbox is not
        /// present, returns an error.
        pub async fn pod_sandbox_status(
            &mut self,
            request: impl tonic::IntoRequest<super::PodSandboxStatusRequest>,
        ) -> Result<tonic::Response<super::PodSandboxStatusResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/PodSandboxStatus",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ListPodSandbox returns a list of PodSandboxes.
        pub async fn list_pod_sandbox(
            &mut self,
            request: impl tonic::IntoRequest<super::ListPodSandboxRequest>,
        ) -> Result<tonic::Response<super::ListPodSandboxResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ListPodSandbox",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// CreateContainer creates a new container in specified PodSandbox
        pub async fn create_container(
            &mut self,
            request: impl tonic::IntoRequest<super::CreateContainerRequest>,
        ) -> Result<tonic::Response<super::CreateContainerResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/CreateContainer",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// StartContainer starts the container.
        pub async fn start_container(
            &mut self,
            request: impl tonic::IntoRequest<super::StartContainerRequest>,
        ) -> Result<tonic::Response<super::StartContainerResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/StartContainer",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// StopContainer stops a running container with a grace period (i.e., timeout).
        /// This call is idempotent, and must not return an error if the container has
        /// already been stopped.
        /// The runtime must forcibly kill the container after the grace period is
        /// reached.
        pub async fn stop_container(
            &mut self,
            request: impl tonic::IntoRequest<super::StopContainerRequest>,
        ) -> Result<tonic::Response<super::StopContainerResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/StopContainer",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// RemoveContainer removes the container. If the container is running, the
        /// container must be forcibly removed.
        /// This call is idempotent, and must not return an error if the container has
        /// already been removed.
        pub async fn remove_container(
            &mut self,
            request: impl tonic::IntoRequest<super::RemoveContainerRequest>,
        ) -> Result<tonic::Response<super::RemoveContainerResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/RemoveContainer",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ListContainers lists all containers by filters.
        pub async fn list_containers(
            &mut self,
            request: impl tonic::IntoRequest<super::ListContainersRequest>,
        ) -> Result<tonic::Response<super::ListContainersResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ListContainers",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ContainerStatus returns status of the container. If the container is not
        /// present, returns an error.
        pub async fn container_status(
            &mut self,
            request: impl tonic::IntoRequest<super::ContainerStatusRequest>,
        ) -> Result<tonic::Response<super::ContainerStatusResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ContainerStatus",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// UpdateContainerResources updates ContainerConfig of the container synchronously.
        /// If runtime fails to transactionally update the requested resources, an error is returned.
        pub async fn update_container_resources(
            &mut self,
            request: impl tonic::IntoRequest<super::UpdateContainerResourcesRequest>,
        ) -> Result<
            tonic::Response<super::UpdateContainerResourcesResponse>,
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
                "/runtime.v1.RuntimeService/UpdateContainerResources",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ReopenContainerLog asks runtime to reopen the stdout/stderr log file
        /// for the container. This is often called after the log file has been
        /// rotated. If the container is not running, container runtime can choose
        /// to either create a new log file and return nil, or return an error.
        /// Once it returns error, new container log file MUST NOT be created.
        pub async fn reopen_container_log(
            &mut self,
            request: impl tonic::IntoRequest<super::ReopenContainerLogRequest>,
        ) -> Result<tonic::Response<super::ReopenContainerLogResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ReopenContainerLog",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ExecSync runs a command in a container synchronously.
        pub async fn exec_sync(
            &mut self,
            request: impl tonic::IntoRequest<super::ExecSyncRequest>,
        ) -> Result<tonic::Response<super::ExecSyncResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ExecSync",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// Exec prepares a streaming endpoint to execute a command in the container.
        pub async fn exec(
            &mut self,
            request: impl tonic::IntoRequest<super::ExecRequest>,
        ) -> Result<tonic::Response<super::ExecResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/Exec",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// Attach prepares a streaming endpoint to attach to a running container.
        pub async fn attach(
            &mut self,
            request: impl tonic::IntoRequest<super::AttachRequest>,
        ) -> Result<tonic::Response<super::AttachResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/Attach",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// PortForward prepares a streaming endpoint to forward ports from a PodSandbox.
        pub async fn port_forward(
            &mut self,
            request: impl tonic::IntoRequest<super::PortForwardRequest>,
        ) -> Result<tonic::Response<super::PortForwardResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/PortForward",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ContainerStats returns stats of the container. If the container does not
        /// exist, the call returns an error.
        pub async fn container_stats(
            &mut self,
            request: impl tonic::IntoRequest<super::ContainerStatsRequest>,
        ) -> Result<tonic::Response<super::ContainerStatsResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ContainerStats",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ListContainerStats returns stats of all running containers.
        pub async fn list_container_stats(
            &mut self,
            request: impl tonic::IntoRequest<super::ListContainerStatsRequest>,
        ) -> Result<tonic::Response<super::ListContainerStatsResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ListContainerStats",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// PodSandboxStats returns stats of the pod sandbox. If the pod sandbox does not
        /// exist, the call returns an error.
        pub async fn pod_sandbox_stats(
            &mut self,
            request: impl tonic::IntoRequest<super::PodSandboxStatsRequest>,
        ) -> Result<tonic::Response<super::PodSandboxStatsResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/PodSandboxStats",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ListPodSandboxStats returns stats of the pod sandboxes matching a filter.
        pub async fn list_pod_sandbox_stats(
            &mut self,
            request: impl tonic::IntoRequest<super::ListPodSandboxStatsRequest>,
        ) -> Result<tonic::Response<super::ListPodSandboxStatsResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/ListPodSandboxStats",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// UpdateRuntimeConfig updates the runtime configuration based on the given request.
        pub async fn update_runtime_config(
            &mut self,
            request: impl tonic::IntoRequest<super::UpdateRuntimeConfigRequest>,
        ) -> Result<tonic::Response<super::UpdateRuntimeConfigResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/UpdateRuntimeConfig",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// Status returns the status of the runtime.
        pub async fn status(
            &mut self,
            request: impl tonic::IntoRequest<super::StatusRequest>,
        ) -> Result<tonic::Response<super::StatusResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/Status",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// CheckpointContainer checkpoints a container
        pub async fn checkpoint_container(
            &mut self,
            request: impl tonic::IntoRequest<super::CheckpointContainerRequest>,
        ) -> Result<tonic::Response<super::CheckpointContainerResponse>, tonic::Status> {
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
                "/runtime.v1.RuntimeService/CheckpointContainer",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// GetContainerEvents gets container events from the CRI runtime
        pub async fn get_container_events(
            &mut self,
            request: impl tonic::IntoRequest<super::GetEventsRequest>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::ContainerEventResponse>>,
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
                "/runtime.v1.RuntimeService/GetContainerEvents",
            );
            self.inner.server_streaming(request.into_request(), path, codec).await
        }
    }
}
/// Generated client implementations.
pub mod image_service_client {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    use tonic::codegen::http::Uri;
    /// ImageService defines the public APIs for managing images.
    #[derive(Debug, Clone)]
    pub struct ImageServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl ImageServiceClient<tonic::transport::Channel> {
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
    impl<T> ImageServiceClient<T>
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
        ) -> ImageServiceClient<InterceptedService<T, F>>
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
            ImageServiceClient::new(InterceptedService::new(inner, interceptor))
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
        /// ListImages lists existing images.
        pub async fn list_images(
            &mut self,
            request: impl tonic::IntoRequest<super::ListImagesRequest>,
        ) -> Result<tonic::Response<super::ListImagesResponse>, tonic::Status> {
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
                "/runtime.v1.ImageService/ListImages",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ImageStatus returns the status of the image. If the image is not
        /// present, returns a response with ImageStatusResponse.Image set to
        /// nil.
        pub async fn image_status(
            &mut self,
            request: impl tonic::IntoRequest<super::ImageStatusRequest>,
        ) -> Result<tonic::Response<super::ImageStatusResponse>, tonic::Status> {
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
                "/runtime.v1.ImageService/ImageStatus",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// PullImage pulls an image with authentication config.
        pub async fn pull_image(
            &mut self,
            request: impl tonic::IntoRequest<super::PullImageRequest>,
        ) -> Result<tonic::Response<super::PullImageResponse>, tonic::Status> {
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
                "/runtime.v1.ImageService/PullImage",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// RemoveImage removes the image.
        /// This call is idempotent, and must not return an error if the image has
        /// already been removed.
        pub async fn remove_image(
            &mut self,
            request: impl tonic::IntoRequest<super::RemoveImageRequest>,
        ) -> Result<tonic::Response<super::RemoveImageResponse>, tonic::Status> {
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
                "/runtime.v1.ImageService/RemoveImage",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
        /// ImageFSInfo returns information of the filesystem that is used to store images.
        pub async fn image_fs_info(
            &mut self,
            request: impl tonic::IntoRequest<super::ImageFsInfoRequest>,
        ) -> Result<tonic::Response<super::ImageFsInfoResponse>, tonic::Status> {
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
                "/runtime.v1.ImageService/ImageFsInfo",
            );
            self.inner.unary(request.into_request(), path, codec).await
        }
    }
}
/// Generated server implementations.
pub mod runtime_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with RuntimeServiceServer.
    #[async_trait]
    pub trait RuntimeService: Send + Sync + 'static {
        /// Version returns the runtime name, runtime version, and runtime API version.
        async fn version(
            &self,
            request: tonic::Request<super::VersionRequest>,
        ) -> Result<tonic::Response<super::VersionResponse>, tonic::Status>;
        /// RunPodSandbox creates and starts a pod-level sandbox. Runtimes must ensure
        /// the sandbox is in the ready state on success.
        async fn run_pod_sandbox(
            &self,
            request: tonic::Request<super::RunPodSandboxRequest>,
        ) -> Result<tonic::Response<super::RunPodSandboxResponse>, tonic::Status>;
        /// StopPodSandbox stops any running process that is part of the sandbox and
        /// reclaims network resources (e.g., IP addresses) allocated to the sandbox.
        /// If there are any running containers in the sandbox, they must be forcibly
        /// terminated.
        /// This call is idempotent, and must not return an error if all relevant
        /// resources have already been reclaimed. kubelet will call StopPodSandbox
        /// at least once before calling RemovePodSandbox. It will also attempt to
        /// reclaim resources eagerly, as soon as a sandbox is not needed. Hence,
        /// multiple StopPodSandbox calls are expected.
        async fn stop_pod_sandbox(
            &self,
            request: tonic::Request<super::StopPodSandboxRequest>,
        ) -> Result<tonic::Response<super::StopPodSandboxResponse>, tonic::Status>;
        /// RemovePodSandbox removes the sandbox. If there are any running containers
        /// in the sandbox, they must be forcibly terminated and removed.
        /// This call is idempotent, and must not return an error if the sandbox has
        /// already been removed.
        async fn remove_pod_sandbox(
            &self,
            request: tonic::Request<super::RemovePodSandboxRequest>,
        ) -> Result<tonic::Response<super::RemovePodSandboxResponse>, tonic::Status>;
        /// PodSandboxStatus returns the status of the PodSandbox. If the PodSandbox is not
        /// present, returns an error.
        async fn pod_sandbox_status(
            &self,
            request: tonic::Request<super::PodSandboxStatusRequest>,
        ) -> Result<tonic::Response<super::PodSandboxStatusResponse>, tonic::Status>;
        /// ListPodSandbox returns a list of PodSandboxes.
        async fn list_pod_sandbox(
            &self,
            request: tonic::Request<super::ListPodSandboxRequest>,
        ) -> Result<tonic::Response<super::ListPodSandboxResponse>, tonic::Status>;
        /// CreateContainer creates a new container in specified PodSandbox
        async fn create_container(
            &self,
            request: tonic::Request<super::CreateContainerRequest>,
        ) -> Result<tonic::Response<super::CreateContainerResponse>, tonic::Status>;
        /// StartContainer starts the container.
        async fn start_container(
            &self,
            request: tonic::Request<super::StartContainerRequest>,
        ) -> Result<tonic::Response<super::StartContainerResponse>, tonic::Status>;
        /// StopContainer stops a running container with a grace period (i.e., timeout).
        /// This call is idempotent, and must not return an error if the container has
        /// already been stopped.
        /// The runtime must forcibly kill the container after the grace period is
        /// reached.
        async fn stop_container(
            &self,
            request: tonic::Request<super::StopContainerRequest>,
        ) -> Result<tonic::Response<super::StopContainerResponse>, tonic::Status>;
        /// RemoveContainer removes the container. If the container is running, the
        /// container must be forcibly removed.
        /// This call is idempotent, and must not return an error if the container has
        /// already been removed.
        async fn remove_container(
            &self,
            request: tonic::Request<super::RemoveContainerRequest>,
        ) -> Result<tonic::Response<super::RemoveContainerResponse>, tonic::Status>;
        /// ListContainers lists all containers by filters.
        async fn list_containers(
            &self,
            request: tonic::Request<super::ListContainersRequest>,
        ) -> Result<tonic::Response<super::ListContainersResponse>, tonic::Status>;
        /// ContainerStatus returns status of the container. If the container is not
        /// present, returns an error.
        async fn container_status(
            &self,
            request: tonic::Request<super::ContainerStatusRequest>,
        ) -> Result<tonic::Response<super::ContainerStatusResponse>, tonic::Status>;
        /// UpdateContainerResources updates ContainerConfig of the container synchronously.
        /// If runtime fails to transactionally update the requested resources, an error is returned.
        async fn update_container_resources(
            &self,
            request: tonic::Request<super::UpdateContainerResourcesRequest>,
        ) -> Result<
            tonic::Response<super::UpdateContainerResourcesResponse>,
            tonic::Status,
        >;
        /// ReopenContainerLog asks runtime to reopen the stdout/stderr log file
        /// for the container. This is often called after the log file has been
        /// rotated. If the container is not running, container runtime can choose
        /// to either create a new log file and return nil, or return an error.
        /// Once it returns error, new container log file MUST NOT be created.
        async fn reopen_container_log(
            &self,
            request: tonic::Request<super::ReopenContainerLogRequest>,
        ) -> Result<tonic::Response<super::ReopenContainerLogResponse>, tonic::Status>;
        /// ExecSync runs a command in a container synchronously.
        async fn exec_sync(
            &self,
            request: tonic::Request<super::ExecSyncRequest>,
        ) -> Result<tonic::Response<super::ExecSyncResponse>, tonic::Status>;
        /// Exec prepares a streaming endpoint to execute a command in the container.
        async fn exec(
            &self,
            request: tonic::Request<super::ExecRequest>,
        ) -> Result<tonic::Response<super::ExecResponse>, tonic::Status>;
        /// Attach prepares a streaming endpoint to attach to a running container.
        async fn attach(
            &self,
            request: tonic::Request<super::AttachRequest>,
        ) -> Result<tonic::Response<super::AttachResponse>, tonic::Status>;
        /// PortForward prepares a streaming endpoint to forward ports from a PodSandbox.
        async fn port_forward(
            &self,
            request: tonic::Request<super::PortForwardRequest>,
        ) -> Result<tonic::Response<super::PortForwardResponse>, tonic::Status>;
        /// ContainerStats returns stats of the container. If the container does not
        /// exist, the call returns an error.
        async fn container_stats(
            &self,
            request: tonic::Request<super::ContainerStatsRequest>,
        ) -> Result<tonic::Response<super::ContainerStatsResponse>, tonic::Status>;
        /// ListContainerStats returns stats of all running containers.
        async fn list_container_stats(
            &self,
            request: tonic::Request<super::ListContainerStatsRequest>,
        ) -> Result<tonic::Response<super::ListContainerStatsResponse>, tonic::Status>;
        /// PodSandboxStats returns stats of the pod sandbox. If the pod sandbox does not
        /// exist, the call returns an error.
        async fn pod_sandbox_stats(
            &self,
            request: tonic::Request<super::PodSandboxStatsRequest>,
        ) -> Result<tonic::Response<super::PodSandboxStatsResponse>, tonic::Status>;
        /// ListPodSandboxStats returns stats of the pod sandboxes matching a filter.
        async fn list_pod_sandbox_stats(
            &self,
            request: tonic::Request<super::ListPodSandboxStatsRequest>,
        ) -> Result<tonic::Response<super::ListPodSandboxStatsResponse>, tonic::Status>;
        /// UpdateRuntimeConfig updates the runtime configuration based on the given request.
        async fn update_runtime_config(
            &self,
            request: tonic::Request<super::UpdateRuntimeConfigRequest>,
        ) -> Result<tonic::Response<super::UpdateRuntimeConfigResponse>, tonic::Status>;
        /// Status returns the status of the runtime.
        async fn status(
            &self,
            request: tonic::Request<super::StatusRequest>,
        ) -> Result<tonic::Response<super::StatusResponse>, tonic::Status>;
        /// CheckpointContainer checkpoints a container
        async fn checkpoint_container(
            &self,
            request: tonic::Request<super::CheckpointContainerRequest>,
        ) -> Result<tonic::Response<super::CheckpointContainerResponse>, tonic::Status>;
        /// Server streaming response type for the GetContainerEvents method.
        type GetContainerEventsStream: futures_core::Stream<
                Item = Result<super::ContainerEventResponse, tonic::Status>,
            >
            + Send
            + 'static;
        /// GetContainerEvents gets container events from the CRI runtime
        async fn get_container_events(
            &self,
            request: tonic::Request<super::GetEventsRequest>,
        ) -> Result<tonic::Response<Self::GetContainerEventsStream>, tonic::Status>;
    }
    /// Runtime service defines the public APIs for remote container runtimes
    #[derive(Debug)]
    pub struct RuntimeServiceServer<T: RuntimeService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: RuntimeService> RuntimeServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for RuntimeServiceServer<T>
    where
        T: RuntimeService,
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
                "/runtime.v1.RuntimeService/Version" => {
                    #[allow(non_camel_case_types)]
                    struct VersionSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::VersionRequest>
                    for VersionSvc<T> {
                        type Response = super::VersionResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::VersionRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).version(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = VersionSvc(inner);
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
                "/runtime.v1.RuntimeService/RunPodSandbox" => {
                    #[allow(non_camel_case_types)]
                    struct RunPodSandboxSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::RunPodSandboxRequest>
                    for RunPodSandboxSvc<T> {
                        type Response = super::RunPodSandboxResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::RunPodSandboxRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).run_pod_sandbox(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = RunPodSandboxSvc(inner);
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
                "/runtime.v1.RuntimeService/StopPodSandbox" => {
                    #[allow(non_camel_case_types)]
                    struct StopPodSandboxSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::StopPodSandboxRequest>
                    for StopPodSandboxSvc<T> {
                        type Response = super::StopPodSandboxResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StopPodSandboxRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).stop_pod_sandbox(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = StopPodSandboxSvc(inner);
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
                "/runtime.v1.RuntimeService/RemovePodSandbox" => {
                    #[allow(non_camel_case_types)]
                    struct RemovePodSandboxSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::RemovePodSandboxRequest>
                    for RemovePodSandboxSvc<T> {
                        type Response = super::RemovePodSandboxResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::RemovePodSandboxRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).remove_pod_sandbox(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = RemovePodSandboxSvc(inner);
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
                "/runtime.v1.RuntimeService/PodSandboxStatus" => {
                    #[allow(non_camel_case_types)]
                    struct PodSandboxStatusSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::PodSandboxStatusRequest>
                    for PodSandboxStatusSvc<T> {
                        type Response = super::PodSandboxStatusResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PodSandboxStatusRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).pod_sandbox_status(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = PodSandboxStatusSvc(inner);
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
                "/runtime.v1.RuntimeService/ListPodSandbox" => {
                    #[allow(non_camel_case_types)]
                    struct ListPodSandboxSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ListPodSandboxRequest>
                    for ListPodSandboxSvc<T> {
                        type Response = super::ListPodSandboxResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListPodSandboxRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).list_pod_sandbox(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListPodSandboxSvc(inner);
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
                "/runtime.v1.RuntimeService/CreateContainer" => {
                    #[allow(non_camel_case_types)]
                    struct CreateContainerSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::CreateContainerRequest>
                    for CreateContainerSvc<T> {
                        type Response = super::CreateContainerResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::CreateContainerRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).create_container(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = CreateContainerSvc(inner);
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
                "/runtime.v1.RuntimeService/StartContainer" => {
                    #[allow(non_camel_case_types)]
                    struct StartContainerSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::StartContainerRequest>
                    for StartContainerSvc<T> {
                        type Response = super::StartContainerResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StartContainerRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).start_container(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = StartContainerSvc(inner);
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
                "/runtime.v1.RuntimeService/StopContainer" => {
                    #[allow(non_camel_case_types)]
                    struct StopContainerSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::StopContainerRequest>
                    for StopContainerSvc<T> {
                        type Response = super::StopContainerResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StopContainerRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).stop_container(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = StopContainerSvc(inner);
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
                "/runtime.v1.RuntimeService/RemoveContainer" => {
                    #[allow(non_camel_case_types)]
                    struct RemoveContainerSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::RemoveContainerRequest>
                    for RemoveContainerSvc<T> {
                        type Response = super::RemoveContainerResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::RemoveContainerRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).remove_container(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = RemoveContainerSvc(inner);
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
                "/runtime.v1.RuntimeService/ListContainers" => {
                    #[allow(non_camel_case_types)]
                    struct ListContainersSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ListContainersRequest>
                    for ListContainersSvc<T> {
                        type Response = super::ListContainersResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListContainersRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).list_containers(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListContainersSvc(inner);
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
                "/runtime.v1.RuntimeService/ContainerStatus" => {
                    #[allow(non_camel_case_types)]
                    struct ContainerStatusSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ContainerStatusRequest>
                    for ContainerStatusSvc<T> {
                        type Response = super::ContainerStatusResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ContainerStatusRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).container_status(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ContainerStatusSvc(inner);
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
                "/runtime.v1.RuntimeService/UpdateContainerResources" => {
                    #[allow(non_camel_case_types)]
                    struct UpdateContainerResourcesSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::UpdateContainerResourcesRequest>
                    for UpdateContainerResourcesSvc<T> {
                        type Response = super::UpdateContainerResourcesResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<
                                super::UpdateContainerResourcesRequest,
                            >,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).update_container_resources(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = UpdateContainerResourcesSvc(inner);
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
                "/runtime.v1.RuntimeService/ReopenContainerLog" => {
                    #[allow(non_camel_case_types)]
                    struct ReopenContainerLogSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ReopenContainerLogRequest>
                    for ReopenContainerLogSvc<T> {
                        type Response = super::ReopenContainerLogResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ReopenContainerLogRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).reopen_container_log(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ReopenContainerLogSvc(inner);
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
                "/runtime.v1.RuntimeService/ExecSync" => {
                    #[allow(non_camel_case_types)]
                    struct ExecSyncSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ExecSyncRequest>
                    for ExecSyncSvc<T> {
                        type Response = super::ExecSyncResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ExecSyncRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).exec_sync(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ExecSyncSvc(inner);
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
                "/runtime.v1.RuntimeService/Exec" => {
                    #[allow(non_camel_case_types)]
                    struct ExecSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ExecRequest> for ExecSvc<T> {
                        type Response = super::ExecResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ExecRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).exec(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ExecSvc(inner);
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
                "/runtime.v1.RuntimeService/Attach" => {
                    #[allow(non_camel_case_types)]
                    struct AttachSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::AttachRequest>
                    for AttachSvc<T> {
                        type Response = super::AttachResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::AttachRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).attach(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = AttachSvc(inner);
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
                "/runtime.v1.RuntimeService/PortForward" => {
                    #[allow(non_camel_case_types)]
                    struct PortForwardSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::PortForwardRequest>
                    for PortForwardSvc<T> {
                        type Response = super::PortForwardResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PortForwardRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).port_forward(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = PortForwardSvc(inner);
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
                "/runtime.v1.RuntimeService/ContainerStats" => {
                    #[allow(non_camel_case_types)]
                    struct ContainerStatsSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ContainerStatsRequest>
                    for ContainerStatsSvc<T> {
                        type Response = super::ContainerStatsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ContainerStatsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).container_stats(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ContainerStatsSvc(inner);
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
                "/runtime.v1.RuntimeService/ListContainerStats" => {
                    #[allow(non_camel_case_types)]
                    struct ListContainerStatsSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ListContainerStatsRequest>
                    for ListContainerStatsSvc<T> {
                        type Response = super::ListContainerStatsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListContainerStatsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).list_container_stats(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListContainerStatsSvc(inner);
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
                "/runtime.v1.RuntimeService/PodSandboxStats" => {
                    #[allow(non_camel_case_types)]
                    struct PodSandboxStatsSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::PodSandboxStatsRequest>
                    for PodSandboxStatsSvc<T> {
                        type Response = super::PodSandboxStatsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PodSandboxStatsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).pod_sandbox_stats(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = PodSandboxStatsSvc(inner);
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
                "/runtime.v1.RuntimeService/ListPodSandboxStats" => {
                    #[allow(non_camel_case_types)]
                    struct ListPodSandboxStatsSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::ListPodSandboxStatsRequest>
                    for ListPodSandboxStatsSvc<T> {
                        type Response = super::ListPodSandboxStatsResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListPodSandboxStatsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).list_pod_sandbox_stats(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListPodSandboxStatsSvc(inner);
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
                "/runtime.v1.RuntimeService/UpdateRuntimeConfig" => {
                    #[allow(non_camel_case_types)]
                    struct UpdateRuntimeConfigSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::UpdateRuntimeConfigRequest>
                    for UpdateRuntimeConfigSvc<T> {
                        type Response = super::UpdateRuntimeConfigResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::UpdateRuntimeConfigRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).update_runtime_config(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = UpdateRuntimeConfigSvc(inner);
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
                "/runtime.v1.RuntimeService/Status" => {
                    #[allow(non_camel_case_types)]
                    struct StatusSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::StatusRequest>
                    for StatusSvc<T> {
                        type Response = super::StatusResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::StatusRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).status(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = StatusSvc(inner);
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
                "/runtime.v1.RuntimeService/CheckpointContainer" => {
                    #[allow(non_camel_case_types)]
                    struct CheckpointContainerSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::UnaryService<super::CheckpointContainerRequest>
                    for CheckpointContainerSvc<T> {
                        type Response = super::CheckpointContainerResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::CheckpointContainerRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).checkpoint_container(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = CheckpointContainerSvc(inner);
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
                "/runtime.v1.RuntimeService/GetContainerEvents" => {
                    #[allow(non_camel_case_types)]
                    struct GetContainerEventsSvc<T: RuntimeService>(pub Arc<T>);
                    impl<
                        T: RuntimeService,
                    > tonic::server::ServerStreamingService<super::GetEventsRequest>
                    for GetContainerEventsSvc<T> {
                        type Response = super::ContainerEventResponse;
                        type ResponseStream = T::GetContainerEventsStream;
                        type Future = BoxFuture<
                            tonic::Response<Self::ResponseStream>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::GetEventsRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).get_container_events(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = GetContainerEventsSvc(inner);
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
    impl<T: RuntimeService> Clone for RuntimeServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: RuntimeService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: RuntimeService> tonic::server::NamedService for RuntimeServiceServer<T> {
        const NAME: &'static str = "runtime.v1.RuntimeService";
    }
}
/// Generated server implementations.
pub mod image_service_server {
    #![allow(unused_variables, dead_code, missing_docs, clippy::let_unit_value)]
    use tonic::codegen::*;
    /// Generated trait containing gRPC methods that should be implemented for use with ImageServiceServer.
    #[async_trait]
    pub trait ImageService: Send + Sync + 'static {
        /// ListImages lists existing images.
        async fn list_images(
            &self,
            request: tonic::Request<super::ListImagesRequest>,
        ) -> Result<tonic::Response<super::ListImagesResponse>, tonic::Status>;
        /// ImageStatus returns the status of the image. If the image is not
        /// present, returns a response with ImageStatusResponse.Image set to
        /// nil.
        async fn image_status(
            &self,
            request: tonic::Request<super::ImageStatusRequest>,
        ) -> Result<tonic::Response<super::ImageStatusResponse>, tonic::Status>;
        /// PullImage pulls an image with authentication config.
        async fn pull_image(
            &self,
            request: tonic::Request<super::PullImageRequest>,
        ) -> Result<tonic::Response<super::PullImageResponse>, tonic::Status>;
        /// RemoveImage removes the image.
        /// This call is idempotent, and must not return an error if the image has
        /// already been removed.
        async fn remove_image(
            &self,
            request: tonic::Request<super::RemoveImageRequest>,
        ) -> Result<tonic::Response<super::RemoveImageResponse>, tonic::Status>;
        /// ImageFSInfo returns information of the filesystem that is used to store images.
        async fn image_fs_info(
            &self,
            request: tonic::Request<super::ImageFsInfoRequest>,
        ) -> Result<tonic::Response<super::ImageFsInfoResponse>, tonic::Status>;
    }
    /// ImageService defines the public APIs for managing images.
    #[derive(Debug)]
    pub struct ImageServiceServer<T: ImageService> {
        inner: _Inner<T>,
        accept_compression_encodings: EnabledCompressionEncodings,
        send_compression_encodings: EnabledCompressionEncodings,
    }
    struct _Inner<T>(Arc<T>);
    impl<T: ImageService> ImageServiceServer<T> {
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
    impl<T, B> tonic::codegen::Service<http::Request<B>> for ImageServiceServer<T>
    where
        T: ImageService,
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
                "/runtime.v1.ImageService/ListImages" => {
                    #[allow(non_camel_case_types)]
                    struct ListImagesSvc<T: ImageService>(pub Arc<T>);
                    impl<
                        T: ImageService,
                    > tonic::server::UnaryService<super::ListImagesRequest>
                    for ListImagesSvc<T> {
                        type Response = super::ListImagesResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ListImagesRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).list_images(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ListImagesSvc(inner);
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
                "/runtime.v1.ImageService/ImageStatus" => {
                    #[allow(non_camel_case_types)]
                    struct ImageStatusSvc<T: ImageService>(pub Arc<T>);
                    impl<
                        T: ImageService,
                    > tonic::server::UnaryService<super::ImageStatusRequest>
                    for ImageStatusSvc<T> {
                        type Response = super::ImageStatusResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ImageStatusRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).image_status(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ImageStatusSvc(inner);
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
                "/runtime.v1.ImageService/PullImage" => {
                    #[allow(non_camel_case_types)]
                    struct PullImageSvc<T: ImageService>(pub Arc<T>);
                    impl<
                        T: ImageService,
                    > tonic::server::UnaryService<super::PullImageRequest>
                    for PullImageSvc<T> {
                        type Response = super::PullImageResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::PullImageRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).pull_image(request).await };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = PullImageSvc(inner);
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
                "/runtime.v1.ImageService/RemoveImage" => {
                    #[allow(non_camel_case_types)]
                    struct RemoveImageSvc<T: ImageService>(pub Arc<T>);
                    impl<
                        T: ImageService,
                    > tonic::server::UnaryService<super::RemoveImageRequest>
                    for RemoveImageSvc<T> {
                        type Response = super::RemoveImageResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::RemoveImageRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).remove_image(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = RemoveImageSvc(inner);
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
                "/runtime.v1.ImageService/ImageFsInfo" => {
                    #[allow(non_camel_case_types)]
                    struct ImageFsInfoSvc<T: ImageService>(pub Arc<T>);
                    impl<
                        T: ImageService,
                    > tonic::server::UnaryService<super::ImageFsInfoRequest>
                    for ImageFsInfoSvc<T> {
                        type Response = super::ImageFsInfoResponse;
                        type Future = BoxFuture<
                            tonic::Response<Self::Response>,
                            tonic::Status,
                        >;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ImageFsInfoRequest>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move {
                                (*inner).image_fs_info(request).await
                            };
                            Box::pin(fut)
                        }
                    }
                    let accept_compression_encodings = self.accept_compression_encodings;
                    let send_compression_encodings = self.send_compression_encodings;
                    let inner = self.inner.clone();
                    let fut = async move {
                        let inner = inner.0;
                        let method = ImageFsInfoSvc(inner);
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
    impl<T: ImageService> Clone for ImageServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self {
                inner,
                accept_compression_encodings: self.accept_compression_encodings,
                send_compression_encodings: self.send_compression_encodings,
            }
        }
    }
    impl<T: ImageService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: ImageService> tonic::server::NamedService for ImageServiceServer<T> {
        const NAME: &'static str = "runtime.v1.ImageService";
    }
}
