# This script is for preparing the environment to run containers with quark in minikube
# First start a minikube with containerd as high level runtime with `minikube start --container-runtime=containerd`
# And use this script in host machine to make it ready to use quark as low level container runtime
set -e

MINIKUBE_DOCKER_ID=`docker ps | grep minikube | awk '{print $1}'`

## install quark binary
docker exec $MINIKUBE_DOCKER_ID rm -f /usr/local/bin/quark_d
docker exec $MINIKUBE_DOCKER_ID rm -f /usr/local/bin/qkernel.bin
minikube cp /usr/local/bin/quark_d minikube:/usr/local/bin/quark_d
minikube cp /usr/local/bin/qkernel_d.bin minikube:/usr/local/bin/qkernel_d.bin
docker exec $MINIKUBE_DOCKER_ID chmod 755 /usr/local/bin/quark_d
docker exec $MINIKUBE_DOCKER_ID chmod 755 /usr/local/bin/qkernel_d.bin

# copy config
docker exec $MINIKUBE_DOCKER_ID mkdir -p /etc/quark/
minikube cp /etc/quark/config.json minikube:/etc/quark/config.json

## create quark log directory if haven't
docker exec $MINIKUBE_DOCKER_ID mkdir -p /var/log/quark
#docker exec $MINIKUBE_DOCKER_ID rm /var/log/quark/quark.log

## copy vdso.so, as qkernel needs it
# adhoc only
docker exec $MINIKUBE_DOCKER_ID rm -f /usr/local/bin/vdso.so
minikube cp /usr/local/bin/vdso.so minikube:/usr/local/bin/vdso.so
docker exec $MINIKUBE_DOCKER_ID chmod 755 /usr/local/bin/vdso.so

## copy runsc-shim so that we can try to use it with quark
minikube cp /usr/local/bin/containerd-shim-runsc-v1 minikube:/usr/local/bin/containerd-shim-runsc-v1
docker exec $MINIKUBE_DOCKER_ID chmod 755 /usr/local/bin/containerd-shim-runsc-v1

## copy runsc into the env too for comparision
minikube cp /usr/local/bin/runsc minikube:/usr/local/bin/runsc
docker exec $MINIKUBE_DOCKER_ID chmod 755 /usr/local/bin/runsc

## clean and rewrite containerd config
docker exec $MINIKUBE_DOCKER_ID rm /etc/containerd/config.toml
cat  <<EOF > /tmp/containerd.toml
version = 2
root = "/var/lib/containerd"
state = "/run/containerd"
oom_score = 0
[grpc]
  address = "/run/containerd/containerd.sock"
  uid = 0
  gid = 0
  max_recv_message_size = 16777216
  max_send_message_size = 16777216

[debug]
  address = ""
  uid = 0
  gid = 0
  level = "debug"

[metrics]
  address = ""
  grpc_histogram = false

[cgroup]
  path = ""

[proxy_plugins]
# fuse-overlayfs is used for rootless
[proxy_plugins."fuse-overlayfs"]
  type = "snapshot"
  address = "/run/containerd-fuse-overlayfs.sock"

[plugins]
  [plugins."io.containerd.runtime.v1.linux"]
    shim_debug = true
  [plugins."io.containerd.monitor.v1.cgroups"]
    no_prometheus = false
  [plugins."io.containerd.grpc.v1.cri"]
    stream_server_address = ""
    stream_server_port = "10010"
    enable_selinux = false
    sandbox_image = "k8s.gcr.io/pause:3.5"
    stats_collect_period = 10
    enable_tls_streaming = false
    max_container_log_line_size = 16384
    restrict_oom_score_adj = false
      [plugins."io.containerd.grpc.v1.cri".containerd]
            snapshotter = "overlayfs"
            default_runtime_name = "runc"
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
            runtime_type = "io.containerd.runc.v2"
            [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
              SystemdCgroup = false
              BinaryName = "quark_d"
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runsc]
            runtime_type = "io.containerd.runsc.v1"
            [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runsc.options]
              TypeUrl = "io.containerd.runsc.v1.options"
              ConfigPath = "/etc/containerd/runsc.toml"
      [plugins."io.containerd.grpc.v1.cri".containerd.untrusted_workload_runtime]
        runtime_type = ""
        runtime_engine = ""
        runtime_root = ""
    [plugins."io.containerd.grpc.v1.cri".cni]
      bin_dir = "/opt/cni/bin"
      conf_dir = "/etc/cni/net.mk"
      conf_template = ""
    [plugins."io.containerd.grpc.v1.cri".registry]
      [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
        [plugins."io.containerd.grpc.v1.cri".registry.mirrors."docker.io"]
          endpoint = ["https://registry-1.docker.io"]
  [plugins."io.containerd.gc.v1.scheduler"]
    pause_threshold = 0.02
    deletion_threshold = 0
    mutation_threshold = 100
    schedule_delay = "0s"
    startup_delay = "100ms"
  [plugins."io.containerd.service.v1.diff-service"]
    default = ["walking"]
EOF

docker cp /tmp/containerd.toml $MINIKUBE_DOCKER_ID:/etc/containerd/config.toml

# runsc config
docker exec $MINIKUBE_DOCKER_ID rm -f /etc/containerd/runsc.toml
cat  <<EOF > /tmp/runsc.toml
binary_name="quark_d"
EOF
# command or uncomment the following line to run with quark/gvisor
docker cp /tmp/runsc.toml $MINIKUBE_DOCKER_ID:/etc/containerd/runsc.toml


docker exec $MINIKUBE_DOCKER_ID systemctl restart containerd