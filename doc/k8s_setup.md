# Kubernates Quick Start
This document records how to use quark container with Kubernates. Quark uses containerd as high level container runtime with running on k8s.

Follow the steps to use quark runtime with kubernates.
### 1. Build Quark Runtime with shim mode
When quark is used as a container runtime, qvisor process will need to serve as the shim for the runtime as specified by containerd [shim-api](https://github.com/containerd/containerd/blob/main/runtime/v2/README.md).

To build Quark with shim mode, change the following configuration in config.json
```
........
  "ShimMode"      : true,
......
```
and run 
```
make clean; make
```
in a terminal to rebuild quark binary

### 2. Install Quark binary to each k8s nodes
An example scripts when running k8s on minikube:

```
# write binaries to minikube containerd
# cleanup old binaries if any
docker exec minikube rm -f /usr/local/bin/qkernel.bin
docker exec minikube rm -f /usr/local/bin/qkernel_d.bin
docker exec minikube rm -f /usr/local/bin/quark
docker exec minikube rm -f /usr/local/bin/quark_d
docker exec minikube rm -f /usr/bin/containerd-shim-runc-v2
# copy new binaries to target directory
minikube cp  ./build/qkernel.bin minikube:/usr/local/bin/qkernel.bin
minikube cp  ./build/qkernel_d.bin minikube:/usr/local/bin/qkernel_d.bin
minikube cp  ./target/release/quark minikube:/usr/local/bin/quark
minikube cp  ./target/debug/quark minikube:/usr/local/bin/quark_d
minikube cp  ./target/debug/quark minikube:/usr/local/bin/containerd-shim-quarkd-v1

# change permissions
docker exec minikube chmod 755 /usr/local/bin/qkernel_d.bin
docker exec minikube chmod 755 /usr/local/bin/qkernel.bin
docker exec minikube chmod 755 /usr/local/bin/quark
docker exec minikube chmod 755 /usr/local/bin/quark_d
docker exec minikube chmod 755 /usr/local/bin/containerd-shim-quarkd-v1


# copy config
docker exec minikube mkdir -p /etc/quark/
minikube cp ./config.json minikube:/etc/quark/config.json

## create quark log directory if haven't created
docker exec minikube mkdir -p /var/log/quark
docker exec minikube touch /var/log/quark/quark.log


## copy vdso.so, as qkernel needs it
# adhoc only
docker exec minikube rm -f /usr/local/bin/vdso.so
minikube cp /usr/local/bin/vdso.so minikube:/usr/local/bin/vdso.so
docker exec minikube chmod 755 /usr/local/bin/vdso.so


```
Notice the quark binary is renamed as `containerd-shim-quarkd-v1`, this is to follow containerd's naming convention for shims.

### 3. Config containerd in k8s cluster
This step need to happen on every k8s node with kubelet running.
open `/etc/containerd/config.yaml` and add/modify the following entry in the containerd config
```

[plugins]
...
  [plugins."io.containerd.grpc.v1.cri"]
  ...
      [plugins."io.containerd.grpc.v1.cri".containerd]
      ...
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
        ...
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.quarkd]
            runtime_type = "io.containerd.quarkd.v1"
```
And restart the containerd service with `systemctl restart containerd`

### 4. Add quark as a Runtime Resource to K8S
Now we can use Quark as a container runtime in K8S
first add Quark as a K8S resources, with kubectl:
```
cat <<EOF | minikube kubectl -- apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: quark
handler: quarkd
EOF
```
Then you can use Quark like this
```
cat <<EOF | minikube kubectl -- apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: nginx-quark
spec:
  runtimeClassName: quark
  containers:
  - name: nginx
    image: nginx
EOF
```




