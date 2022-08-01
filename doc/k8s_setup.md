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


```
make install
```
Notice the quark binary is renamed as `containerd-shim-quarkd-v1`, this is to follow containerd's naming convention for shims.

### 3. Config containerd in k8s cluster
This step need to happen on every k8s node with kubelet running.
open `/etc/containerd/config.yaml` and add/modify the following entry in the containerd config
```
cat <<EOF | sudo tee /etc/containerd/config.toml
version = 2
[plugins."io.containerd.runtime.v1.linux"]
  shim_debug = true
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
  runtime_type = "io.containerd.runc.v2"
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runsc]
  runtime_type = "io.containerd.runsc.v1"
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.quarkd]
  runtime_type = "io.containerd.quarkd.v1"
EOF
```
And restart the containerd service with `systemctl restart containerd`


### 4. Start a k8s cluster
There are multiple ways to start a k8s cluster. We recommend using kubeadm to start a production k8s cluster. Please check [kubeadm](https://kubernetes.io/docs/reference/setup-tools/kubeadm/) on how to install and use kubeadm.

For kubeadm init and join command, need to set parameter "--cri-socket=/var/run/containerd/containerd.sock".

Following is sample kubeadm command to init a cluster.
```
# Execute on master node
sudo kubeadm init --cri-socket=/var/run/containerd/containerd.sock --pod-network-cidr=10.244.0.0/16

sudo rm $HOME/.kube/config
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Optional, make master node runable for pod:
kubectl taint nodes --all node-role.kubernetes.io/master-
```

```
# Execute on worker node
# Need to replace token and cert with the real one in the master node. 
# The data can be found in master node's kubeadm init log.
sudo kubeadm join 10.218.233.29:6443 --cri-socket=/var/run/containerd/containerd.sock --token qy2r1j.t0y5ekx71t0tcfiq \
        --discovery-token-ca-cert-hash sha256:78a23762652befd90bbcd3506ca9309c5243371360d7a66fc131cb1a4b255553
```

### 5. Add CNI to K8S
Container Network Interface (CNI) provides networking to k8s. Following example use flannel as CNI for test purpose.
```
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml
```

### 6. Add quark as a Runtime Resource to K8S
Now we can use Quark as a container runtime in K8S
first add Quark as a K8S resources, with kubectl:
```
cat <<EOF | kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: quark
handler: quarkd
EOF
```
Then you can use Quark like this
```
cat <<EOF | kubectl apply -f -
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

### 7. Use RDMA networking

Quark container now support TCP over RDMA which means if you have RDMA NIC on all the nodes, with proper configration, Quark container can bypass TCP/IP stack and use RDMA NIC to send/receive network traffic without exiting TCP/IP application change. 

In each node, change MTU to 4200 using following command:
```
sudo ifconfig [RDMA NIC name] mtu 4200
```

Remove previous flannel CNI if there is:
```
sudo rm /etc/cni/net.d/10-flannel.conflist
```

Change the following configuration in config.json
```
........
  "EnableRDMA"      : true,
......
```
and run in each node:
```
make; make install
```

Then use kubectl to install quarkcm CNI:
```
kubectl apply -f https://raw.githubusercontent.com/QuarkContainer/quarkcm/main/deploy/deploy-quarkcm.yaml
```

Now the network communication between pods will be through RDMA.
