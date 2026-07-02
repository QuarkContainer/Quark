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

## clean and rewrite containerd config (containerd 2.x)
docker exec $MINIKUBE_DOCKER_ID bash -c 'containerd config default > /tmp/containerd.toml'
docker exec $MINIKUBE_DOCKER_ID python3 << 'PY'
from pathlib import Path
p = Path("/tmp/containerd.toml")
text = p.read_text()
extra = """
      [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.quark]
        runtime_type = 'io.containerd.quark.v1'
        sandboxer = 'podsandbox'
      [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runsc]
        runtime_type = 'io.containerd.runsc.v1'
"""
if "runtimes.quark" not in text:
    runc_end = text.find("\n\n", text.find("runtimes.runc.options"))
    if runc_end == -1:
        raise SystemExit("could not locate end of runc runtime block")
    text = text[:runc_end] + "\n" + extra + text[runc_end:]
# Dev-only: run workload via runsc shim pointing at quark_d
text = text.replace(
    "runtime_type = 'io.containerd.runc.v2'",
    "runtime_type = 'io.containerd.runc.v2'\n        [plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.runc.options]\n          BinaryName = 'quark_d'",
    1,
)
p.write_text(text)
PY
docker cp /tmp/containerd.toml $MINIKUBE_DOCKER_ID:/etc/containerd/config.toml 2>/dev/null || \
  docker exec $MINIKUBE_DOCKER_ID cp /tmp/containerd.toml /etc/containerd/config.toml

# runsc config
docker exec $MINIKUBE_DOCKER_ID rm -f /etc/containerd/runsc.toml
cat  <<EOF > /tmp/runsc.toml
binary_name="quark_d"
EOF
# command or uncomment the following line to run with quark/gvisor
docker cp /tmp/runsc.toml $MINIKUBE_DOCKER_ID:/etc/containerd/runsc.toml


docker exec $MINIKUBE_DOCKER_ID systemctl restart containerd