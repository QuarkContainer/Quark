cat << EOF | sudo tee /etc/cni/net.d/10-containerd-net.conflist
{
  "cniVersion": "0.4.0",
    "name": "containerd-net",
    "plugins": [
    {
      "type": "bridge",
      "bridge": "cni0",
      "isGateway": true,
      "ipMasq": true,
      "promiscMode": true,
      "ipam": {
        "type": "host-local",
        "ranges": [
          [{
            "subnet": "10.22.0.0/16"

          }]

        ],
        "routes": [
        { "dst": "0.0.0.0/0" }
        ]
      }
    },
    {
      "type": "portmap",
      "capabilities": {"portMappings": true}
    }
  ]
}
EOF
sudo systemctl restart containerd
sudo chmod 777 /run/containerd/containerd.sock
