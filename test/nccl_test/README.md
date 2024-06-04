# Quark Runtime Container Setup

Start a container with Quark runtime. The container must have MPI (not CUDA-aware at this point) installed. The host path should include Quark and NCCL (clone and build). If NCCL is installed in the image, this is not required.

## OpenMPI Installation

Install OpenMPI using the following command:

\`\`\`bash
sudo apt install openmpi-bin openmpi-dev openmpi-common openmpi-doc libopenmpi-dev
\`\`\`

## Docker Container Setup

Run the following commands to start the Docker container:

\`\`\`bash
ID=$(sudo docker run -it -d --runtime=quark_d -v /hostpath/:/countainerpath/ --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 vllmd:mpi bash 2>&1 | tee docker.log)
sudo docker exec -it ${ID} bash -c "cd .. && cd /countainerpath/Quark/test/nccl_test/ && export LD_LIBRARY_PATH=\"/countainerpath/nccl/build/lib/:\$LD_LIBRARY_PATH\" && export LD_LIBRARY_PATH=\"/countainerpath/Quark/target/debug/:\$LD_LIBRARY_PATH\" && bash" 2>&1 | tee container_mab.log
\`\`\`

## Building Test Code

Preferably run with a container without Quark runtime.

### Single Process Multi Device

\`\`\`bash
nvcc -cudart shared  nccl_test_sp.cpp -o nccl_te_sp -lcuda -lnccl
\`\`\`

### Multiprocess, One Device per Process

\`\`\`bash
nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -cudart shared nccl_test_mp.cpp -o nccl_te_mp -lcuda -lnccl
\`\`\`

## Running Test File

\`\`\`bash
LD_PRELOAD=/countainerpath/Quark/target/debug/libcudaproxy.so ./nccl_te_sp
\`\`\`

### Multiprocess

\`\`\`bash
mpirun -x LD_PRELOAD=/countainerpath/Quark/target/debug/libcudaproxy.so --host localhost,localhost -np 2 --allow-run-as-root ./nccl_te_mp
\`\`\`
