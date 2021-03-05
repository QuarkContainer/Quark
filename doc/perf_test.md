#Performance Test Result and Comparison

Quark Container is a secure container runtime with OCI interface. There are 2 other open source project provides similar functions: 
1. [Gvisor](https://gvisor.dev/)
2. [Kata Containers](https://github.com/kata-containers/)

Here we compare the startup time and memory overhead of container runtime.
1. Startup time: This test will test how much overhead time will take for container runtime between start a container application and the first set of system calls are executed by the application.
We are using `date` to do such test. The test script is as below.
```sh
date +%s%N; docker run --rm -it ubuntu /bin/date +%s%N
date +%s%N; docker run --runtime=quark --rm -it ubuntu /bin/date +%s%N
date +%s%N; docker run --runtime=runsc  --rm -it ubuntu /bin/date +%s%N
date +%s%N; docker run --runtime=kata-runtime --rm -it ubuntu /bin/date +%s%N
```

Runtime     | Runc      |   Quark       |   Runsc   |   Kata
____________|___________|_______________|___________|________
Startup (ms)| 607       |  625          |  708      |   1747


2. Memory usage overhead:

We also did performance test and comparison based on some popular open source service benchmarks. 
1. Etcd
2. Redis
3. Nginx
4. dd
5. MariaDB
6. Mysql

