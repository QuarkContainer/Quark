#Performance Test Result and Comparison

Quark Container is a secure container runtime with OCI interface. There are 2 other open source project provides similar functions: 
1. [Gvisor](https://gvisor.dev/)
2. [Kata Containers](https://github.com/kata-containers/)

Here we compare the startup time and memory overhead of container runtime.
1. Startup time: Test how much overhead time will take for container runtime between start a container application and the first set of system calls are executed by the application.
We are using `date` to do such test. The test script is as below.
```sh
date +%s%N; docker run --rm -it ubuntu /bin/date +%s%N
date +%s%N; docker run --runtime=quark --rm -it ubuntu /bin/date +%s%N
date +%s%N; docker run --runtime=runsc  --rm -it ubuntu /bin/date +%s%N
date +%s%N; docker run --runtime=kata-runtime --rm -it ubuntu /bin/date +%s%N
```


|              | Runc | Quark | gVisor | Kata |
|--------------|------|-------|-------|------|
| Startup (ms) | 607  | 625   | 708   | 1747 |


2. Memory usage overhead: Test how much memory overhead the container runtime will consume. We are using busybox to test. 

We used "busybox" to measure the memory usage of container runtime. The test script is as below. Note: the memory usage only measure the processes other than `docker run -it --name some_busybox`.

```sh
docker run -it --name some_busybox --rm busybox
docker run -it --name some_busybox --runtime=quark --rm busybox
docker run -it --name some_busybox --runtime=runsc  --rm busybox
docker run -it --name some_busybox --runtime=kata-runtime --rm busybox
```

|                      | Runc | Quark | gVisor | Kata  |
|----------------------|------|-------|--------|-------|
| Memory Overhead (MB) | 0    | 11.8  | 28.1   | 184.3 |

We also did performance test and comparison based on some popular open source service benchmarks. 
1. Etcd3.0: The test is based on ETCD benchmark. 

The etcd startup script is as below.

```sh
"sudo docker run -p 2379:2379 -p 2380:2380 --rm --name etcd0 quay.io/coreos/etcd:v3.0.0 /usr/local/bin/etcd --name my-etcd-1 --listen-client-urls http://0.0.0.0:2379 \
                --advertise-client-urls http://0.0.0.0:2379 \
                --listen-peer-urls http://0.0.0.0:2380 \
                --initial-advertise-peer-urls http://0.0.0.0:2380 \
                --initial-cluster my-etcd-1=http://0.0.0.0:2380 \
                --initial-cluster-token my-etcd-token \
                --initial-cluster-state new"
"sudo docker run  -p 2379:2379 -p 2380:2380 --runtime=quark --rm --name etcd0 quay.io/coreos/etcd:v3.0.0 /usr/local/bin/etcd --name my-etcd-1 --listen-client-urls http://0.0.0.0:2379 \
                --advertise-client-urls http://0.0.0.0:2379 \
                --listen-peer-urls http://0.0.0.0:2380 \
                --initial-advertise-peer-urls http://0.0.0.0:2380 \
                --initial-cluster my-etcd-1=http://0.0.0.0:2380 \
                --initial-cluster-token my-etcd-token \
                --initial-cluster-state new"
"sudo docker run --runtime=runsc -p 2379:2379 -p 2380:2380 --rm --name etcd0 quay.io/coreos/etcd:v3.0.0 /usr/local/bin/etcd --name my-etcd-1 --listen-client-urls http://0.0.0.0:2379 \
                --advertise-client-urls http://0.0.0.0:2379 \
                --listen-peer-urls http://0.0.0.0:2380 \
                --initial-advertise-peer-urls http://0.0.0.0:2380 \
                --initial-cluster my-etcd-1=http://0.0.0.0:2380 \
                --initial-cluster-token my-etcd-token \
                --initial-cluster-state new"
"sudo docker run --runtime=kata-runtime -p 2379:2379 -p 2380:2380 --rm --name etcd0 quay.io/coreos/etcd:v3.0.0 /usr/local/bin/etcd --name my-etcd-1 --listen-client-urls http://0.0.0.0:2379 \
                --advertise-client-urls http://0.0.0.0:2379 \
                --listen-peer-urls http://0.0.0.0:2380 \
                --initial-advertise-peer-urls http://0.0.0.0:2380 \
                --initial-cluster my-etcd-1=http://0.0.0.0:2380 \
                --initial-cluster-token my-etcd-token \
                --initial-cluster-state new"
```

The benchmark script is as below.
```sh
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  put --key-size=8 --sequential-keys --total=10000 --val-size=256
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  lease-keepalive --total=10000
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  mvcc put --total=10000
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  range 123 1234567890 --total=10000
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  stm --total=10000
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  txn-put --total=10000
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  watch
./benchmark --endpoints=http://localhost:2379 --target-leader --conns=10 --clients=10  watch-get
```

The result is as below.
|                 | Runc   | Quark  | gVisor | Kata   |
|-----------------|--------|--------|--------|--------|
| Put             | 3741   | 3471   | 2408   | 769    |
| lease-keepalive | 17883  | 6802   | 5647   | 4982   |
| mvcc Put        | 127377 | 98250  | 105232 | 101256 |
| range           | 7856   | 5117   | 1837   | 932    |
| stm             | 7877   | 3544   | 2379   | 2059   |
| txn-put         | 4933   | 3359   | 2189   | 746    |
| watch           | 243688 | 116216 | 134686 | 28035  |
| watch-get       | 3418   | 1941   | 1072   | 1774   |

2. Redis
The Redis startup scripts are as below.
```sh
docker run -p 6379:6379  --name some-redis --rm  -it redis
docker run -p 6379:6379  --runtime=quark --name some-redis --rm  -it redis
docker run -p 6379:6379  --runtime=runsc --name some-redis --rm  -it redis
docker run -p 6379:6379  --runtime=kata-runtime --name some-redis --rm  -it redis
```

The benchmark execution scripts are as below.
```sh
redis-benchmark -n 100000 -c 20
```

Test result is as below.
|             | Runc  | Quark | gVisor | Kata  |
|-------------|-------|-------|--------|-------|
| PING_INLINE | 38476 | 36791 | 8204   | 31446 |
| PING_BULK   | 37579 | 38240 | 8481   | 33602 |
| SET         | 38595 | 37864 | 8348   | 34698 |
| GET         | 38624 | 37425 | 8374   | 34048 |
| INCR        | 38520 | 37037 | 8316   | 34867 |
| LPUSH       | 39231 | 37509 | 8214   | 33840 |
| RPUSH       | 38955 | 37950 | 8259   | 32393 |
| LPOP        | 37355 | 37341 | 8212   | 32948 |
| RPOP        | 38270 | 38431 | 8269   | 34506 |
| SADD        | 39370 | 39354 | 8297   | 35310 |
| HSET        | 39541 | 37821 | 8205   | 33579 |
| SPOP        | 37341 | 37878 | 8467   | 34782 |
| LPUSH       | 39370 | 37921 | 8206   | 33211 |
| LRANGE_100  | 25177 | 24142 | 6955   | 24703 |
| LRANGE_300  | 13159 | 13504 | 5014   | 12980 |
| LRANGE_500  | 9090  | 9573  | 4231   | 9044  |
| LRANGE_600  | 8328  | 7937  | 3384   | 7176  |
| MSET        | 38402 | 37707 | 7884   | 30147 |


3. Nginx: We use Nginx to test http get performance

The Nginx startup script is as below.
```sh
docker run --name some-nginx -it --runtime=runc --rm -p 80:80  --rm -v /home/brad/website/:/usr/share/nginx/html:ro nginx
docker run --name some-nginx -it --runtime=quark --rm -p 80:80  --rm -v /home/brad/website/:/usr/share/nginx/html:ro nginx
docker run --name some-nginx -it --runtime=runsc --rm -p 80:80  --rm -v /home/brad/website/:/usr/share/nginx/html:ro nginx
docker run --name some-nginx -it --runtime=kata-runtime --rm -p 80:80  --rm -v /home/brad/website/:/usr/share/nginx/html:ro nginx
```

The index.html is [this](doc/index.html)

The benchmark script is as below.
```sh
ab -n 10000 -c 10 http://localhost/index.html
```

The test is as below.
|                          | Runc | Quark   | gVisor | Kata   |
|--------------------------|------|---------|--------|--------|
| RPS (Request Per Second) | 6136 | 2274.74 | 1296.7 | 928.96 |

4. dd: We use dd to test disk write performance

The test script is as below.
```sh
docker run -P --mount type=bind,source="/home/brad/rust/quark/test",target=/test --rm -it ubuntu /bin/dd if=/dev/zero of=/test/fio-rand-read bs=4k count=2500
docker run -P --runtime=quark --mount type=bind,source="/home/brad/rust/quark/test",target=/test --rm -it ubuntu /bin/dd if=/dev/zero of=/test/fio-rand-read bs=4k count=2500
docker run -P --runtime=runsc --mount type=bind,source="/home/brad/rust/quark/test",target=/test --rm -it ubuntu /bin/dd if=/dev/zero of=/test/fio-rand-read bs=4k count=2500
docker run -P --runtime=kata-runtime --mount type=bind,source="/home/brad/rust/quark/test",target=/test --rm -it ubuntu /bin/dd if=/dev/zero of=/test/fio-rand-read bs=4k count=2500
```

The test result is as below.
|        | Runc | Quark | gVisor | Kata |
|--------|------|-------|--------|------|
| MB/Sec | 397  | 179   | 30.5   | 103  |


5. MariaDB initialization time: MariaDB's initialization is complex as it includes much IO/CPU/Memory operation. So we use as benchmark.

The test script is as below.
```sh
sudo docker run --net=host --rm --name some-mariadb -e MYSQL_ROOT_PASSWORD=123 -it mariadb
sudo docker run --net=host  --runtime=quark  --rm --name some-mariadb -e MYSQL_ROOT_PASSWORD=123 -it mariadb
sudo docker run --runtime=runsc  --rm --name some-mariadb -e MYSQL_ROOT_PASSWORD=123 -it mariadb
sudo docker run --runtime=kata-runtime --rm --name some-mariadb -e MYSQL_ROOT_PASSWORD=123 -it mariadb
```


The test result is as below.
|      | Runc | Quark | gVisor | Kata |
|------|------|-------|--------|------|
| Sec  | 8    | 10    | 14     | 12   |

6. Mysql initialization time: We also measure the MySql initialization time.
 ```sh
sudo docker run -P --runtime=runc  --rm --name some-mysql -e MYSQL_ROOT_PASSWORD=123 -it mysql
sudo docker run -P --runtime=quark  --rm --name some-mysql -e MYSQL_ROOT_PASSWORD=123 -it mysql
sudo docker run -P --runtime=runsc --rm --name some-mysql -e MYSQL_ROOT_PASSWORD=123 -it mysql
sudo docker run -P --runtime=kata-runtime --rm --name some-mysql -e MYSQL_ROOT_PASSWORD=123 -it mysql
 ```
 

The test result is as below. The Kata can't run mysql with error "ERROR 14 (HY000) at line 147118: Can't change size of file (OS errno 2 - No such file or directory)"
|      | Runc | Quark | gVisor | Kata |
|------|------|-------|--------|------|
| Sec  | 18   | 20    | 32     | N/A  |