all: std server client unixcli unixsrv socketpair stat dev fork signal futex multithread epoll mkdir fifo timerfd eventfd seek gettimeofday server_benchmark client_benchmark epoll_client epoll_server multithread_client multithread_server multithread_pp_client multithread_pp_server

std: std.c
	gcc -o std std.c
server: server.c
	gcc -o server server.c
client: client.c
	gcc -o client client.c
epoll_client: epoll_client.c
	gcc -o epollc epoll_client.c
epoll_server: epoll_server.c
	gcc -o epolls epoll_server.c
multithread_client: multithread_client.c
	gcc -pthread -o mc multithread_client.c
multithread_server: multithread_server.c
	gcc -pthread -o ms multithread_server.c
multithread_pp_client: multithread_pp_client.c
	gcc -pthread -o mpc multithread_pp_client.c
multithread_pp_server: multithread_pp_server.c
	gcc -pthread -o mps multithread_pp_server.c
multithread_epoll_pp_client: multithread_epoll_pp_client.c
	gcc -pthread -o mepc multithread_epoll_pp_client.c
multithread_epoll_pp_server: multithread_epoll_pp_server.c
	gcc -pthread -o meps multithread_epoll_pp_server.c
server_benchmark: server_benchmark.c
	gcc -o serverb server_benchmark.c
client_benchmark: client_benchmark.c
	gcc -o clientb client_benchmark.c
unixcli: unixcli.c
	gcc -o unixcli unixcli.c
unixsrv: unixsrv.c
	gcc -o unixsrv unixsrv.c
socketpair: socketpair.c
	gcc -o socketpair socketpair.c
stat: stat.c
	gcc -o stat stat.c
dev: dev.c
	gcc -o dev dev.c
fork: fork.c
	gcc -o fork fork.c
signal: signal.c
	gcc -o signal signal.c
futex: futex.c
	gcc -o futex futex.c
multithread: multithread.c
	gcc -pthread -o multithread multithread.c
epoll: epoll.c
	gcc -o epoll epoll.c
mkdir: mkdir.c
	gcc -o mkdir mkdir.c
fifo: fifo.c
	gcc -o fifo fifo.c
timerfd: timerfd.c
	gcc -o timerfd timerfd.c
eventfd: eventfd.c
	gcc -o eventfd eventfd.c -lpthread
seek: seek.c
	gcc -o seek seek.c
gettimeofday: gettimeofday.c
	gcc -o gettimeofday gettimeofday.c
clean:
	rm std server client unixcli unixsrv socketpair stat dev fork signal futex multithread epoll mkdir fifo timerfd eventfd seek gettimeofday
