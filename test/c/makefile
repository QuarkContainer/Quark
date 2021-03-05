all: std server client unixcli unixsrv socketpair stat dev fork signal futex multithread epoll mkdir fifo timerfd eventfd seek

std: std.c
	gcc -o std std.c
server: server.c
	gcc -o server server.c
client: client.c
	gcc -o client client.c
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
clean:
	rm std server client unixcli unixsrv socketpair stat dev fork signal futex multithread epoll mkdir fifo timerfd eventfd seek
