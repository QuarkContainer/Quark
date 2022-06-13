#define MAX_EVENTS 5
#define READ_SIZE 10
#include <stdio.h>     // for fprintf()
#include <unistd.h>    // for close(), read()
#include <sys/epoll.h> // for epoll_create1(), epoll_ctl(), struct epoll_event
#include <string.h>    // for strncmp
#include <sys/socket.h>
#include <sys/poll.h>

int main()
{
    printf("poll.c\n");
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    printf("sock is %d\n", sock);
    struct pollfd fds[1];
    fds[0].fd = sock;
	fds[0].events = 0x4;
    fds[0].revents = 0;

    int ret = poll(fds, 1, 0);
    printf("ret: %d, revents: %x\n", ret, fds[0].revents);
}