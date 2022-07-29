#include <netinet/in.h>
#include <sys/socket.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <errno.h>

#define MAXSIZE     1024
#define IPADDRESS   "127.0.0.1"
#define SERV_PORT   8787
#define FDSIZE        1024
#define EPOLLEVENTS 20
#define PORT 9987
#define BUFFERNUM 1024*32

static void handle_connection(int sockfd);
static void
handle_events(int epollfd,struct epoll_event *events,int num,int sockfd,char *buf);
static void do_read(int epollfd,int fd,int sockfd,char *buf);
static void do_read(int epollfd,int fd,int sockfd,char *buf);
static void do_write(int epollfd,int fd,int sockfd,char *buf);
static void add_event(int epollfd,int fd,int state);
static void add_event2(int epollfd,int fd, int index, int state);
static void delete_event(int epollfd,int fd,int state);
static void modify_event(int epollfd,int fd,int state);

int main(int argc,char *argv[])
{
    char *addr = "127.0.0.1";
    // if (argc < 2)
    // {
    //     printf("read count must be specified\n");
    //     return;
    // }
    if (argc > 1)
    {
        addr = argv[1];
    }

    printf("add is %s\n", addr);
    long long readCount = 1000000;
    if (argc > 2)
    {
        readCount = atoll(argv[2]);
    }
    printf("readCount: %lld\n", readCount);
    

    
    int buffernum = BUFFERNUM;
    if (argc > 3)
    {
        buffernum = atoi(argv[3]);
    }
    printf("buffer size is %d\n", buffernum);

    int connectionNum = 1;
    
    if (argc > 4)
    {
        connectionNum = atoi(argv[4]);
    }
    printf("connection num is %d\n", connectionNum);

    int port = PORT;

    if (argc > 5)
    {
        port = atoi(argv[5]);
    }
    printf("port is %d\n", port);

    int log = 0;
    if (argc > 6)
    {
        log = strcmp(argv[6], "log") == 0 ? 1 : 0;
    }

    printf("log is %d\n", log);

    long long bytes = buffernum * readCount;
    printf("bytes is %llu\n", bytes);

    int sock = 0, valread;
    struct sockaddr_in serv_addr; 
    char *hello = "Hello from client"; 
    
    

    char* buffer = malloc(buffernum); 
    // if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    // { 
    //     printf("\n Socket creation error \n"); 
    //     return -1; 
    // } 

    printf("start to connect \n");
    sleep(1);

    serv_addr.sin_family = AF_INET;
    
    serv_addr.sin_port = htons(port); 

      
    // Convert IPv4 and IPv6 addresses from text to binary form 
    if(inet_pton(AF_INET, addr, &serv_addr.sin_addr)<=0)  
    { 
        printf("\nInvalid address/ Address not supported \n"); 
        return -1; 
    } 

    int *fds = malloc(connectionNum * sizeof(int));
    int *readNum = malloc(connectionNum * sizeof(int));
    for (int i = 0; i < connectionNum; i ++)
    {
        int socketfd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(socketfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) 
        {
            perror("connect failed\n");
            return;
        }
        printf("connected to server, socketfd is: %d\n", socketfd);
        fds[i] = socketfd;
        readNum[i] = 0;
    }

    //do reading
    int epollfd;
    struct epoll_event events[EPOLLEVENTS];
    int ret;
    epollfd = epoll_create(FDSIZE);
    for (int i = 0; i < connectionNum; i++)
    {
        add_event2(epollfd, fds[i], i, EPOLLIN);
    }

    struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    for (int i = 0; i < connectionNum; i++)
    {
        int nwrite = write(fds[i], buffer, buffernum);
        if (nwrite != buffernum) {
            printf("fd: %d, nwrite: %d, buffernum: %d\n", fds[i], nwrite, buffernum);
            return;
        }
    }

    
    int leftConnectionNum = connectionNum;
    while (leftConnectionNum > 0)
    {
        if (log)
        {
            printf("wait for epoll notificaiton\n");
        }
        
        ret = epoll_wait(epollfd,events,EPOLLEVENTS,-1);
        if (log) 
        {
            printf("get epoll notificaiton\n");
        }
        
        for (int i = 0;i < ret;i++)
        {
            int idx = events[i].data.u32;
            int fd = fds[idx];
            if (log)
            {
                printf("fd: %d got notification\n", fd);
            }
            
            int nread = read(fd, buffer, buffernum);
            if (log)
            {
                printf("read %d \n", nread);
            }
            
            if (nread == -1 || nread != buffernum)
            {
                perror("Error printed by perror");
                return;
            }
            readNum[idx] += 1;
            if (readNum[idx] == readCount) {
                // close(fd);
                leftConnectionNum -= 1;
                continue;
            }

            int nwrite = write(fd, buffer, buffernum);

            if (nwrite == -1 || nwrite != buffernum)
            {
                printf("Error printed by perror, nread: %d\n", nwrite);
                return;
            }

            if (log)
            {
                printf("write %d \n", nwrite);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &tend);
    printf("time used: %lf\n", 10.0);
    double ws = (double)(tend.tv_sec - tstart.tv_sec) * 1.0e6 + (double)(tend.tv_nsec - tstart.tv_nsec)/1.0e3;
    printf("time used: %lf\n", ws);
    printf("time used: %lf\n", ws);
    //double speed = ((double)buffernum * (double)readCount) / (ns);
    long long totalbytes = connectionNum * buffernum * readCount;
    printf("recv bytes: %lld\n", totalbytes);
    double speed = ((double)totalbytes*2) / (ws);
    printf("speed is %lf\n", speed);
    double pps = (2 * totalbytes * 1024 * 1024) / (ws)/buffernum;
    printf("pps is %lf\n", pps);
    double latency = ws / (readCount * 2 * connectionNum);
    close(epollfd);
    for (int i = 0; i < connectionNum; i ++)
    {
        close(fds[i]);
    }

    free(fds);

    return 0;
}


static void handle_connection(int sockfd)
{
    int epollfd;
    struct epoll_event events[EPOLLEVENTS];
    char buf[MAXSIZE];
    int ret;
    epollfd = epoll_create(FDSIZE);
    add_event(epollfd,STDIN_FILENO,EPOLLIN);
    for ( ; ; )
    {
        ret = epoll_wait(epollfd,events,EPOLLEVENTS,-1);
        handle_events(epollfd,events,ret,sockfd,buf);
    }
    close(epollfd);
}

static void
handle_events(int epollfd,struct epoll_event *events,int num,int sockfd,char *buf)
{
    int fd;
    int i;
    for (i = 0;i < num;i++)
    {
        fd = events[i].data.fd;
        if (events[i].events & EPOLLIN)
            do_read(epollfd,fd,sockfd,buf);
        else if (events[i].events & EPOLLOUT)
            do_write(epollfd,fd,sockfd,buf);
    }
}

static void do_read(int epollfd,int fd,int sockfd,char *buf)
{
    int nread;
    nread = read(fd,buf,MAXSIZE);
        if (nread == -1)
    {
        perror("read error:");
        close(fd);
    }
    else if (nread == 0)
    {
        fprintf(stderr,"server close.\n");
        close(fd);
    }
    else
    {
        if (fd == STDIN_FILENO)
            add_event(epollfd,sockfd,EPOLLOUT);
        else
        {
            delete_event(epollfd,sockfd,EPOLLIN);
            add_event(epollfd,STDOUT_FILENO,EPOLLOUT);
        }
    }
}

static void do_write(int epollfd,int fd,int sockfd,char *buf)
{
    int nwrite;
    nwrite = write(fd,buf,strlen(buf));
    if (nwrite == -1)
    {
        perror("write error:");
        close(fd);
    }
    else
    {
        if (fd == STDOUT_FILENO)
            delete_event(epollfd,fd,EPOLLOUT);
        else
            modify_event(epollfd,fd,EPOLLIN);
    }
    memset(buf,0,MAXSIZE);
}

static void add_event(int epollfd,int fd, int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.fd = fd;
    epoll_ctl(epollfd,EPOLL_CTL_ADD,fd,&ev);
}

static void add_event2(int epollfd,int fd, int index, int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.fd = fd;
    ev.data.u32 = index;
    epoll_ctl(epollfd,EPOLL_CTL_ADD,fd,&ev);
}

static void delete_event(int epollfd,int fd,int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.fd = fd;
    epoll_ctl(epollfd,EPOLL_CTL_DEL,fd,&ev);
}

static void modify_event(int epollfd,int fd,int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.fd = fd;
    epoll_ctl(epollfd,EPOLL_CTL_MOD,fd,&ev);
}