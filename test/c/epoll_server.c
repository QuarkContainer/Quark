// Copyright (c) 2021 Quark Container Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/types.h>

#define IPADDRESS   "127.0.0.1"
#define PORT        8787
#define MAXSIZE     1024
#define LISTENQ     5
#define FDSIZE      1000
#define EPOLLEVENTS 100

#define BUFFERNUM 10000

//函数声明
//创建套接字并进行绑定
static int socket_bind(const char* ip,int port);
//IO多路复用epoll
static void do_epoll(int * fds, int connectNum, char * buf, int bufferSize);
//事件处理函数
static void
handle_events(int epollfd,struct epoll_event *events,int num,char *buf, int bufSize);
//处理接收到的连接
static void handle_accpet(int epollfd,int listenfd);
//读处理
static void do_read(int epollfd,int fd,char *buf);
//写处理
static void do_write(int epollfd,int fd,char *buf);
//添加事件
static void add_event(int epollfd,int fd,int state);
//修改事件
static void modify_event(int epollfd,int fd,int state);
//删除事件
static void delete_event(int epollfd,int fd,int state);

int main(int argc,char *argv[])
{
    int server_fd, new_socket, valread; 
    struct sockaddr_in address; 
    int opt = 1; 
    int addrlen = sizeof(address); 
    char *hello = "Hello from server";

    long long writeCount = 1000000;
    if (argc >1)
    {
        writeCount = atoi(argv[1]);
    }
    printf("writeCount: %lld\n", writeCount);

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
   
    int buffernum = BUFFERNUM;
    if (argc >2)
    {
        buffernum = atoi(argv[2]);
    }
    printf("buffer len is %d\n", buffernum);
    
    int connectionNum = 1;
    
    if (argc >3)
    {
        connectionNum = atoi(argv[3]);
    }
    printf("connection num is %d\n", connectionNum);

    int port = PORT;
    if (argc >4)
    {
        port = atoi(argv[4]);
    }
    printf("sin_port is %d\n", port);

    int log = 0;
    if (argc >5)
    {
        log = strcmp(argv[5], "log") == 0 ? 1 : 0;
    }
    printf("log is %d\n", log);

    address.sin_port = htons(port);

    char* buffer = malloc(buffernum);
    //memset(buffer, 'a', buffernum);
    for (int i = 0; i < buffernum; i ++)
    {
        buffer[i] = 'A' + random() % 26;
    }
    
    long long bytes = buffernum * writeCount;
    printf("bytes is %llu\n", bytes);

       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address,  
                                 sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    printf("bind successfully\n");
    if (listen(server_fd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 
    printf("listen successfully\n");

    int *fds = malloc(connectionNum * sizeof(int));
    for (int i = 0; i < connectionNum; i ++)
    {
        int socketfd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (socketfd < 0)
        {
            printf("error when accept\n");
            return;
        }
        printf("new connection is accepted, socketfd is: %d\n", socketfd);
        fds[i] = socketfd;
    }

    int epollfd;
    struct epoll_event events[EPOLLEVENTS];
    int ret;
    epollfd = epoll_create(FDSIZE);
    for (int i = 0; i < connectionNum; i++)
    {
        add_event(epollfd, fds[i], EPOLLOUT);
    }
    
    while (bytes > 0)
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
            int fd = events[i].data.fd;
            if (log)
            {
                printf("fd: %d got notification\n", fd);
            }

            int nwrite = write(fd, buffer, buffernum);
            if (log)
            {
                printf("write %d \n", nwrite);
            }
            
            bytes -= nwrite;
        }
        
    }
    close(epollfd);
    sleep(2);
    for (int i = 0; i < connectionNum; i++)
    {
        close(fds[i]);
    }
    close(server_fd);
    free(fds);
    return 0;
}

static int socket_bind(const char* ip,int port)
{
    int  listenfd;
    struct sockaddr_in servaddr;
    listenfd = socket(AF_INET,SOCK_STREAM,0);
    if (listenfd == -1)
    {
        perror("socket error:");
        exit(1);
    }
    bzero(&servaddr,sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    inet_pton(AF_INET,ip,&servaddr.sin_addr);
    servaddr.sin_port = htons(port);
    if (bind(listenfd,(struct sockaddr*)&servaddr,sizeof(servaddr)) == -1)
    {
        perror("bind error: ");
        exit(1);
    }
    return listenfd;
}

static void do_epoll(int *fds, int connectNum, char* buf, int bufferSize)
{
    int epollfd;
    struct epoll_event events[EPOLLEVENTS];
    int ret;
    epollfd = epoll_create(FDSIZE);
    for (int i = 0; i < connectNum; i++)
    {
        add_event(epollfd, fds[i], EPOLLOUT);
    }
    
    for ( ; ; )
    {
        ret = epoll_wait(epollfd,events,EPOLLEVENTS,-1);
        handle_events(epollfd,events,ret, buf, bufferSize);
    }
    close(epollfd);
}

static void
handle_events(int epollfd,struct epoll_event *events,int num, char *buf, int bufSize)
{
    int i;
    int fd;

    for (i = 0;i < num;i++)
    {
        fd = events[i].data.fd;

        int nwrite = write(fd,buf, bufSize);
    }
}
static void handle_accpet(int epollfd,int listenfd)
{
    int clifd;
    struct sockaddr_in cliaddr;
    socklen_t  cliaddrlen;
    clifd = accept(listenfd,(struct sockaddr*)&cliaddr,&cliaddrlen);
    if (clifd == -1)
        perror("accpet error:");
    else
    {
        printf("accept a new client: %s:%d\n",inet_ntoa(cliaddr.sin_addr),cliaddr.sin_port);
        //添加一个客户描述符和事件
        add_event(epollfd,clifd,EPOLLIN);
    }
}

static void do_read(int epollfd,int fd,char *buf)
{
    int nread;
    nread = read(fd,buf,MAXSIZE);
    if (nread == -1)
    {
        perror("read error:");
        close(fd);
        delete_event(epollfd,fd,EPOLLIN);
    }
    else if (nread == 0)
    {
        fprintf(stderr,"client close.\n");
        close(fd);
        delete_event(epollfd,fd,EPOLLIN);
    }
    else
    {
        printf("read message is : %s",buf);
        //修改描述符对应的事件，由读改为写
        modify_event(epollfd,fd,EPOLLOUT);
    }
}

static void do_write(int epollfd,int fd,char *buf)
{
    int nwrite;
    nwrite = write(fd,buf,strlen(buf));
    if (nwrite == -1)
    {
        perror("write error:");
        close(fd);
        delete_event(epollfd,fd,EPOLLOUT);
    }
    else
        modify_event(epollfd,fd,EPOLLIN);
    memset(buf,0,MAXSIZE);
}

static void add_event(int epollfd,int fd,int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.fd = fd;
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