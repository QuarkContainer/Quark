#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <errno.h>
#include <wait.h>
#include <signal.h>
#include <pthread.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>

#define PORT 9987
#define BUFFERNUM 133
#define CORENUM 10
#define EPOLLEVENTS 100
#define FDSIZE 1000

struct config_t
{
    long long buffer_size;
    long long bytestotalsent;
    long long bytestotalrecv;
    long long totalbytes;
    int log;
    int count;
    int connectionPerThread;
};

struct config_t config =
    {
        32768,
        0,
        0,
        32768 * 10000,
        0, // log
        0,
        0,
};

char *send_buffer;
char **recv_buffers;
int *clientsocks;
int *readNum;
int *epollfds;
int *readCountLeft;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void add_event(int epollfd,int fd,int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.fd = fd;
    epoll_ctl(epollfd,EPOLL_CTL_ADD,fd,&ev);
}

static void add_event2(int epollfd, int fd, int index, int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.u32 = index;
    epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &ev);
}

void *socketThread(void *arg)
{
    // int newSocket = *((int *)arg);
    // int newSocket = (int)arg;
    int index = (int)arg;
    if (config.log)
    {
        printf("epollfd is: %d\n", epollfds[index]);
    }

    struct epoll_event events[EPOLLEVENTS];

    int leftConnectionNum = config.connectionPerThread;

    while (leftConnectionNum > 0)
    {
        if (config.log)
        {
            printf("wait for epoll notificaiton\n");
        }
        int ret = epoll_wait(epollfds[index], events, EPOLLEVENTS, -1);
        if (config.log)
        {
            printf("get epoll notificaiton, ret: %d\n", ret);
            if (ret < 0 )
            {
                perror("fail to wait");
            }
        }
        for (int i = 0; i < ret; i++)
        {
            int idx = events[i].data.u32;
            int fd = clientsocks[idx];

            // printf(" i: %d, ret: %d\n", i, ret);
            if (config.log)
            {
                // printf("fd: %d got notification\n", fd);
            }

            int nread = read(fd, recv_buffers[index], readCountLeft[idx]);

            if (nread == 0)
            {
                leftConnectionNum -= 1;
                // close(fd);
                // printf("*****close fd: %d got notification, leftConnectionNum: %d, i: %d, ret: %d\n", fd, leftConnectionNum, i, ret);
                continue;
            }

            // printf("**** i: %d, ret: %d\n", i, ret);


            if (nread == -1)
            {
                printf("Error printed by perror, nread: %d\n", nread);
                return;
            }

            if (nread < readCountLeft[idx])
            {
                readCountLeft[idx] -= nread;
                continue;
            }

            readNum[idx] += 1;
            if (config.log)
            {
                printf("read %dth: %d \n", readNum[idx], config.buffer_size);
            }

            readCountLeft[idx] = config.buffer_size;

            int nwrite = write(fd, send_buffer, config.buffer_size);

            if (nwrite == -1 || nwrite != config.buffer_size)
            {
                printf("Error printed by perror, nread: %d\n", nwrite);
                return;
            }

            if (config.log)
            {
                printf("write %dth: %d \n", readNum[idx], nwrite);
            }
        }
    }

    printf("after loop");

    pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char *hello = "Hello from server";

    long long writeCount = 1000000;
    if (argc >= 2)
    {
        writeCount = atoll(argv[1]);
    }
    printf("writeCount: %d\n", writeCount);
    config.count = writeCount;
    // if (argc < 2)
    // {
    //     printf("must specify write count\n");
    //     return;
    // }
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // // Forcefully attaching socket to the port 8080
    // if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
    //                                               &opt, sizeof(opt)))
    // {
    //     perror("setsockopt");
    //     exit(EXIT_FAILURE);
    // }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;

    int buffernum = BUFFERNUM;
    if (argc >= 3)
    {
        buffernum = atoi(argv[2]);
    }
    printf("buffer len is %d\n", buffernum);
    config.buffer_size = buffernum;

    int threadnum = 1;
    if (argc > 3)
    {
        threadnum = atoi(argv[3]);
    }
    config.connectionPerThread = threadnum;
    printf("thread number is %d\n", threadnum);

    int nodelay = 0;
    if (argc > 4)
    {
        nodelay = atoi(argv[4]);
    }
    printf("nodelay is %d\n", nodelay);

    int port = PORT;
    if (argc > 5)
    {
        port = atoi(argv[5]);
    }
    printf("sin_port is %d\n", port);

    int coreNum = CORENUM;
    if (argc> 6) {
        coreNum = atoi(argv[6]);
    }

    printf("coreNum is %d\n", coreNum);

    int log = 0;
    if (argc > 7)
    {
        log = strcmp(argv[7], "log") == 0 ? 1 : 0;
    }

    printf("log is %d\n", log);
    config.log = log;

    address.sin_port = htons(port);

    send_buffer = malloc(buffernum);
    recv_buffers = malloc(coreNum * sizeof(char *));

    // memset(buffer, 'a', buffernum);
    for (int i = 0; i < buffernum; i++)
    {
        send_buffer[i] = 'A' + random() % 26;
    }

    for (int i = 0; i < threadnum; i++)
    {
        recv_buffers[i] = malloc(config.buffer_size);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int)) < 0) {
        perror("setsockopt failed"); 
        exit(EXIT_FAILURE); 
    }

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address,
             sizeof(address)) < 0)
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
    long long bytes = buffernum * writeCount;
    printf("bytes is %llu\n", bytes);
    config.totalbytes = bytes;

    int connectionNum = coreNum * threadnum;
    pthread_t *tid = malloc(sizeof(pthread_t) * coreNum);
    clientsocks = malloc(connectionNum * sizeof(int));
    readCountLeft = malloc(connectionNum * sizeof(int));
    readNum = malloc(connectionNum * sizeof(int));
    epollfds = malloc(coreNum * sizeof(int));
    for (int i = 0; i < connectionNum; i++)
    {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                                 (socklen_t *)&addrlen)) < 0)
        {
            printf("accept %d", errno);
        }
        printf("get connection, socket is: %d\n", new_socket);

        if (nodelay)
        {
            int one = 1;
            setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }

        clientsocks[i] = new_socket;
        readCountLeft[i] = config.buffer_size;
        readNum[i] = 0;
    }

    for (int i = 0; i < coreNum; i++)
    {
        epollfds[i] = epoll_create(FDSIZE);
        for (int j = 0; j < threadnum; j++)
        {
            int idx = i * threadnum + j;
            add_event2(epollfds[i], clientsocks[idx], idx, EPOLLIN|EPOLLET);
            // printf("epollfd: %d watches sock: %d\n", epollfds[i], clientsocks[idx]);
        }
    }

    for (int i = 0; i < coreNum; i++)
    {
        // if( pthread_create(&tid[i], NULL, socketThread, &new_socket) != 0 )
        if (pthread_create(&tid[i], NULL, socketThread, (void *)i) != 0)
        {
            printf("Failed to create thread\n");
        }
    }

    for (int i = 0; i < threadnum; i++)
    {
        pthread_join(tid[i], NULL);
    }

    // printf("before sleep\n");
    // sleep(2);
    printf("after data transfer 1\n");
    for(int i=0; i<connectionNum; i++)
    {
        close(clientsocks[i]);
    }
    free(clientsocks);
    close(server_fd);
    printf("after data transfer 2\n");
    free(tid);
    free(send_buffer);
    printf("after data transfer 3\n");
    for (int i = 0; i < coreNum; i++)
    {
        close(epollfds[i]);
        free(recv_buffers[i]);
    }
    free(epollfds);
    printf("after data transfer 4\n");

    free(recv_buffers);
}