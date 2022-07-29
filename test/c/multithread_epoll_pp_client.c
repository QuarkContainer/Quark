#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>

#define PORT 9987
#define BUFFERNUM 1024 * 32
#define CORENUM 10
#define EPOLLEVENTS 100
#define FDSIZE 1000

struct config_t
{
    long long buffer_size;
    long long bytestotalsent;
    long long bytestotalrecv;
    long long totalbytes;
    int port;
    char *addr;
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
        25026,       // port
        "127.0.0.1", // addr
        0,           // log
        0,
        0,
};

char **recv_buffers;
char *send_buffer;
int *clientsocks;
int *epollfds;
int *readNum;
int *readCountLeft;
int *writeCountLeft;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void add_event2(int epollfd, int fd, int index, int state)
{
    struct epoll_event ev;
    ev.events = state;
    ev.data.u32 = index;
    epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &ev);
}

void *clientThread(void *arg)
{
    // int index = *((int *)arg);
    int index = (int)arg;
    struct epoll_event events[EPOLLEVENTS];
    if (config.log)
    {
        printf("index is: %d, efd: %d\n", index, epollfds[index]);
    }

    for (int i = 0; i < config.connectionPerThread; i++)
    {
        int sockIdx = index * config.connectionPerThread + i;
        int nwrite = write(clientsocks[sockIdx], send_buffer, config.buffer_size);
        if (nwrite != config.buffer_size)
        {
            printf("fd: %d, nwrite: %d, buffernum: %d\n", clientsocks[sockIdx], nwrite, config.buffer_size);
            return;
        }
    }

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
        }

        for (int i = 0; i < ret; i++)
        {
            int idx = events[i].data.u32;
            int fd = clientsocks[idx];
            if (config.log)
            {
                printf("fd: %d got notification\n", fd);
            }

            int nread = read(fd, recv_buffers[index], readCountLeft[idx]);

            if (nread == -1)
            {
                perror("Error printed by perror");
                return;
            }

            if (nread < readCountLeft[idx])
            {
                readCountLeft[idx] -= nread;
                continue;
            }

            readCountLeft[idx] = config.buffer_size;

            readNum[idx] += 1;
            if (config.log)
            {
                printf("read %dth: %d \n", readNum[idx], config.buffer_size);
            }

            if (readNum[idx] == config.count)
            {
                // close(fd);
                leftConnectionNum -= 1;
                continue;
            }

            int nwrite = write(fd, send_buffer, config.buffer_size);

            if (nwrite == -1 || nwrite != config.buffer_size)
            {
                printf("Error printed by perror, nread: %d\n", nwrite);
                return;
            }

            if (config.log)
            {
                printf("write %dth, %d \n", readNum[idx], nwrite);
            }
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char const *argv[])
{
    // printf("xxxxxx\n");
    // return Send();

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
    config.addr = addr;

    long long readCount = 1000000;
    if (argc > 2)
    {
        readCount = atoll(argv[2]);
    }
    printf("readCount: %lld\n", readCount);
    config.count = readCount;

    int buffernum = BUFFERNUM;
    if (argc > 3)
    {
        buffernum = atoi(argv[3]);
    }
    printf("buffer size is %d\n", buffernum);
    config.buffer_size = buffernum;

    int threadnum = 1;
    if (argc > 4)
    {
        threadnum = atoi(argv[4]);
    }
    config.connectionPerThread = threadnum;
    printf("thread num is %d\n", threadnum);

    int nodelay = 0;
    if (argc > 5)
    {
        nodelay = atoi(argv[5]);
    }
    printf("nodelay is %d\n", nodelay);

    int port = PORT;

    if (argc > 6)
    {
        port = atoi(argv[6]);
    }
    printf("port is %d\n", port);

    int coreNum = CORENUM;
    if (argc > 7)
    {
        coreNum = atoi(argv[7]);
    }

    printf("coreNum is %d\n", coreNum);

    int log = 0;
    if (argc > 8)
    {
        log = strcmp(argv[8], "log") == 0 ? 1 : 0;
    }

    printf("log is %d\n", log);
    config.log = log;

    send_buffer = malloc(buffernum);
    recv_buffers = malloc(threadnum * sizeof(char *));
    for (int i = 0; i < buffernum; i++)
    {
        send_buffer[i] = 'A' + random() % 26;
    }
    for (int i = 0; i < threadnum; i++)
    {
        recv_buffers[i] = malloc(config.buffer_size);
    }

    int connectionNum = coreNum * threadnum;
    clientsocks = malloc(connectionNum * sizeof(int));
    readNum = malloc(connectionNum * sizeof(int));
    readCountLeft = malloc(connectionNum * sizeof(int));
    writeCountLeft = malloc(connectionNum * sizeof(int));
    epollfds = malloc(coreNum * sizeof(int));

    config.totalbytes = config.buffer_size * readCount;
    for (int i = 0; i < connectionNum; i++)
    {
        int sock = 0, valread;
        struct sockaddr_in serv_addr;
        char *hello = "Hello from client";
        char *buffer = malloc(buffernum);
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        {
            printf("\n Socket creation error \n");
            return -1;
        }

        printf("start to connect with sock: %d\n", sock);

        serv_addr.sin_family = AF_INET;

        serv_addr.sin_port = htons(port);

        // Convert IPv4 and IPv6 addresses from text to binary form
        if (inet_pton(AF_INET, addr, &serv_addr.sin_addr) <= 0)
        {
            printf("\nInvalid address/ Address not supported \n");
            return -1;
        }

        if (nodelay)
        {
            int one = 1;
            setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }

        struct timespec tstart1 = {0, 0}, tend1 = {0, 0};
        clock_gettime(CLOCK_MONOTONIC, &tstart1);

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        {
            printf("\nConnection Failed \n");
            return -1;
        }

        clock_gettime(CLOCK_MONOTONIC, &tend1);

        double cws = (double)(tend1.tv_sec - tstart1.tv_sec) * 1.0e6 + (double)(tend1.tv_nsec - tstart1.tv_nsec) / 1.0e3;

        printf("connected! used time: %f\n", cws);
        clientsocks[i] = sock;
        readNum[i] = 0;
        readCountLeft[i] = config.buffer_size;
        writeCountLeft[i] = config.buffer_size;
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

    pthread_t *tid = malloc(sizeof(pthread_t) * coreNum);

    struct timespec tstart = {0, 0}, tend = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    int index = 0;
    for (int i = 0; i < coreNum; i++)
    {
        index = i;
        if (config.log)
        {
            printf("create thread with index: %d\n", index);
        }
        // if( pthread_create(&tid[i], NULL, clientThread, &index) != 0 )
        if (pthread_create(&tid[i], NULL, clientThread, (void *)i) != 0)
        {
            printf("Failed to create thread\n");
        }
    }

    for (int i = 0; i < coreNum; i++)
    {
        pthread_join(tid[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &tend);
    double ns = (double)(tend.tv_sec - tstart.tv_sec) * 1.0e6 + (double)(tend.tv_nsec - tstart.tv_nsec) / 1.0e3;
    long long totalbytes = config.buffer_size * readCount * connectionNum;
    printf("bytes received: %lld, bytes sent: %lld\n", totalbytes, totalbytes);
    printf("time used: %lf\n", ns);
    double speed = (2 * totalbytes) / (ns);
    printf("throughput is %lf\n", speed);
    double pps = (2 * totalbytes * 1024 * 1024) / (ns) / config.buffer_size;
    printf("pps is %lf\n", pps);
    double latency = ns / (readCount * 2 * threadnum);
    printf("latency is %lf\n", latency);

    // printf("before sleep\n");
    // sleep(2);
    // printf("after sleep\n");

    free(tid);
    for (int i = 0; i < connectionNum; i++)
    {
        // shutdown(clientsocks[i], 1);
        // if (read(clientsocks[i], recv_buffers[i], config.buffer_size) == 0)
        // {
            // printf("read == 0\n");
            close(clientsocks[i]);
        // }
    }
    free(clientsocks);
    free(send_buffer);
    for (int i = 0; i < coreNum; i++)
    {
        close(epollfds[i]);
        free(recv_buffers[i]);
    }
    free(epollfds);

    free(recv_buffers);

    return 0;
}