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
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <time.h>
#include <pthread.h>
#include <stdlib.h> 
#define PORT 9987
#define BUFFERNUM 1024*32

struct config_t
{
    long long buffer_size;
    long long bytestotalsent;
    long long totalbytes;
    int port;
    char *addr;
    int log;
};

struct config_t config =
{
    32768,
    0,
    32768 * 10000,
    25026, // port
    "127.0.0.1", //addr
    0, // log
};

char **buffers;
int *clientsocks;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void * clientThread(void *arg)
{
    //int index = *((int *)arg);
    int index = (int) arg;
    if (config.log)
    {
        printf("index is: %d\n", index);
        printf("client sock is: %d\n", clientsocks[index]);
    }
    while (1)
    {
        if (config.bytestotalsent >= config.totalbytes)
        {
            break;
        }

        int n = read(clientsocks[index], buffers[index], config.buffer_size);
        if (config.log)
        {
            printf("sock: %d, read: %d\n", clientsocks[index], n);
        }
        pthread_mutex_lock(&lock);
        config.bytestotalsent += n;
        pthread_mutex_unlock(&lock);
    }
    pthread_exit(NULL);
}

int main(int argc, char const *argv[]) 
{
    //printf("xxxxxx\n");
    //return Send();

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
    printf("thread num is %d\n", threadnum);

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
    config.log = log;

    buffers = malloc(threadnum * sizeof(char*));
    for(int i=0; i<threadnum; i++)
    {
        buffers[i] = malloc(config.buffer_size);
    }

    config.totalbytes = config.buffer_size * readCount;

    clientsocks = malloc(threadnum * sizeof(int));
    for(int i=0; i<threadnum; i++)
    {
        int sock = 0, valread;
        struct sockaddr_in serv_addr; 
        char *hello = "Hello from client"; 
        
        

        char* buffer = malloc(buffernum); 
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
        { 
            printf("\n Socket creation error \n"); 
            return -1; 
        } 

        printf("start to connect with sock: %d\n", sock);

        serv_addr.sin_family = AF_INET;
        
        serv_addr.sin_port = htons(port); 

        
        // Convert IPv4 and IPv6 addresses from text to binary form 
        if(inet_pton(AF_INET, addr, &serv_addr.sin_addr)<=0)  
        { 
            printf("\nInvalid address/ Address not supported \n"); 
            return -1; 
        } 

        struct timespec tstart1={0,0}, tend1={0,0};
        clock_gettime(CLOCK_MONOTONIC, &tstart1);
    
        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) 
        { 
            printf("\nConnection Failed \n"); 
            return -1; 
        }

        clock_gettime(CLOCK_MONOTONIC, &tend1);

        double cws = (double)(tend1.tv_sec - tstart1.tv_sec) * 1.0e6 + (double)(tend1.tv_nsec - tstart1.tv_nsec)/1.0e3;
        
        printf("connected! used time: %f\n", cws);
        clientsocks[i] = sock;

    }

    pthread_t *tid = malloc(sizeof(pthread_t) * threadnum);

    struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    int index = 0;
    for(int i=0; i<threadnum; i++)
    {
        index = i;
        if(config.log)
        {
            printf("create thread with index: %d\n", index);
        }
        //if( pthread_create(&tid[i], NULL, clientThread, &index) != 0 )
        if( pthread_create(&tid[i], NULL, clientThread, (void*)i) != 0 )
        {
            printf("Failed to create thread\n");
        }
    }

    for (int i=0; i<threadnum; i++)
    {
        pthread_join(tid[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &tend);
    double ns = (double)(tend.tv_sec - tstart.tv_sec) * 1.0e6 + (double)(tend.tv_nsec - tstart.tv_nsec)/1.0e3;
    printf("bytes received: %lld\n", config.bytestotalsent);
    printf("time used: %lf\n", ns);
    double speed = config.bytestotalsent / (ns);
    printf("speed is %lf\n", speed);

    free(tid);
    for(int i=0; i<threadnum; i++)
    {
        close(clientsocks[i]);
    }
    free(clientsocks);
    for(int i=0; i<threadnum; i++)
    {
        free(buffers[i]);
    }

    free(buffers);

    return 0;
}