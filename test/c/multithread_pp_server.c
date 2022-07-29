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

#define PORT 9987
#define BUFFERNUM 133

struct config_t
{
    long long buffer_size;
    long long bytestotalsent;
    long long bytestotalrecv;
    long long totalbytes;
    int log;
    int count;
};

struct config_t config =
{
    32768,
    0,
    0,
    32768 * 10000,
    0, // log
    0,
};

char *send_buffer;
char **recv_buffers;
int *clientsocks;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void * socketThread(void *arg)
{
    //int newSocket = *((int *)arg);
    //int newSocket = (int)arg;
    int index = (int) arg;
    if (config.log)
    {
        printf("newSocket is: %d\n", clientsocks[index]);
    }

    // for(int i=0; i<config.count;i++)
    int i = 0;
    for (;;)
    {
        int readcount = 0;
        while (readcount < config.buffer_size) {
            int curreadcount = read(clientsocks[index], recv_buffers[index], config.buffer_size - readcount);
            if (config.log)
            {
                printf("sock: %d, cur read: %d\n", clientsocks[index], curreadcount);
            }
            if (curreadcount == 0) {
                close(clientsocks[index]);
                return;
            }
            readcount += curreadcount;
        }
        
        int writecount = 0;
        while (writecount < config.buffer_size)
        {
            int curwritecount = write(clientsocks[index], send_buffer, config.buffer_size - writecount);
            if (config.log)
            {
                printf("sock: %d, cur write %d\n", clientsocks[index], curwritecount);
            }
            
            writecount += curwritecount;
        }

        // if (readcount == -1)
        // {
        //     perror("read error");
        // }

        // if (writecount == -1)
        // {
        //     perror("write error");
        // }

        if(config.log)
        {
            printf("sock: %d, nth: %d, total write %d\n", clientsocks[index], ++i, writecount);
            printf("sock: %d, nth: %d, total read %d\n", clientsocks[index], i, readcount);
        }
    }

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

    int log = 0;
    if (argc > 6)
    {
        log = strcmp(argv[6], "log") == 0 ? 1 : 0;
    }
    printf("log is %d\n", log);
    config.log = log;

    address.sin_port = htons(port);

    send_buffer = malloc(buffernum);
    recv_buffers = malloc(threadnum * sizeof(char*));
    //memset(buffer, 'a', buffernum);
    for (int i = 0; i < buffernum; i ++)
    {
        send_buffer[i] = 'A' + random() % 26;
    }

    for(int i=0; i<threadnum; i++)
    {
        recv_buffers[i] = malloc(config.buffer_size);
    }
       
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
    long long bytes = buffernum * writeCount;
    printf("bytes is %llu\n", bytes);
    config.totalbytes = bytes;

    pthread_t *tid = malloc(sizeof(pthread_t) * threadnum);
    clientsocks = malloc(threadnum * sizeof(int));
    for(int i=0; i<threadnum; i++)
    {
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                        (socklen_t*)&addrlen))<0) {
            printf("accept %d", errno);
        }
        printf("get connection, socket is: %d\n", new_socket);

        if (nodelay)
        {
            int one = 1;
            setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }
        
        clientsocks[i] = new_socket;
    }

    for(int i=0; i<threadnum; i++)
    {
        //if( pthread_create(&tid[i], NULL, socketThread, &new_socket) != 0 )
        if( pthread_create(&tid[i], NULL, socketThread, (void*)i) != 0 )
        {
            printf("Failed to create thread\n");
        }
    }
    
    for (int i=0; i<threadnum; i++)
    {
        pthread_join(tid[i], NULL);
    }

    // printf("before sleep\n");
    // sleep(2);
    // printf("after sleep\n");
    // for(int i=0; i<threadnum; i++)
    // {
    //     close(clientsocks[i]);
    // }
    free(clientsocks);
    close(server_fd);
    free(tid);
    free(send_buffer);
    for(int i=0; i<threadnum; i++)
    {
        free(recv_buffers[i]);
    }

    free(recv_buffers);
} 