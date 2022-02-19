#include <unistd.h> 
#include <stdio.h> 
#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <string.h>
#include <errno.h>
#include <wait.h>
#include <signal.h>
#include <netinet/tcp.h>

#define PORT 9987
#define BUFFERNUM 133

void handler1(int sig)
{
    char *hello = "Hello from signal\n";
    //write(1, hello, strlen(hello));
    printf("sig..... = %d\n", sig);
    write(1, hello, strlen(hello));
        /* Flushes the printed string to stdout */
    fflush(stdout);
}

int main(int argc, char const *argv[]) 
{ 
    int server_fd, new_socket, valread; 
    struct sockaddr_in address; 
    int opt = 1; 
    int addrlen = sizeof(address); 
    char *hello = "Hello from server";

    long long writeCount = 1000000;
    if (argc >1)
    {
        writeCount = atoll(argv[1]);
    }
    printf("writeCount: %d\n", writeCount);
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
    if (argc >2)
    {
        buffernum = atoi(argv[2]);
    }
    printf("buffer len is %d\n", buffernum);

    int nodelay = 0;
    if (argc > 3)
    {
        nodelay = atoi(argv[3]);
    }
    printf("nodelay is %d\n", nodelay);

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
    // int i;
    // printf("n is: %d", 1024*32-1);
    // printf("before write %d\n", strlen(buffer));
    // printf("edge is %d\n", buffer[1024*32]);
    // for (i = 1024*32-1; i > 1024*32-10; i--)
    // {
    //     printf("i is %d\n", i);
    //     printf("buffer[%d]: %c\n", i, buffer[i]);
    // }
    // printf("before write %d\n", strlen(buffer));
    // return;

       
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
    while (1)
    {
        while ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                        (socklen_t*)&addrlen))<0) {
            printf("accept %d", errno);
        }
        printf("get connection\n");
        //sleep(1);

        if (nodelay)
        {
            int one = 1;
            setsockopt(new_socket, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        }


        // printf("before write hello\n");
        // int n = write(new_socket , hello, strlen(hello));
        // printf("after write hello\n");
        // printf("before write\n");
        // int n = write(new_socket , buffer , 1024*32);
        // printf("after write %d\n", n);

        bytes = buffernum * writeCount;
        printf("bytes is %llu\n", bytes);
        long long i = 0;
        //for(int i = 0; i < writeCount; i++)
        while (bytes > 0)
        {
            if (log)
            {
                printf("before write %llu batch\n", i+1);
            }
            
            int n = write(new_socket , buffer , buffernum);
            bytes -= n;
            // if (n < buffernum)
            // {
            //     printf("write %d th, %d was writtern which is less than %d\n", i, n, buffernum);
            // }
            if (log)
            {
                printf("after write %llu batch, write: %d\n", i+1, n);
            }

            i += 1;
            // printf("before write %d hello\n", i);
            // int n = write(new_socket , hello, strlen(hello));
            // printf("after write %d hello\n", i);
        }

        printf("iteration is %d\n", i);

    }
    

    close(new_socket);
    close(server_fd);
    free(buffer);

    return 222;
} 