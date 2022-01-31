#include <stdio.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#include <time.h>
#define PORT 9987
#define BUFFERNUM 1024*32

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
    int readCount = 1000000;
    if (argc > 1)
    {
        readCount = atoi(argv[1]);
    }
    printf("readCount: %d\n", readCount);
    if (argc > 2)
    {
        addr = argv[2];
    }

    printf("add is %s\n", addr);

    int port = PORT;

    if (argc > 3)
    {
        port = atoi(argv[3]);
    }
    printf("port is %d\n", port);
    int buffernum = BUFFERNUM;
    if (argc > 4)
    {
        buffernum = atoi(argv[4]);
    }
    printf("buffer size is %d\n", buffernum);

    int log = 0;
    if (argc > 5)
    {
        log = strcmp(argv[5], "log") == 0 ? 1 : 0;
    }

    printf("log is %d\n", log);



    int sock = 0, valread;
    struct sockaddr_in serv_addr; 
    char *hello = "Hello from client"; 
    
    

    char* buffer = malloc(buffernum); 
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return -1; 
    } 

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
   
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) 
    { 
        printf("\nConnection Failed \n"); 
        return -1; 
    }

    printf("connected!\n");

    // printf("before read \n");
    // valread = read(sock , buffer, 1024*32);
    // printf("read %d\n",valread);
    // printf("after read \n");

    struct timespec tstart={0,0}, tend={0,0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    for (int i = 0; i < readCount; i++)
    {
        if (log) 
        {
            printf("before read %d batch\n", i+1);
        }

        valread = read(sock , buffer, buffernum);
        //printf("%s\n",buffer );
        if(log)
        {
            printf("after read %d batch, read: %d\n", i+1, valread);
        }
        
    }
    clock_gettime(CLOCK_MONOTONIC, &tend);
    double ns = (double)(tend.tv_sec - tstart.tv_sec) * 1.0e6 + (double)(tend.tv_nsec - tstart.tv_nsec)/1.0e3;
    printf("time used: %lf\n", ns);
    double speed = ((double)buffernum * (double)readCount) / (ns);
    printf("speed is %lf\n", speed);
    
    int ret = close(sock);
    printf("close return value is %d\n", ret);
    free(buffer);

    return 0;
} 
