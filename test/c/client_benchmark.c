#include <stdio.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <unistd.h> 
#include <string.h> 
#define PORT 9987

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
    printf("add is %d\n", port);

    int sock = 0, valread;
    struct sockaddr_in serv_addr; 
    char *hello = "Hello from client"; 
    char buffer[1024 * 32] = {0}; 
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

    // printf("before read \n");
    // valread = read(sock , buffer, 1024*32);
    // printf("read %d\n",valread);
    // printf("after read \n");

    for (int i = 0; i < readCount; i++)
    {
        printf("before read %d batch\n", i);
        valread = read(sock , buffer, 1024*32);
        //printf("%s\n",buffer );
        printf("after read %d batch, read: %d\n", i, valread);
    }
    
    int ret = close(sock);
    printf("close return value is %d\n", ret);

    return 0;
} 
