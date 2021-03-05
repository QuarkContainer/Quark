#include <unistd.h> 
#include <stdio.h> 
#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <string.h>
#include <errno.h>
#include <wait.h>
#include <signal.h>

#define PORT 8080

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
    char buffer[1024] = {0}; 
    char *hello = "Hello from server";

    //write(1, hello, strlen(hello));
    //printf("sig..... = %d\n", 123);

    signal(SIGUSR1, handler1);
    kill(getpid(), SIGUSR1);

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, 
                                                  &opt, sizeof(opt))) 
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons( PORT ); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address,  
                                 sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (listen(server_fd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 
    while ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                       (socklen_t*)&addrlen))<0) {
        printf("accept %d", errno);
    }
    printf("get connection\n");
    sleep(1);

    int n = write(new_socket , hello , strlen(hello));
    printf("Server::write Hello message sent %d\n", n);
    valread = read( new_socket , buffer, 1024);
    printf("Server::read get message  %d: %s\n", valread, buffer);

    valread = recv( new_socket , buffer, 1024, 0);
    printf("Server::recv %d get message recv: %s\n", valread, buffer);
    n = send(new_socket, hello , strlen(hello) , 0 );
    printf("Server::send %d Server: Hello message sent\n", n);

    struct iovec iov[1];
    iov[0].iov_base = buffer;
    iov[0].iov_len = sizeof(buffer);

    struct msghdr mh;
    mh.msg_name = 0;
    mh.msg_namelen = 0;
    mh.msg_iov = iov;
    mh.msg_iovlen = 1;
    mh.msg_control = NULL;
    mh.msg_controllen = 0;
    mh.msg_flags = 0;


    while (valread != 0) {
        valread = recvmsg(new_socket, &mh, 0);
        printf("Server::recmsg %d %s\n", valread, buffer);
    }

    close(new_socket);

    return 222;
} 
