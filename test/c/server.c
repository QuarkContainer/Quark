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
//#include <netinet/in.h> 
#include <arpa/inet.h>
#include <string.h>
#include <errno.h>
#include <wait.h>
#include <signal.h>

#define PORT 5678

void handler1(int sig)
{
    char *hello = "Hello from signal\n";
    //write(1, hello, strlen(hello));
    printf("sig..... = %d\n", sig);
    write(1, hello, strlen(hello));
        /* Flushes the printed string to stdout */
    fflush(stdout);
}

void test()
{
    printf("test begin\n");
    int sock = 0, valread;
    struct sockaddr_in serv_addr; 
    char *hello = "Hello from client"; 
    char buffer[1024] = {0}; 
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    { 
        printf("\n Socket creation error \n"); 
        return -1; 
    }

    struct sockaddr_in sa;
    int sa_len;
    sa_len = sizeof(sa);
    printf("sa_len: %d\n", sa_len);
    printf("sock is %d\n", sock);
    if (getsockname(sock, &sa, &sa_len) == -1) {
          perror("getsockname() failed");
          return -1;
    }
    printf("Local IP address is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port is: %d\n", (int) ntohs(sa.sin_port));
    printf("test end\n");
}

int main(int argc, char const *argv[]) 
{ 
    //test();
    int port = PORT;
    if (argc > 1)
    {
        port = atoi(argv[1]);
    }
    int server_fd=0, new_socket, valread; 
    struct sockaddr_in address; 
    int opt = 1; 
    int addrlen = sizeof(address); 
    char buffer[1024] = {0}; 
    char *hello = "Hello from server";

    write(1, hello, strlen(hello));
    printf("sig..... = %d\n", 123);

    signal(SIGUSR1, handler1);
    kill(getpid(), SIGUSR1);

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 

    struct sockaddr_in sa;
    int sa_len;
    sa_len = sizeof(sa);
    memset (&sa, 0, sa_len);
    printf("sa_len: %d\n", sa_len);
    printf("sock is %d\n", server_fd);
    printf("after calling socket()******************************\n");
    if (getsockname(server_fd, &sa, &sa_len) == -1) {
          perror("getsockname() failed");
          return -1;
    }
    printf("Local IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port  for server_fd is: %d\n", (int) ntohs(sa.sin_port));

    if (getpeername(server_fd, &sa, &sa_len) == -1) {
        printf("errorno: %d\n", errno);
            //   perror("getsockname() failed");
            //   return -1;
    }
    else {
        printf("Remote IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for server_fd is: %d\n", (int) ntohs(sa.sin_port));
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
    address.sin_port = htons( port ); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address,  
                                 sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    }
    printf("after calling bind()******************************\n"); 
    if (getsockname(server_fd, &sa, &sa_len) == -1) {
          perror("getsockname() failed");
          return -1;
    }
    printf("Local IP address is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port is: %d\n", (int) ntohs(sa.sin_port));
    if (getpeername(server_fd, &sa, &sa_len) == -1) {
        printf("getpeername for server_fd, errorno: %d\n", errno);
            //   perror("getsockname() failed");
            //   return -1;
    }
    else {
        printf("Remote IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for server_fd is: %d\n", (int) ntohs(sa.sin_port));
    }
    if (listen(server_fd, 3) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 

    printf("after calling listen()******************************\n");
    if (getsockname(server_fd, &sa, &sa_len) == -1) {
          perror("getsockname() failed");
          return -1;
    }
    printf("Local IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port for server_fd is: %d\n", (int) ntohs(sa.sin_port));

    if (getpeername(server_fd, &sa, &sa_len) == -1) {
        printf("getpeername for server_fd, errorno: %d\n", errno);
            //   perror("getsockname() failed");
            //   return -1;
    }
    else {
        printf("Remote IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for server_fd is: %d\n", (int) ntohs(sa.sin_port));
    }

    // if (getpeername(server_fd, &sa, &sa_len) == -1) {
    //           perror("getsockname() failed");
    //           return -1;
    // }
    // printf("Remote IP address is: %s\n", inet_ntoa(sa.sin_addr));
    // printf("Remote port is: %d\n", (int) ntohs(sa.sin_port));
    while ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                       (socklen_t*)&addrlen))<0) {
        printf("accept %d", errno);
    }
    printf("get connection\n");
    sleep(1);
    printf("after calling accept()******************************\n");
    if (getsockname(server_fd, &sa, &sa_len) == -1) {
          perror("getsockname() failed");
          return -1;
    }
    
    printf("Local IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port for server_fd is: %d\n", (int) ntohs(sa.sin_port));

    if (getpeername(server_fd, &sa, &sa_len) == -1) {
        printf("getpeername for server_fd, errorno: %d\n", errno);
            //   perror("getsockname() failed");
            //   return -1;
    }
    else {
        printf("Remote IP address for server_fd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for server_fd is: %d\n", (int) ntohs(sa.sin_port));
    }
    if (getsockname(new_socket, &sa, &sa_len) == -1) {
          perror("getsockname() failed");
          return -1;
    }
    printf("Local IP address is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port is: %d\n", (int) ntohs(sa.sin_port));

    if (getpeername(new_socket, &sa, &sa_len) == -1) {
              perror("getsockname() failed");
              return -1;
    }

    printf("Remote IP address is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Remote port is: %d\n", (int) ntohs(sa.sin_port));

    int n = write(new_socket , hello , strlen(hello));
    printf("1 Server::write Hello message sent %d\n", n);
    valread = read( new_socket , buffer, 1024);
    printf("2 Server::read get message  %d: %s\n", valread, buffer);
    printf("qq******* 1\n");
    // n = send(new_socket, hello , strlen(hello) , 0 );
    n = write(new_socket , hello , strlen(hello));
    printf("3 Server::send %d Server: Hello message sent\n", n);
    valread = recv( new_socket , buffer, 1024, 0);
    printf("4 Server::recv %d get message recv: %s\n", valread, buffer);
    n = send(new_socket, hello , strlen(hello) , 0 );
    printf("5 Server::send %d Server: Hello message sent\n", n);

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
