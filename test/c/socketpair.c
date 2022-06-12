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

//#include "stderr.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

static
void Sendfd(int socket, int fd)  // send fd by socket
{
    struct msghdr msg = { 0 };
    char buf[CMSG_SPACE(sizeof(fd))];
    memset(buf, '\0', sizeof(buf));
    struct iovec io = { .iov_base = "ABC", .iov_len = 4 };

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(fd));

    *((int *) CMSG_DATA(cmsg)) = fd;

    msg.msg_controllen = CMSG_SPACE(sizeof(fd));

    if (sendmsg(socket, &msg, 0) < 0)
        printf("Failed to send message\n");
}

static
int Recvfd(int socket)  // receive fd from socket
{
    struct msghdr msg = {0};

    char m_buffer[256];
    struct iovec io = { .iov_base = m_buffer, .iov_len = sizeof(m_buffer) };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;

    char c_buffer[256];
    msg.msg_control = c_buffer;
    msg.msg_controllen = sizeof(c_buffer);

    int n = 0;
    if ((n=recvmsg(socket, &msg, 0)) < 0)
        printf("Failed to receive message\n");

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);

    unsigned char * data = CMSG_DATA(cmsg);

    printf("About to extract fd\n");
    int fd = *((int*) data);
    printf("Extracted fd %d\n", fd);

    printf("get data %s, len is %d\n", m_buffer, n);
    return fd;
}

struct ucred {
    pid_t pid;    /* Process ID of the sending process */
    uid_t uid;    /* User ID of the sending process */
    gid_t gid;    /* Group ID of the sending process */
};

#define SCM_CREDENTIALS 2

static
void SendCred(int socket)  // send fd by socket
{
    union {
        char   buf[CMSG_SPACE(sizeof(struct ucred))];
        struct cmsghdr align;
    } controlMsg;

    struct msghdr msg = { 0 };
    struct iovec io = { .iov_base = "xyz", .iov_len = 4 };

    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = controlMsg.buf;
    msg.msg_controllen = sizeof(controlMsg.buf);
    memset(controlMsg.buf, 0, sizeof(controlMsg.buf));

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    printf("SendCred cmsg is %lx, buf is %lx, data is %lx\n", (long)cmsg, (long)controlMsg.buf, (long)CMSG_DATA(cmsg));
    cmsg->cmsg_len = CMSG_LEN(sizeof(struct ucred));
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_CREDENTIALS;

    struct ucred creds;
    creds.pid = getpid();
    creds.uid = getuid();
    creds.gid = 2; //getgid();

    printf("Send credentials pid=%ld, uid=%ld, gid=%ld\n",
                    (long) creds.pid, (long) creds.uid, (long) creds.gid);

    memcpy(CMSG_DATA(cmsg), &creds, sizeof(struct ucred));

    printf("before send message \n");
    if (sendmsg(socket, &msg, 0) < 0)
        printf("SendCred Failed to send message\n");
    printf("after send message \n");
}

static
void RecvCred(int socket)  // receive fd from socket
{
    union {
        //char   buf[CMSG_SPACE(sizeof(struct ucred))];
        char buf[1024];
                        /* Space large enough to hold a 'ucred' structure */
        struct cmsghdr align;
    } controlMsg;
    struct msghdr msg = {0};
    struct ucred rcred;

    char m_buffer[256];
    struct iovec io = { .iov_base = m_buffer, .iov_len = sizeof(m_buffer) };
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;

    msg.msg_control = controlMsg.buf;
    msg.msg_controllen = sizeof(controlMsg.buf);

    int n = 0;
    printf("before recvmsg message \n");
    if ((n=recvmsg(socket, &msg, 0)) < 0)
        printf("Failed to receive message\n");

    struct cmsghdr * cmsg = CMSG_FIRSTHDR(&msg);
    printf("cmsg len is %lx\n", (long)cmsg);
    //printf("cmsg len is %ld\n", cmsg->cmsg_len);

    memcpy(&rcred, CMSG_DATA(cmsg), sizeof(struct ucred));

    printf("Received credentials pid=%ld, uid=%ld, gid=%ld\n",
                    (long) rcred.pid, (long) rcred.uid, (long) rcred.gid);

    printf("get data %s, len is %d\n", m_buffer, n);
}

int main(int argc, char **argv)
{
    const char *filename = "./server.c";

    //err_setarg0(argv[0]);
    //err_setlogopts(ERR_PID);
    if (argc > 1)
        filename = argv[1];
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) != 0)
        printf("Failed to create Unix-domain socket pair\n");

    int pid = fork();
    if (pid > 0)  // in parent
    {
        printf("Parent at work\n");
        close(sv[1]);
        int sock = sv[0];

        int fd = open(filename, O_RDONLY);
        if (fd < 0)
            printf("Failed to open file %s for reading\n", filename);

        SendCred(sock);
        Sendfd(sock, fd);

        close(fd);
        nanosleep(&(struct timespec){ .tv_sec = 1, .tv_nsec = 500000000}, 0);
        printf("Parent exits\n");
    }
    else  // in child
    {
        printf("Child at play\n");
        close(sv[0]);
        int sock = sv[1];

        nanosleep(&(struct timespec){ .tv_sec = 0, .tv_nsec = 500000000}, 0);

        RecvCred(sock);
        int fd = Recvfd(sock);
        printf("Read %d!\n", fd);
        char buffer[256];
        ssize_t nbytes;
        while ((nbytes = read(fd, buffer, sizeof(buffer))) > 0)
            write(1, buffer, nbytes);
        printf("Done!\n");
        close(fd);
    }
    return 0;
}