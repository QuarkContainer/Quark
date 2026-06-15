// Copyright (c) 2026 Quark Container Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static void fail(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(void)
{
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        fail("socket(listener)");
    }

    int one = 1;
    if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
        fail("setsockopt(SO_REUSEADDR)");
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (bind(listener, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fail("bind(listener)");
    }
    if (listen(listener, 1) < 0) {
        fail("listen(listener)");
    }

    socklen_t addr_len = sizeof(addr);
    if (getsockname(listener, (struct sockaddr *)&addr, &addr_len) < 0) {
        fail("getsockname(listener)");
    }

    pid_t child = fork();
    if (child < 0) {
        fail("fork");
    }
    if (child == 0) {
        int accepted = accept(listener, NULL, NULL);
        if (accepted < 0) {
            fail("accept");
        }
        char byte;
        (void)read(accepted, &byte, sizeof(byte));
        close(accepted);
        close(listener);
        return 0;
    }

    int client = socket(AF_INET, SOCK_STREAM, 0);
    if (client < 0) {
        fail("socket(client)");
    }
    int flags = fcntl(client, F_GETFL, 0);
    if (flags < 0 || fcntl(client, F_SETFL, flags | O_NONBLOCK) < 0) {
        fail("fcntl(O_NONBLOCK)");
    }

    int rc = connect(client, (struct sockaddr *)&addr, sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) {
        fail("connect(client)");
    }

    fd_set readfds;
    fd_set writefds;
    fd_set exceptfds;
    FD_ZERO(&readfds);
    FD_ZERO(&writefds);
    FD_ZERO(&exceptfds);
    FD_SET(client, &readfds);
    FD_SET(client, &writefds);
    FD_SET(client, &exceptfds);

    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    rc = select(client + 1, &readfds, &writefds, &exceptfds, &timeout);
    if (rc < 0) {
        fail("select");
    }
    if (rc == 0) {
        fprintf(stderr, "select timed out\n");
        return 1;
    }

    int err = 0;
    socklen_t err_len = sizeof(err);
    if (getsockopt(client, SOL_SOCKET, SO_ERROR, &err, &err_len) < 0) {
        fail("getsockopt(SO_ERROR)");
    }

    printf("select rc=%d read=%d write=%d except=%d so_error=%d\n",
           rc,
           FD_ISSET(client, &readfds) != 0,
           FD_ISSET(client, &writefds) != 0,
           FD_ISSET(client, &exceptfds) != 0,
           err);

    if (err != 0) {
        fprintf(stderr, "SO_ERROR=%d (%s)\n", err, strerror(err));
        return 1;
    }
    if (!FD_ISSET(client, &writefds)) {
        fprintf(stderr, "client socket was not in writefds\n");
        return 1;
    }
    if (FD_ISSET(client, &exceptfds)) {
        fprintf(stderr, "client socket was incorrectly left in exceptfds\n");
        return 1;
    }

    (void)write(client, "x", 1);
    close(client);
    close(listener);

    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        fail("waitpid");
    }
    return status == 0 ? 0 : 1;
}
