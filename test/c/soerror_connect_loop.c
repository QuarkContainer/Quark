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

static int check_refused_connect(void)
{
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        fail("socket(refused listener)");
    }

    int one = 1;
    if (setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
        fail("setsockopt(refused SO_REUSEADDR)");
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (bind(listener, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        fail("bind(refused listener)");
    }

    socklen_t addr_len = sizeof(addr);
    if (getsockname(listener, (struct sockaddr *)&addr, &addr_len) < 0) {
        fail("getsockname(refused listener)");
    }
    close(listener);

    int client = socket(AF_INET, SOCK_STREAM, 0);
    if (client < 0) {
        fail("socket(refused client)");
    }

    int flags = fcntl(client, F_GETFL, 0);
    if (flags < 0 || fcntl(client, F_SETFL, flags | O_NONBLOCK) < 0) {
        fail("fcntl(refused O_NONBLOCK)");
    }

    int connect_rc = connect(client, (struct sockaddr *)&addr, sizeof(addr));
    if (connect_rc == 0) {
        fprintf(stderr, "refused connect unexpectedly succeeded\n");
        close(client);
        return 1;
    }
    if (errno != EINPROGRESS && errno != ECONNREFUSED) {
        fail("connect(refused client)");
    }

    fd_set writefds;
    FD_ZERO(&writefds);
    FD_SET(client, &writefds);

    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    int select_rc = select(client + 1, NULL, &writefds, NULL, &timeout);
    if (select_rc < 0) {
        fail("select(refused)");
    }
    if (select_rc == 0) {
        fprintf(stderr, "refused connect select timed out\n");
        close(client);
        return 1;
    }

    int err = 0;
    socklen_t err_len = sizeof(err);
    if (getsockopt(client, SOL_SOCKET, SO_ERROR, &err, &err_len) < 0) {
        fail("getsockopt(refused SO_ERROR)");
    }
    if (err != ECONNREFUSED) {
        fprintf(stderr, "expected ECONNREFUSED, got %d (%s)\n", err, strerror(err));
        close(client);
        return 1;
    }

    int cleared = -1;
    err_len = sizeof(cleared);
    if (getsockopt(client, SOL_SOCKET, SO_ERROR, &cleared, &err_len) < 0) {
        fail("getsockopt(refused SO_ERROR clear)");
    }
    if (cleared != 0) {
        fprintf(stderr, "SO_ERROR did not clear after read: %d\n", cleared);
        close(client);
        return 1;
    }

    close(client);
    return 0;
}

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 100;
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
    if (listen(listener, 128) < 0) {
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
        for (int i = 0; i < iters; i++) {
            int accepted = accept(listener, NULL, NULL);
            if (accepted >= 0) {
                close(accepted);
            }
        }
        close(listener);
        return 0;
    }

    int bad_immediate = 0;
    int bad_except = 0;
    int bad_write = 0;
    int bad_after = 0;
    int bad_peer = 0;

    for (int i = 1; i <= iters; i++) {
        int client = socket(AF_INET, SOCK_STREAM, 0);
        if (client < 0) {
            fail("socket(client)");
        }

        int flags = fcntl(client, F_GETFL, 0);
        if (flags < 0 || fcntl(client, F_SETFL, flags | O_NONBLOCK) < 0) {
            fail("fcntl(O_NONBLOCK)");
        }

        int connect_rc = connect(client, (struct sockaddr *)&addr, sizeof(addr));
        if (connect_rc < 0 && errno != EINPROGRESS) {
            fail("connect(client)");
        }

        int immediate = 0;
        socklen_t err_len = sizeof(immediate);
        if (getsockopt(client, SOL_SOCKET, SO_ERROR, &immediate, &err_len) < 0) {
            fail("getsockopt(SO_ERROR immediate)");
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
        int select_rc = select(client + 1, &readfds, &writefds, &exceptfds, &timeout);
        if (select_rc < 0) {
            fail("select");
        }

        int after = 0;
        err_len = sizeof(after);
        if (getsockopt(client, SOL_SOCKET, SO_ERROR, &after, &err_len) < 0) {
            fail("getsockopt(SO_ERROR after)");
        }

        struct sockaddr_in peer;
        memset(&peer, 0, sizeof(peer));
        socklen_t peer_len = sizeof(peer);
        int peer_ok = getpeername(client, (struct sockaddr *)&peer, &peer_len) == 0 &&
                      peer.sin_family == AF_INET &&
                      peer.sin_addr.s_addr == addr.sin_addr.s_addr &&
                      peer.sin_port == addr.sin_port;

        int read_ready = FD_ISSET(client, &readfds) != 0;
        int write_ready = FD_ISSET(client, &writefds) != 0;
        int except_ready = FD_ISSET(client, &exceptfds) != 0;
        int bad = immediate != 0 || except_ready || !write_ready || after != 0 || !peer_ok;
        if (bad && bad_immediate + bad_except + bad_write + bad_after + bad_peer < 10) {
            printf(
                "iter=%d connect_rc=%d immediate=%d select_rc=%d read=%d write=%d except=%d after=%d peer_ok=%d\n",
                i,
                connect_rc,
                immediate,
                select_rc,
                read_ready,
                write_ready,
                except_ready,
                after,
                peer_ok);
        }

        bad_immediate += immediate != 0;
        bad_except += except_ready;
        bad_write += !write_ready;
        bad_after += after != 0;
        bad_peer += !peer_ok;
        close(client);
    }

    close(listener);
    int status = 0;
    waitpid(child, &status, 0);
    int refused_status = check_refused_connect();
    printf(
        "summary bad_immediate=%d/%d bad_except=%d/%d bad_missing_write=%d/%d bad_after=%d/%d bad_peer=%d/%d child_status=%d refused_status=%d\n",
        bad_immediate,
        iters,
        bad_except,
        iters,
        bad_write,
        iters,
        bad_after,
        iters,
        bad_peer,
        iters,
        status,
        refused_status);

    return bad_immediate == 0 && bad_except == 0 && bad_write == 0 && bad_after == 0 &&
                   bad_peer == 0 && status == 0 && refused_status == 0
               ? 0
               : 1;
}
