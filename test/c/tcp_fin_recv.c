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

#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define DEFAULT_HOST "127.0.0.1"
#define DEFAULT_PORT 47050
#define DEFAULT_ITERS 32

static int run_once(const char *host, int port, const char *expected) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "invalid IPv4 address: %s\n", host);
        close(fd);
        return 1;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(fd);
        return 1;
    }

    if (write(fd, "ping\n", 5) != 5) {
        perror("write");
        close(fd);
        return 1;
    }

    char buf[64];
    ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n == 0) {
        fprintf(stderr, "first recv returned EOF\n");
        close(fd);
        return 2;
    }
    if (n < 0) {
        perror("recv");
        close(fd);
        return 1;
    }
    if (expected != NULL && ((size_t)n != strlen(expected) || memcmp(buf, expected, (size_t)n) != 0)) {
        fprintf(stderr, "first recv returned unexpected payload length=%zd\n", n);
        close(fd);
        return 3;
    }

    n = recv(fd, buf, sizeof(buf), 0);
    if (n != 0) {
        if (n < 0) {
            perror("second recv");
        } else {
            fprintf(stderr, "second recv returned data after expected EOF length=%zd\n", n);
        }
        close(fd);
        return 4;
    }

    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    const char *host = argc > 1 ? argv[1] : DEFAULT_HOST;
    int port = argc > 2 ? atoi(argv[2]) : DEFAULT_PORT;
    int iters = argc > 3 ? atoi(argv[3]) : DEFAULT_ITERS;
    const char *expected = argc > 4 ? argv[4] : NULL;

    if (port <= 0 || iters <= 0) {
        fprintf(stderr, "usage: %s [host] [port] [iterations] [expected-payload]\n", argv[0]);
        return 1;
    }

    for (int i = 0; i < iters; i++) {
        int ret = run_once(host, port, expected);
        if (ret != 0) {
            fprintf(stderr, "tcp_fin_recv failed at iteration %d\n", i + 1);
            return ret;
        }
    }

    printf("tcp_fin_recv: %d iterations ok\n", iters);
    return 0;
}
