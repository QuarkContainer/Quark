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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define DEFAULT_ITERS 8

static int write_all(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, buf, len);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("write");
            return 1;
        }

        buf += n;
        len -= (size_t)n;
    }

    return 0;
}

static int recv_reply(int fd, int expect_fd) {
    char buf[4096];
    char control[CMSG_SPACE(sizeof(int))];
    size_t used = 0;
    int received_fd = -1;

    while (used < sizeof(buf)) {
        struct iovec iov;
        memset(&iov, 0, sizeof(iov));
        iov.iov_base = buf + used;
        iov.iov_len = sizeof(buf) - used;

        struct msghdr msg;
        memset(&msg, 0, sizeof(msg));
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = sizeof(control);

        ssize_t n = recvmsg(fd, &msg, 0);
        if (n == 0) {
            fprintf(stderr, "EOF before newline, partial length=%zu\n", used);
            if (received_fd >= 0) {
                close(received_fd);
            }
            return 2;
        }
        if (n < 0) {
            perror("recvmsg");
            if (received_fd >= 0) {
                close(received_fd);
            }
            return 1;
        }

        for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
             cmsg != NULL;
             cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
                cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
                memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(received_fd));
            }
        }

        used += (size_t)n;
        if (memchr(buf, '\n', used) != NULL) {
            if (buf[0] != '{') {
                fprintf(stderr, "reply has unexpected leading byte 0x%02x\n",
                        (unsigned char)buf[0]);
                if (received_fd >= 0) {
                    close(received_fd);
                }
                return 3;
            }

            if (expect_fd) {
                struct stat st;
                if (received_fd < 0 || fstat(received_fd, &st) < 0) {
                    perror("received fd");
                    if (received_fd >= 0) {
                        close(received_fd);
                    }
                    return 4;
                }
            }

            if (received_fd >= 0) {
                close(received_fd);
            }
            return 0;
        }
    }

    fprintf(stderr, "reply exceeded buffer without newline\n");
    if (received_fd >= 0) {
        close(received_fd);
    }
    return 1;
}

static int run_once(const char *path, int expect_fd) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "socket path too long: %s\n", path);
        close(fd);
        return 1;
    }
    strcpy(addr.sun_path, path);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(fd);
        return 1;
    }

    const char *request = "{\"op\":\"ping\"}\n";
    if (write_all(fd, request, strlen(request)) != 0) {
        close(fd);
        return 1;
    }

    int ret = recv_reply(fd, expect_fd);
    close(fd);
    return ret;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <unix-socket-path> [iterations] [expect-fd]\n", argv[0]);
        return 1;
    }

    int iters = argc == 3 ? atoi(argv[2]) : DEFAULT_ITERS;
    if (iters <= 0) {
        fprintf(stderr, "iterations must be positive\n");
        return 1;
    }
    int expect_fd = 0;
    if (argc == 4) {
        if (strcmp(argv[3], "expect-fd") != 0) {
            fprintf(stderr, "unknown mode: %s\n", argv[3]);
            return 1;
        }
        expect_fd = 1;
    }

    for (int i = 0; i < iters; i++) {
        int ret = run_once(argv[1], expect_fd);
        if (ret != 0) {
            fprintf(stderr, "host_uds_recv failed at iteration %d\n", i + 1);
            return ret;
        }
    }

    printf("host_uds_recv: %d iterations ok\n", iters);
    return 0;
}
