#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

void TestZero() {
    char buf[10];
    int bytesRead;
    int fp;
    fp = open("/dev/zero", O_RDWR, "rb");

    bytesRead = read(fp, buf, 10);
    for(int i=0; i<10; i++) {
        if (buf[i] != 0) {
            printf("get non zero value from zero device, index is %d, value is %d\n", i, buf[i]);
            exit(1);
        }
    }

    printf("get 10 zero from /dev/zero\n");

    close(fp);
}

void TestNull() {
    char buf[10];
    int bytesWrite;
    int fp;
    fp = open("/dev/null", O_RDWR, "rb");

    bytesWrite = write(fp, buf, 10);
    if (bytesWrite != 10) {
        printf("write %d zero to /dev/null, fail.....\n", bytesWrite);
        exit(1);
    }

    printf("write %d zero to /dev/null, success.....\n", bytesWrite);

    close(fp);
}

void TestFull() {
    char buf[10];
    int bytesWrite;
    int fp;
    fp = open("/dev/full", O_RDWR, "rb");

    bytesWrite = write(fp, buf, 10);
    if (bytesWrite != -1 || errno!=28 /*ENOSPC*/) {
        printf("write %d zero to /dev/full, fail.....\n", bytesWrite);
        exit(1);
    }

    printf("write %d zero to /dev/full, errno is %d success.....\n", 0, errno);

    close(fp);
}

void TestProcFd() {
    char *buf = "test 123\n";
    char *dev = "/proc/self/fd/1";
    int bytesWrite;
    int fp;
    fp = open(dev, O_RDWR, "rb");

    bytesWrite = write(fp, buf, strlen(buf));
    if (bytesWrite != strlen(buf)) {
        printf("write to fd %d,  '%s' to /proc/self/1, fail..... errno is %d \n", fp, buf, errno);
        exit(1);
    }

    printf("write str '%s' to %s, success.....\n", buf, dev);

    close(fp);
}

void main(int argc, char * argv[])
{
    TestZero();
    TestNull();
    TestFull();
    TestProcFd();
    exit(0);
}

