#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

char *socket_path = "/tmp/socket";
//char *socket_path = "/test/c/socket";
//char *socket_path = "./socket";
//char *socket_path = "\0hidden";

int main(int argc, char *argv[]) {
  struct sockaddr_un addr;
  //char buf[100];
  int fd,rc;

  if (argc > 1) socket_path=argv[1];

  if ( (fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    perror("socket error");
    exit(-1);
  }

  printf("the socket_path is %s\n", socket_path);

  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (*socket_path == '\0') {
    *addr.sun_path = '\0';
    strncpy(addr.sun_path+1, socket_path+1, sizeof(addr.sun_path)-2);
  } else {
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path)-1);
  }

  if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    perror("connect error");
    exit(-1);
  }

  char *buf = "01234567890123456789012345678901234567890123456789012345678901234567890123456789\n";
  rc = strlen(buf);
  //rc=read(STDIN_FILENO, buf, strlen(buf)+1);
  for (int i=0; i<1000; i++) {
    if (write(fd, buf, rc) != rc) {
      if (rc > 0) fprintf(stderr,"partial write");
      else {
        perror("write error");
        exit(-1);
      }
    }
  }

  return 0;
}

