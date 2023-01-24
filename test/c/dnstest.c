#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>

void gethostbynamecall(char* name) {
   struct hostent *hp = gethostbyname(name);

    if (hp == NULL) {
       printf("gethostbyname() failed\n");
    } else {
       printf("%s = ", hp->h_name);
       unsigned int i=0;
       while ( hp -> h_addr_list[i] != NULL) {
          printf( "%s ", inet_ntoa( *( struct in_addr*)( hp -> h_addr_list[i])));
          i++;
       }
       printf("\n");
    }
}

int getaddrinfocall(char* host) {
   struct addrinfo hints, *res, *result;
  int errcode;
  char addrstr[100];
  void *ptr;

  memset (&hints, 0, sizeof (hints));
  hints.ai_family = PF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags |= AI_CANONNAME;

  errcode = getaddrinfo (host, NULL, &hints, &result);
  if (errcode != 0)
    {
      printf("getaddrinfo ret is %d\n", errcode);
      perror ("getaddrinfo");
      return -1;
    }
  
  res = result;

  printf ("Host: %s\n", host);
  while (res)
    {
      inet_ntop (res->ai_family, res->ai_addr->sa_data, addrstr, 100);

      switch (res->ai_family)
        {
        case AF_INET:
          ptr = &((struct sockaddr_in *) res->ai_addr)->sin_addr;
          break;
        case AF_INET6:
          ptr = &((struct sockaddr_in6 *) res->ai_addr)->sin6_addr;
          break;
        }
      inet_ntop (res->ai_family, ptr, addrstr, 100);
      printf ("IPv%d address: %s (%s)\n", res->ai_family == PF_INET6 ? 6 : 4,
              addrstr, res->ai_canonname);
      res = res->ai_next;
    }
  
  freeaddrinfo(result);

  return 0;

}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s hostname", argv[0]);
        exit(-1);
    }

    // int i;
    // struct hostent *he;
    // struct in_addr **addr_list;
    // struct in_addr addr;

    // // get the addresses of www.yahoo.com:

    // he = gethostbyname("www.yahoo.com");
    // if (he == NULL) { // do some error checking
    //     herror("gethostbyname"); // herror(), NOT perror()
    //     exit(1);
    // }

    // // print information about this host:
    // printf("Official name is: %s\n", he->h_name);
    // printf("IP address: %s\n", inet_ntoa(*(struct in_addr*)he->h_addr));
    // printf("All addresses: ");
    // addr_list = (struct in_addr **)he->h_addr_list;
    // for(i = 0; addr_list[i] != NULL; i++) {
    //     printf("%s ", inet_ntoa(*addr_list[i]));
    // }
    // printf("\n");

    // // get the host name of 66.94.230.32:

    // inet_aton("66.94.230.32", &addr);
    // he = gethostbyaddr(&addr, sizeof(addr), AF_INET);

    // printf("Host name: %s\n", he->h_name);

    // struct hostent *hp = gethostbyname("www.google.com");
    getaddrinfocall(argv[1]);
    
}

