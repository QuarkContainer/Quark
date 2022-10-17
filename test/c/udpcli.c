#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <getopt.h>
#include <errno.h>

#define PORT 8888
#define MAXLINE 1024

static void usage(const char *argv0)
{
    printf("Usage:\n");
    printf("  %s <host>     connect to server at <host>\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  -p, --port=<port>      listen on port <port> (default 8888)\n");
}

int main(int argc, char *argv[])
{
    int port = PORT;
    char *servername = NULL;
    while (1)
    {
        int c;

        static struct option long_options[] = {
            {.name = "port", .has_arg = 1, .val = 'p'},
            {}};

        c = getopt_long(argc, argv, "p:", long_options,
                        NULL);
        if (c == -1)
            break;

        switch (c)
        {
        case 'p':
            port = strtol(optarg, NULL, 0);
            if (port > 65535)
            {
                usage(argv[0]);
                return 1;
            }
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    if (optind == argc - 1)
        servername = strdupa(argv[optind]);
    else if (optind < argc)
    {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    int sockfd;
    char buffer[MAXLINE];
    char *hello = "Hello Quark from Client";
    struct sockaddr_in servaddr;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in sa;
    int sa_len;
    sa_len = sizeof(sa);
    memset(&sa, 0, sa_len);
    printf("sa_len: %d\n", sa_len);
    printf("sock is %d\n", sockfd);
    printf("after calling socket()******************************\n");
    if (getsockname(sockfd, &sa, &sa_len) == -1)
    {
        perror("getsockname() failed");
        return -1;
    }
    printf("Local IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port  for sockfd is: %d\n", (int)ntohs(sa.sin_port));

    if (getpeername(sockfd, &sa, &sa_len) == -1)
    {
        printf("getpeername, errorno: %d\n", errno);
        //   perror("getsockname() failed");
        //   return -1;
    }
    else
    {
        printf("Remote IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for sockfd is: %d\n", (int)ntohs(sa.sin_port));
    }

    memset(&servaddr, 0, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = INADDR_ANY;
    if (servername)
    {
        if (inet_pton(AF_INET, servername, &servaddr.sin_addr) <= 0)
        {
            printf("\nInvalid address/ Address not supported \n");
            return -1;
        }
    }

    int n, len;

    sendto(sockfd, (const char *)hello, strlen(hello),
           0, (const struct sockaddr *)&servaddr,
           sizeof(servaddr));
    printf("Hello Quark message sent 1...\n");

    printf("after calling sendto()******************************\n");
    memset(&sa, 0, sa_len);
    if (getsockname(sockfd, &sa, &sa_len) == -1)
    {
        perror("getsockname() failed");
        return -1;
    }
    printf("Local IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port  for sockfd is: %d\n", (int)ntohs(sa.sin_port));

    memset(&sa, 0, sa_len);
    if (getpeername(sockfd, &sa, &sa_len) == -1)
    {
        printf("getpeername, errorno: %d\n", errno);
    }
    else
    {
        printf("Remote IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for sockfd is: %d\n", (int)ntohs(sa.sin_port));
    }

    sendto(sockfd, (const char *)hello, strlen(hello),
           0, (const struct sockaddr *)&servaddr,
           sizeof(servaddr));
    printf("Hello Quark message sent 2...\n");
    sendto(sockfd, (const char *)hello, strlen(hello),
           0, (const struct sockaddr *)&servaddr,
           sizeof(servaddr));
    printf("Hello Quark message sent 3...\n");
    sendto(sockfd, (const char *)hello, strlen(hello),
           0, (const struct sockaddr *)&servaddr,
           sizeof(servaddr));
    printf("Hello Quark message sent 4...\n");
    sendto(sockfd, (const char *)hello, strlen(hello),
           0, (const struct sockaddr *)&servaddr,
           sizeof(servaddr));
    printf("Hello Quark message sent 5...\n");

	printf("after calling sendto()******************************\n");
    if (getsockname(sockfd, &sa, &sa_len) == -1)
    {
        perror("getsockname() failed");
        return -1;
    }
    printf("Local IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port  for sockfd is: %d\n", (int)ntohs(sa.sin_port));

    if (getpeername(sockfd, &sa, &sa_len) == -1)
    {
        printf("getpeername, errorno: %d\n", errno);
        //   perror("getsockname() failed");
        //   return -1;
    }
    else
    {
        printf("Remote IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for sockfd is: %d\n", (int)ntohs(sa.sin_port));
    }

    len = sizeof(servaddr);
    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                 MSG_WAITALL, (struct sockaddr *)&servaddr,
                 &len);
    buffer[n] = '\0';
    printf("Client : %s 1\n", buffer);

    printf("after calling recvfrom()******************************\n");
    memset(&sa, 0, sa_len);
    if (getsockname(sockfd, &sa, &sa_len) == -1)
    {
        perror("getsockname() failed");
        return -1;
    }
    printf("Local IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
    printf("Local port  for sockfd is: %d\n", (int)ntohs(sa.sin_port));

    memset(&sa, 0, sa_len);
    if (getpeername(sockfd, &sa, &sa_len) == -1)
    {
        printf("getpeername, errorno: %d\n", errno);
    }
    else
    {
        printf("Remote IP address for sockfd is: %s\n", inet_ntoa(sa.sin_addr));
        printf("Remote port for sockfd is: %d\n", (int)ntohs(sa.sin_port));
    }

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                 MSG_WAITALL, (struct sockaddr *)&servaddr,
                 &len);
    buffer[n] = '\0';
    printf("Client : %s 2\n", buffer);

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                 MSG_WAITALL, (struct sockaddr *)&servaddr,
                 &len);
    buffer[n] = '\0';
    printf("Client : %s 3\n", buffer);

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                 MSG_WAITALL, (struct sockaddr *)&servaddr,
                 &len);
    buffer[n] = '\0';
    printf("Client : %s 4\n", buffer);

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                 MSG_WAITALL, (struct sockaddr *)&servaddr,
                 &len);
    buffer[n] = '\0';
    printf("Client : %s 5\n", buffer);

    close(sockfd);
    return 0;
}