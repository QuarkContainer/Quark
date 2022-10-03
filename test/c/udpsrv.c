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

#define PORT 8888
#define MAXLINE 1024

static void usage(const char *argv0)
{
    printf("Usage:\n");
    printf("  %s            start a UDP server and wait for connection\n", argv0);
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
            printf("optarg: %s\n", optarg);
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

    printf("optind: %d, argc-1: %d\n", optind, argc -1);
    if (optind < argc)
    {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    int sockfd;
    char buffer[MAXLINE];
    char *hello = "Hello Quarkfrom server";
    struct sockaddr_in servaddr, cliaddr;

    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));

    servaddr.sin_family = AF_INET; // IPv4
    servaddr.sin_addr.s_addr = INADDR_ANY;
    servaddr.sin_port = htons(PORT);

    if (bind(sockfd, (const struct sockaddr *)&servaddr,
             sizeof(servaddr)) < 0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    int len, n;

    len = sizeof(cliaddr);

    n = recvfrom(sockfd, (char *)buffer, MAXLINE,
                 MSG_WAITALL, (struct sockaddr *)&cliaddr,
                 &len);
    buffer[n] = '\0';
    printf("Client : %s\n", buffer);
    sendto(sockfd, (const char *)hello, strlen(hello),
           MSG_CONFIRM, (const struct sockaddr *)&cliaddr,
           len);
    printf("Hello message sent.\n");

    return 0;
}
