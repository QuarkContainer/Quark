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

/*#include <stdio.h>
#include <unistd.h>
#include <sys/poll.h>

#define TIMEOUT 5

int main (void)
{
	struct pollfd fds[2];
	int ret;

	fds[0].fd = STDIN_FILENO;
	fds[0].events = POLLIN;

	fds[1].fd = STDOUT_FILENO;
	fds[1].events = POLLOUT;

	ret = poll(fds, 2, TIMEOUT * 1000);

	if (ret == -1) {
		perror ("poll");
		return 1;
	}

	if (!ret) {
		printf ("%d seconds elapsed.\n", TIMEOUT);
		return 0;
	}

	if (fds[0].revents & POLLIN)
		printf ("stdin is readable\n");

	if (fds[1].revents & POLLOUT)
		printf ("stdout is writable\n");

	return 0;

}*/


#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <errno.h>
#include <wait.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>

#define SECS_IN_DAY (24 * 60 * 60)

void handler1(int sig)
{
    char *hello = "Hello from signal\n";
    //write(1, hello, strlen(hello));
    printf("sig..... = %d\n", sig);
    fflush(stdout);
}

static void
displayClock(clockid_t clock, const char *name, bool showRes)
{
   struct timespec ts;

   if (clock_gettime(clock, &ts) == -1) {
       perror("clock_gettime");
       exit(EXIT_FAILURE);
   }

   printf("%-15s: %10jd.%03ld (", name,
           (intmax_t) ts.tv_sec, ts.tv_nsec / 1000000);

   long days = ts.tv_sec / SECS_IN_DAY;
   if (days > 0)
       printf("%ld days + ", days);

   printf("%2dh %2dm %2ds",
           (int) (ts.tv_sec % SECS_IN_DAY) / 3600,
           (int) (ts.tv_sec % 3600) / 60,
           (int) ts.tv_sec % 60);
   printf(")\n");

   if (clock_getres(clock, &ts) == -1) {
       perror("clock_getres");
       exit(EXIT_FAILURE);
   }

   if (showRes)
       printf("     resolution: %10jd.%09ld\n",
               (intmax_t) ts.tv_sec, ts.tv_nsec);
}

int main( ) {
    bool showRes = true;
    displayClock(CLOCK_REALTIME, "CLOCK_REALTIME", showRes);
    displayClock(CLOCK_MONOTONIC, "CLOCK_MONOTONIC", showRes);

   int c = 1;

   //signal(SIGUSR1, handler1);
   //kill(getpid(), SIGUSR1);

   printf( "Enter a value :");
   c = getchar( );

   printf( "\nYou entered: ");
   putchar( c );
   printf( "\n");

   return 123;
}