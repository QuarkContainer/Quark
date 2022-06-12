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

#include <stdio.h>
#include <stdlib.h>
#include <wait.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/syscall.h>

pid_t pid;
int counter = 0;
void handler1(int sig)
{
    counter++;
    printf("counter..... = %d\n", counter);
    /* Flushes the printed string to stdout */
    fflush(stdout);
    kill(pid, SIGUSR1);
}
void handler2(int sig)
{
    counter += 3;
    printf("child: counter = %d\n", counter);
}

int main()
{
    pid_t p;
    int status;

    int cpu = 1;
    syscall(SYS_getcpu, &cpu, 0, 0);
    printf("the cpu is %d \n", cpu);

    signal(SIGUSR1, handler1);
    if ((pid = fork()) == 0)
    {
        printf("in child, pid is %d, tid is %ld\n", getpid(), syscall(SYS_gettid));
        signal(SIGUSR1, handler2);
        kill(getppid(), SIGUSR1);
        int remain = sleep(3);
        printf("sleep remain is %d\n", remain);
        exit(0x2);
    }
    printf("in parent, pid is %d, tid is %ld\n", getpid(), syscall(SYS_gettid));
    if ((p = wait(&status)) > 0)
    {
        counter += 4;
        printf("counter = %d, status is %d\n", counter, status);
    }

    exit(456);
}