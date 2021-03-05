#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int
main(int argc, char *argv[])
{
   int pipefd[2];
   pid_t cpid;
   char buf[1024];

   if (pipe(pipefd) == -1) {
       perror("pipe");
       exit(EXIT_FAILURE);
   }

   cpid = fork();
   if (cpid == -1) {
       perror("fork");
       exit(EXIT_FAILURE);
   }

   if (cpid == 0) {    /* Child reads from pipe */
       close(pipefd[1]);          /* Close unused write end */

       int size;
       while ((size = read(pipefd[0], buf, 1024)) > 0) {
            printf("out %d bytes \n", size);
            write(STDOUT_FILENO, buf, size);
       }

       write(STDOUT_FILENO, "\n", 1);
       close(pipefd[0]);
       _exit(EXIT_SUCCESS);

   } else {            /* Parent writes argv[1] to pipe */
       close(pipefd[0]);          /* Close unused read end */
       char* buff = "01234567890\n";
       for (int i =0; i<1000; i++) {
            write(pipefd[1], buff, strlen(buff));
       }
       close(pipefd[1]);          /* Reader will see EOF */
       wait(NULL);                /* Wait for child */
       exit(EXIT_SUCCESS);
   }
}