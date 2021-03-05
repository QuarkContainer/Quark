#include <stdio.h>
#include <sys/types.h> 
#include <unistd.h>
#include<sys/wait.h>

void forkexample() 
{ 
    // child process because return value zero 
    if (fork() == 0) 
        printf("Hello from Child!\n"); 
  
    // parent process because return value non-zero. 
    else {
        printf("Hello from Parent!\n");
        wait(0);
    }
} 
int main() 
{ 
    forkexample(); 
    return 0; 
}
