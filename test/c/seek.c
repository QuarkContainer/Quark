// C program to read nth byte of a file and
// copy it to another file using lseek
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>

int main()
{
    int file=0;
    char* filename = "./testfile1234.txt";
    if((file=open(filename,O_RDWR | O_CREAT | O_TRUNC )) < -1) {
        printf("open fail ...");
        return 1;
    }

    int len = strlen(filename) + 1;
    printf("write %ld bytes\n", write (file, filename, len));

    if(lseek(file, 0, SEEK_SET) < 0) {
      printf("seek1 fail ...\n");
      return 1;
    }

    char buffer[100];
    int n;
    if((n=read(file,buffer,100)) != len)  {
        printf("read fail ...  file is %d, len is %d , n is %d, errno is %d\n", file, len, n, errno);
        return 1;
    }
    printf("%s\n",buffer);

    if(lseek(file, 10, SEEK_SET) < 0) {
      printf("seek fail ...\n");
      return 1;
    }

    if(read(file,buffer,100) < 0) {
      printf("read2 fail ...\n");
      return 1;
    }
    printf("%s\n",buffer);

    printf("current offset is %ld\n", lseek(file, 0, SEEK_CUR));
    ftruncate(file, 10);
    printf("after truncate current offset is %ld\n", lseek(file, 0, SEEK_CUR));

    close(file);
    return 0;
}