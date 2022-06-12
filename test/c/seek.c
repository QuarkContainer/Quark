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